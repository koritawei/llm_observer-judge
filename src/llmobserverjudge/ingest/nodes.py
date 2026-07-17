"""Ingest pipeline nodes: Sampling → Index → Write (SDD-IG-12)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.shared.mq import MessageQueue
from llmobserverjudge.shared.storage import SpanStore


@dataclass
class IngestState:
    span: dict[str, Any]
    sampled: bool = True
    sample_reason: str = "default"
    storage_location: str | None = None
    written: bool = False
    consumer_lag: int = 0
    effective_sample_rate: float = 1.0
    errors: list[str] = field(default_factory=list)

    @property
    def tenant_id(self) -> str:
        return str(self.span["tenant_id"])

    @property
    def trace_id(self) -> str:
        return str(self.span["trace_id"])

    @property
    def span_id(self) -> str:
        return str(self.span["span_id"])

    def trace_summary(self) -> dict[str, Any]:
        tags = self.span.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        # Keep prompt/response for Judge (inline spans); empty when offloaded to OSS.
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "tenant_id": self.tenant_id,
            "project_id": str(self.span.get("project_id") or "default"),
            "name": self.span.get("name"),
            "status": self.span.get("status"),
            "model": self.span.get("model"),
            "tags": list(tags),
            "has_error": self.span.get("status") == "ERROR",
            "storage_location": self.storage_location,
            "sample_reason": self.sample_reason,
            "prompt": self.span.get("prompt") or "",
            "response": self.span.get("response") or "",
            "kind": self.span.get("kind"),
        }


class IngestNode(Protocol):
    async def run(self, state: IngestState) -> IngestState: ...


@dataclass
class SamplingConfig:
    rate: float = 1.0
    event_tags: frozenset[str] = frozenset({"safety", "user_feedback"})


class SamplingNode:
    """Error-first / event-trigger / adaptive lag hysteresis."""

    LAG_HIGH = 10_000
    LAG_LOW = 2_000

    def __init__(
        self,
        mq: MessageQueue,
        *,
        config: SamplingConfig | None = None,
        force_sample: bool | None = None,
    ) -> None:
        self.mq = mq
        self.config = config or SamplingConfig()
        self.force_sample = force_sample
        self._last_rate = self.config.rate

    async def run(self, state: IngestState) -> IngestState:
        if self.force_sample is not None:
            state.sampled = self.force_sample
            state.sample_reason = "forced"
            state.effective_sample_rate = 1.0 if self.force_sample else 0.0
            return state

        lag = await self.mq.consumer_lag(Topic.SPANS)
        state.consumer_lag = lag

        if state.span.get("status") == "ERROR" or state.span.get("has_error"):
            state.sampled = True
            state.sample_reason = "error_first"
            state.effective_sample_rate = 1.0
            return state

        tags = state.span.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if any(t in self.config.event_tags for t in tags):
            state.sampled = True
            state.sample_reason = "event_triggered"
            state.effective_sample_rate = 1.0
            return state

        # Honor client sampled=False only when not overridden by rules above
        if state.span.get("sampled") is False:
            state.sampled = False
            state.sample_reason = "client_unsampled"
            state.effective_sample_rate = 0.0
            return state

        base = self.config.rate
        if lag > self.LAG_HIGH:
            effective = min(base, 0.1)
        elif lag < self.LAG_LOW:
            effective = base
        else:
            effective = self._last_rate or base
        self._last_rate = effective
        state.effective_sample_rate = effective

        digest = hashlib.md5(state.trace_id.encode()).hexdigest()
        hash_val = int(digest, 16) % 10000
        state.sampled = hash_val < int(effective * 10000)
        state.sample_reason = "adaptive" if state.sampled else "skipped"
        return state


class IndexNode:
    """Large payload → OSS location; CH keeps summary."""

    INLINE_MAX_BYTES = 1_000_000  # 1MB

    def __init__(self, *, inline_max_bytes: int = INLINE_MAX_BYTES) -> None:
        self.inline_max_bytes = inline_max_bytes

    async def run(self, state: IngestState) -> IngestState:
        prompt = str(state.span.get("prompt") or "")
        response = str(state.span.get("response") or "")
        size = len(prompt.encode()) + len(response.encode())
        if size > self.inline_max_bytes:
            state.storage_location = (
                f"oss://llmjudge/{state.tenant_id}/{state.trace_id}/{state.span_id}.json"
            )
            # Strip bulky fields from durable CH row (kept in OSS conceptually)
            state.span = {
                **state.span,
                "prompt": "",
                "response": "",
                "storage_location": state.storage_location,
                "attributes": {
                    **(state.span.get("attributes") or {}),
                    "offloaded_bytes": size,
                },
            }
        return state


class WriteNode:
    """Persist span to SpanStore (memory/CH)."""

    def __init__(self, store: SpanStore) -> None:
        self.store = store

    async def run(self, state: IngestState) -> IngestState:
        row = dict(state.span)
        if state.storage_location:
            row["storage_location"] = state.storage_location
        row["sampled"] = state.sampled
        row["sample_reason"] = state.sample_reason
        await self.store.write_span(row)
        state.written = True
        return state
