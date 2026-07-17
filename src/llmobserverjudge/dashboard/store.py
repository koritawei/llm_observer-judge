"""Query-side stores for Dashboard (in-memory stand-in for CH / OTel)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class EvalResultRecord:
    trace_id: str
    tenant_id: str
    final_score: float
    bad_case_detected: bool
    eval_scores: list[dict[str, Any]]
    eval_execution_layers: list[list[str]]
    eval_results: list[dict[str, Any]]
    node_errors: dict[str, str]
    optional_skipped: list[str]
    prompt_version_id: str | None = None
    execution_graph: list[dict[str, Any]] = field(default_factory=list)
    project_id: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
            "final_score": self.final_score,
            "bad_case_detected": self.bad_case_detected,
            "eval_scores": self.eval_scores,
            "eval_execution_layers": self.eval_execution_layers,
            "eval_results": self.eval_results,
            "node_errors": self.node_errors,
            "optional_skipped": self.optional_skipped,
            "prompt_version_id": self.prompt_version_id,
            "execution_graph": self.execution_graph,
            "created_at": self.created_at,
        }


@dataclass
class EvalResultStore:
    _by_trace: dict[str, EvalResultRecord] = field(default_factory=dict)
    _by_key: dict[str, str] = field(default_factory=dict)

    def _agg_key(self, tenant_id: str, project_id: str, trace_id: str) -> str:
        return f"{tenant_id}:{project_id}:{trace_id}"

    async def save(
        self, record: EvalResultRecord, *, idempotency_key: str | None = None
    ) -> None:
        project_id = getattr(record, "project_id", None) or "default"
        record.project_id = project_id
        key = self._agg_key(record.tenant_id, project_id, record.trace_id)
        self._by_trace[key] = record
        # also index by bare trace for backward-compatible get(trace_id)
        self._by_trace[record.trace_id] = record
        if idempotency_key:
            self._by_key[idempotency_key] = key

    async def get(
        self, trace_id: str, *, tenant_id: str | None = None, project_id: str | None = None
    ) -> EvalResultRecord | None:
        if tenant_id is not None and project_id is not None:
            return self._by_trace.get(self._agg_key(tenant_id, project_id, trace_id))
        return self._by_trace.get(trace_id)

    async def get_by_idempotency(self, key: str) -> EvalResultRecord | None:
        tid = self._by_key.get(key)
        return self._by_trace.get(tid) if tid else None

    async def list_all(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        limit: int = 500,
    ) -> list[EvalResultRecord]:
        # Prefer composite keys (contain ':')
        seen: set[str] = set()
        rows: list[EvalResultRecord] = []
        for k, r in self._by_trace.items():
            if ":" not in k and k == r.trace_id:
                # skip bare-trace duplicates when composite exists
                composite = self._agg_key(
                    r.tenant_id, getattr(r, "project_id", None) or "default", r.trace_id
                )
                if composite in self._by_trace:
                    continue
            identity = self._agg_key(
                r.tenant_id, getattr(r, "project_id", None) or "default", r.trace_id
            )
            if identity in seen:
                continue
            seen.add(identity)
            if tenant_id is not None and r.tenant_id != tenant_id:
                continue
            if project_id is not None and (getattr(r, "project_id", None) or "default") != project_id:
                continue
            rows.append(r)
        rows.sort(key=lambda r: r.created_at, reverse=True)
        return rows[:limit]


@dataclass
class SamplingStatsStore:
    """Counters for sampling panel (design §10.2)."""

    total_seen: int = 0
    sampled: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    async def record(self, *, sampled: bool, reason: str | None) -> None:
        self.total_seen += 1
        if sampled:
            self.sampled += 1
        key = reason or ("sampled" if sampled else "dropped")
        self.by_reason[key] = self.by_reason.get(key, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        rate = (self.sampled / self.total_seen) if self.total_seen else 0.0
        return {
            "total_seen": self.total_seen,
            "sampled": self.sampled,
            "effective_sample_rate": rate,
            "by_reason": dict(self.by_reason),
        }


@dataclass
class DashboardContext:
    """Shared handles wired by DevStack / tests."""

    span_store: Any
    eval_results: EvalResultStore
    sampling: SamplingStatsStore
    annotations: Any = None
    attributions: Any = None
    benchmark_store: Any = None
    event_bus: Any = None
    mq: Any = None
    telemetry: Any = None
    stack_info: Any = None  # callable → dict
    prompt_service: Any = None
