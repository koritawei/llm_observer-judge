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
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
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

    async def save(self, record: EvalResultRecord) -> None:
        self._by_trace[record.trace_id] = record

    async def get(self, trace_id: str) -> EvalResultRecord | None:
        return self._by_trace.get(trace_id)

    async def list_all(
        self, *, tenant_id: str | None = None, limit: int = 500
    ) -> list[EvalResultRecord]:
        rows = list(self._by_trace.values())
        if tenant_id is not None:
            rows = [r for r in rows if r.tenant_id == tenant_id]
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
