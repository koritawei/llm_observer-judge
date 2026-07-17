"""Trace context for builtin attribution tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttributionContext:
    """Working set for a single attribution job."""

    tenant_id: str
    trace_id: str
    trace_summary: dict[str, Any] = field(default_factory=dict)
    spans: list[dict[str, Any]] = field(default_factory=list)
    score_snapshot: dict[str, Any] = field(default_factory=dict)
    eval_agent_id: str = ""


@dataclass
class InMemoryAttributionContextStore:
    """Optional enrichment keyed by tenant:trace (tests / future SpanStore join)."""

    _rows: dict[str, AttributionContext] = field(default_factory=dict)

    def _key(self, tenant_id: str, trace_id: str) -> str:
        return f"{tenant_id}:{trace_id}"

    async def put(self, ctx: AttributionContext) -> None:
        self._rows[self._key(ctx.tenant_id, ctx.trace_id)] = ctx

    async def get(self, tenant_id: str, trace_id: str) -> AttributionContext | None:
        return self._rows.get(self._key(tenant_id, trace_id))
