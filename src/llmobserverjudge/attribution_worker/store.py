"""Attribution result persistence protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from llmobserverjudge.contracts.models import AttributionResult


@runtime_checkable
class AttributionStore(Protocol):
    async def save(
        self, result: AttributionResult, *, idempotency_key: str | None = None
    ) -> None: ...

    async def get_by_idempotency(
        self, idempotency_key: str
    ) -> AttributionResult | None: ...

    async def list_for_trace(
        self,
        tenant_id: str,
        trace_id: str,
        *,
        project_id: str | None = None,
    ) -> list[AttributionResult]: ...


@dataclass
class InMemoryAttributionStore:
    _by_id: dict[str, AttributionResult] = field(default_factory=dict)
    _by_key: dict[str, str] = field(default_factory=dict)

    async def save(self, result: AttributionResult, *, idempotency_key: str | None = None) -> None:
        self._by_id[result.attribution_id] = result
        if idempotency_key:
            self._by_key[idempotency_key] = result.attribution_id

    async def get_by_idempotency(self, idempotency_key: str) -> AttributionResult | None:
        aid = self._by_key.get(idempotency_key)
        if not aid:
            return None
        return self._by_id.get(aid)

    async def list_for_trace(
        self,
        tenant_id: str,
        trace_id: str,
        *,
        project_id: str | None = None,
    ) -> list[AttributionResult]:
        rows = [
            r
            for r in self._by_id.values()
            if r.tenant_id == tenant_id and r.trace_id == trace_id
        ]
        if project_id is not None:
            rows = [
                r
                for r in rows
                if (getattr(r, "project_id", None) or "default") == project_id
            ]
        return rows

    async def list_recent(self, *, limit: int = 500) -> list[AttributionResult]:
        rows = list(self._by_id.values())
        return rows[:limit]
