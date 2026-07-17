"""In-memory checkpoint for ReAct subgraph (ADR-0004)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryCheckpointStore:
    """Dev/test stand-in for PostgresSaver — only used by Attribution ReAct."""

    _rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _key(
        self, trace_id: str, *, tenant_id: str = "", project_id: str = "default"
    ) -> str:
        return f"{tenant_id}:{project_id}:{trace_id}"

    async def save(
        self,
        trace_id: str,
        payload: dict[str, Any],
        *,
        tenant_id: str = "",
        project_id: str = "default",
    ) -> None:
        self._rows[self._key(trace_id, tenant_id=tenant_id, project_id=project_id)] = (
            dict(payload)
        )

    async def load(
        self,
        trace_id: str,
        *,
        tenant_id: str = "",
        project_id: str = "default",
    ) -> dict[str, Any] | None:
        row = self._rows.get(
            self._key(trace_id, tenant_id=tenant_id, project_id=project_id)
        )
        return dict(row) if row else None
