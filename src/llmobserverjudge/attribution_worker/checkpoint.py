"""In-memory checkpoint for ReAct subgraph (ADR-0004)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InMemoryCheckpointStore:
    """Dev/test stand-in for PostgresSaver — only used by Attribution ReAct."""

    _rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def save(self, trace_id: str, payload: dict[str, Any]) -> None:
        self._rows[trace_id] = dict(payload)

    async def load(self, trace_id: str) -> dict[str, Any] | None:
        row = self._rows.get(trace_id)
        return dict(row) if row else None
