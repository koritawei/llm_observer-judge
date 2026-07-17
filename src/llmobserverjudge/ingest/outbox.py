"""Transactional Outbox protocol (ADR-0011 / SDD-IG-13).

Ingest writes TraceClosed intent only — Dispatch is sole eval_jobs publisher.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OutboxStore(Protocol):
    async def insert_trace_closed(
        self,
        *,
        tenant_id: str,
        trace_id: str,
        trace_summary: dict[str, Any],
        prompt_version_id: str | None = None,
    ) -> str: ...

    async def fetch_unpublished(self, limit: int = 100) -> list[dict[str, Any]]: ...

    async def mark_published(self, row_id: str) -> None: ...

    async def bump_attempt(self, row_id: str) -> None: ...


@dataclass
class InMemoryOutbox:
    """Memory driver. Production: PostgreSQL pending_eval_jobs."""

    _rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def insert_trace_closed(
        self,
        *,
        tenant_id: str,
        trace_id: str,
        trace_summary: dict[str, Any],
        prompt_version_id: str | None = None,
    ) -> str:
        key = f"{tenant_id}:{trace_id}"
        existing = self._rows.get(key)
        if existing is not None:
            # Merge summary tags/span refs; keep same id
            old = dict(existing["trace_summary"])
            old_tags = list(old.get("tags") or [])
            new_tags = list(trace_summary.get("tags") or [])
            summary = {**old, **trace_summary}
            tags = list({*old_tags, *new_tags})
            if tags:
                summary["tags"] = tags
            existing["trace_summary"] = summary
            return existing["id"]

        row_id = str(uuid.uuid4())
        self._rows[key] = {
            "id": row_id,
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "trace_summary": dict(trace_summary),
            "prompt_version_id": prompt_version_id,
            "published": False,
            "attempt_count": 0,
        }
        return row_id

    async def fetch_unpublished(self, limit: int = 100) -> list[dict[str, Any]]:
        out = [r for r in self._rows.values() if not r["published"]]
        out.sort(key=lambda r: r["id"])
        return out[:limit]

    async def mark_published(self, row_id: str) -> None:
        for row in self._rows.values():
            if row["id"] == row_id:
                row["published"] = True
                return

    async def bump_attempt(self, row_id: str) -> None:
        for row in self._rows.values():
            if row["id"] == row_id:
                row["attempt_count"] = int(row.get("attempt_count", 0)) + 1
                return
