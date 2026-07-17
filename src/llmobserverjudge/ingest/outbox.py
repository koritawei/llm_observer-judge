"""Transactional Outbox protocol (ADR-0011 / SDD-IG-13).

Ingest writes TraceClosed intent only — Dispatch is sole eval_jobs publisher.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from llmobserverjudge.shared.deployment import resolve_outbox_claim_lease_seconds


@runtime_checkable
class OutboxStore(Protocol):
    async def insert_trace_closed(
        self,
        *,
        tenant_id: str,
        trace_id: str,
        trace_summary: dict[str, Any],
        prompt_version_id: str | None = None,
        project_id: str | None = None,
    ) -> str: ...

    async def fetch_unpublished(self, limit: int = 100) -> list[dict[str, Any]]: ...

    async def claim_unpublished(self, limit: int = 100) -> list[dict[str, Any]]: ...

    async def mark_published(self, row_id: str) -> bool: ...

    async def release_claim(self, row_id: str) -> None: ...

    async def unclaim(self, row_id: str) -> None: ...

    async def bump_attempt(self, row_id: str) -> None: ...


@dataclass
class InMemoryOutbox:
    """Memory driver. Production: PostgreSQL pending_eval_jobs."""

    _rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    _claim_lease_s: int = field(default_factory=resolve_outbox_claim_lease_seconds)

    def _claimable(self, row: dict[str, Any]) -> bool:
        if row.get("published"):
            return False
        claimed = row.get("claimed_at")
        if claimed is None:
            return True
        if isinstance(claimed, str):
            claimed = datetime.fromisoformat(claimed)
        stale = datetime.now(UTC) - timedelta(seconds=self._claim_lease_s)
        return claimed < stale

    async def insert_trace_closed(
        self,
        *,
        tenant_id: str,
        trace_id: str,
        trace_summary: dict[str, Any],
        prompt_version_id: str | None = None,
        project_id: str | None = None,
    ) -> str:
        pid = project_id or str(trace_summary.get("project_id") or "default")
        key = f"{tenant_id}:{pid}:{trace_id}"
        existing = self._rows.get(key)
        if existing is not None:
            old = dict(existing["trace_summary"])
            old_tags = list(old.get("tags") or [])
            new_tags = list(trace_summary.get("tags") or [])
            summary = {**old, **trace_summary}
            tags = list({*old_tags, *new_tags})
            if tags:
                summary["tags"] = tags
            summary["project_id"] = pid
            existing["trace_summary"] = summary
            existing["project_id"] = pid
            return existing["id"]

        row_id = str(uuid.uuid4())
        summary = dict(trace_summary)
        summary["project_id"] = pid
        self._rows[key] = {
            "id": row_id,
            "tenant_id": tenant_id,
            "project_id": pid,
            "trace_id": trace_id,
            "trace_summary": summary,
            "prompt_version_id": prompt_version_id,
            "published": False,
            "attempt_count": 0,
            "claimed_at": None,
        }
        return row_id

    async def fetch_unpublished(self, limit: int = 100) -> list[dict[str, Any]]:
        out = [r for r in self._rows.values() if self._claimable(r)]
        out.sort(key=lambda r: r["id"])
        return [dict(r) for r in out[:limit]]

    async def claim_unpublished(self, limit: int = 100) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        out = [r for r in self._rows.values() if self._claimable(r)]
        out.sort(key=lambda r: r["id"])
        claimed = out[:limit]
        result: list[dict[str, Any]] = []
        for row in claimed:
            row["claimed_at"] = now
            result.append(dict(row))
        return result

    async def mark_published(self, row_id: str) -> bool:
        for row in self._rows.values():
            if row["id"] == row_id:
                if row["published"]:
                    return False
                row["published"] = True
                row["claimed_at"] = None
                return True
        return False

    async def release_claim(self, row_id: str) -> None:
        for row in self._rows.values():
            if row["id"] == row_id and not row["published"]:
                row["claimed_at"] = None
                return

    async def unclaim(self, row_id: str) -> None:
        await self.release_claim(row_id)

    async def bump_attempt(self, row_id: str) -> None:
        for row in self._rows.values():
            if row["id"] == row_id:
                row["attempt_count"] = int(row.get("attempt_count", 0)) + 1
                return
