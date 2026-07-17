"""PostgreSQL/SQLite OutboxStore — pending_eval_jobs with claim semantics."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.shared.db import is_postgres_url, resolve_database_url
from llmobserverjudge.shared.deployment import resolve_outbox_claim_lease_seconds
from llmobserverjudge.shared.tables import PendingEvalJobRow


class SqlOutboxStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory
        self._postgres = is_postgres_url(resolve_database_url())
        self._claim_lease_s = resolve_outbox_claim_lease_seconds()

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
        async with self._sf() as session:
            q = await session.execute(
                select(PendingEvalJobRow).where(
                    PendingEvalJobRow.tenant_id == tenant_id,
                    PendingEvalJobRow.project_id == pid,
                    PendingEvalJobRow.trace_id == trace_id,
                )
            )
            existing = q.scalar_one_or_none()
            if existing is not None:
                old = dict(existing.trace_summary or {})
                old_tags = list(old.get("tags") or [])
                new_tags = list(trace_summary.get("tags") or [])
                summary = {**old, **trace_summary}
                tags = list({*old_tags, *new_tags})
                if tags:
                    summary["tags"] = tags
                summary["project_id"] = pid
                existing.trace_summary = summary
                if prompt_version_id is not None:
                    existing.prompt_version_id = prompt_version_id
                await session.commit()
                return existing.id

            row_id = str(uuid.uuid4())
            summary = dict(trace_summary)
            summary["project_id"] = pid
            session.add(
                PendingEvalJobRow(
                    id=row_id,
                    tenant_id=tenant_id,
                    project_id=pid,
                    trace_id=trace_id,
                    trace_summary=summary,
                    prompt_version_id=prompt_version_id,
                    published=False,
                    attempt_count=0,
                    claimed_at=None,
                )
            )
            await session.commit()
            return row_id

    def _row_dict(self, r: PendingEvalJobRow) -> dict[str, Any]:
        claimed = getattr(r, "claimed_at", None)
        return {
            "id": r.id,
            "tenant_id": r.tenant_id,
            "project_id": getattr(r, "project_id", None) or "default",
            "trace_id": r.trace_id,
            "trace_summary": dict(r.trace_summary or {}),
            "prompt_version_id": r.prompt_version_id,
            "published": r.published,
            "attempt_count": r.attempt_count,
            "claimed_at": claimed.isoformat() if claimed is not None else None,
        }

    def _claimable_filter(self, *, stale_before: datetime):
        return (
            PendingEvalJobRow.published.is_(False),
            or_(
                PendingEvalJobRow.claimed_at.is_(None),
                PendingEvalJobRow.claimed_at < stale_before,
            ),
        )

    async def fetch_unpublished(self, limit: int = 100) -> list[dict[str, Any]]:
        """Peek unpublished rows without claiming."""
        stale_before = datetime.now(UTC) - timedelta(seconds=self._claim_lease_s)
        async with self._sf() as session:
            q = await session.execute(
                select(PendingEvalJobRow)
                .where(*self._claimable_filter(stale_before=stale_before))
                .order_by(PendingEvalJobRow.id)
                .limit(limit)
            )
            return [self._row_dict(r) for r in q.scalars().all()]

    async def claim_unpublished(self, limit: int = 100) -> list[dict[str, Any]]:
        """Claim rows with a time-bounded lease (``claimed_at``).

        Caller must ``mark_published`` after MQ publish succeeds, or
        ``release_claim`` on failure. Expired leases are reclaimable by
        other Dispatch replicas (multi-node / multi-region shared DB).
        """
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=self._claim_lease_s)
        async with self._sf() as session:
            stmt = (
                select(PendingEvalJobRow)
                .where(*self._claimable_filter(stale_before=stale_before))
                .order_by(PendingEvalJobRow.id)
                .limit(limit)
            )
            if self._postgres:
                stmt = stmt.with_for_update(skip_locked=True)
            q = await session.execute(stmt)
            rows = list(q.scalars().all())
            out: list[dict[str, Any]] = []
            for r in rows:
                r.claimed_at = now
                out.append(self._row_dict(r))
            await session.commit()
            return out

    async def release_claim(self, row_id: str) -> None:
        async with self._sf() as session:
            await session.execute(
                update(PendingEvalJobRow)
                .where(
                    PendingEvalJobRow.id == row_id,
                    PendingEvalJobRow.published.is_(False),
                )
                .values(claimed_at=None)
            )
            await session.commit()

    async def unclaim(self, row_id: str) -> None:
        await self.release_claim(row_id)

    async def mark_published(self, row_id: str) -> bool:
        """Atomically mark published; returns False if already published."""
        async with self._sf() as session:
            result = await session.execute(
                update(PendingEvalJobRow)
                .where(
                    PendingEvalJobRow.id == row_id,
                    PendingEvalJobRow.published.is_(False),
                )
                .values(published=True, claimed_at=None)
            )
            await session.commit()
            return bool(getattr(result, "rowcount", 0))

    async def bump_attempt(self, row_id: str) -> None:
        async with self._sf() as session:
            q = await session.execute(
                select(PendingEvalJobRow).where(PendingEvalJobRow.id == row_id)
            )
            row = q.scalar_one_or_none()
            if row is not None:
                row.attempt_count = int(row.attempt_count or 0) + 1
                await session.commit()
