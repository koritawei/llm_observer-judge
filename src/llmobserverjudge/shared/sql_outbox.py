"""PostgreSQL/SQLite OutboxStore — pending_eval_jobs."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.shared.tables import PendingEvalJobRow


class SqlOutboxStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def insert_trace_closed(
        self,
        *,
        tenant_id: str,
        trace_id: str,
        trace_summary: dict[str, Any],
        prompt_version_id: str | None = None,
    ) -> str:
        async with self._sf() as session:
            q = await session.execute(
                select(PendingEvalJobRow).where(
                    PendingEvalJobRow.tenant_id == tenant_id,
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
                existing.trace_summary = summary
                if prompt_version_id is not None:
                    existing.prompt_version_id = prompt_version_id
                await session.commit()
                return existing.id

            row_id = str(uuid.uuid4())
            session.add(
                PendingEvalJobRow(
                    id=row_id,
                    tenant_id=tenant_id,
                    trace_id=trace_id,
                    trace_summary=dict(trace_summary),
                    prompt_version_id=prompt_version_id,
                    published=False,
                    attempt_count=0,
                )
            )
            await session.commit()
            return row_id

    async def fetch_unpublished(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._sf() as session:
            q = await session.execute(
                select(PendingEvalJobRow)
                .where(PendingEvalJobRow.published.is_(False))
                .order_by(PendingEvalJobRow.id)
                .limit(limit)
            )
            rows = q.scalars().all()
            return [
                {
                    "id": r.id,
                    "tenant_id": r.tenant_id,
                    "trace_id": r.trace_id,
                    "trace_summary": dict(r.trace_summary or {}),
                    "prompt_version_id": r.prompt_version_id,
                    "published": r.published,
                    "attempt_count": r.attempt_count,
                }
                for r in rows
            ]

    async def mark_published(self, row_id: str) -> None:
        async with self._sf() as session:
            await session.execute(
                update(PendingEvalJobRow)
                .where(PendingEvalJobRow.id == row_id)
                .values(published=True)
            )
            await session.commit()

    async def bump_attempt(self, row_id: str) -> None:
        async with self._sf() as session:
            q = await session.execute(
                select(PendingEvalJobRow).where(PendingEvalJobRow.id == row_id)
            )
            row = q.scalar_one_or_none()
            if row is not None:
                row.attempt_count = int(row.attempt_count or 0) + 1
                await session.commit()
