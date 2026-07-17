"""PostgreSQL/SQLite OutboxStore — pending_eval_jobs with claim semantics."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.shared.db import is_postgres_url, resolve_database_url
from llmobserverjudge.shared.tables import PendingEvalJobRow


class SqlOutboxStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory
        self._postgres = is_postgres_url(resolve_database_url())

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
                )
            )
            await session.commit()
            return row_id

    def _row_dict(self, r: PendingEvalJobRow) -> dict[str, Any]:
        return {
            "id": r.id,
            "tenant_id": r.tenant_id,
            "project_id": getattr(r, "project_id", None) or "default",
            "trace_id": r.trace_id,
            "trace_summary": dict(r.trace_summary or {}),
            "prompt_version_id": r.prompt_version_id,
            "published": r.published,
            "attempt_count": r.attempt_count,
        }

    async def fetch_unpublished(self, limit: int = 100) -> list[dict[str, Any]]:
        """Peek unpublished rows without claiming."""
        async with self._sf() as session:
            q = await session.execute(
                select(PendingEvalJobRow)
                .where(PendingEvalJobRow.published.is_(False))
                .order_by(PendingEvalJobRow.id)
                .limit(limit)
            )
            return [self._row_dict(r) for r in q.scalars().all()]

    async def claim_unpublished(self, limit: int = 100) -> list[dict[str, Any]]:
        """Claim unpublished rows (mark published=True under row lock).

        On MQ publish failure, caller should ``unclaim`` to retry.
        Postgres uses ``FOR UPDATE SKIP LOCKED`` so multiple Dispatch
        replicas do not claim the same row.
        """
        async with self._sf() as session:
            stmt = (
                select(PendingEvalJobRow)
                .where(PendingEvalJobRow.published.is_(False))
                .order_by(PendingEvalJobRow.id)
                .limit(limit)
            )
            if self._postgres:
                stmt = stmt.with_for_update(skip_locked=True)
            q = await session.execute(stmt)
            rows = list(q.scalars().all())
            out: list[dict[str, Any]] = []
            for r in rows:
                r.published = True
                out.append(self._row_dict(r))
            await session.commit()
            return out

    async def unclaim(self, row_id: str) -> None:
        async with self._sf() as session:
            await session.execute(
                update(PendingEvalJobRow)
                .where(PendingEvalJobRow.id == row_id)
                .values(published=False)
            )
            await session.commit()

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
