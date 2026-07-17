"""SQL-backed SpanStore for multi-process / shared Dashboard."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.shared.storage import InMemorySpanStore
from llmobserverjudge.shared.tables import SpanRow


class SqlSpanStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def write_span(self, span: dict[str, Any]) -> None:
        tenant_id = str(span["tenant_id"])
        span_id = str(span["span_id"])
        project_id = str(span.get("project_id") or "default")
        trace_id = str(span.get("trace_id") or "")
        payload = dict(span)
        payload["project_id"] = project_id
        async with self._sf() as session:
            q = await session.execute(
                select(SpanRow).where(
                    SpanRow.tenant_id == tenant_id,
                    SpanRow.project_id == project_id,
                    SpanRow.span_id == span_id,
                )
            )
            existing = q.scalar_one_or_none()
            if existing is not None:
                existing.payload = payload
                existing.trace_id = trace_id
            else:
                session.add(
                    SpanRow(
                        id=str(uuid4()),
                        tenant_id=tenant_id,
                        project_id=project_id,
                        trace_id=trace_id,
                        span_id=span_id,
                        payload=payload,
                    )
                )
            await session.commit()

    async def get_span(
        self, tenant_id: str, span_id: str, *, project_id: str = "default"
    ) -> dict[str, Any] | None:
        async with self._sf() as session:
            q = await session.execute(
                select(SpanRow).where(
                    SpanRow.tenant_id == tenant_id,
                    SpanRow.project_id == project_id,
                    SpanRow.span_id == span_id,
                )
            )
            row = q.scalar_one_or_none()
            return dict(row.payload) if row else None

    async def list_spans_by_trace(
        self,
        trace_id: str,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        async with self._sf() as session:
            stmt = select(SpanRow).where(SpanRow.trace_id == trace_id)
            if tenant_id is not None:
                stmt = stmt.where(SpanRow.tenant_id == tenant_id)
            if project_id is not None:
                stmt = stmt.where(SpanRow.project_id == project_id)
            q = await session.execute(stmt)
            out = [dict(r.payload) for r in q.scalars().all()]
        out.sort(key=lambda r: str(r.get("timestamp") or r.get("span_id") or ""))
        return out

    async def list_traces(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with self._sf() as session:
            stmt = select(SpanRow)
            if tenant_id is not None:
                stmt = stmt.where(SpanRow.tenant_id == tenant_id)
            if project_id is not None:
                stmt = stmt.where(SpanRow.project_id == project_id)
            q = await session.execute(stmt)
            rows = [dict(r.payload) for r in q.scalars().all()]
        mem = InMemorySpanStore()
        for row in rows:
            await mem.write_span(row)
        return await mem.list_traces(
            tenant_id=tenant_id, project_id=project_id, limit=limit
        )
