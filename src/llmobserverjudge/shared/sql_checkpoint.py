"""SQL-backed ReAct attribution checkpoint store."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.shared.tables import AttributionCheckpointRow


class SqlCheckpointStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def save(
        self,
        trace_id: str,
        payload: dict[str, Any],
        *,
        tenant_id: str = "",
        project_id: str = "default",
    ) -> None:
        async with self._sf() as session:
            q = await session.execute(
                select(AttributionCheckpointRow).where(
                    AttributionCheckpointRow.tenant_id == tenant_id,
                    AttributionCheckpointRow.project_id == project_id,
                    AttributionCheckpointRow.trace_id == trace_id,
                )
            )
            row = q.scalar_one_or_none()
            if row is None:
                session.add(
                    AttributionCheckpointRow(
                        id=str(uuid4()),
                        tenant_id=tenant_id,
                        project_id=project_id,
                        trace_id=trace_id,
                        payload=dict(payload),
                    )
                )
            else:
                row.payload = dict(payload)
            await session.commit()

    async def load(
        self,
        trace_id: str,
        *,
        tenant_id: str = "",
        project_id: str = "default",
    ) -> dict[str, Any] | None:
        async with self._sf() as session:
            q = await session.execute(
                select(AttributionCheckpointRow).where(
                    AttributionCheckpointRow.tenant_id == tenant_id,
                    AttributionCheckpointRow.project_id == project_id,
                    AttributionCheckpointRow.trace_id == trace_id,
                )
            )
            row = q.scalar_one_or_none()
            return dict(row.payload) if row else None
