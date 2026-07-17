"""SQL-backed AttributionStore."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.contracts.models import AttributionResult
from llmobserverjudge.shared.tables import AttributionResultRow


class SqlAttributionStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    def _to_model(self, row: AttributionResultRow) -> AttributionResult:
        return AttributionResult(
            attribution_id=row.attribution_id,
            trace_id=row.trace_id,
            tenant_id=row.tenant_id,
            eval_agent_id=row.eval_agent_id,
            attribution_agent=row.attribution_agent,
            root_cause=row.root_cause,
            confidence=row.confidence,
            evidence=list(row.evidence or []),
            tools_called=list(row.tools_called or []),
            rounds=row.rounds,
            early_stop=row.early_stop,
            status=row.status,
            error=row.error,
            duration_ms=row.duration_ms,
        )

    async def save(
        self, result: AttributionResult, *, idempotency_key: str | None = None
    ) -> None:
        async with self._sf() as session:
            if idempotency_key:
                q = await session.execute(
                    select(AttributionResultRow).where(
                        AttributionResultRow.idempotency_key == idempotency_key
                    )
                )
                if q.scalar_one_or_none() is not None:
                    return
            session.add(
                AttributionResultRow(
                    attribution_id=result.attribution_id,
                    idempotency_key=idempotency_key,
                    trace_id=result.trace_id,
                    tenant_id=result.tenant_id,
                    eval_agent_id=result.eval_agent_id,
                    attribution_agent=result.attribution_agent,
                    root_cause=result.root_cause,
                    confidence=result.confidence,
                    evidence=list(result.evidence),
                    tools_called=list(result.tools_called),
                    rounds=result.rounds,
                    early_stop=result.early_stop,
                    status=result.status,
                    error=result.error,
                    duration_ms=result.duration_ms,
                )
            )
            await session.commit()

    async def get_by_idempotency(self, idempotency_key: str) -> AttributionResult | None:
        async with self._sf() as session:
            q = await session.execute(
                select(AttributionResultRow).where(
                    AttributionResultRow.idempotency_key == idempotency_key
                )
            )
            row = q.scalar_one_or_none()
            return self._to_model(row) if row else None

    async def list_for_trace(
        self, tenant_id: str, trace_id: str
    ) -> list[AttributionResult]:
        async with self._sf() as session:
            q = await session.execute(
                select(AttributionResultRow).where(
                    AttributionResultRow.tenant_id == tenant_id,
                    AttributionResultRow.trace_id == trace_id,
                )
            )
            return [self._to_model(r) for r in q.scalars().all()]

    async def list_recent(self, *, limit: int = 500) -> list[AttributionResult]:
        async with self._sf() as session:
            q = await session.execute(select(AttributionResultRow).limit(limit))
            return [self._to_model(r) for r in q.scalars().all()]
