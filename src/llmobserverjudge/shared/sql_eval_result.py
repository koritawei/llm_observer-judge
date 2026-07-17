"""SQL-backed EvalResultStore for multi-process Dashboard."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.dashboard.store import EvalResultRecord
from llmobserverjudge.shared.tables import EvalResultRow


def _record_from_payload(payload: dict[str, Any]) -> EvalResultRecord:
    return EvalResultRecord(
        trace_id=str(payload["trace_id"]),
        tenant_id=str(payload["tenant_id"]),
        project_id=str(payload.get("project_id") or "default"),
        final_score=float(payload.get("final_score") or 0.0),
        bad_case_detected=bool(payload.get("bad_case_detected")),
        eval_scores=list(payload.get("eval_scores") or []),
        eval_execution_layers=list(payload.get("eval_execution_layers") or []),
        eval_results=list(payload.get("eval_results") or []),
        node_errors=dict(payload.get("node_errors") or {}),
        optional_skipped=list(payload.get("optional_skipped") or []),
        prompt_version_id=payload.get("prompt_version_id"),
        execution_graph=list(payload.get("execution_graph") or []),
        created_at=str(payload.get("created_at") or ""),
    )


class SqlEvalResultStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def save(
        self, record: EvalResultRecord, *, idempotency_key: str | None = None
    ) -> None:
        payload = record.to_dict()
        project_id = str(getattr(record, "project_id", None) or "default")
        async with self._sf() as session:
            if idempotency_key:
                qk = await session.execute(
                    select(EvalResultRow).where(
                        EvalResultRow.idempotency_key == idempotency_key
                    )
                )
                by_key = qk.scalar_one_or_none()
                if by_key is not None:
                    by_key.payload = payload
                    by_key.tenant_id = record.tenant_id
                    by_key.project_id = project_id
                    by_key.trace_id = record.trace_id
                    await session.commit()
                    return
            q = await session.execute(
                select(EvalResultRow).where(
                    EvalResultRow.tenant_id == record.tenant_id,
                    EvalResultRow.project_id == project_id,
                    EvalResultRow.trace_id == record.trace_id,
                )
            )
            existing = q.scalar_one_or_none()
            if existing is not None:
                existing.payload = payload
                if idempotency_key:
                    existing.idempotency_key = idempotency_key
            else:
                session.add(
                    EvalResultRow(
                        id=str(uuid4()),
                        tenant_id=record.tenant_id,
                        project_id=project_id,
                        trace_id=record.trace_id,
                        idempotency_key=idempotency_key,
                        payload=payload,
                    )
                )
            await session.commit()

    async def get(
        self, trace_id: str, *, tenant_id: str | None = None, project_id: str | None = None
    ) -> EvalResultRecord | None:
        async with self._sf() as session:
            stmt = (
                select(EvalResultRow)
                .where(EvalResultRow.trace_id == trace_id)
                .order_by(EvalResultRow.created_at.desc())
                .limit(1)
            )
            if tenant_id is not None:
                stmt = stmt.where(EvalResultRow.tenant_id == tenant_id)
            if project_id is not None:
                stmt = stmt.where(EvalResultRow.project_id == project_id)
            q = await session.execute(stmt)
            row = q.scalar_one_or_none()
            return _record_from_payload(dict(row.payload)) if row else None

    async def get_by_idempotency(self, key: str) -> EvalResultRecord | None:
        async with self._sf() as session:
            q = await session.execute(
                select(EvalResultRow).where(EvalResultRow.idempotency_key == key)
            )
            row = q.scalar_one_or_none()
            return _record_from_payload(dict(row.payload)) if row else None

    async def list_all(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        limit: int = 500,
    ) -> list[EvalResultRecord]:
        async with self._sf() as session:
            stmt = select(EvalResultRow).order_by(EvalResultRow.created_at.desc())
            if tenant_id is not None:
                stmt = stmt.where(EvalResultRow.tenant_id == tenant_id)
            if project_id is not None:
                stmt = stmt.where(EvalResultRow.project_id == project_id)
            stmt = stmt.limit(limit)
            q = await session.execute(stmt)
            return [_record_from_payload(dict(r.payload)) for r in q.scalars().all()]
