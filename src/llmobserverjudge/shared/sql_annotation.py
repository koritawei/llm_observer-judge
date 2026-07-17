"""SQL-backed AnnotationStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.annotation_worker.store import (
    BACKLOG_ALERT_THRESHOLD,
    DEFAULT_PENDING_TTL,
)
from llmobserverjudge.contracts.models import AnnotationStatus, HumanAnnotation
from llmobserverjudge.shared.tables import HumanAnnotationRow


class SqlAnnotationStore:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        pending_ttl: timedelta = DEFAULT_PENDING_TTL,
    ) -> None:
        self._sf = session_factory
        self.pending_ttl = pending_ttl

    def _to_model(self, row: HumanAnnotationRow) -> HumanAnnotation:
        return HumanAnnotation(
            annotation_id=row.annotation_id,
            trace_id=row.trace_id,
            tenant_id=row.tenant_id,
            eval_agent_id=row.eval_agent_id,
            trigger_reason=row.trigger_reason,
            status=AnnotationStatus(row.status),
            auto_score=row.auto_score,
            auto_label=row.auto_label,
            conflict_detected=bool(row.conflict_detected),
            idempotency_key=row.idempotency_key,
            annotator_id=row.annotator_id,
            human_score=row.human_score,
            human_label=row.human_label,
            score_delta=row.score_delta,
            label_correct=row.label_correct,
            human_root_cause=row.human_root_cause,
            evidence_sufficient=row.evidence_sufficient,
            fix_feasible=row.fix_feasible,
            expires_at=row.expires_at,
            created_at=row.created_at,
            completed_at=row.completed_at,
            annotation_time_ms=row.annotation_time_ms,
        )

    async def upsert_pending(self, record: HumanAnnotation) -> HumanAnnotation:
        key = record.idempotency_key or record.annotation_id
        async with self._sf() as session:
            if record.idempotency_key:
                q = await session.execute(
                    select(HumanAnnotationRow).where(
                        HumanAnnotationRow.idempotency_key == key
                    )
                )
                existing = q.scalar_one_or_none()
                if existing is not None:
                    return self._to_model(existing)
            q2 = await session.execute(
                select(HumanAnnotationRow).where(
                    HumanAnnotationRow.annotation_id == record.annotation_id
                )
            )
            by_id = q2.scalar_one_or_none()
            if by_id is not None:
                return self._to_model(by_id)

            now = datetime.now(UTC)
            row = HumanAnnotationRow(
                annotation_id=record.annotation_id,
                trace_id=record.trace_id,
                tenant_id=record.tenant_id,
                eval_agent_id=record.eval_agent_id,
                trigger_reason=record.trigger_reason,
                status=AnnotationStatus.PENDING.value,
                auto_score=record.auto_score,
                auto_label=record.auto_label,
                conflict_detected=record.conflict_detected,
                idempotency_key=key,
                created_at=record.created_at or now,
                expires_at=record.expires_at or (now + self.pending_ttl),
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return self._to_model(row)

    async def get(self, annotation_id: str) -> HumanAnnotation | None:
        async with self._sf() as session:
            q = await session.execute(
                select(HumanAnnotationRow).where(
                    HumanAnnotationRow.annotation_id == annotation_id
                )
            )
            row = q.scalar_one_or_none()
            return self._to_model(row) if row else None

    async def get_by_idempotency(self, key: str) -> HumanAnnotation | None:
        async with self._sf() as session:
            q = await session.execute(
                select(HumanAnnotationRow).where(HumanAnnotationRow.idempotency_key == key)
            )
            row = q.scalar_one_or_none()
            return self._to_model(row) if row else None

    async def list(
        self,
        *,
        tenant_id: str | None = None,
        status: AnnotationStatus | None = None,
        limit: int = 100,
    ) -> list[HumanAnnotation]:
        async with self._sf() as session:
            stmt = select(HumanAnnotationRow).order_by(HumanAnnotationRow.created_at)
            if tenant_id is not None:
                stmt = stmt.where(HumanAnnotationRow.tenant_id == tenant_id)
            if status is not None:
                stmt = stmt.where(HumanAnnotationRow.status == status.value)
            stmt = stmt.limit(limit)
            q = await session.execute(stmt)
            return [self._to_model(r) for r in q.scalars().all()]

    async def claim(self, annotation_id: str, annotator_id: str) -> HumanAnnotation:
        async with self._sf() as session:
            q = await session.execute(
                select(HumanAnnotationRow).where(
                    HumanAnnotationRow.annotation_id == annotation_id
                )
            )
            row = q.scalar_one_or_none()
            if row is None:
                raise KeyError(f"annotation not found: {annotation_id}")
            if row.status == AnnotationStatus.EXPIRED.value:
                raise ValueError("annotation expired")
            if row.status == AnnotationStatus.COMPLETED.value:
                raise ValueError("annotation already completed")
            if (
                row.status == AnnotationStatus.IN_PROGRESS.value
                and row.annotator_id
                and row.annotator_id != annotator_id
            ):
                raise ValueError(f"claimed by {row.annotator_id}")
            row.status = AnnotationStatus.IN_PROGRESS.value
            row.annotator_id = annotator_id
            await session.commit()
            await session.refresh(row)
            return self._to_model(row)

    async def complete(
        self,
        annotation_id: str,
        *,
        annotator_id: str,
        human_score: float | None = None,
        human_label: str | None = None,
        human_root_cause: str | None = None,
        evidence_sufficient: bool | None = None,
        fix_feasible: bool | None = None,
    ) -> HumanAnnotation:
        async with self._sf() as session:
            q = await session.execute(
                select(HumanAnnotationRow).where(
                    HumanAnnotationRow.annotation_id == annotation_id
                )
            )
            row = q.scalar_one_or_none()
            if row is None:
                raise KeyError(f"annotation not found: {annotation_id}")
            if row.status == AnnotationStatus.EXPIRED.value:
                raise ValueError("annotation expired")
            if row.status == AnnotationStatus.COMPLETED.value:
                return self._to_model(row)
            if row.annotator_id and row.annotator_id != annotator_id:
                raise ValueError(f"claimed by {row.annotator_id}")

            now = datetime.now(UTC)
            created = row.created_at or now
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            score_delta = None
            label_correct = None
            if human_score is not None and row.auto_score is not None:
                score_delta = round(human_score - row.auto_score, 4)
            if human_label is not None and row.auto_label is not None:
                label_correct = human_label == row.auto_label

            row.status = AnnotationStatus.COMPLETED.value
            row.annotator_id = annotator_id
            row.human_score = human_score
            row.human_label = human_label
            row.human_root_cause = human_root_cause
            row.evidence_sufficient = evidence_sufficient
            row.fix_feasible = fix_feasible
            row.score_delta = score_delta
            row.label_correct = label_correct
            row.completed_at = now
            row.annotation_time_ms = int((now - created).total_seconds() * 1000)
            await session.commit()
            await session.refresh(row)
            return self._to_model(row)

    async def expire_pending(self, *, now: datetime | None = None) -> int:
        now = now or datetime.now(UTC)
        async with self._sf() as session:
            q = await session.execute(
                select(HumanAnnotationRow).where(
                    HumanAnnotationRow.status == AnnotationStatus.PENDING.value
                )
            )
            count = 0
            for row in q.scalars().all():
                expires = row.expires_at
                created = row.created_at
                if expires is not None and expires.tzinfo is None:
                    expires = expires.replace(tzinfo=UTC)
                if created is not None and created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                overdue = False
                if expires is not None and expires <= now:
                    overdue = True
                elif (
                    expires is None
                    and created is not None
                    and created + self.pending_ttl <= now
                ):
                    overdue = True
                if overdue:
                    row.status = AnnotationStatus.EXPIRED.value
                    count += 1
            await session.commit()
            return count

    async def pending_count(self, tenant_id: str) -> int:
        async with self._sf() as session:
            q = await session.execute(
                select(func.count())
                .select_from(HumanAnnotationRow)
                .where(
                    HumanAnnotationRow.tenant_id == tenant_id,
                    HumanAnnotationRow.status == AnnotationStatus.PENDING.value,
                )
            )
            return int(q.scalar_one())


# re-export for convenience
__all__ = ["SqlAnnotationStore", "BACKLOG_ALERT_THRESHOLD"]
