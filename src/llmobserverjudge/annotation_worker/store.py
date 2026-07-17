"""Human annotation persistence (design §9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from llmobserverjudge.contracts.models import AnnotationStatus, HumanAnnotation

DEFAULT_PENDING_TTL = timedelta(days=7)
BACKLOG_ALERT_THRESHOLD = 1000


@runtime_checkable
class AnnotationStore(Protocol):
    async def upsert_pending(self, record: HumanAnnotation) -> HumanAnnotation: ...

    async def get(self, annotation_id: str) -> HumanAnnotation | None: ...

    async def get_by_idempotency(self, key: str) -> HumanAnnotation | None: ...

    async def list(
        self,
        *,
        tenant_id: str | None = None,
        status: AnnotationStatus | None = None,
        limit: int = 100,
    ) -> list[HumanAnnotation]: ...

    async def claim(
        self, annotation_id: str, annotator_id: str
    ) -> HumanAnnotation: ...

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
    ) -> HumanAnnotation: ...

    async def expire_pending(self, *, now: datetime | None = None) -> int: ...

    async def pending_count(self, tenant_id: str) -> int: ...


@dataclass
class InMemoryAnnotationStore:
    _by_id: dict[str, HumanAnnotation] = field(default_factory=dict)
    _by_key: dict[str, str] = field(default_factory=dict)
    pending_ttl: timedelta = DEFAULT_PENDING_TTL

    async def upsert_pending(self, record: HumanAnnotation) -> HumanAnnotation:
        key = record.idempotency_key or record.annotation_id
        existing_id = self._by_key.get(key)
        if existing_id and existing_id in self._by_id:
            return self._by_id[existing_id]
        if record.annotation_id in self._by_id:
            return self._by_id[record.annotation_id]

        now = datetime.now(UTC)
        row = record.model_copy(
            update={
                "status": AnnotationStatus.PENDING,
                "created_at": record.created_at or now,
                "expires_at": record.expires_at or (now + self.pending_ttl),
            }
        )
        self._by_id[row.annotation_id] = row
        self._by_key[key] = row.annotation_id
        return row

    async def get(self, annotation_id: str) -> HumanAnnotation | None:
        return self._by_id.get(annotation_id)

    async def get_by_idempotency(self, key: str) -> HumanAnnotation | None:
        aid = self._by_key.get(key)
        return self._by_id.get(aid) if aid else None

    async def list(
        self,
        *,
        tenant_id: str | None = None,
        status: AnnotationStatus | None = None,
        limit: int = 100,
    ) -> list[HumanAnnotation]:
        rows = list(self._by_id.values())
        if tenant_id is not None:
            rows = [r for r in rows if r.tenant_id == tenant_id]
        if status is not None:
            rows = [r for r in rows if r.status == status]
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        rows.sort(key=lambda r: r.created_at or epoch)
        return rows[:limit]

    async def claim(self, annotation_id: str, annotator_id: str) -> HumanAnnotation:
        row = self._by_id.get(annotation_id)
        if row is None:
            raise KeyError(f"annotation not found: {annotation_id}")
        if row.status == AnnotationStatus.EXPIRED:
            raise ValueError("annotation expired")
        if row.status == AnnotationStatus.COMPLETED:
            raise ValueError("annotation already completed")
        if (
            row.status == AnnotationStatus.IN_PROGRESS
            and row.annotator_id
            and row.annotator_id != annotator_id
        ):
            raise ValueError(f"claimed by {row.annotator_id}")
        updated = row.model_copy(
            update={
                "status": AnnotationStatus.IN_PROGRESS,
                "annotator_id": annotator_id,
            }
        )
        self._by_id[annotation_id] = updated
        return updated

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
        row = self._by_id.get(annotation_id)
        if row is None:
            raise KeyError(f"annotation not found: {annotation_id}")
        if row.status == AnnotationStatus.EXPIRED:
            raise ValueError("annotation expired")
        if row.status == AnnotationStatus.COMPLETED:
            return row
        if row.status == AnnotationStatus.PENDING:
            # allow complete without explicit claim
            pass
        elif row.annotator_id and row.annotator_id != annotator_id:
            raise ValueError(f"claimed by {row.annotator_id}")

        now = datetime.now(UTC)
        created = row.created_at or now
        score_delta = None
        label_correct = None
        if human_score is not None and row.auto_score is not None:
            score_delta = round(human_score - row.auto_score, 4)
        if human_label is not None and row.auto_label is not None:
            label_correct = human_label == row.auto_label

        updated = row.model_copy(
            update={
                "status": AnnotationStatus.COMPLETED,
                "annotator_id": annotator_id,
                "human_score": human_score,
                "human_label": human_label,
                "human_root_cause": human_root_cause,
                "evidence_sufficient": evidence_sufficient,
                "fix_feasible": fix_feasible,
                "score_delta": score_delta,
                "label_correct": label_correct,
                "completed_at": now,
                "annotation_time_ms": int((now - created).total_seconds() * 1000),
            }
        )
        self._by_id[annotation_id] = updated
        return updated

    async def expire_pending(self, *, now: datetime | None = None) -> int:
        now = now or datetime.now(UTC)
        count = 0
        for aid, row in list(self._by_id.items()):
            if row.status != AnnotationStatus.PENDING:
                continue
            expires = row.expires_at
            created = row.created_at
            overdue = False
            if expires is not None and expires <= now:
                overdue = True
            elif expires is None and created is not None and created + self.pending_ttl <= now:
                overdue = True
            if overdue:
                self._by_id[aid] = row.model_copy(update={"status": AnnotationStatus.EXPIRED})
                count += 1
        return count

    async def pending_count(self, tenant_id: str) -> int:
        return sum(
            1
            for r in self._by_id.values()
            if r.tenant_id == tenant_id and r.status == AnnotationStatus.PENDING
        )


def backlog_alert(pending: int, threshold: int = BACKLOG_ALERT_THRESHOLD) -> bool:
    return pending > threshold
