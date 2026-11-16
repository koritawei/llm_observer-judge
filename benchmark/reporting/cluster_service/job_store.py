"""Cluster Report PostgreSQL/SQLite job store."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from tortoise.exceptions import IntegrityError
from tortoise.expressions import F, Q

from benchmark.records.models.cluster_report_job import ClusterReportJob


@dataclass(frozen=True)
class ReportJobCreate:
    job_type: str
    source_batch_tags: list[str]
    job_request: dict[str, Any]
    evaluator_name: str | None = None


@dataclass(frozen=True)
class ReportJobFinish:
    job_id: str
    owner: str
    expected_attempt: int
    status: str
    phase: str
    error: str = ""


class ClusterReportJobStore:
    """Coordinate multi-instance report execution with a database unique key and a lease."""

    def __init__(self, *, environment: str, evaluator_name: str, lease_seconds: int = 60) -> None:
        self._environment = environment
        self._evaluator_name = evaluator_name
        self._lease_seconds = lease_seconds

    @staticmethod
    def build_idempotency_key(
        *, job_type: str, source_batch_tags: list[str], job_request: dict[str, Any]
    ) -> str:
        """Map the same source, type, and request parameters stably onto one job."""

        payload = json.dumps(
            {
                "job_type": job_type,
                "source_batch_tags": sorted(source_batch_tags),
                "job_request": job_request,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def create_or_get(self, request: ReportJobCreate) -> tuple[ClusterReportJob, bool]:
        key = self.build_idempotency_key(
            job_type=request.job_type,
            source_batch_tags=request.source_batch_tags,
            job_request=request.job_request,
        )
        resolved_evaluator_name = request.evaluator_name or self._evaluator_name
        defaults = {
            "source_batch_tags": request.source_batch_tags,
            "job_request": request.job_request,
            "status": "pending",
        }
        try:
            return await ClusterReportJob.get_or_create(
                environment=self._environment,
                evaluator_name=resolved_evaluator_name,
                job_type=request.job_type,
                idempotency_key=key,
                defaults=defaults,
            )
        except IntegrityError:
            job = await ClusterReportJob.get(
                environment=self._environment,
                evaluator_name=resolved_evaluator_name,
                job_type=request.job_type,
                idempotency_key=key,
            )
            return (job, False)

    async def get(self, job_id: str) -> ClusterReportJob | None:
        try:
            return await ClusterReportJob.get_or_none(job_id=UUID(job_id))
        except ValueError:
            return None

    async def acquire(self, *, job_id: str, owner: str) -> ClusterReportJob | None:
        """Atomically claim a pending job or a job whose lease has expired."""

        now = datetime.now(UTC)
        try:
            identifier = UUID(job_id)
        except ValueError:
            return None
        updated = (
            await ClusterReportJob.filter(job_id=identifier)
            .filter(Q(status="pending") | (Q(status="running") & Q(lease_expire_at__lt=now)))
            .update(
                status="running",
                lease_owner=owner,
                lease_expire_at=now + timedelta(seconds=self._lease_seconds),
                attempt=F("attempt") + 1,
                error="",
                updated_at=now,
            )
        )
        if updated != 1:
            return None
        return await ClusterReportJob.get(job_id=identifier)

    async def renew(self, *, job_id: str, owner: str, expected_attempt: int) -> bool:
        try:
            identifier = UUID(job_id)
        except ValueError:
            return False
        now = datetime.now(UTC)
        updated = await ClusterReportJob.filter(
            job_id=identifier,
            status="running",
            lease_owner=owner,
            attempt=expected_attempt,
            lease_expire_at__gt=now,
        ).update(
            lease_expire_at=now + timedelta(seconds=self._lease_seconds),
            updated_at=now,
        )
        return updated == 1

    async def update_owned(
        self, *, job_id: str, owner: str, expected_attempt: int, **fields: object
    ) -> bool:
        """Only the lease holder may write the execution result, so a lapsed instance cannot overwrite the successor."""

        try:
            identifier = UUID(job_id)
        except ValueError:
            return False
        now = datetime.now(UTC)
        fields["updated_at"] = now
        return (
            await ClusterReportJob.filter(
                job_id=identifier,
                status="running",
                lease_owner=owner,
                attempt=expected_attempt,
                lease_expire_at__gt=now,
            ).update(**fields)
        ) == 1

    async def finish(self, request: ReportJobFinish, **fields: object) -> bool:
        fields.update(
            status=request.status,
            phase=request.phase,
            error=request.error,
            lease_owner="",
            lease_expire_at=None,
        )
        return await self.update_owned(
            job_id=request.job_id,
            owner=request.owner,
            expected_attempt=request.expected_attempt,
            **fields,
        )

    async def requeue(
        self,
        job_id: str,
        *,
        force: bool = False,
        job_request_updates: dict[str, Any] | None = None,
    ) -> ClusterReportJob | None:
        """Requeue a failed job, or force-requeue a completed/failed job."""

        try:
            identifier = UUID(job_id)
        except ValueError:
            return None
        statuses = ("failed", "completed") if force else ("failed",)
        job = await ClusterReportJob.get_or_none(job_id=identifier, status__in=statuses)
        if job is None:
            return None
        request = dict(job.job_request or {})
        request.update(job_request_updates or {})
        fields: dict[str, Any] = {
            "status": "pending",
            "phase": "",
            "error": "",
            "lease_owner": "",
            "lease_expire_at": None,
            "job_request": request,
            "updated_at": datetime.now(UTC),
        }
        if force:
            revision = _next_force_revision(job)
            request.update({"force_revision": revision, "force_dispatch": True})
            fields.update(
                artifacts={},
                result_summary={},
                dispatch={"status": "pending", "history": _dispatch_history(job)},
            )
        updated = await ClusterReportJob.filter(job_id=identifier, status__in=statuses).update(
            **fields
        )
        if updated != 1:
            return None
        return await ClusterReportJob.get(job_id=identifier)

    async def requeue_failed(
        self,
        job_id: str,
        *,
        job_request_updates: dict[str, Any] | None = None,
    ) -> ClusterReportJob | None:
        """Failed-only retry entry kept for older callers."""

        return await self.requeue(job_id, job_request_updates=job_request_updates)

    async def recoverable_job_ids(self) -> list[str]:
        now = datetime.now(UTC)
        jobs = await ClusterReportJob.filter(
            Q(status="pending") | (Q(status="running") & Q(lease_expire_at__lt=now))
        ).only("job_id")
        return [str(job.job_id) for job in jobs]

    @staticmethod
    def serialize(job: ClusterReportJob) -> dict[str, Any]:
        return {
            "jobId": str(job.job_id),
            "jobType": job.job_type,
            "evaluatorName": job.evaluator_name,
            "sourceBatchTags": list(job.source_batch_tags or []),
            "status": job.status,
            "phase": job.phase,
            "attempt": job.attempt,
            "artifacts": dict(job.artifacts or {}),
            "resultSummary": dict(job.result_summary or {}),
            "evaluatorConfigSnapshot": dict(
                (job.job_request or {}).get("evaluator_config_snapshot") or {}
            ),
            "dispatch": dict(job.dispatch or {}) or None,
            "error": job.error or "",
            "createdAt": job.created_at.isoformat() if job.created_at else "",
            "updatedAt": job.updated_at.isoformat() if job.updated_at else "",
        }


def _dispatch_history(job: ClusterReportJob) -> list[dict[str, Any]]:
    dispatch = dict(job.dispatch or {})
    history = [item for item in dispatch.pop("history", []) if isinstance(item, dict)]
    revision = int((job.job_request or {}).get("force_revision") or 0)
    if (dispatch or job.artifacts) and not any(
        int(item.get("revision") or 0) == revision for item in history
    ):
        history.append(
            {
                "revision": revision,
                "attempt": job.attempt,
                "trigger": "force" if revision else "initial",
                "createdAt": job.updated_at.isoformat() if job.updated_at else "",
                "artifacts": dict(job.artifacts or {}),
                "dispatch": dispatch,
            }
        )
    return history


def _next_force_revision(job: ClusterReportJob) -> int:
    revisions = [int((job.job_request or {}).get("force_revision") or 0)]
    revisions.extend(
        int(item.get("revision") or 0) for item in _dispatch_history(job) if isinstance(item, dict)
    )
    return max(revisions) + 1


__all__ = ["ClusterReportJobStore"]
