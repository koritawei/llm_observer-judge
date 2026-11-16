"""Local SQLite persistence and lease tests for the Cluster Report JobStore."""

from datetime import UTC, datetime, timedelta

import pytest
from benchmark.records.models.cluster_report_job import ClusterReportJob
from benchmark.services import TraceRecordService
from benchmark.reporting.cluster_service.job_store import (
    ClusterReportJobStore,
    ReportJobCreate,
    ReportJobFinish,
)


@pytest.fixture()
def repo() -> TraceRecordService:
    """Trigger the shared conftest to initialize an isolated SQLite."""

    return TraceRecordService()


async def test_sqlite_job_store_deduplicates_same_request(repo: TraceRecordService) -> None:
    del repo
    store = ClusterReportJobStore(environment="test", evaluator_name="vehicle_agent")
    request = {"expected_total": 2, "batch_meta": {"model": "mock"}}

    first, created = await store.create_or_get(
        ReportJobCreate(
            job_type="normal", source_batch_tags=["run-20260908-100000"], job_request=request
        )
    )
    second, second_created = await store.create_or_get(
        ReportJobCreate(
            job_type="normal", source_batch_tags=["run-20260908-100000"], job_request=request
        )
    )

    assert created is True
    assert second_created is False
    assert first.job_id == second.job_id


async def test_sqlite_job_store_allows_only_one_lease_owner(repo: TraceRecordService) -> None:
    del repo
    store = ClusterReportJobStore(environment="test", evaluator_name="vehicle_agent")
    job, _ = await store.create_or_get(
        ReportJobCreate(
            job_type="normal",
            source_batch_tags=["run-20260908-100000"],
            job_request={"expected_total": 1},
        )
    )

    first = await store.acquire(job_id=str(job.job_id), owner="instance-a")
    second = await store.acquire(job_id=str(job.job_id), owner="instance-b")

    assert first is not None
    assert second is None
    assert first.lease_owner == "instance-a"


async def test_sqlite_job_store_updates_timestamp_for_atomic_writes(
    repo: TraceRecordService,
) -> None:
    """QuerySet.update() does not fire auto_now, so JobStore must maintain updated_at explicitly."""
    del repo
    store = ClusterReportJobStore(environment="test", evaluator_name="vehicle_agent")
    job, _ = await store.create_or_get(
        ReportJobCreate(
            job_type="normal", source_batch_tags=["run-20260908-100000"], job_request={}
        )
    )
    old_timestamp = datetime(2000, 1, 1, tzinfo=UTC)

    await ClusterReportJob.filter(job_id=job.job_id).update(updated_at=old_timestamp)
    acquired = await store.acquire(job_id=str(job.job_id), owner="instance-a")
    assert acquired is not None and acquired.updated_at > old_timestamp

    await ClusterReportJob.filter(job_id=job.job_id).update(updated_at=old_timestamp)
    assert await store.renew(
        job_id=str(job.job_id), owner="instance-a", expected_attempt=acquired.attempt
    )
    renewed = await ClusterReportJob.get(job_id=job.job_id)
    assert renewed.updated_at > old_timestamp

    await ClusterReportJob.filter(job_id=job.job_id).update(updated_at=old_timestamp)
    assert await store.update_owned(
        job_id=str(job.job_id),
        owner="instance-a",
        expected_attempt=acquired.attempt,
        phase="build_report",
    )
    updated = await ClusterReportJob.get(job_id=job.job_id)
    assert updated.updated_at > old_timestamp

    assert await store.finish(
        ReportJobFinish(
            job_id=str(job.job_id),
            owner="instance-a",
            expected_attempt=acquired.attempt,
            status="failed",
            phase="failed",
        )
    )
    await ClusterReportJob.filter(job_id=job.job_id).update(updated_at=old_timestamp)
    requeued = await store.requeue_failed(str(job.job_id))
    assert requeued is not None and requeued.updated_at > old_timestamp


async def test_sqlite_job_store_recovers_expired_lease(repo: TraceRecordService) -> None:
    del repo
    store = ClusterReportJobStore(environment="test", evaluator_name="vehicle_agent")
    job, _ = await store.create_or_get(
        ReportJobCreate(
            job_type="stability",
            source_batch_tags=["run-20260908-100000", "run-20260908-101000"],
            job_request={},
        )
    )
    acquired = await store.acquire(job_id=str(job.job_id), owner="crashed-instance")
    assert acquired is not None
    await ClusterReportJob.filter(job_id=job.job_id).update(
        lease_expire_at=datetime.now(UTC) - timedelta(seconds=1)
    )

    recovered = await store.acquire(job_id=str(job.job_id), owner="replacement-instance")

    assert recovered is not None
    assert recovered.lease_owner == "replacement-instance"
    assert recovered.attempt == 2


async def test_force_requeue_preserves_dispatch_history_and_increments_revision(
    repo: TraceRecordService,
) -> None:
    del repo
    store = ClusterReportJobStore(environment="test", evaluator_name="vehicle_agent")
    job, _ = await store.create_or_get(
        ReportJobCreate(
            job_type="normal", source_batch_tags=["run-20260908-100000"], job_request={}
        )
    )
    acquired = await store.acquire(job_id=str(job.job_id), owner="instance-a")
    assert acquired is not None
    assert await store.finish(
        ReportJobFinish(
            job_id=str(job.job_id),
            owner="instance-a",
            expected_attempt=acquired.attempt,
            status="completed",
            phase="done",
        ),
        artifacts={"report.json": "oss://initial/report.json"},
        dispatch={"status": "created", "workitem_id": "old-workitem"},
    )

    forced = await store.requeue(str(job.job_id), force=True)

    assert forced is not None
    assert forced.status == "pending"
    assert forced.job_request["force_revision"] == 1
    assert forced.artifacts == {}
    assert forced.dispatch["history"][0]["revision"] == 0
    assert forced.dispatch["history"][0]["dispatch"]["workitem_id"] == "old-workitem"
    # status is required: a force snapshot missing status makes GET /{job_id} fail deserialization with a 500.
    assert forced.dispatch["status"] == "pending"
    assert await store.requeue(str(job.job_id), force=True) is None


async def test_job_store_serializes_multi_batch_stability_request(repo: TraceRecordService) -> None:
    del repo
    store = ClusterReportJobStore(environment="test", evaluator_name="vehicle_agent")
    job, _ = await store.create_or_get(
        ReportJobCreate(
            job_type="stability",
            source_batch_tags=["run-20260908-101000", "run-20260908-100000"],
            job_request={"meta": {"purpose": "stability"}},
        )
    )

    serialized = store.serialize(job)

    assert serialized["jobType"] == "stability"
    assert serialized["sourceBatchTags"] == [
        "run-20260908-101000",
        "run-20260908-100000",
    ]


async def test_job_store_serializes_persisted_evaluator_snapshot(repo: TraceRecordService) -> None:
    del repo
    store = ClusterReportJobStore(environment="test", evaluator_name="vehicle_agent")
    snapshot = {
        "version": 1,
        "evaluator_name": "vehicle_agent",
        "source": "remote_config",
        "policy": {"collection": {"model_allowlist": ["vehicle-model"]}},
    }
    job, _ = await store.create_or_get(
        ReportJobCreate(
            job_type="normal",
            source_batch_tags=["run-20260908-100000"],
            job_request={"evaluator_config_snapshot": snapshot},
        )
    )

    serialized = store.serialize(job)

    assert serialized["evaluatorConfigSnapshot"] == snapshot


@pytest.mark.parametrize("new_owner", ["same-instance", "replacement-instance"])
async def test_old_attempt_cannot_mutate_same_owner_reacquired_lease(
    repo: TraceRecordService,
    new_owner: str,
) -> None:
    del repo
    store = ClusterReportJobStore(environment="test", evaluator_name="vehicle_agent")
    job, _ = await store.create_or_get(ReportJobCreate("normal", ["run-20261030-100000"], {}))
    old = await store.acquire(job_id=str(job.job_id), owner="same-instance")
    assert old is not None
    await ClusterReportJob.filter(job_id=job.job_id).update(
        lease_expire_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    current = await store.acquire(job_id=str(job.job_id), owner=new_owner)
    assert current is not None and current.attempt == old.attempt + 1
    old_lease = {
        "job_id": str(job.job_id),
        "owner": "same-instance",
        "expected_attempt": old.attempt,
    }
    assert await store.renew(**old_lease) is False
    assert (
        await store.update_owned(**old_lease, phase="stale", artifacts={"report": "old"}) is False
    )
    assert (
        await store.finish(
            ReportJobFinish(**old_lease, status="completed", phase="done"),
            artifacts={"report": "old"},
        )
        is False
    )
    current_lease = {**old_lease, "owner": new_owner, "expected_attempt": current.attempt}
    assert await store.renew(**current_lease) is True
    assert (
        await store.finish(
            ReportJobFinish(**current_lease, status="completed", phase="done"),
            artifacts={"report": "new"},
        )
        is True
    )
    saved = await store.get(str(job.job_id))
    assert saved is not None and saved.artifacts == {"report": "new"}


async def test_expired_lease_cannot_write_without_reacquisition(repo: TraceRecordService) -> None:
    del repo
    store = ClusterReportJobStore(environment="test", evaluator_name="vehicle_agent")
    job, _ = await store.create_or_get(ReportJobCreate("normal", ["run-20261030-110000"], {}))
    acquired = await store.acquire(job_id=str(job.job_id), owner="expired")
    assert acquired is not None
    await ClusterReportJob.filter(job_id=job.job_id).update(
        lease_expire_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    lease = {"job_id": str(job.job_id), "owner": "expired", "expected_attempt": acquired.attempt}
    assert await store.renew(**lease) is False
    assert await store.update_owned(**lease, phase="stale") is False
    assert await store.finish(ReportJobFinish(**lease, status="completed", phase="done")) is False
