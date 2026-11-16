"""Unit tests for running_healer — the D16 guard + cleanup/requeue of a sync orphan running."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.run_stop import STOP_RECONCILE_GRACE_S
from benchmark.workers.running_healer import (
    HealerCallbacks,
    RunningHealer,
    collect_active_run_ids,
    recover_stale_with_queue_guard,
)


async def _list_via_get_jobs(
    queue: MagicMock,
    types: list[str],
    **_: Any,
) -> list[Any]:
    """Test double: the old mock is still attached to ``queue.getJobs``."""
    return await queue.getJobs(types, 0, -1)


def _patch_list_queue_jobs() -> Any:
    return patch(
        "benchmark.workers.running_healer.list_queue_jobs",
        new_callable=AsyncMock,
        side_effect=_list_via_get_jobs,
    )


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "sync_slot_heal_interval_s": 60,
        "external_slot_heal_interval_s": 60,
        "eval_queue_lock_duration_ms": 1_200_000,
        "external_wait_lock_duration_ms": 300_000,
        "external_running_ttl_ms": 86_400_000,
        "eval_queue_max_attempts": 3,
        "sync_retry_max_duration_ms": 7_200_000,
        "external_retry_max_duration_ms": 7_200_000,
        "eval_queue_backoff_initial_ms": 5000,
        "eval_queue_backoff_max_ms": 25000,
        "external_run_max_duration_ms": 43_200_000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _job(
    *,
    job_id: str,
    name: str,
    data: dict[str, Any],
    timestamp: int,
) -> MagicMock:
    job = MagicMock()
    job.id = job_id
    job.name = name
    job.data = data
    job.timestamp = timestamp
    job.remove = AsyncMock()
    job.getState = AsyncMock(return_value="waiting")
    return job


@pytest.mark.asyncio
async def test_collect_active_run_ids_from_wait_and_running() -> None:
    wait_job = _job(
        job_id="bm_wait:i1",
        name="case",
        data={"run_id": "bm_wait", "item_id": "i1"},
        timestamp=1,
    )
    run_job = _job(
        job_id="bm_1:i2",
        name="inflight",
        data={"run_id": "bm_1", "item_id": "i2"},
        timestamp=1,
    )

    def _queue_for(name: str) -> MagicMock:
        q = MagicMock()
        if "cases" in name or name.endswith("wait"):
            q.getJobs = AsyncMock(return_value=[wait_job] if name.startswith("benchmark") else [])
        else:
            q.getJobs = AsyncMock(return_value=[run_job] if "benchmark" in name else [])
        return q

    qm = MagicMock()
    qm.get_queue = MagicMock(side_effect=_queue_for)

    with _patch_list_queue_jobs():
        run_ids = await collect_active_run_ids(qm)
    assert "bm_wait" in run_ids
    assert "bm_1" in run_ids


@pytest.mark.asyncio
async def test_recover_stale_with_queue_guard_heartbeats_active_runs() -> None:
    qm = MagicMock()
    with patch(
        "benchmark.workers.running_healer.collect_active_run_ids",
        new_callable=AsyncMock,
        return_value={"de_alive", "bm_alive"},
    ):
        heartbeat = AsyncMock()
        recover = AsyncMock(return_value=1)
        n = await recover_stale_with_queue_guard(
            queue_manager=qm,
            recover_fn=recover,
            heartbeat_fn=heartbeat,
            stale_timeout_s=600,
        )
    assert n == 1
    assert heartbeat.await_count == 2
    recover.assert_awaited_once_with(stale_timeout_s=600)


@pytest.mark.asyncio
async def test_heal_sync_orphan_cleaned_when_wait_still_present() -> None:
    """wait has already been requeued because it stalled: only clear running + the slot, and do not increment retry."""
    now = 2_000_000
    running_job = _job(
        job_id="de_1:i1",
        name="inflight",
        data={
            "run_id": "de_1",
            "item_id": "i1",
            "slot_key": "slot:de_1",
            "retry_count": 0,
            "first_enqueued_at": now - 100_000,
        },
        timestamp=now - 1_300_000,  # > lockDuration grace
    )
    wait_job = _job(
        job_id="de_1:i1",
        name="case",
        data={"run_id": "de_1", "item_id": "i1"},
        timestamp=now,
    )
    wait_job.getState = AsyncMock(return_value="waiting")

    wait_q = MagicMock()
    wait_q.getJob = AsyncMock(return_value=wait_job)
    running_q = MagicMock()
    running_q.getJobs = AsyncMock(return_value=[running_job])

    qm = MagicMock()

    def _get_queue(name: str) -> MagicMock:
        if name.endswith("running"):
            return running_q
        empty = MagicMock()
        empty.getJobs = AsyncMock(return_value=[])
        empty.getJob = AsyncMock(return_value=None)
        if "dataset-cases" in name or name == "dataset-cases":
            return wait_q
        return empty

    qm.get_queue = MagicMock(side_effect=_get_queue)

    with (
        _patch_list_queue_jobs(),
        patch("benchmark.workers.running_healer.time.time", return_value=now / 1000),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
    ):
        healer = RunningHealer(queue_manager=qm, settings=_settings())
        # only exercise dataset pair via tick internals
        cleaned, requeued = await healer._heal_sync_pair(
            wait_name="dataset-cases",
            running_name="dataset-running",
            mark_failed=AsyncMock(),
        )

    assert cleaned == 1
    assert requeued == 0
    running_job.remove.assert_awaited_once()
    release.assert_awaited_once_with("slot:de_1")
    requeue.assert_not_awaited()


@pytest.mark.asyncio
async def test_heal_sync_orphan_requeues_when_wait_missing() -> None:
    now = 2_000_000
    running_job = _job(
        job_id="de_1:i1",
        name="inflight",
        data={
            "run_id": "de_1",
            "item_id": "i1",
            "slot_key": "slot:de_1",
            "retry_count": 0,
            "first_enqueued_at": now - 100_000,
        },
        timestamp=now - 1_300_000,
    )
    wait_q = MagicMock()
    wait_q.getJob = AsyncMock(return_value=None)
    running_q = MagicMock()
    running_q.getJobs = AsyncMock(return_value=[running_job])

    qm = MagicMock()
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    mark_failed = AsyncMock()

    with (
        _patch_list_queue_jobs(),
        patch("benchmark.workers.running_healer.time.time", return_value=now / 1000),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="requeued",
        ) as requeue,
    ):
        healer = RunningHealer(queue_manager=qm, settings=_settings())
        cleaned, requeued = await healer._heal_sync_pair(
            wait_name="dataset-cases",
            running_name="dataset-running",
            mark_failed=mark_failed,
        )

    assert cleaned == 0
    assert requeued == 1
    requeue.assert_awaited_once()
    assert requeue.await_args.args[1] == "de_1:i1"


@pytest.mark.asyncio
async def test_heal_skips_young_inflight() -> None:
    now = 2_000_000
    running_job = _job(
        job_id="de_1:i1",
        name="inflight",
        data={"run_id": "de_1", "item_id": "i1", "slot_key": "s"},
        timestamp=now - 10_000,  # younger than lockDuration
    )
    running_q = MagicMock()
    running_q.getJobs = AsyncMock(return_value=[running_job])
    wait_q = MagicMock()
    wait_q.getJob = AsyncMock()

    qm = MagicMock()
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    with patch(
        "benchmark.workers.running_healer.time.time",
        return_value=now / 1000,
    ):
        healer = RunningHealer(queue_manager=qm, settings=_settings())
        cleaned, requeued = await healer._heal_sync_pair(
            wait_name="dataset-cases",
            running_name="dataset-running",
            mark_failed=AsyncMock(),
        )

    assert cleaned == 0
    assert requeued == 0
    wait_q.getJob.assert_not_called()
    running_job.remove.assert_not_called()


@pytest.mark.asyncio
async def test_run_deadline_fails_overdue_external_run() -> None:
    """run.created_at is past 12h → clear the queue and fail_run, independent of the single-case 1h window."""
    run = SimpleNamespace(run_id="ieval_external-1")
    qs = MagicMock()
    qs.only = MagicMock(return_value=_AwaitableList([run]))

    svc = MagicMock()
    svc.fail_run = AsyncMock()
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(external_run_max_duration_ms=43_200_000),
        callbacks=HealerCallbacks(benchmark_run_service=svc),
    )
    healer._drain_queue_jobs_for_run = AsyncMock(return_value=2)

    with patch(
        "benchmark.records.models.benchmark_run.BenchmarkRun.filter",
        return_value=qs,
    ):
        n = await healer._heal_external_run_deadline()

    assert n == 1
    svc.fail_run.assert_awaited_once_with(
        run_id="ieval_external-1",
        reason="run_max_duration",
    )
    healer._drain_queue_jobs_for_run.assert_awaited_once_with(
        "ieval_external-1",
    )


class _AwaitableList(list):
    def __await__(self):
        async def _inner() -> list[Any]:
            return list(self)

        return _inner().__await__()


@pytest.mark.asyncio
async def test_reconcile_stuck_stop_defers_within_grace() -> None:
    """Inside the mark→align grace period: even with nothing in flight, do not push stop_target / finalize."""
    now = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)
    run = SimpleNamespace(
        run_id="bm-grace",
        evaluated_items=2,
        failed_items=0,
        config_json={
            "stop_requested_at": (now - timedelta(seconds=5)).isoformat(),
            "stop_target_count": 10,
        },
    )
    svc = MagicMock()
    svc.get_run = AsyncMock(return_value=run)
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
        callbacks=HealerCallbacks(benchmark_run_service=svc),
    )
    with (
        patch(
            "benchmark.workers.running_healer.stop_reconcile_in_grace",
            return_value=True,
        ),
        patch.object(
            healer,
            "_run_has_stop_inflight",
            new=AsyncMock(return_value=False),
        ) as has_inflight,
        patch(
            "benchmark.workers.running_healer.reconcile_stop_target_to_processed",
            new=AsyncMock(),
        ) as reconcile,
        patch.object(
            healer,
            "_finalize_after_stop_reconcile",
            new=AsyncMock(),
        ) as finalize,
    ):
        ok = await healer._reconcile_one_stuck_stop("bm-grace", svc)

    assert ok is False
    has_inflight.assert_not_awaited()
    reconcile.assert_not_awaited()
    finalize.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_stuck_stop_after_grace_when_no_inflight() -> None:
    """Outside the grace period and nothing is really in flight: push the target and finalize."""
    aged = (datetime.now(UTC) - timedelta(seconds=STOP_RECONCILE_GRACE_S + 5)).isoformat()
    run = SimpleNamespace(
        run_id="bm-aged",
        evaluated_items=3,
        failed_items=0,
        config_json={
            "stop_requested_at": aged,
            "stop_target_count": 99,
        },
    )
    svc = MagicMock()
    svc.get_run = AsyncMock(return_value=run)
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
        callbacks=HealerCallbacks(benchmark_run_service=svc),
    )
    with (
        patch.object(
            healer,
            "_run_has_stop_inflight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "benchmark.workers.running_healer.count_running_benchmark_items",
            new=AsyncMock(return_value=0),
        ),
        patch(
            "benchmark.workers.running_healer.reconcile_stop_target_to_processed",
            new=AsyncMock(return_value={"stop_target_count": 3}),
        ) as reconcile,
        patch.object(
            healer,
            "_finalize_after_stop_reconcile",
            new=AsyncMock(),
        ) as finalize,
    ):
        ok = await healer._reconcile_one_stuck_stop("bm-aged", svc)

    assert ok is True
    reconcile.assert_awaited_once_with("bm-aged")
    finalize.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_stuck_stop_waits_running_items() -> None:
    """Nothing is in flight on the queue but running items remain: do not push the target / finalize."""
    aged = (datetime.now(UTC) - timedelta(seconds=STOP_RECONCILE_GRACE_S + 5)).isoformat()
    run = SimpleNamespace(
        run_id="bm-items",
        evaluated_items=1,
        failed_items=0,
        config_json={
            "stop_requested_at": aged,
            "stop_target_count": 9,
        },
    )
    svc = MagicMock()
    svc.get_run = AsyncMock(return_value=run)
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
        callbacks=HealerCallbacks(benchmark_run_service=svc),
    )
    with (
        patch.object(
            healer,
            "_run_has_stop_inflight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "benchmark.workers.running_healer.count_running_benchmark_items",
            new=AsyncMock(return_value=2),
        ),
        patch(
            "benchmark.workers.running_healer.reconcile_stop_target_to_processed",
            new=AsyncMock(),
        ) as reconcile,
        patch.object(
            healer,
            "_finalize_after_stop_reconcile",
            new=AsyncMock(),
        ) as finalize,
    ):
        ok = await healer._reconcile_one_stuck_stop("bm-items", svc)

    assert ok is False
    reconcile.assert_not_awaited()
    finalize.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_stuck_stop_waits_attribution_pending() -> None:
    """Nothing is in flight on the queue and there are no running items, but attribution is not complete: do not push the target / finalize."""
    aged = (datetime.now(UTC) - timedelta(seconds=STOP_RECONCILE_GRACE_S + 5)).isoformat()
    run = SimpleNamespace(
        run_id="bm-attr",
        evaluated_items=2,
        failed_items=0,
        config_json={
            "stop_requested_at": aged,
            "stop_target_count": 2,
            "attribution_expected_count": 2,
            "attribution_received_count": 0,
            "attribution_expected_item_ids": ["a", "b"],
            "attribution_received_item_ids": [],
        },
    )
    svc = MagicMock()
    svc.get_run = AsyncMock(return_value=run)
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
        callbacks=HealerCallbacks(benchmark_run_service=svc),
    )
    with (
        patch.object(
            healer,
            "_run_has_stop_inflight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "benchmark.workers.running_healer.count_running_benchmark_items",
            new=AsyncMock(return_value=0),
        ),
        patch(
            "benchmark.workers.running_healer.reconcile_stop_target_to_processed",
            new=AsyncMock(),
        ) as reconcile,
        patch.object(
            healer,
            "_finalize_after_stop_reconcile",
            new=AsyncMock(),
        ) as finalize,
    ):
        ok = await healer._reconcile_one_stuck_stop("bm-attr", svc)

    assert ok is False
    reconcile.assert_not_awaited()
    finalize.assert_not_awaited()
