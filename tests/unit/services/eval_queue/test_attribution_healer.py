"""Unit tests for the attribution healer — _heal_attribution / callback_timeout / inflight / slot_counters / the distributed lock.

Covers the v2.1 multi-node safety fixes:
- remove-before-release (prevents a double DECR across nodes)
- the distributed lock (prevents nodes from sweeping twice)
- SCAN of orphan keys (prevents a counter leak)
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.workers.running_healer import (
    HealerCallbacks,
    HealJobContext,
    RunningHealer,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "sync_slot_heal_interval_s": 60,
        "external_slot_heal_interval_s": 60,
        "eval_queue_lock_duration_ms": 1_200_000,
        "external_wait_lock_duration_ms": 300_000,
        "external_running_ttl_ms": 86_400_000,
        "eval_queue_max_attempts": 3,
        "sync_retry_max_duration_ms": 7_200_000,
        "external_retry_max_duration_ms": 3_600_000,
        "eval_queue_backoff_initial_ms": 5000,
        "eval_queue_backoff_max_ms": 25000,
        "attr_callback_timeout_ms": 600_000,
        "attr_callback_url": "",
        "external_callback_timeout_ms": 600_000,
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
    job.updateData = AsyncMock()
    return job


async def _list_via_get_jobs(
    queue: MagicMock,
    types: list[str],
    **_: Any,
) -> list[Any]:
    return await queue.getJobs(types, 0, -1)


def _patch_list_queue_jobs() -> Any:
    return patch(
        "benchmark.workers.running_healer.list_queue_jobs",
        new_callable=AsyncMock,
        side_effect=_list_via_get_jobs,
    )


# ── _heal_attribution: dispatch ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heal_attribution_dispatches_callback_timeout() -> None:
    """pending_callback past callback_timeout → call _heal_attribution_callback_timeout."""
    now = 2_000_000
    job = _job(
        job_id="bm_1:i1",
        name="pending_callback",
        data={
            "run_id": "bm_1",
            "item_id": "i1",
            "attribution_job_id": "attr-task-001",
            "slot_key": "pre:inflight:attr:node1:bm_1",
            "attribution_provider": "external",
        },
        timestamp=now - 87_000_000,  # Past the 24h timeout
    )
    running_q = MagicMock()
    running_q.getJobs = AsyncMock(return_value=[job])
    wait_q = MagicMock()

    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with (
        _patch_list_queue_jobs(),
        patch.object(
            healer,
            "_heal_attribution_callback_timeout",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_cb,
    ):
        timeout_n, clean_n = await healer._heal_attribution()

    assert timeout_n == 1
    assert clean_n == 0
    mock_cb.assert_awaited_once()


@pytest.mark.asyncio
async def test_heal_attribution_dispatches_inflight() -> None:
    """inflight past grace_ms → call _heal_attribution_inflight."""
    now = 2_000_000
    job = _job(
        job_id="bm_1:i1",
        name="inflight",
        data={
            "run_id": "bm_1",
            "item_id": "i1",
            "slot_key": "pre:inflight:attr:node1:bm_1",
        },
        timestamp=now - 400_000,  # Past the 300s grace
    )
    running_q = MagicMock()
    running_q.getJobs = AsyncMock(return_value=[job])
    wait_q = MagicMock()

    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with (
        _patch_list_queue_jobs(),
        patch.object(
            healer,
            "_heal_attribution_inflight",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_inf,
    ):
        timeout_n, clean_n = await healer._heal_attribution()

    assert timeout_n == 0
    assert clean_n == 1
    mock_inf.assert_awaited_once()


@pytest.mark.asyncio
async def test_heal_attribution_skips_young_jobs() -> None:
    """A job that has not timed out and is inside the 1h window is skipped."""
    now = 2_000_000
    job = _job(
        job_id="bm_1:i1",
        name="pending_callback",
        data={
            "run_id": "bm_1",
            "item_id": "i1",
            "first_enqueued_at": now - 100_000,
        },
        timestamp=now - 100_000,  # Far below callback_timeout
    )
    running_q = MagicMock()
    running_q.getJobs = AsyncMock(return_value=[job])
    wait_q = MagicMock()

    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with (
        _patch_list_queue_jobs(),
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
    ):
        timeout_n, clean_n = await healer._heal_attribution()

    assert timeout_n == 0
    assert clean_n == 0


# ── _heal_attribution_callback_timeout ───────────────────────────────────────


@pytest.mark.asyncio
async def test_callback_timeout_removes_then_releases_and_increments() -> None:
    """remove-before-release: remove the job first → release the slot → increment the counter → finalize."""
    now = 2_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "attribution_job_id": "attr-task-001",
        "slot_key": "pre:inflight:attr:node1:bm_1",
        "attribution_provider": "external",
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="pending_callback",
        data=job_data,
        timestamp=now - 87_000_000,
    )

    running_q = MagicMock()
    running_q.getJob = AsyncMock(return_value=running_job)
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    bench_service = MagicMock()
    bench_service.increment_attribution_received_once = AsyncMock(
        return_value=(1, {}, True),
    )
    bench_service.is_attribution_received = AsyncMock(return_value=False)

    healer = RunningHealer(
        queue_manager=qm,
        settings=_settings(),
        callbacks=HealerCallbacks(benchmark_run_service=bench_service),
    )

    with (
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
        patch(
            "benchmark.workers.running_healer.try_finalize_external",
            new_callable=AsyncMock,
        ) as finalize,
        patch(
            "benchmark.workers.running_healer.get_queue_job",
            new_callable=AsyncMock,
            return_value=running_job,
        ),
    ):
        result = await healer._heal_attribution_callback_timeout(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is True
    running_job.remove.assert_awaited_once()
    release.assert_awaited_once_with("pre:inflight:attr:node1:bm_1")
    release_cluster.assert_awaited_once_with("attr")
    bench_service.increment_attribution_received_once.assert_awaited_once_with(
        "bm_1",
        "i1",
    )
    finalize.assert_awaited_once_with("bm_1", bench_service)


@pytest.mark.asyncio
async def test_callback_timeout_force_fail_when_job_already_removed() -> None:
    """Another healer already removed the job → still release the slot + force_fail (idempotent), and return True.

    Aligned with the judge TTL path: ``_fail_attribution_pending`` releases the slot before
    ``force_fail_from_running``, and always returns True (it does not check whether the job was already deleted).
    """
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "attribution_job_id": "attr-task-001",
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="pending_callback",
        data=job_data,
        timestamp=1,
    )

    running_q = MagicMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with (
        patch(
            "benchmark.workers.running_healer.get_queue_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
        patch.object(
            healer,
            "_purge_case_from_other_running",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.force_fail_from_running",
            new_callable=AsyncMock,
        ) as force_fail,
    ):
        result = await healer._heal_attribution_callback_timeout(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is True
    release_cluster.assert_awaited_once()
    force_fail.assert_awaited_once()
    running_job.remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_attr_callback_timeout_force_fail_when_case_window_exceeded() -> None:
    """More than 1h since first_enqueued_at → force_fail even if the provider is still running."""
    now = 5_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "attribution_job_id": "attr-task-001",
        "slot_key": "pre:inflight:attr:node1:bm_1",
        "attribution_provider": "external",
        "first_enqueued_at": now - 3_600_001,
        "retry_count": 0,
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="pending_callback",
        data=job_data,
        timestamp=now - 700_000,
    )
    running_q = MagicMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    bench_service = MagicMock()
    bench_service.increment_attribution_received_once = AsyncMock(
        return_value=(1, {}, True),
    )
    bench_service.is_attribution_received = AsyncMock(return_value=False)
    provider = MagicMock()
    provider.query_job_status = AsyncMock()
    registry = MagicMock()
    registry.resolve_for_job = MagicMock(return_value=provider)

    healer = RunningHealer(
        queue_manager=qm,
        settings=_settings(),
        callbacks=HealerCallbacks(
            benchmark_run_service=bench_service,
            attribution_provider_registry=registry,
        ),
    )

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.try_finalize_external",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.get_queue_job",
            new_callable=AsyncMock,
            return_value=running_job,
        ),
        patch(
            "benchmark.queue.retry_requeue.get_queue_job",
            new_callable=AsyncMock,
            return_value=running_job,
        ),
        patch.object(
            healer,
            "_purge_case_from_other_running",
            new_callable=AsyncMock,
        ),
    ):
        result = await healer._heal_attribution_callback_timeout(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is True
    provider.query_job_status.assert_not_called()
    running_job.remove.assert_awaited_once()
    release.assert_awaited_once()
    bench_service.increment_attribution_received_once.assert_awaited_once_with(
        "bm_1",
        "i1",
    )


@pytest.mark.asyncio
async def test_attr_callback_skips_completed_when_callback_in_flight() -> None:
    """The provider is completed but PG has not written it, and a full observation round has not finished → skip."""
    now = 2_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "attribution_job_id": "attr-task-001",
        "slot_key": "pre:inflight:attr:node1:bm_1",
        "attribution_provider": "external",
        "first_enqueued_at": now - 100_000,
        "retry_count": 0,
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="pending_callback",
        data=job_data,
        timestamp=now - 700_000,
    )
    running_q = MagicMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    provider = MagicMock()
    provider.query_job_status = AsyncMock(
        return_value=SimpleNamespace(status="completed", error=None),
    )
    registry = MagicMock()
    registry.resolve_for_job = MagicMock(return_value=provider)
    bench_service = MagicMock()
    bench_service.is_attribution_received = AsyncMock(return_value=False)

    healer = RunningHealer(
        queue_manager=qm,
        settings=_settings(),
        callbacks=HealerCallbacks(
            attribution_provider_registry=registry,
            benchmark_run_service=bench_service,
        ),
    )

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release,
    ):
        result = await healer._heal_attribution_callback_timeout(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is False
    requeue.assert_not_called()
    release.assert_not_called()
    running_job.updateData.assert_awaited_once()
    stamped = running_job.updateData.await_args.args[0]
    assert stamped["callback_completed_seen_at"] == now


@pytest.mark.asyncio
async def test_attr_callback_skips_completed_when_already_applied() -> None:
    """The provider is completed and attribution received is already counted → a normal completed, and do not requeue."""
    now = 2_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "attribution_job_id": "attr-task-001",
        "first_enqueued_at": now - 100_000,
        "retry_count": 0,
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="pending_callback",
        data=job_data,
        timestamp=now - 700_000,
    )
    provider = MagicMock()
    provider.query_job_status = AsyncMock(
        return_value=SimpleNamespace(status="completed", error=None),
    )
    registry = MagicMock()
    registry.resolve_for_job = MagicMock(return_value=provider)
    bench_service = MagicMock()
    bench_service.is_attribution_received = AsyncMock(return_value=True)

    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
        callbacks=HealerCallbacks(
            attribution_provider_registry=registry,
            benchmark_run_service=bench_service,
        ),
    )

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
    ):
        result = await healer._heal_attribution_callback_timeout(
            HealJobContext(
                job=running_job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=job_data,
            ),
        )

    assert result is False
    provider.query_job_status.assert_not_called()
    requeue.assert_not_called()


@pytest.mark.asyncio
async def test_attr_callback_requeues_when_provider_completed() -> None:
    """The provider is completed, the observation window has elapsed, and PG still has not written it → a lost callback, requeue onto wait."""
    now = 2_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "attribution_job_id": "attr-task-001",
        "slot_key": "pre:inflight:attr:node1:bm_1",
        "attribution_provider": "external",
        "first_enqueued_at": now - 100_000,
        "retry_count": 0,
        "callback_completed_seen_at": now - 600_000,
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="pending_callback",
        data=job_data,
        timestamp=now - 700_000,
    )
    running_q = MagicMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    provider = MagicMock()
    provider.query_job_status = AsyncMock(
        return_value=SimpleNamespace(status="completed", error=None),
    )
    registry = MagicMock()
    registry.resolve_for_job = MagicMock(return_value=provider)
    bench_service = MagicMock()
    bench_service.is_attribution_received = AsyncMock(return_value=False)

    healer = RunningHealer(
        queue_manager=qm,
        settings=_settings(),
        callbacks=HealerCallbacks(
            attribution_provider_registry=registry,
            benchmark_run_service=bench_service,
        ),
    )

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="requeued",
        ) as requeue,
    ):
        result = await healer._heal_attribution_callback_timeout(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is True
    requeue.assert_awaited_once()
    assert requeue.await_args.args[2].wait_job_name == "attr_case"


# ── _heal_attribution_inflight ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inflight_cleans_when_wait_has_stale_job() -> None:
    """wait has a job with the same ID but it is not active → remove inflight + release the slot."""
    now = 2_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "slot_key": "pre:inflight:attr:node1:bm_1",
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="inflight",
        data=job_data,
        timestamp=now - 400_000,
    )
    wait_job = MagicMock()
    wait_job.getState = AsyncMock(return_value="waiting")

    running_q = MagicMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with (
        patch(
            "benchmark.workers.running_healer.get_queue_job",
            new_callable=AsyncMock,
            return_value=wait_job,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
    ):
        result = await healer._heal_attribution_inflight(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is True
    running_job.remove.assert_awaited_once()
    release.assert_awaited_once()
    release_cluster.assert_awaited_once_with("attr")


@pytest.mark.asyncio
async def test_inflight_skips_when_wait_job_active() -> None:
    """wait has a job with the same ID and it is active → skip (the worker is processing it)."""
    now = 2_000_000
    job_data = {"run_id": "bm_1", "item_id": "i1"}
    running_job = _job(
        job_id="bm_1:i1",
        name="inflight",
        data=job_data,
        timestamp=now - 400_000,
    )
    wait_job = MagicMock()
    wait_job.getState = AsyncMock(return_value="active")

    running_q = MagicMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with patch(
        "benchmark.workers.running_healer.get_queue_job",
        new_callable=AsyncMock,
        return_value=wait_job,
    ):
        result = await healer._heal_attribution_inflight(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is False
    running_job.remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_inflight_requeues_when_wait_missing() -> None:
    """wait has no job → release the slot + requeue_or_fail_from_running puts it back on wait."""
    now = 2_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "slot_key": "pre:inflight:attr:node1:bm_1",
        "retry_count": 0,
        "first_enqueued_at": now - 100_000,
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="inflight",
        data=job_data,
        timestamp=now - 400_000,
    )
    running_q = MagicMock()
    wait_q = MagicMock()
    wait_q.add = AsyncMock()
    qm = MagicMock()
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with (
        patch(
            "benchmark.workers.running_healer.get_queue_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="requeued",
        ) as requeue,
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
    ):
        result = await healer._heal_attribution_inflight(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is True
    requeue.assert_awaited_once()
    assert requeue.await_args.args[1] == "bm_1:i1"
    assert requeue.await_args.args[2].wait_job_name == "attr_case"
    wait_q.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_inflight_terminal_fail_on_max_attempts() -> None:
    """retry_count >= max_attempts → requeue_or_fail ends in failure (increment the counter, do not requeue)."""
    now = 2_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "retry_count": 3,  # == max_attempts (3)
        "first_enqueued_at": now - 100_000,
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="inflight",
        data=job_data,
        timestamp=now - 400_000,
    )
    running_q = MagicMock()
    wait_q = MagicMock()
    wait_q.add = AsyncMock()
    qm = MagicMock()
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    bench_service = MagicMock()
    bench_service.increment_attribution_received_once = AsyncMock(
        return_value=(1, {}, True),
    )
    bench_service.is_attribution_received = AsyncMock(return_value=False)

    healer = RunningHealer(
        queue_manager=qm,
        settings=_settings(),
        callbacks=HealerCallbacks(benchmark_run_service=bench_service),
    )

    with (
        patch(
            "benchmark.workers.running_healer.get_queue_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "benchmark.queue.retry_requeue.get_queue_job",
            new_callable=AsyncMock,
            return_value=running_job,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.try_finalize_external",
            new_callable=AsyncMock,
        ) as finalize,
        patch.object(
            healer,
            "_purge_case_from_other_running",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
    ):
        result = await healer._heal_attribution_inflight(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is True
    running_job.remove.assert_awaited_once()
    wait_q.add.assert_not_awaited()
    bench_service.increment_attribution_received_once.assert_awaited_once_with(
        "bm_1",
        "i1",
    )
    finalize.assert_awaited_once_with("bm_1", bench_service)


@pytest.mark.asyncio
async def test_inflight_calls_requeue_when_first_enqueued_at_missing() -> None:
    """first_enqueued_at is missing → still go through requeue_or_fail_from_running (aligned with judge)."""
    now = 2_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "retry_count": 0,
    }
    running_job = _job(
        job_id="bm_1:i1",
        name="inflight",
        data=job_data,
        timestamp=now - 400_000,
    )
    running_q = MagicMock()
    wait_q = MagicMock()
    wait_q.add = AsyncMock()
    qm = MagicMock()
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with (
        patch(
            "benchmark.workers.running_healer.get_queue_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="requeued",
        ) as requeue,
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
    ):
        result = await healer._heal_attribution_inflight(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is True
    requeue.assert_awaited_once()
    assert requeue.await_args.args[1] == "bm_1:i1"
    assert requeue.await_args.args[2].wait_job_name == "attr_case"
    wait_q.add.assert_not_awaited()


# ── _heal_attr_slot_counters ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_attr_slot_counters_clamps_when_running_empty() -> None:
    """ATTRIBUTION_RUNNING is empty → clamp to 0."""
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    running_q = MagicMock()
    qm.get_queue = MagicMock(return_value=running_q)

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with (
        patch(
            "benchmark.workers.running_healer.list_queue_jobs",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "benchmark.workers.running_healer.build_cluster_slot_key",
            return_value="pre:bullmq:inflight:cluster:attr",
        ),
        patch(
            "benchmark.workers.running_healer.scan_slot_keys",
            new_callable=AsyncMock,
            return_value=["pre:bullmq:inflight:attr:node1:run1"],
        ),
        patch(
            "benchmark.workers.running_healer.clamp_slot_count",
            new_callable=AsyncMock,
            side_effect=[4, 2],
        ) as clamp,
    ):
        n = await healer._heal_attr_slot_counters()

    assert n == 6
    assert clamp.await_count == 2
    clamp.assert_any_await(
        "pre:bullmq:inflight:cluster:attr",
        expected=0,
        ttl_s=86400,
    )
    clamp.assert_any_await(
        "pre:bullmq:inflight:attr:node1:run1",
        expected=0,
        ttl_s=86400,
    )


@pytest.mark.asyncio
async def test_attr_slot_counters_keeps_expected_from_running() -> None:
    """ATTRIBUTION_RUNNING has jobs holding slots → clamp to the real number of held slots."""
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(return_value=MagicMock())
    healer = RunningHealer(queue_manager=qm, settings=_settings())
    job = _job(
        job_id="bm_1:i1",
        name="pending_callback",
        data={
            "run_id": "run1",
            "slot_key": "pre:bullmq:inflight:attr:node1:run1",
        },
        timestamp=1,
    )

    with (
        patch(
            "benchmark.workers.running_healer.list_queue_jobs",
            new_callable=AsyncMock,
            return_value=[job],
        ),
        patch(
            "benchmark.workers.running_healer.build_cluster_slot_key",
            return_value="pre:bullmq:inflight:cluster:attr",
        ),
        patch(
            "benchmark.workers.running_healer.scan_slot_keys",
            new_callable=AsyncMock,
            return_value=["pre:bullmq:inflight:attr:node1:run1"],
        ),
        patch(
            "benchmark.workers.running_healer.clamp_slot_count",
            new_callable=AsyncMock,
            return_value=0,
        ) as clamp,
    ):
        n = await healer._heal_attr_slot_counters()

    assert n == 0
    clamp.assert_any_await(
        "pre:bullmq:inflight:cluster:attr",
        expected=1,
        ttl_s=86400,
    )
    clamp.assert_any_await(
        "pre:bullmq:inflight:attr:node1:run1",
        expected=1,
        ttl_s=86400,
    )


# ── distributed lock ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_try_acquire_heal_lock_success() -> None:
    """SET NX EX succeeds → return True."""
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
    )
    mock_redis = MagicMock()
    mock_redis.set = AsyncMock(return_value=True)

    with patch(
        "benchmark.workers.running_healer.get_redis",
        return_value=mock_redis,
    ):
        result = await healer._try_acquire_heal_lock(
            "pre:heal:lock:attr",
            ttl_s=30,
        )

    assert result is True
    mock_redis.set.assert_awaited_once_with(
        "pre:heal:lock:attr",
        "1",
        nx=True,
        ex=30,
    )


@pytest.mark.asyncio
async def test_try_acquire_heal_lock_already_held() -> None:
    """SET NX EX fails (already held) → return False."""
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
    )
    mock_redis = MagicMock()
    mock_redis.set = AsyncMock(return_value=None)

    with patch(
        "benchmark.workers.running_healer.get_redis",
        return_value=mock_redis,
    ):
        result = await healer._try_acquire_heal_lock(
            "pre:heal:lock:attr",
            ttl_s=30,
        )

    assert result is False


@pytest.mark.asyncio
async def test_release_heal_lock() -> None:
    """Release the lock → DELETE the key."""
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
    )
    mock_redis = MagicMock()
    mock_redis.delete = AsyncMock()

    with patch(
        "benchmark.workers.running_healer.get_redis",
        return_value=mock_redis,
    ):
        await healer._release_heal_lock("pre:heal:lock:attr")

    mock_redis.delete.assert_awaited_once_with("pre:heal:lock:attr")


# ── _release_attribution_slots ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_release_attribution_slots_uses_attr_kind() -> None:
    """_release_attribution_slots is fixed at kind=attr (so it does not default to judge)."""
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
    )
    data = {"slot_key": "pre:inflight:attr:node1:run1"}

    with (
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
    ):
        await healer._release_attribution_slots(data)

    release.assert_awaited_once_with("pre:inflight:attr:node1:run1")
    release_cluster.assert_awaited_once_with("attr")


@pytest.mark.asyncio
async def test_release_attribution_slots_no_slot_key() -> None:
    """slot_key is missing → release only the cluster slot."""
    healer = RunningHealer(
        queue_manager=MagicMock(),
        settings=_settings(),
    )
    data = {}

    with (
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
    ):
        await healer._release_attribution_slots(data)

    release.assert_not_awaited()
    release_cluster.assert_awaited_once_with("attr")


@pytest.mark.asyncio
async def test_attr_callback_replays_from_get_result() -> None:
    """GET completed and metadata.attribution can be restored → go through handle_callback."""
    from benchmark.runs.common.external_client import (
        ExternalJobStatusResponse,
    )

    now = 2_000_000
    job_data = {
        "run_id": "bm_1",
        "item_id": "i1",
        "attribution_job_id": "attr-task-001",
        "slot_key": "pre:inflight:attr:node1:bm_1",
        "attribution_provider": "external",
        "first_enqueued_at": now - 100_000,
        "retry_count": 0,
    }
    running_job = _job(
        job_id="attr-task-001",
        name="pending_callback",
        data=job_data,
        timestamp=now - 700_000,
    )
    running_q = MagicMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    provider = MagicMock()
    provider.query_job_status = AsyncMock(
        return_value=ExternalJobStatusResponse.model_validate(
            {
                "job_id": "attr-task-001",
                "status": "completed",
                "params": {
                    "run_id": "bm_1",
                    "item_id": "i1",
                    "job_id": "judge-job-1",
                },
                "result": {
                    "metadata": {
                        "attribution": {
                            "is_eval_misjudge": False,
                            "primary_issue_layer": "tool",
                            "primary_issue_category": "tool_exec_error",
                            "primary_issue_detail": "参数错误",
                            "affected_dims": ["D2"],
                            "severity": "major",
                            "root_cause_chain": "工具失败",
                            "improvement_priority": "P1",
                            "improvement_suggestion": "校验参数",
                        },
                    },
                },
            },
        ),
    )
    registry = MagicMock()
    registry.resolve_for_job = MagicMock(return_value=provider)
    bench_service = MagicMock()
    bench_service.is_attribution_received = AsyncMock(return_value=False)
    handler = AsyncMock()

    healer = RunningHealer(
        queue_manager=qm,
        settings=_settings(),
        callbacks=HealerCallbacks(
            attribution_provider_registry=registry,
            benchmark_run_service=bench_service,
            external_callback_handler=handler,
        ),
    )

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.get_queue_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release,
    ):
        result = await healer._heal_attribution_callback_timeout(
            HealJobContext(
                job=running_job,
                running_q=running_q,
                wait_q=wait_q,
                data=job_data,
            ),
        )

    assert result is True
    handler.assert_awaited_once()
    payload = handler.await_args.args[0]
    assert payload.type == "attribution"
    assert payload.ieval_job_id == "attr-task-001"
    assert payload.attribution.primary_issue_layer == "tool"
    requeue.assert_not_called()
    release.assert_not_called()
    running_job.updateData.assert_not_called()
