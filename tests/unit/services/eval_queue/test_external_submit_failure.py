"""external_worker submit failure branches — retryable requeue / non-retryable force_fail."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.workers.external_worker import (
    ExternalWorkerDeps,
    process_external_wait_case,
)
from benchmark.utils.http_transport import ConnectError


def _base_job(*, retry_count: int = 0) -> MagicMock:
    job = MagicMock()
    job.id = "run1:item1"
    job.data = {
        "run_id": "run1",
        "item_id": "item1",
        "retry_count": retry_count,
        "first_enqueued_at": int(time.time() * 1000),
        "input": {"query": "q"},
        "output": {"response_text": "a"},
        "metadata": {},
        "ds": "20260817",
    }
    return job


def _settings() -> MagicMock:
    s = MagicMock()
    s.external_running_ttl_ms = 3_600_000
    s.external_cluster_notification_soft_cap = 0
    s.external_slots_per_run_per_node = 1
    s.external_slot_busy_delay_ms = 1000
    s.max_slot_probes_per_run = 100
    s.external_retry_max_duration_ms = 86_400_000
    s.eval_queue_max_attempts = 5
    s.eval_queue_backoff_initial_ms = 1000
    s.eval_queue_backoff_max_ms = 10_000
    return s


@pytest.mark.asyncio
async def test_submit_retryable_calls_requeue() -> None:
    job = _base_job()
    wait_q = MagicMock()
    running_q = MagicMock()
    running_q.add = AsyncMock()
    running_q.getJob = AsyncMock(return_value=None)

    qm = MagicMock()
    qm.prefix = "p"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    client = MagicMock()
    client.submit_evaluation = AsyncMock(
        side_effect=ConnectError("down"),
    )
    run_service = MagicMock()
    run_service.heartbeat = AsyncMock()

    with (
        patch(
            "benchmark.workers.external_worker._is_run_terminal",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.workers.external_worker.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.external_worker.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="requeued",
        ) as requeue,
        patch(
            "benchmark.workers.external_worker.force_fail_from_running",
            new_callable=AsyncMock,
        ) as force,
        patch(
            "benchmark.workers.external_worker.tick_and_maybe_finalize",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
        patch(
            "benchmark.workers.external_worker.prewrite_running_fused_item",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        await process_external_wait_case(
            job,
            "tok-test",
            queue_manager=qm,
            deps=ExternalWorkerDeps(
                run_service=run_service,
                external_judge_client=client,
                settings=_settings(),
            ),
        )

    release.assert_awaited()
    requeue.assert_awaited_once()
    force.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_non_retryable_calls_force_fail() -> None:
    job = _base_job()
    wait_q = MagicMock()
    running_q = MagicMock()
    running_q.add = AsyncMock()
    running_q.getJob = AsyncMock(return_value=None)

    qm = MagicMock()
    qm.prefix = "p"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    run_service = MagicMock()
    run_service.heartbeat = AsyncMock()

    with (
        patch(
            "benchmark.workers.external_worker._is_run_terminal",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.workers.external_worker.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.external_worker.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
        patch(
            "benchmark.workers.external_worker.force_fail_from_running",
            new_callable=AsyncMock,
            return_value="failed",
        ) as force,
        patch(
            "benchmark.workers.external_worker.tick_and_maybe_finalize",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
        patch(
            "benchmark.workers.external_worker.prewrite_running_fused_item",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        await process_external_wait_case(
            job,
            "tok-test",
            queue_manager=qm,
            deps=ExternalWorkerDeps(
                run_service=run_service,
                external_judge_client=None,  # → RuntimeError not configured
                settings=_settings(),
            ),
        )

    force.assert_awaited_once()
    requeue.assert_not_awaited()
