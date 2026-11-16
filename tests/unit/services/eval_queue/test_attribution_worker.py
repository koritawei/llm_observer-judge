"""Unit tests for attribution_worker — an independent attr slot, an unknown provider, and an early callback."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
AttributionSubmitResult = None
from benchmark.workers.attribution_worker import (
    AttributionWorkerDeps,
    process_attribution_wait_case,
)


def _job() -> MagicMock:
    job = MagicMock()
    job.id = "run1:item1"
    job.data = {
        "run_id": "run1",
        "item_id": "item1",
        "retry_count": 0,
        "first_enqueued_at": int(time.time() * 1000),
        "source_trace_id": "judge-job-1",
        "attribution_provider": "external",
        "adiu": "adiu-1",
        "ds_start": "20260817",
        "issue": {"criterion_scores": {}},
        "query": "测试问题",
        "response": "测试回答",
        "input": {},
        "output": {},
        "metadata": {},
    }
    return job


def _settings() -> MagicMock:
    s = MagicMock()
    s.external_running_ttl_ms = 86_400_000
    s.attr_cluster_soft_cap = 4
    s.attr_slots_per_run_per_node = 2
    s.external_cluster_notification_soft_cap = 99
    s.external_slots_per_run_per_node = 99
    s.external_slot_busy_delay_ms = 1000
    s.max_slot_probes_per_run = 2
    s.external_retry_max_duration_ms = 7_200_000
    s.eval_queue_max_attempts = 3
    s.eval_queue_backoff_initial_ms = 1000
    s.eval_queue_backoff_max_ms = 5000
    s.attr_callback_url = "https://example/callback/{run_id}"
    return s


@pytest.mark.asyncio
async def test_worker_uses_independent_attr_slot_limits() -> None:
    """The attribution worker uses the attr_* limits, and does not reuse the judge cluster/per-node cap."""
    captured: dict[str, object] = {}

    async def _capture(job: object, token: str, runtime: object, **_: object) -> None:
        handlers = runtime.handlers  # type: ignore[attr-defined]
        captured["kind"] = handlers.cluster_slot_kind  # type: ignore[attr-defined]
        captured["soft_cap"] = handlers.slot_config.cluster_soft_cap  # type: ignore[attr-defined]
        captured["slots"] = handlers.slot_config.slots_per_run_per_node  # type: ignore[attr-defined]

    with patch(
        "benchmark.workers.attribution_worker.process_sync_eval_case",
        side_effect=_capture,
    ):
        await process_attribution_wait_case(
            _job(),
            "tok",
            queue_manager=MagicMock(),
            deps=AttributionWorkerDeps(
                run_service=MagicMock(),
                attribution_provider_registry=MagicMock(),
                settings=_settings(),
            ),
        )

    assert captured["kind"] == "attr"
    assert captured["soft_cap"] == 4
    assert captured["slots"] == 2


@pytest.mark.asyncio
async def test_unknown_provider_is_non_retryable_force_fail() -> None:
    """An unregistered provider → ValueError → force_fail, and do not requeue."""
    job = _job()
    job.data["attribution_provider"] = "missing"
    running_q = MagicMock()
    running_q.add = AsyncMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    registry = MagicMock()
    registry.resolve_for_job.return_value = None
    run_service = MagicMock()
    run_service.heartbeat = AsyncMock()
    run_service.increment_attribution_received_once = AsyncMock(
        return_value=(1, {}, True),
    )
    run_service.get_run = AsyncMock(return_value=MagicMock(status="running"))

    with (
        patch(
            "benchmark.workers.attribution_worker._is_run_terminal",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_cluster_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.workers.attribution_worker.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.attribution_worker.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
        patch(
            "benchmark.workers.attribution_worker.force_fail_from_running",
            new_callable=AsyncMock,
        ) as force_fail,
        patch(
            "benchmark.workers.attribution_worker.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
        patch(
            "benchmark.workers.attribution_worker.tick_and_maybe_finalize",
            new_callable=AsyncMock,
        ),
    ):
        await process_attribution_wait_case(
            job,
            "tok",
            queue_manager=qm,
            deps=AttributionWorkerDeps(
                run_service=run_service,
                attribution_provider_registry=registry,
                settings=_settings(),
            ),
        )

    force_fail.assert_awaited_once()
    requeue.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_success_promotes_pending_callback() -> None:
    """Submit succeeds → write attribution_job_id → inflight is promoted to pending_callback."""
    job = _job()
    inflight = MagicMock()
    inflight.data = {"slot_key": "pre:bullmq:inflight:attr:n1:run1", "run_id": "run1"}
    inflight.remove = AsyncMock()
    running_q = MagicMock()
    running_q.add = AsyncMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    provider = MagicMock()
    provider.submit = AsyncMock(
        return_value=AttributionSubmitResult(
            job_id="attr-task-1",
            response_item_id="run1:item1",
        ),
    )
    registry = MagicMock()
    registry.resolve_for_job.return_value = provider
    run_service = MagicMock()
    run_service.heartbeat = AsyncMock()
    run_service.update_attribution_job_id = AsyncMock()
    run_service.is_attribution_received = AsyncMock(return_value=False)
    run_service.get_run = AsyncMock(return_value=MagicMock(status="running"))

    with (
        patch(
            "benchmark.workers.attribution_worker._is_run_terminal",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_cluster_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
        patch(
            "benchmark.workers.attribution_worker.get_queue_job",
            new_callable=AsyncMock,
            return_value=inflight,
        ),
        patch(
            "benchmark.workers.attribution_worker.add_running_with_retry",
            new_callable=AsyncMock,
        ) as add_running,
        patch(
            "benchmark.workers.attribution_worker.tick_and_maybe_finalize",
            new_callable=AsyncMock,
        ),
    ):
        await process_attribution_wait_case(
            job,
            "tok",
            queue_manager=qm,
            deps=AttributionWorkerDeps(
                run_service=run_service,
                attribution_provider_registry=registry,
                settings=_settings(),
            ),
        )

    run_service.update_attribution_job_id.assert_awaited_once_with(
        run_id="run1",
        item_id="item1",
        attribution_job_id="attr-task-1",
    )
    add_running.assert_awaited()
    # The BullMQ jobId of pending_callback uses case_job_id (= {run_id}:{item_id}),
    # not the attribution_task_id returned by Aone, so jobs of different cases do not overwrite each other
    assert add_running.await_args.args[1].job_id == "run1:item1"
    assert add_running.await_args.args[1].name == "pending_callback"
    inflight.remove.assert_awaited_once()


@pytest.mark.asyncio
async def test_early_callback_releases_slot_after_pending_add() -> None:
    """The callback arrives before pending_callback → the worker releases the slot immediately after the promotion."""
    job = _job()
    inflight = MagicMock()
    inflight.data = {"slot_key": "pre:bullmq:inflight:attr:n1:run1"}
    inflight.remove = AsyncMock()
    pending = MagicMock()
    pending.remove = AsyncMock()
    running_q = MagicMock()
    running_q.add = AsyncMock()
    wait_q = MagicMock()
    wait_q.add = AsyncMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    provider = MagicMock()
    provider.submit = AsyncMock(
        return_value=AttributionSubmitResult(
            job_id="attr-task-1",
            response_item_id="run1:item1",
        ),
    )
    registry = MagicMock()
    registry.resolve_for_job.return_value = provider
    run_service = MagicMock()
    run_service.heartbeat = AsyncMock()
    run_service.update_attribution_job_id = AsyncMock()
    run_service.is_attribution_received = AsyncMock(return_value=True)
    run_service.get_run = AsyncMock(return_value=MagicMock(status="running"))

    # pending_callback and inflight share case_job_id;
    # the first lookup is inflight, and after remove the lookup is pending
    removed = set()

    async def _remove_inflight() -> None:
        removed.add("run1:item1")

    inflight.remove = AsyncMock(side_effect=_remove_inflight)

    async def _get_job(_q: object, job_id: str) -> MagicMock | None:
        if job_id == "run1:item1":
            return pending if "run1:item1" in removed else inflight
        return None

    with (
        patch(
            "benchmark.workers.attribution_worker._is_run_terminal",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_cluster_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.workers.attribution_worker.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.attribution_worker.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
        patch(
            "benchmark.workers.attribution_worker.get_queue_job",
            side_effect=_get_job,
        ),
        patch(
            "benchmark.workers.attribution_worker.add_running_with_retry",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.attribution_worker.promote_delayed_for_run",
            new_callable=AsyncMock,
            return_value=1,
        ) as promote,
        patch(
            "benchmark.workers.attribution_worker.tick_and_maybe_finalize",
            new_callable=AsyncMock,
        ),
    ):
        await process_attribution_wait_case(
            job,
            "tok",
            queue_manager=qm,
            deps=AttributionWorkerDeps(
                run_service=run_service,
                attribution_provider_registry=registry,
                settings=_settings(),
            ),
        )

    pending.remove.assert_awaited_once()
    release.assert_awaited()
    release_cluster.assert_any_await("attr")
    promote.assert_awaited()


@pytest.mark.asyncio
async def test_early_callback_skips_release_when_pending_already_gone() -> None:
    """pending was already deleted by the callback → the worker must not DECR again (so a double release does not punch through the limit)."""
    job = _job()
    inflight = MagicMock()
    inflight.data = {"slot_key": "pre:bullmq:inflight:attr:n1:run1"}
    inflight.remove = AsyncMock()
    running_q = MagicMock()
    running_q.add = AsyncMock()
    wait_q = MagicMock()
    wait_q.add = AsyncMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    provider = MagicMock()
    provider.submit = AsyncMock(
        return_value=AttributionSubmitResult(
            job_id="attr-task-1",
            response_item_id="run1:item1",
        ),
    )
    registry = MagicMock()
    registry.resolve_for_job.return_value = provider
    run_service = MagicMock()
    run_service.heartbeat = AsyncMock()
    run_service.update_attribution_job_id = AsyncMock()
    run_service.is_attribution_received = AsyncMock(return_value=True)
    run_service.get_run = AsyncMock(return_value=MagicMock(status="running"))

    # pending was already removed by the callback: only inflight can be found
    # After inflight is removed, case_job_id finds no job
    removed = set()

    async def _remove_inflight() -> None:
        removed.add("run1:item1")

    inflight.remove = AsyncMock(side_effect=_remove_inflight)

    async def _get_job(_q: object, job_id: str) -> MagicMock | None:
        if job_id in removed:
            return None
        return inflight if job_id == "run1:item1" else None

    with (
        patch(
            "benchmark.workers.attribution_worker._is_run_terminal",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_cluster_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.workers.attribution_worker.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.attribution_worker.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
        patch(
            "benchmark.workers.attribution_worker.get_queue_job",
            side_effect=_get_job,
        ),
        patch(
            "benchmark.workers.attribution_worker.add_running_with_retry",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.attribution_worker.promote_delayed_for_run",
            new_callable=AsyncMock,
            return_value=1,
        ) as promote,
        patch(
            "benchmark.workers.attribution_worker.tick_and_maybe_finalize",
            new_callable=AsyncMock,
        ),
    ):
        await process_attribution_wait_case(
            job,
            "tok",
            queue_manager=qm,
            deps=AttributionWorkerDeps(
                run_service=run_service,
                attribution_provider_registry=registry,
                settings=_settings(),
            ),
        )

    release.assert_not_awaited()
    release_cluster.assert_not_awaited()
    promote.assert_not_awaited()


@pytest.mark.asyncio
async def test_response_item_id_mismatch_still_uses_local_item_id_for_pg() -> None:
    """When response_item_id does not match task_id, PG is still written with the local item_id."""
    job = _job()
    inflight = MagicMock()
    inflight.data = {"slot_key": "pre:bullmq:inflight:attr:n1:run1"}
    inflight.remove = AsyncMock()
    running_q = MagicMock()
    running_q.add = AsyncMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    provider = MagicMock()
    # The task_id echoed by Aone does not match what was expected (for example a dedup hit on another case)
    provider.submit = AsyncMock(
        return_value=AttributionSubmitResult(
            job_id="attr-task-1",
            response_item_id="run1:item_other",
        ),
    )
    registry = MagicMock()
    registry.resolve_for_job.return_value = provider
    run_service = MagicMock()
    run_service.heartbeat = AsyncMock()
    run_service.update_attribution_job_id = AsyncMock()
    run_service.is_attribution_received = AsyncMock(return_value=False)
    run_service.get_run = AsyncMock(return_value=MagicMock(status="running"))

    with (
        patch(
            "benchmark.workers.attribution_worker._is_run_terminal",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.acquire_cluster_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
        patch(
            "benchmark.workers.attribution_worker.get_queue_job",
            new_callable=AsyncMock,
            return_value=inflight,
        ),
        patch(
            "benchmark.workers.attribution_worker.add_running_with_retry",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.attribution_worker.tick_and_maybe_finalize",
            new_callable=AsyncMock,
        ),
    ):
        await process_attribution_wait_case(
            job,
            "tok",
            queue_manager=qm,
            deps=AttributionWorkerDeps(
                run_service=run_service,
                attribution_provider_registry=registry,
                settings=_settings(),
            ),
        )

    # The PG write always uses the local item_id, not response_item_id (= task_id)
    run_service.update_attribution_job_id.assert_awaited_once_with(
        run_id="run1",
        item_id="item1",
        attribution_job_id="attr-task-1",
    )
