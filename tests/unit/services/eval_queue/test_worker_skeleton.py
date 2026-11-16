"""Unit tests for the shared worker_skeleton processor.

Covers three key paths:
1. retry deadline → the wait path fails terminally and writes input/output
2. a job is delayed when cluster soft_cap > 0
3. behavior is unchanged when cluster soft_cap = 0
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.queue.worker_skeleton import (
    SKIP_HOLD_SLOT_FLAG,
    JudgeWriteCall,
    SlotConfig,
    SyncEvalHandlers,
    SyncEvalRuntime,
    process_sync_eval_case,
)

# ── helpers ─────────────────────────────────────────────────────────────


def _base_job(*, first_enqueued_at: int | None = None, retry_count: int = 0) -> MagicMock:
    job = MagicMock()
    job.id = "run1:item1"
    job.data = {
        "run_id": "run1",
        "item_id": "item1",
        "retry_count": retry_count,
        "first_enqueued_at": first_enqueued_at or int(time.time() * 1000),
        "input": {"query": "test_query"},
        "output": {"response_text": "test_response"},
        "metadata": {},
    }
    job.updateData = AsyncMock()
    return job


def _settings() -> MagicMock:
    s = MagicMock()
    s.max_slot_probes_per_run = 2
    s.eval_queue_max_attempts = 3
    s.eval_queue_backoff_initial_ms = 5000
    s.eval_queue_backoff_max_ms = 25000
    return s


def _slot_config(
    *,
    cluster_soft_cap: int = 0,
    retry_max_duration_ms: int = 7_200_000,
) -> SlotConfig:
    return SlotConfig(
        retry_max_duration_ms=retry_max_duration_ms,
        cluster_soft_cap=cluster_soft_cap,
        slots_per_run_per_node=2,
        slot_busy_delay_ms=3000,
        slot_ttl_s=1200,
    )


def _handlers(
    *,
    slot_config: SlotConfig | None = None,
    judge_and_write: AsyncMock | None = None,
    mark_failed_from_wait: AsyncMock | None = None,
    is_run_terminal: AsyncMock | None = None,
    finalize_fn: AsyncMock | None = None,
) -> SyncEvalHandlers:
    return SyncEvalHandlers(
        eval_type="test",
        wait_queue_name="test-wait",
        running_queue_name="test-running",
        slot_config=slot_config or _slot_config(),
        is_run_terminal=is_run_terminal or AsyncMock(return_value=False),
        heartbeat=AsyncMock(),
        judge_and_write=judge_and_write or AsyncMock(),
        mark_failed_from_wait=mark_failed_from_wait or AsyncMock(),
        mark_failed_from_running=AsyncMock(),
        finalize_fn=finalize_fn or AsyncMock(),
    )


def _queue_manager() -> MagicMock:
    qm = MagicMock()
    qm.prefix = "test"
    running_q = MagicMock()
    running_q.add = AsyncMock()
    wait_q = MagicMock()
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    return qm


# ── Test 1: retry deadline → the wait path fails terminally and writes input/output ────────────────


@pytest.mark.asyncio
async def test_retry_deadline_calls_mark_failed_from_wait_with_input_output() -> None:
    """When the retry deadline fires, mark_failed_from_wait receives data (including input/output)."""
    # first_enqueued_at is far in the past, so the time window is already exhausted
    past = int(time.time() * 1000) - 10_000_000
    job = _base_job(first_enqueued_at=past)
    sc = _slot_config(retry_max_duration_ms=1)  # 1ms → almost certainly already past the window

    mark_wait = AsyncMock()
    finalize = AsyncMock()
    handlers = _handlers(
        slot_config=sc,
        mark_failed_from_wait=mark_wait,
        finalize_fn=finalize,
    )

    await process_sync_eval_case(
        job,
        "tok-test",
        SyncEvalRuntime(
            handlers=handlers,
            queue_manager=_queue_manager(),
            deps=SimpleNamespace(settings=_settings()),
        ),
    )

    # mark_failed_from_wait is called, carrying data + reason="retry_deadline"
    mark_wait.assert_awaited_once()
    call_kwargs = mark_wait.call_args.kwargs
    assert call_kwargs["reason"] == "retry_deadline"
    assert call_kwargs["data"]["input"] == {"query": "test_query"}
    assert call_kwargs["data"]["output"] == {"response_text": "test_response"}

    # finalize is called
    finalize.assert_awaited_once()


# ── Test 2: cluster soft_cap > 0 → delay ────────────────────────────────


@pytest.mark.asyncio
async def test_cluster_soft_cap_positive_delays_when_busy() -> None:
    """When cluster soft_cap > 0 and the slot is busy, call delay_for_slot_busy (do not fail terminally)."""
    job = _base_job()
    sc = _slot_config(cluster_soft_cap=4)

    mark_wait = AsyncMock()
    handlers = _handlers(
        slot_config=sc,
        mark_failed_from_wait=mark_wait,
    )

    with (
        patch(
            "benchmark.queue.worker_skeleton.acquire_cluster_slot",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "benchmark.queue.worker_skeleton.delay_for_slot_busy",
            new_callable=AsyncMock,
        ) as delay,
    ):
        await process_sync_eval_case(
            job,
            "tok-test",
            SyncEvalRuntime(
                handlers=handlers,
                queue_manager=_queue_manager(),
                deps=SimpleNamespace(settings=_settings()),
            ),
        )

    # delay is called (the job is delayed for a retry, and it does not count as a failure)
    delay.assert_awaited_once()
    # mark_failed_from_wait is not called (this is not a terminal failure)
    mark_wait.assert_not_awaited()


# ── Test 3: cluster soft_cap = 0 → behavior unchanged ──────────────────────────────


@pytest.mark.asyncio
async def test_cluster_soft_cap_zero_skips_cluster_slot() -> None:
    """When cluster soft_cap = 0, do not call acquire_cluster_slot, and go straight to the per-node slot."""
    job = _base_job()
    sc = _slot_config(cluster_soft_cap=0)

    judge = AsyncMock()
    handlers = _handlers(
        slot_config=sc,
        judge_and_write=judge,
    )

    with (
        patch(
            "benchmark.queue.worker_skeleton.acquire_cluster_slot",
            new_callable=AsyncMock,
        ) as acquire_cluster,
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
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
            "benchmark.queue.worker_skeleton.get_queue_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        await process_sync_eval_case(
            job,
            "tok-test",
            SyncEvalRuntime(
                handlers=handlers,
                queue_manager=_queue_manager(),
                deps=SimpleNamespace(settings=_settings()),
            ),
        )

    # acquire_cluster_slot is not called (soft_cap=0 → skip the cluster slot)
    acquire_cluster.assert_not_awaited()
    # judge_and_write is called (the normal flow continues)
    judge.assert_awaited_once()


@pytest.mark.asyncio
async def test_attr_slot_key_matches_design_without_eval_type() -> None:
    """The async attr slot key is {prefix}:inflight:attr:{node}:{run}, with no eval_type."""
    job = _base_job()
    handlers = _handlers()
    handlers.eval_type = "attribution"
    handlers.cluster_slot_kind = "attr"
    handlers.hold_slot_on_success = True
    handlers.remove_running_on_success = False

    with (
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ) as acquire,
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
            "benchmark.queue.worker_skeleton.WORKER_NAME",
            "node-a",
        ),
    ):
        await process_sync_eval_case(
            job,
            "tok-test",
            SyncEvalRuntime(
                handlers=handlers,
                queue_manager=_queue_manager(),
                deps=SimpleNamespace(settings=_settings()),
            ),
        )

    acquire.assert_awaited()
    slot_key = acquire.await_args.args[0]
    assert slot_key == "test:inflight:attr:node-a:run1"
    assert ":attribution:" not in slot_key


@pytest.mark.asyncio
async def test_dataset_slot_key_includes_eval_type() -> None:
    """The sync dataset slot key is {prefix}:inflight:sync:dataset:{node}:{run}."""
    job = _base_job()
    handlers = _handlers()
    handlers.eval_type = "dataset"
    handlers.cluster_slot_kind = "sync"

    with (
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ) as acquire,
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
            "benchmark.queue.worker_skeleton.WORKER_NAME",
            "node-a",
        ),
        patch(
            "benchmark.queue.worker_skeleton.get_queue_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        await process_sync_eval_case(
            job,
            "tok-test",
            SyncEvalRuntime(
                handlers=handlers,
                queue_manager=_queue_manager(),
                deps=SimpleNamespace(settings=_settings()),
            ),
        )

    acquire.assert_awaited()
    slot_key = acquire.await_args.args[0]
    assert slot_key == "test:inflight:sync:dataset:node-a:run1"


# ── Test 6: a custom handle_exception fails → fall back to the default logic ───────────────


class _PermanentJudgeError(Exception):
    """A 4xx permanent-failure exception that carries a code (the default branch takes mark_failed_from_running)."""

    code = 400


@pytest.mark.asyncio
async def test_custom_handle_exception_failure_falls_back_to_default() -> None:
    """When the custom handler raises, fall back to the default logic: release the slot + report the original judge exception."""
    job = _base_job()
    judge_exc = _PermanentJudgeError("bad request")
    handlers = _handlers(
        judge_and_write=AsyncMock(side_effect=judge_exc),
    )
    handlers.handle_exception = AsyncMock(side_effect=RuntimeError("handler crashed"))
    mark_running = AsyncMock()
    handlers.mark_failed_from_running = mark_running

    with (
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
    ):
        # Must not raise (leaking it would leave the running inflight job a zombie until the TTL)
        await process_sync_eval_case(
            job,
            "tok-test",
            SyncEvalRuntime(
                handlers=handlers,
                queue_manager=_queue_manager(),
                deps=SimpleNamespace(settings=_settings()),
            ),
        )

    # Fall back to the default logic: release the slot + a permanent fail writes PG (reporting the original judge exception)
    release.assert_awaited()
    mark_running.assert_awaited_once()
    assert mark_running.call_args.kwargs["exc"] is judge_exc


@pytest.mark.asyncio
async def test_custom_handle_exception_success_skips_default() -> None:
    """When the custom handler finishes normally, do not enter the default branch (the handler disposes of the slot itself)."""
    job = _base_job()
    handlers = _handlers(
        judge_and_write=AsyncMock(side_effect=RuntimeError("judge boom")),
    )
    mark_running = AsyncMock()
    handlers.mark_failed_from_running = mark_running

    async def _custom_handler(ctx: object) -> None:
        # Simulate the external worker: it manages the slot itself (the declaration is kept, and finally does not release it)
        ctx.state.held = False

    handlers.handle_exception = _custom_handler

    with (
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
    ):
        await process_sync_eval_case(
            job,
            "tok-test",
            SyncEvalRuntime(
                handlers=handlers,
                queue_manager=_queue_manager(),
                deps=SimpleNamespace(settings=_settings()),
            ),
        )

    # The default branch did not fire: no permanent fail and no slot release (decided by the handler/finally semantics)
    mark_running.assert_not_awaited()
    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_skip_hold_slot_flag_releases_slot_and_removes_inflight() -> None:
    """When the fused row is already terminal, do not hold the slot: clear inflight, and finally releases this worker's slot."""
    job = _base_job()
    handlers = _handlers()
    handlers.hold_slot_on_success = True
    handlers.remove_running_on_success = False

    async def _judge(call: JudgeWriteCall) -> None:
        call.data[SKIP_HOLD_SLOT_FLAG] = True

    handlers.judge_and_write = _judge
    running_job = MagicMock()
    running_job.remove = AsyncMock()

    with (
        patch(
            "benchmark.queue.worker_skeleton.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.worker_skeleton.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.queue.worker_skeleton.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.queue.worker_skeleton.EvalCase.from_payload",
            return_value=MagicMock(input={}, output={}),
        ),
        patch(
            "benchmark.queue.worker_skeleton.get_queue_job",
            new_callable=AsyncMock,
            return_value=running_job,
        ),
    ):
        await process_sync_eval_case(
            job,
            "tok-test",
            SyncEvalRuntime(
                handlers=handlers,
                queue_manager=_queue_manager(),
                deps=SimpleNamespace(settings=_settings()),
            ),
        )

    running_job.remove.assert_awaited_once()
    release.assert_awaited_once()
