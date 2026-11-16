"""Unit tests for retry_requeue — aligned with design doc §6.4 / §14.

Covers: requeue while under the limit / over the attempt limit / the time window is hit first / delay is clamped /
no running is rejected / first_enqueued_at is missing / force_fail /
the worker path uses moveToDelayed (so an add with the same jobId does not drop the job).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.queue.retry_requeue import (
    RequeueDeps,
    RequeueOpts,
    RetryPolicy,
    eval_backoff_ms,
    force_fail_from_running,
    is_retry_window_exhausted,
    requeue_or_fail_from_running,
    stamp_first_running_at,
)
from bullmq import WaitingChildrenError


def _job(*, data: dict[str, Any], job_id: str = "run1:item1") -> MagicMock:
    job = MagicMock()
    job.id = job_id
    job.data = data
    job.remove = AsyncMock()
    return job


def _queues(
    *,
    running_job: MagicMock | None,
) -> tuple[MagicMock, MagicMock]:
    running_q = MagicMock()
    running_q.getJob = AsyncMock(return_value=running_job)
    wait_q = MagicMock()
    wait_q.add = AsyncMock()
    return running_q, wait_q


def _deps(
    running_q: MagicMock,
    wait_q: MagicMock,
    mark_failed: AsyncMock,
    *,
    backoff_initial_ms: int = 5000,
    backoff_max_ms: int = 5000,
) -> RequeueDeps:
    """The requeue dependency bundle (default 3 attempts / a 2h window / 5s backoff)."""
    return RequeueDeps(
        running_queue=running_q,
        wait_queue=wait_q,
        policy=RetryPolicy(
            max_attempts=3,
            retry_max_duration_ms=7_200_000,
            backoff_initial_ms=backoff_initial_ms,
            backoff_max_ms=backoff_max_ms,
        ),
        mark_failed=mark_failed,
    )


# ── is_retry_window_exhausted ─────────────────────────────────────────────


def test_is_retry_window_exhausted_missing_first_enqueued_at() -> None:
    """Not in running yet → the clock has not started, so it is not past the window."""
    assert is_retry_window_exhausted({}, max_duration_ms=7200000) is False


def test_is_retry_window_exhausted_within_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "benchmark.queue.retry_requeue.time.time",
        lambda: 1_000.0,
    )
    data = {"first_enqueued_at": 1_000_000 - 60_000}  # 1 minute ago
    assert is_retry_window_exhausted(data, max_duration_ms=3_600_000) is False


def test_is_retry_window_exhausted_over_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "benchmark.queue.retry_requeue.time.time",
        lambda: 1_000.0,
    )
    data = {"first_enqueued_at": 1_000_000 - 3_600_001}
    assert is_retry_window_exhausted(data, max_duration_ms=3_600_000) is True


def test_stamp_first_running_at_writes_once() -> None:
    data: dict[str, Any] = {}
    first = stamp_first_running_at(data, now_ms=100)
    assert first == 100
    assert data["first_enqueued_at"] == 100
    assert stamp_first_running_at(data, now_ms=999) == 100


# ── eval_backoff_ms ───────────────────────────────────────────────────────


def test_eval_backoff_ms_exponential_capped() -> None:
    assert eval_backoff_ms(1, initial_ms=5000, max_ms=25000) == 5000
    assert eval_backoff_ms(2, initial_ms=5000, max_ms=25000) == 10000
    assert eval_backoff_ms(3, initial_ms=5000, max_ms=25000) == 20000
    assert eval_backoff_ms(4, initial_ms=5000, max_ms=25000) == 25000


# ── requeue_or_fail_from_running ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_requeue_no_running_job_returns_failed() -> None:
    running_q, wait_q = _queues(running_job=None)
    mark_failed = AsyncMock()

    outcome = await requeue_or_fail_from_running(
        _deps(running_q, wait_q, mark_failed),
        "run1:item1",
    )

    assert outcome == "failed"
    wait_q.add.assert_not_called()
    mark_failed.assert_not_called()


@pytest.mark.asyncio
async def test_requeue_missing_first_enqueued_at_stamps_and_requeues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """first_enqueued_at is missing at requeue time → stamp it from the running time, and do not fail terminally."""
    monkeypatch.setattr(
        "benchmark.queue.retry_requeue.time.time",
        lambda: 1_000.0,
    )
    job = _job(data={"run_id": "run1", "item_id": "item1", "retry_count": 0})
    job.timestamp = 900_000
    running_q, wait_q = _queues(running_job=job)
    mark_failed = AsyncMock()

    outcome = await requeue_or_fail_from_running(
        _deps(running_q, wait_q, mark_failed),
        "run1:item1",
    )

    assert outcome == "requeued"
    mark_failed.assert_not_called()
    wait_q.add.assert_awaited_once()
    _name, data, _opts = wait_q.add.await_args.args
    assert data["first_enqueued_at"] == 900_000
    assert data["retry_count"] == 1


@pytest.mark.asyncio
async def test_requeue_under_limits_adds_wait_with_same_first_enqueued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Healer/callback path: no wait_job → Queue.add."""
    monkeypatch.setattr(
        "benchmark.queue.retry_requeue.time.time",
        lambda: 1_000.0,
    )
    first = 1_000_000 - 60_000
    job = _job(
        data={
            "run_id": "run1",
            "item_id": "item1",
            "retry_count": 0,
            "first_enqueued_at": first,
        },
    )
    running_q, wait_q = _queues(running_job=job)
    mark_failed = AsyncMock()

    outcome = await requeue_or_fail_from_running(
        _deps(running_q, wait_q, mark_failed),
        "run1:item1",
    )

    assert outcome == "requeued"
    mark_failed.assert_not_called()
    wait_q.add.assert_awaited_once()
    name, data, opts = wait_q.add.await_args.args
    assert name == "case"
    assert data["retry_count"] == 1
    assert data["first_enqueued_at"] == first  # Rewriting is forbidden
    assert opts["jobId"] == "run1:item1"
    assert opts["attempts"] == 1
    assert opts["delay"] == 5000


@pytest.mark.asyncio
async def test_requeue_worker_path_moves_to_delayed_not_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker path: moveToDelayed on the active wait job, and an add with the same jobId is forbidden."""
    monkeypatch.setattr(
        "benchmark.queue.retry_requeue.time.time",
        lambda: 1_000.0,
    )
    first = 1_000_000 - 60_000
    running_job = _job(
        data={
            "run_id": "run1",
            "item_id": "item1",
            "retry_count": 0,
            "first_enqueued_at": first,
        },
    )
    running_q, wait_q = _queues(running_job=running_job)

    wait_job = MagicMock()
    wait_job.id = "run1:item1"
    wait_job.updateData = AsyncMock()
    wait_job.scripts = MagicMock()
    wait_job.scripts.moveToDelayed = AsyncMock()

    with pytest.raises(WaitingChildrenError):
        await requeue_or_fail_from_running(
            _deps(running_q, wait_q, AsyncMock()),
            "run1:item1",
            RequeueOpts(wait_job=wait_job, token="tok-1"),
        )

    wait_q.add.assert_not_called()
    wait_job.updateData.assert_awaited_once()
    updated = wait_job.updateData.await_args.args[0]
    assert updated["retry_count"] == 1
    assert updated["first_enqueued_at"] == first
    wait_job.scripts.moveToDelayed.assert_awaited_once()
    move_args = wait_job.scripts.moveToDelayed.await_args.args
    assert move_args[0] == "run1:item1"
    assert move_args[2] == 5000  # delay
    assert move_args[3] == "tok-1"
    running_job.remove.assert_awaited_once()


@pytest.mark.asyncio
async def test_requeue_max_attempts_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "benchmark.queue.retry_requeue.time.time",
        lambda: 1_000.0,
    )
    job = _job(
        data={
            "run_id": "run1",
            "item_id": "item1",
            "retry_count": 2,  # +1 → 3 >= max_attempts
            "first_enqueued_at": 1_000_000 - 60_000,
        },
    )
    running_q, wait_q = _queues(running_job=job)
    mark_failed = AsyncMock()

    outcome = await requeue_or_fail_from_running(
        _deps(running_q, wait_q, mark_failed),
        "run1:item1",
    )

    assert outcome == "failed"
    assert mark_failed.await_args.kwargs["reason"] == "max_attempts"
    wait_q.add.assert_not_called()


@pytest.mark.asyncio
async def test_requeue_deadline_before_attempts_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The time window is hit first: it fails terminally even though retry_count is not full."""
    monkeypatch.setattr(
        "benchmark.queue.retry_requeue.time.time",
        lambda: 1_000.0,
    )
    job = _job(
        data={
            "run_id": "run1",
            "item_id": "item1",
            "retry_count": 0,
            "first_enqueued_at": 1_000_000 - 7_200_001,
        },
    )
    running_q, wait_q = _queues(running_job=job)
    mark_failed = AsyncMock()

    outcome = await requeue_or_fail_from_running(
        _deps(running_q, wait_q, mark_failed),
        "run1:item1",
    )

    assert outcome == "failed"
    assert mark_failed.await_args.kwargs["reason"] == "retry_deadline"
    wait_q.add.assert_not_called()


@pytest.mark.asyncio
async def test_requeue_delay_clamped_to_remaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When backoff exceeds the time left in the window, delay is clamped to remaining."""
    monkeypatch.setattr(
        "benchmark.queue.retry_requeue.time.time",
        lambda: 1_000.0,
    )
    # 3s left; backoff would produce a larger delay
    first = 1_000_000 - (7_200_000 - 3_000)
    job = _job(
        data={
            "run_id": "run1",
            "item_id": "item1",
            "retry_count": 0,
            "first_enqueued_at": first,
        },
    )
    running_q, wait_q = _queues(running_job=job)

    outcome = await requeue_or_fail_from_running(
        _deps(
            running_q,
            wait_q,
            AsyncMock(),
            backoff_initial_ms=25_000,
            backoff_max_ms=25_000,
        ),
        "run1:item1",
    )

    assert outcome == "requeued"
    opts = wait_q.add.await_args.args[2]
    assert opts["delay"] == 3_000


@pytest.mark.asyncio
async def test_force_fail_from_running_removes_and_marks() -> None:
    job = _job(data={"run_id": "run1", "item_id": "item1", "retry_count": 1})
    running_q = MagicMock()
    running_q.getJob = AsyncMock(return_value=job)
    mark_failed = AsyncMock()

    outcome = await force_fail_from_running(
        running_queue=running_q,
        running_job_id="run1:item1",
        mark_failed=mark_failed,
    )

    assert outcome == "failed"
    job.remove.assert_awaited_once()
    mark_failed.assert_awaited_once()
    assert mark_failed.await_args.kwargs["retry_count"] == 1


@pytest.mark.asyncio
async def test_requeue_uses_custom_wait_job_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attribution requeue onto wait uses attr_case rather than case."""
    monkeypatch.setattr(
        "benchmark.queue.retry_requeue.time.time",
        lambda: 1_000.0,
    )
    job = _job(
        data={
            "run_id": "run1",
            "item_id": "item1",
            "retry_count": 0,
            "first_enqueued_at": 1_000_000,
        },
    )
    running_q, wait_q = _queues(running_job=job)

    outcome = await requeue_or_fail_from_running(
        _deps(
            running_q,
            wait_q,
            AsyncMock(),
            backoff_initial_ms=1000,
            backoff_max_ms=1000,
        ),
        "run1:item1",
        RequeueOpts(wait_job_name="attr_case"),
    )

    assert outcome == "requeued"
    name, _, opts = wait_q.add.await_args.args
    assert name == "attr_case"
    assert opts["jobId"] == "run1:item1"
    assert opts["attempts"] == 1
