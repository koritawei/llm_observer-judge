"""The real in-flight check for a stop-run (shared by the console and the healer)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.stop_inflight import (
    INFLIGHT_PLACEHOLDER_JOB_NAME,
    PENDING_CALLBACK_JOB_NAME,
    STOP_RUNNING_JUDGE_QUEUES,
    count_stop_inflight_in_queues,
    count_stop_inflight_jobs,
    is_stop_inflight_job,
    job_belongs_to_run,
)


def test_job_belongs_to_run_reads_data() -> None:
    job = MagicMock()
    job.data = {"run_id": "bm-1"}
    assert job_belongs_to_run(job, "bm-1")
    assert not job_belongs_to_run(job, "other")


def test_active_always_inflight() -> None:
    job = MagicMock()
    job.name = "anything"
    assert is_stop_inflight_job(job, listed_as_active=True)


def test_parked_pending_callback_is_inflight() -> None:
    job = MagicMock()
    job.name = PENDING_CALLBACK_JOB_NAME
    assert is_stop_inflight_job(job, listed_as_active=False)


def test_parked_sync_inflight_placeholder_is_inflight() -> None:
    """A sync running placeholder name=inflight (usually parked) must count as in-flight."""
    job = MagicMock()
    job.name = INFLIGHT_PLACEHOLDER_JOB_NAME
    assert is_stop_inflight_job(job, listed_as_active=False)


def test_parked_unknown_name_is_not_inflight() -> None:
    job = MagicMock()
    job.name = "orphan-unknown"
    assert not is_stop_inflight_job(job, listed_as_active=False)


@pytest.mark.asyncio
async def test_count_in_queues_fail_closed_returns_none() -> None:
    """When list_queue_jobs raises, the default is fail-closed → None (it must not be counted as 0)."""
    qm = MagicMock()
    qm.get_queue.return_value = MagicMock()
    with patch(
        "benchmark.runs.stop_inflight.list_queue_jobs",
        new=AsyncMock(side_effect=RuntimeError("redis blip")),
    ):
        assert (
            await count_stop_inflight_in_queues(
                qm,
                "bm-1",
                STOP_RUNNING_JUDGE_QUEUES,
            )
            is None
        )


@pytest.mark.asyncio
async def test_count_in_queues_fail_open_skips_failed_queue() -> None:
    """fail_closed=False: skip a failed queue, and still accumulate the rest."""
    qm = MagicMock()
    qm.get_queue.return_value = MagicMock()
    calls = {"n": 0}

    async def _list(_queue: object, types: list[str]) -> list[object]:
        calls["n"] += 1
        # The first queue's two lists (active+parked) fail; the second succeeds with an empty list
        if calls["n"] <= 2:
            raise RuntimeError("redis blip")
        return []

    with patch(
        "benchmark.runs.stop_inflight.list_queue_jobs",
        new=AsyncMock(side_effect=_list),
    ):
        total = await count_stop_inflight_in_queues(
            qm,
            "bm-1",
            STOP_RUNNING_JUDGE_QUEUES,
            fail_closed=False,
        )
    assert total == 0


@pytest.mark.asyncio
async def test_count_jobs_fail_closed_returns_none() -> None:
    qm = MagicMock()
    qm.get_queue.return_value = MagicMock()
    with patch(
        "benchmark.runs.stop_inflight.list_queue_jobs",
        new=AsyncMock(side_effect=RuntimeError("redis blip")),
    ):
        assert await count_stop_inflight_jobs(qm, "bm-1") is None
