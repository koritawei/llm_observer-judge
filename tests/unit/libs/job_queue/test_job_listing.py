"""list_queue_jobs — an empty queue does not crash."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.libs.jobqueue.job_listing import list_queue_jobs


@pytest.mark.asyncio
async def test_list_queue_jobs_empty_ranges_returns_empty() -> None:
    queue = MagicMock()
    queue.sanitizeJobTypes = MagicMock(return_value=["waiting"])
    queue.scripts.getRanges = AsyncMock(return_value=[])

    jobs = await list_queue_jobs(queue, ["waiting"])

    assert jobs == []
    queue.scripts.getRanges.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_queue_jobs_loads_ids_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = MagicMock()
    queue.sanitizeJobTypes = MagicMock(return_value=["waiting"])
    queue.scripts.getRanges = AsyncMock(return_value=["a", "b"])
    queue.emit = MagicMock()

    class _Job:
        def __init__(self, job_id: str) -> None:
            self.id = job_id

    async def _from_id(_queue: object, job_id: str) -> _Job:
        return _Job(job_id)

    monkeypatch.setattr(
        "benchmark.libs.jobqueue.job_listing.Job.fromId",
        _from_id,
    )
    monkeypatch.setattr(
        "benchmark.libs.jobqueue.job_listing.extract_result",
        lambda task, _emit: task.result(),
    )

    jobs = await list_queue_jobs(queue, ["waiting"])

    assert [job.id for job in jobs] == ["a", "b"]


@pytest.mark.asyncio
async def test_get_queue_job_uses_from_id_for_real_queue_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.libs.jobqueue.job_listing import get_queue_job
    from bullmq import Queue

    queue = MagicMock(spec=Queue)
    expected = MagicMock()
    expected.id = "j1"

    async def _from_id(_q: object, job_id: str) -> object:
        assert job_id == "j1"
        return expected

    monkeypatch.setattr(
        "benchmark.libs.jobqueue.job_listing.Job.fromId",
        _from_id,
    )

    assert await get_queue_job(queue, "j1") is expected


@pytest.mark.asyncio
async def test_get_queue_job_uses_mock_get_job_on_test_double() -> None:
    from benchmark.libs.jobqueue.job_listing import get_queue_job

    job = MagicMock()
    queue = MagicMock()  # not isinstance Queue
    queue.getJob = AsyncMock(return_value=job)

    assert await get_queue_job(queue, "j1") is job
    queue.getJob.assert_awaited_once_with("j1")
