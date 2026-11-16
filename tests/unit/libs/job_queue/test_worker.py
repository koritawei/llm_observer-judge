"""job_queue.Worker lock renewal must not be wrapped in a Redis MULTI/EXEC."""

from types import SimpleNamespace
from typing import Any

from benchmark.libs.jobqueue.worker import Worker


class _Pipe:
    def __init__(self, *, transaction: bool) -> None:
        self.transaction = transaction
        self.calls: list[tuple[Any, ...]] = []
        self.results: list[Any] = []
        self.raise_on_execute: Exception | None = None

    async def execute(self, raise_on_error: bool = True) -> list[Any]:
        self.calls.append(("execute", raise_on_error))
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        return list(self.results)


def _worker(
    jobs: list[tuple[SimpleNamespace, str]] | None = None,
    *,
    pipe: _Pipe | None = None,
) -> Worker:
    used = pipe or _Pipe(transaction=False)

    def pipeline(*, transaction: bool = True, shard_hint: str | None = None) -> _Pipe:
        del shard_hint
        used.transaction = transaction
        used.calls.append(("pipeline", transaction))
        return used

    async def extend_lock(
        job_id: str,
        token: str,
        duration: int,
        client: _Pipe,
    ) -> None:
        client.calls.append(("extend", job_id, token, duration, client.transaction))

    worker = Worker.__new__(Worker)
    # The list keeps order, so the lock-renewal order can be asserted; the upstream runtime is a set.
    worker.jobs = jobs or []  # type: ignore[assignment]
    worker.opts = {"lockDuration": 1_200_000}
    worker.client = SimpleNamespace(pipeline=pipeline)
    worker.scripts = SimpleNamespace(extendLock=extend_lock)
    worker._test_pipe = used  # type: ignore[attr-defined]
    return worker


async def test_extend_locks_skips_redis_when_idle() -> None:
    worker = _worker([])
    await worker.extendLocks()
    assert worker._test_pipe.calls == []  # type: ignore[attr-defined]


async def test_extend_locks_uses_nontransactional_pipeline() -> None:
    job = SimpleNamespace(id="job-1")
    worker = _worker([(job, "tok")])
    worker._test_pipe.results = [1]  # type: ignore[attr-defined]
    await worker.extendLocks()
    calls = worker._test_pipe.calls  # type: ignore[attr-defined]
    assert ("pipeline", False) in calls
    assert ("extend", "job-1", "tok", 1_200_000, False) in calls
    assert ("execute", False) in calls


async def test_extend_locks_keeps_other_jobs_when_one_evalsha_fails() -> None:
    ok_job = SimpleNamespace(id="ok")
    bad_job = SimpleNamespace(id="bad")
    worker = _worker([(ok_job, "t1"), (bad_job, "t2")])
    worker._test_pipe.results = [1, RuntimeError("EVALSHA rejected")]  # type: ignore[attr-defined]
    await worker.extendLocks()
    calls = worker._test_pipe.calls  # type: ignore[attr-defined]
    extended = [call for call in calls if call[0] == "extend"]
    assert [call[1] for call in extended] == ["ok", "bad"]


async def test_extend_locks_swallows_connection_errors() -> None:
    job = SimpleNamespace(id="job-1")
    worker = _worker([(job, "tok")])
    worker._test_pipe.raise_on_execute = ConnectionError("redis down")  # type: ignore[attr-defined]
    await worker.extendLocks()
