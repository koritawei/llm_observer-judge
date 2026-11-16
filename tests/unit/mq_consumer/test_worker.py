"""EvalWorkerPool unit tests — fixed concurrency, a bounded queue, blocking enqueue backpressure, and a graceful shutdown."""

import asyncio

import pytest

from benchmark.ingest.mq.runtime import EvalWorkerPool


async def _wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.01) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


def test_invalid_params_rejected() -> None:
    async def _noop(_: int) -> None:
        return None

    with pytest.raises(ValueError):
        EvalWorkerPool(handler=_noop, concurrency=0, queue_size=1)
    with pytest.raises(ValueError):
        EvalWorkerPool(handler=_noop, concurrency=1, queue_size=0)


async def test_processes_all_enqueued() -> None:
    done: list[int] = []

    async def handler(item: int) -> None:
        done.append(item)

    pool = EvalWorkerPool(handler=handler, concurrency=3, queue_size=10)
    pool.start()
    try:
        for i in range(7):
            await pool.enqueue(i)
        assert await _wait_until(lambda: len(done) == 7)
        assert set(done) == set(range(7))
    finally:
        await pool.stop()


async def test_respects_fixed_concurrency_limit() -> None:
    # A fixed concurrency cap: the number of tasks running at once does not exceed concurrency (unbounded growth is forbidden).
    release = asyncio.Event()
    running = 0
    max_running = 0

    async def handler(_: int) -> None:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        try:
            await release.wait()
        finally:
            running -= 1

    pool = EvalWorkerPool(handler=handler, concurrency=2, queue_size=10)
    pool.start()
    try:
        for i in range(6):
            await pool.enqueue(i)
        assert await _wait_until(lambda: max_running == 2)
        await asyncio.sleep(0.05)
        assert max_running == 2  # Concurrency stays at 2 and does not exceed the cap just because more work is enqueued
    finally:
        release.set()
        await pool.drain(1.0)
        await pool.stop()


async def test_enqueue_blocks_when_queue_full_backpressure() -> None:
    # A full queue → enqueue blocks (backpressure); tasks are not dropped, and memory is bounded.
    release = asyncio.Event()

    async def handler(_: str) -> None:
        await release.wait()

    pool = EvalWorkerPool(handler=handler, concurrency=1, queue_size=1)
    pool.start()
    try:
        await pool.enqueue("a")  # Taken by the only worker and blocked on release
        await asyncio.sleep(0.02)
        await pool.enqueue("b")  # Fill the queue whose capacity is 1
        # The third enqueue: the queue is full and the worker is busy → it must block (a timeout proves it blocks).
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(pool.enqueue("c"), timeout=0.1)
        release.set()
        assert await pool.drain(2.0)
    finally:
        release.set()
        await pool.stop()


async def test_handler_exception_does_not_kill_worker() -> None:
    processed: list[str] = []

    async def handler(item: str) -> None:
        if item == "boom":
            raise RuntimeError("boom")
        processed.append(item)

    pool = EvalWorkerPool(handler=handler, concurrency=1, queue_size=10)
    pool.start()
    try:
        await pool.enqueue("boom")
        await pool.enqueue("ok")
        assert await _wait_until(lambda: processed == ["ok"])
        assert pool.worker_count == 1  # The worker is not killed by a single exception
    finally:
        await pool.stop()


async def test_stop_cancels_running_workers() -> None:
    started = asyncio.Event()

    async def handler(_: str) -> None:
        started.set()
        await asyncio.sleep(3600)  # Blocks forever, and is cancelled by stop()

    pool = EvalWorkerPool(handler=handler, concurrency=1, queue_size=1)
    pool.start()
    await pool.enqueue("x")
    assert await _wait_until(lambda: started.is_set())
    await pool.stop()
    assert pool.worker_count == 0
    assert pool.is_running is False


async def test_enqueue_after_stop_raises() -> None:
    async def handler(_: int) -> None:
        return None

    pool = EvalWorkerPool(handler=handler, concurrency=1, queue_size=1)
    pool.start()
    await pool.stop()
    with pytest.raises(RuntimeError):
        await pool.enqueue(1)
