"""Bounded-concurrency worker pool — decouple MQ pull from eval execution, with a fixed pool size and bounded memory.

Evaluation is LLM-heavy (high latency). A synchronous await inside the MQConsumer per-message loop gives throughput = 1; a
**fixed** ``concurrency`` of workers running together raises throughput by N (higher resource use).

Two gates that "forbid unbounded growth":

1. **Fixed concurrency**: the worker count is fixed at construction and does not grow at runtime.
2. **Bounded queue**: ``queue_size`` is the cap; ``enqueue`` is **blocking** — when the queue is full it waits
   until some worker frees a slot. That backpressures the upstream MQ consume loop (poll slows down). The process
   holds at most ``queue_size + concurrency`` messages, never piles up without bound, and never drops a message.

"Retry until success before advancing" is a **per-task** semantic, guaranteed inside the worker callback (
``service._process_task``): a worker does not ``get`` the next task until the current callback returns.
This pool only provides "bounded concurrency + backpressure + graceful shutdown" and does not care about retry/commit policy (separated duties).

See ``GroupAgentWorkerPool`` in ``services/amap-agent/app/services/group/agent_worker.py``.
"""

import asyncio
from collections.abc import Awaitable, Callable

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger("agent_eval.mq.worker")

#: Callback signature for a worker handling one task (the callback itself retries until success + commit).
type TaskHandler[T] = Callable[[T], Awaitable[None]]


class EvalWorkerPool[T]:
    """In-process worker pool with fixed concurrency and a bounded queue.

    The queue exists only in-process: it lowers MQ consume-loop latency and provides concurrency, but **does not** recover
    after ACK (the same tradeoff as amap-agent group consumption). The real durability boundary is the broker.
    """

    def __init__(
        self,
        *,
        handler: TaskHandler[T],
        concurrency: int,
        queue_size: int,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._handler = handler
        self._concurrency = concurrency
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=queue_size)
        self._workers: list[asyncio.Task[None]] = []
        self._running = False

    # ── properties ──────────────────────────────────────────────────────

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def max_queue_size(self) -> int:
        return self._queue.maxsize

    @property
    def worker_count(self) -> int:
        self._prune_workers()
        return len(self._workers)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for index in range(self._concurrency):
            self._workers.append(
                asyncio.create_task(
                    self._worker_loop(index), name=f"eval-mq-worker-{index}"
                )
            )
        logger.info(
            "EvalWorkerPool started concurrency=%d max_queue=%d",
            self._concurrency,
            self._queue.maxsize,
        )

    async def enqueue(self, task: T) -> None:
        """Blocking enqueue: when the queue is full, wait for a worker to free a slot (backpressure). Do not drop tasks; memory stays bounded."""

        if not self._running:
            raise RuntimeError("EvalWorkerPool is not running")
        await self._queue.put(task)

    async def drain(self, timeout: float) -> bool:
        """Wait until every queued task has been processed; return False on timeout.

        Call before shutdown: stop consumption first (no more enqueue), then drain tasks already queued, so in-flight tasks are not dropped.
        """

        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def stop(self) -> None:
        """Cancel every worker and wait for them to exit. Call this after :meth:`drain`."""

        if not self._running:
            return
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("EvalWorkerPool stopped remaining_queue=%d", self._queue.qsize())

    # ── internals ───────────────────────────────────────────────────────

    def _prune_workers(self) -> None:
        self._workers = [w for w in self._workers if not w.done()]

    async def _worker_loop(self, index: int) -> None:
        while True:
            item = await self._queue.get()
            try:
                await self._handler(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The callback should absorb the outcome itself (retry/skip); this backstop keeps one task from killing the worker.
                logger.exception("eval worker task failed worker=%d", index)
            finally:
                self._queue.task_done()


__all__ = ["EvalWorkerPool", "TaskHandler"]
