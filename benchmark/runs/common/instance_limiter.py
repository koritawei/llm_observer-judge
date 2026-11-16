"""In-process concurrency buckets for multiple instances of the same code
(multi-instance isolation design §12.5 F1).

When several instances of one component share a worker and a model channel, limit
them in buckets by internal agent name: an instance that floods concurrency can only
fill its own bucket and cannot take other instances' evaluation slots (noisy-neighbor
isolation). This stacks orthogonally with the existing global LLM semaphore: the
instance bucket first, then the global bucket, so a request blocked by the instance
bucket does not hold a global slot.

In-process implementation (same level as the global semaphore in ``EvaluatorService``):
in a multi-node deployment each node buckets independently, and the cluster cap is
node count × per-bucket cap. Size the config from the model channel's total limit.
This module only provides per-instance fairness; it does not limit across processes.
"""

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.setting import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)

#: Bucket-table cap: the bucket count is naturally bounded by the instance quota (one component ≤20); here we also
#: add a process-level hard cap so abnormal registration volume cannot exhaust memory; over the cap, evict the oldest
#: bucket with no in-flight request.
_MAX_BUCKETS = 1024


class InstanceConcurrencyLimiter:
    """Async concurrency limiter bucketed by internal agent name.

    ``per_instance_limit <= 0`` means unlimited (pass through). Requests with no internal
    name (evaluations that are not registered) also pass through, so existing behavior stays.
    """

    def __init__(self, per_instance_limit: int, *, max_buckets: int = _MAX_BUCKETS) -> None:
        self._limit = per_instance_limit
        self._max_buckets = max_buckets
        self._semaphores: OrderedDict[str, asyncio.Semaphore] = OrderedDict()
        self._users: dict[str, int] = {}
        self._inflight: dict[str, int] = {}
        self._guard = asyncio.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    def inflight(self, internal_name: str) -> int:
        """Current in-flight count for one instance (observation / tests)."""
        return self._inflight.get(internal_name, 0)

    def bucket_count(self) -> int:
        return len(self._semaphores)

    async def _retain(self, internal_name: str) -> asyncio.Semaphore:
        async with self._guard:
            self._users[internal_name] = self._users.get(internal_name, 0) + 1
            semaphore = self._semaphores.get(internal_name)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._limit)
                self._semaphores[internal_name] = semaphore
                self._evict_idle_buckets()
            else:
                self._semaphores.move_to_end(internal_name)
            return semaphore

    def _evict_idle_buckets(self) -> None:
        """When the bucket count exceeds the hard cap, evict the oldest unused bucket (call while holding _guard)."""
        while len(self._semaphores) > self._max_buckets:
            oldest = next(iter(self._semaphores))
            if self._users.get(oldest, 0) > 0:
                break
            del self._semaphores[oldest]

    async def _release_user(self, internal_name: str) -> None:
        async with self._guard:
            remaining = self._users.get(internal_name, 0) - 1
            if remaining > 0:
                self._users[internal_name] = remaining
            else:
                self._users.pop(internal_name, None)

    @asynccontextmanager
    async def acquire(self, internal_name: str | None) -> "AsyncIterator[None]":
        """Take one instance slot; pass through when there is no internal name or the limit is disabled."""
        if not internal_name or self._limit <= 0:
            yield
            return
        semaphore = await self._retain(internal_name)
        try:
            async with semaphore:
                self._inflight[internal_name] = self._inflight.get(internal_name, 0) + 1
                try:
                    yield
                finally:
                    self._inflight[internal_name] -= 1
        finally:
            await self._release_user(internal_name)


class _LimiterState:
    """Process-level singleton container (fields are replaceable, so no global statement is needed)."""

    def __init__(self) -> None:
        self.limiter: InstanceConcurrencyLimiter | None = None


_state = _LimiterState()


def get_instance_limiter() -> InstanceConcurrencyLimiter:
    """Process-level singleton (the cap comes from Settings ``instance_max_concurrency``)."""
    if _state.limiter is None:
        _state.limiter = InstanceConcurrencyLimiter(settings.instance_max_concurrency)
        logger.info(
            "instance_limiter.ready per_instance_limit=%d",
            settings.instance_max_concurrency,
        )
    return _state.limiter


def reset_instance_limiter(limiter: InstanceConcurrencyLimiter | None = None) -> None:
    """Replace the process-level singleton (for tests)."""
    _state.limiter = limiter


__all__ = [
    "InstanceConcurrencyLimiter",
    "get_instance_limiter",
    "reset_instance_limiter",
]
