"""Bounded async concurrency helpers — reuse instead of ad-hoc Semaphore+gather.

Uses :class:`anyio.CapacityLimiter` (already in the service lockfile) so callers
get a single, testable primitive with clear back-pressure semantics.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import anyio

T = TypeVar("T")
R = TypeVar("R")


async def map_bounded(
    items: Sequence[T],
    worker: Callable[[int, T], Awaitable[R]],
    *,
    limit: int,
    return_exceptions: bool = False,
) -> list[R | BaseException]:
    """Run ``worker(index, item)`` with at most ``limit`` concurrent tasks.

    Results are returned in input order. When ``return_exceptions`` is True,
    per-item failures become exception objects in the result list (same shape
    as ``asyncio.gather(..., return_exceptions=True)``); otherwise the first
    exception propagates.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if not items:
        return []

    limiter = anyio.CapacityLimiter(limit)

    async def _run_one(index: int, item: T) -> R:
        async with limiter:
            return await worker(index, item)

    return list(
        await asyncio.gather(
            *(_run_one(index, item) for index, item in enumerate(items)),
            return_exceptions=return_exceptions,
        )
    )


__all__ = ["map_bounded"]
