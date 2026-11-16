"""Unit tests for map_bounded concurrency helper."""

import asyncio

import pytest
from benchmark.libs.async_runtime import map_bounded


@pytest.mark.asyncio
async def test_map_bounded_preserves_order_and_limit() -> None:
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def worker(index: int, item: int) -> int:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.02)
        async with lock:
            active -= 1
        return item * 10

    results = await map_bounded([1, 2, 3, 4, 5], worker, limit=2)
    assert results == [10, 20, 30, 40, 50]
    assert peak <= 2


@pytest.mark.asyncio
async def test_map_bounded_return_exceptions() -> None:
    async def worker(index: int, item: int) -> int:
        if item == 2:
            raise ValueError("boom")
        return item

    results = await map_bounded([1, 2, 3], worker, limit=2, return_exceptions=True)
    assert results[0] == 1
    assert isinstance(results[1], ValueError)
    assert results[2] == 3


@pytest.mark.asyncio
async def test_map_bounded_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        await map_bounded([1], lambda _i, x: asyncio.sleep(0, result=x), limit=0)
