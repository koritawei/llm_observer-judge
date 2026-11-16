"""async_runtime.heartbeat unit tests."""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest
from benchmark.libs.async_runtime.heartbeat import (
    HeartbeatSpec,
    cancel_and_wait,
    heartbeat_loop,
)


@pytest.mark.asyncio
async def test_cancel_and_wait_swallows_cancelled() -> None:
    async def _sleep() -> None:
        await asyncio.sleep(10)

    task = asyncio.create_task(_sleep())
    await asyncio.sleep(0)
    await cancel_and_wait(task)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_heartbeat_loop_calls_fn() -> None:
    fn = AsyncMock()
    spec = HeartbeatSpec(
        heartbeat_fn=fn,
        logger=logging.getLogger("test"),
        fail_log_message="fail",
    )
    task = asyncio.create_task(heartbeat_loop(spec, interval_s=0.01))
    await asyncio.sleep(0.05)
    await cancel_and_wait(task)
    assert fn.await_count >= 1
