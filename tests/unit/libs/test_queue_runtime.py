"""app/libs/queue_runtime unit tests."""

import pytest
from benchmark.libs import queue_runtime


@pytest.fixture(autouse=True)
async def _reset_queue_runtime() -> None:
    await queue_runtime.close_redis()
    yield
    await queue_runtime.close_redis()


def test_init_redis_idempotent_and_prefix_updates() -> None:
    queue_runtime.init_redis("redis://localhost:6379/0", prefix="daily")
    first = queue_runtime.get_redis()
    queue_runtime.init_redis("redis://localhost:6379/0", prefix="pre")
    second = queue_runtime.get_redis()
    assert first is second
    assert queue_runtime.get_prefix() == "pre"


def test_get_redis_before_init_raises() -> None:
    with pytest.raises(RuntimeError, match="queue_runtime not initialized"):
        queue_runtime.get_redis()
