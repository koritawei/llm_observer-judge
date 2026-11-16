"""Multi-instance concurrency buckets (design §12.5 F1): limit within one instance, instances stay independent, and an unnamed one passes through."""

import asyncio

from benchmark.runs.common.instance_limiter import (
    InstanceConcurrencyLimiter,
)
from benchmark.queue.queue_factory import case_queue_name


async def test_same_instance_capped_at_limit() -> None:
    limiter = InstanceConcurrencyLimiter(2)
    entered: list[str] = []
    release = asyncio.Event()

    async def hold(name: str) -> None:
        async with limiter.acquire(name):
            entered.append(name)
            await release.wait()

    tasks = [asyncio.create_task(hold("agx_example_0000000001")) for _ in range(3)]
    await asyncio.sleep(0.05)
    # The third call is blocked by the bucket: this instance's in-flight count is exactly at the limit
    assert len(entered) == 2
    assert limiter.inflight("agx_example_0000000001") == 2
    release.set()
    await asyncio.gather(*tasks)
    assert len(entered) == 3
    assert limiter.inflight("agx_example_0000000001") == 0


async def test_different_instances_do_not_block_each_other() -> None:
    limiter = InstanceConcurrencyLimiter(1)
    release = asyncio.Event()
    entered: list[str] = []

    async def hold(name: str) -> None:
        async with limiter.acquire(name):
            entered.append(name)
            await release.wait()

    first = asyncio.create_task(hold("agx_example_0000000001"))
    second = asyncio.create_task(hold("agx_example_0000000002"))
    await asyncio.sleep(0.05)
    # Each one takes the only slot in its own bucket, and they do not block each other
    assert sorted(entered) == ["agx_example_0000000001", "agx_example_0000000002"]
    release.set()
    await asyncio.gather(first, second)


async def test_no_name_or_disabled_limit_passes_through() -> None:
    limiter = InstanceConcurrencyLimiter(1)
    async with limiter.acquire(None):
        pass
    assert limiter.bucket_count() == 0

    disabled = InstanceConcurrencyLimiter(0)
    release = asyncio.Event()
    entered = 0

    async def hold() -> None:
        nonlocal entered
        async with disabled.acquire("agx_example_0000000001"):
            entered += 1
            await release.wait()

    tasks = [asyncio.create_task(hold()) for _ in range(3)]
    await asyncio.sleep(0.05)
    assert entered == 3  # limit=0 passes straight through and is not bucketed
    release.set()
    await asyncio.gather(*tasks)


def test_case_queue_name_routes_l2_businesses() -> None:
    assert case_queue_name("hotel", ["hotel"]) == "benchmark-cases-l2"
    assert case_queue_name("ticket", ["hotel"]) == "benchmark-cases"
    assert case_queue_name(None, ["hotel"]) == "benchmark-cases"
    assert case_queue_name("", ["hotel"]) == "benchmark-cases"
    assert case_queue_name("hotel", []) == "benchmark-cases"
