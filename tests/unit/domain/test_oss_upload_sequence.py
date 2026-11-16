"""Same-day sequence number for OSS uploads: starts at 1, independent across days, and concurrent callers do not reuse a number."""

import asyncio

from benchmark.domain.runtime_config_store import next_oss_upload_id


async def test_sequence_starts_at_one_and_increments() -> None:
    assert await next_oss_upload_id("20261021") == 1
    assert await next_oss_upload_id("20261021") == 2
    assert await next_oss_upload_id("20261021") == 3


async def test_sequence_is_independent_per_day() -> None:
    assert await next_oss_upload_id("20261021") == 1
    assert await next_oss_upload_id("20261022") == 1
    assert await next_oss_upload_id("20261021") == 2


async def test_sequence_concurrent_allocations_are_unique() -> None:
    results = await asyncio.gather(
        *(next_oss_upload_id("20261021") for _ in range(10)),
    )
    assert sorted(results) == list(range(1, 11))
