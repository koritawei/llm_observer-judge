"""STEP-14: Redis Lua batch completion signal (fakeredis or real Redis)."""

from __future__ import annotations

import asyncio
import os

import pytest

from llmobserverjudge.benchmark.redis_signal import (
    RedisBatchSignal,
    create_batch_signal,
    create_batch_signal_from_env,
)
from llmobserverjudge.benchmark.signal import InMemoryBatchSignal
from llmobserverjudge.benchmark.service import BenchmarkService
from llmobserverjudge.contracts.models import BatchJobStatus

fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture
async def redis_client():
    """Prefer LLMJUDGE_REDIS_URL; else FakeRedis with Lua."""
    url = (os.environ.get("LLMJUDGE_REDIS_URL") or "").strip()
    if url:
        import redis.asyncio as aioredis

        client = aioredis.from_url(url, decode_responses=True)
        await client.ping()
        try:
            yield client
        finally:
            await client.aclose()
        return

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_redis_batch_signal_dead_complete_once(redis_client) -> None:
    sig = RedisBatchSignal(redis_client, prefix="llmj:test:batch:")
    await sig.init_batch("b1", total=3)
    assert await sig.incr("b1", "completed") == "PENDING"
    assert await sig.incr("b1", "failed") == "PENDING"
    assert await sig.incr("b1", "dead") == "COMPLETE"
    assert await sig.incr("b1", "completed") == "ALREADY_SIGNALED"
    snap = await sig.snapshot("b1")
    assert snap["total"] == 3
    assert snap["completed"] == 2  # +1 after already signaled
    assert snap["failed"] == 1
    assert snap["dead"] == 1
    assert await sig.try_claim_benchmark("b1") is True
    assert await sig.try_claim_benchmark("b1") is False


@pytest.mark.asyncio
async def test_redis_batch_signal_concurrent_complete_once(redis_client) -> None:
    sig = RedisBatchSignal(redis_client, prefix="llmj:test:conc:")
    n = 20
    await sig.init_batch("b-conc", total=n)

    async def one() -> str:
        return await sig.incr("b-conc", "completed")

    results = await asyncio.gather(*[one() for _ in range(n)])
    assert results.count("COMPLETE") == 1
    assert results.count("PENDING") == n - 1
    assert "ALREADY_SIGNALED" not in results


@pytest.mark.asyncio
async def test_benchmark_service_with_redis_signal(redis_client) -> None:
    sig = RedisBatchSignal(redis_client, prefix="llmj:test:svc:")
    svc = BenchmarkService(signal=sig)
    job = await svc.trigger_batch_eval(
        tenant_id="t1",
        name="redis-batch",
        traces=[
            {
                "trace_id": "0123456789abcdef0123456789abcdef",
                "trace_summary": {
                    "status": "OK",
                    "response": "a sufficiently long answer",
                },
            },
            {
                "trace_id": "1123456789abcdef0123456789abcdef",
                "trace_summary": {
                    "status": "OK",
                    "response": "another long enough answer",
                },
            },
        ],
        eval_configs=[{"agent_id": "judge_quality", "model": "default-judge"}],
    )
    assert job.status == BatchJobStatus.COMPLETED
    report = await svc.store.get_report_by_batch(job.batch_job_id)
    assert report is not None
    snap = await sig.snapshot(job.batch_job_id)
    assert snap["completed"] == 2
    assert snap["total"] == 2


@pytest.mark.asyncio
async def test_create_batch_signal_factory() -> None:
    assert isinstance(create_batch_signal(None), InMemoryBatchSignal)
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        assert isinstance(create_batch_signal(client), RedisBatchSignal)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_create_batch_signal_from_env_memory_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLMJUDGE_REDIS_URL", raising=False)
    sig = await create_batch_signal_from_env()
    assert isinstance(sig, InMemoryBatchSignal)


@pytest.mark.asyncio
async def test_create_batch_signal_from_env_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMJUDGE_REDIS_URL", "redis://127.0.0.1:1/0")
    sig = await create_batch_signal_from_env(ping=True)
    assert isinstance(sig, InMemoryBatchSignal)
