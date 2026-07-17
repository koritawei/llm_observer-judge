"""STEP-16: Redis Streams MQ + multi-process consumer groups."""

from __future__ import annotations

import pytest

from llmobserverjudge.contracts.topics import Component, Topic
from llmobserverjudge.dispatch.publisher import DispatchPublisher
from llmobserverjudge.eval_worker.worker import EvalWorker
from llmobserverjudge.ingest.worker import IngestWorker
from llmobserverjudge.shared.mq import InMemoryMQ, create_mq, create_mq_from_env
from llmobserverjudge.shared.persistence import create_persistence
from llmobserverjudge.shared.redis_streams_mq import RedisStreamsMQ

fakeredis = pytest.importorskip("fakeredis")


@pytest.fixture
async def redis_client():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_redis_streams_publish_drain_ack(redis_client) -> None:
    pub = RedisStreamsMQ(redis_client, consumer_group="gateway", prefix="llmj:t:")
    sub = RedisStreamsMQ(redis_client, consumer_group="ingest", prefix="llmj:t:")
    await pub.publish(
        Component.GATEWAY,
        Topic.SPANS,
        {"tenant_id": "t1", "span_id": "s1", "trace_id": "0" * 32},
    )
    assert await pub.depth(Topic.SPANS) == 1
    batch = await sub.drain(Topic.SPANS, max_count=10)
    assert len(batch) == 1
    assert batch[0]["span_id"] == "s1"
    assert "_mq_id" in batch[0]
    await sub.ack(Topic.SPANS, [batch[0]["_mq_id"]])
    assert await sub.drain(Topic.SPANS) == []


@pytest.mark.asyncio
async def test_redis_streams_consumer_groups_independent(redis_client) -> None:
    """Different groups each see the message (fan-out)."""
    pub = RedisStreamsMQ(redis_client, consumer_group="gw", prefix="llmj:fan:")
    a = RedisStreamsMQ(redis_client, consumer_group="group-a", prefix="llmj:fan:")
    b = RedisStreamsMQ(redis_client, consumer_group="group-b", prefix="llmj:fan:")
    await pub.publish(Component.GATEWAY, Topic.SPANS, {"span_id": "x"})
    assert (await a.drain(Topic.SPANS))[0]["span_id"] == "x"
    assert (await b.drain(Topic.SPANS))[0]["span_id"] == "x"


@pytest.mark.asyncio
async def test_create_mq_redis_factory(redis_client) -> None:
    mq = create_mq("redis", redis=redis_client, consumer_group="g1")
    assert isinstance(mq, RedisStreamsMQ)
    await mq.health_check()


def test_create_mq_rocketmq_stub() -> None:
    with pytest.raises(NotImplementedError, match="Redis Streams"):
        create_mq("rocketmq")


def test_create_mq_from_env_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMJUDGE_MQ", raising=False)
    assert isinstance(create_mq_from_env(), InMemoryMQ)


@pytest.mark.asyncio
async def test_multiprocess_path_via_redis_streams(redis_client) -> None:
    """Simulate split processes: gateway→ingest→dispatch→eval over shared Redis."""
    from datetime import UTC, datetime

    gw = RedisStreamsMQ(redis_client, consumer_group="gateway", prefix="llmj:e2e:")
    ingest_mq = RedisStreamsMQ(redis_client, consumer_group="ingest", prefix="llmj:e2e:")
    dispatch_mq = RedisStreamsMQ(
        redis_client, consumer_group="dispatch", prefix="llmj:e2e:"
    )
    eval_mq = RedisStreamsMQ(redis_client, consumer_group="eval", prefix="llmj:e2e:")

    bundle = await create_persistence("memory")
    ingest = IngestWorker(ingest_mq, outbox=bundle.outbox, force_sample=True)
    dispatch = DispatchPublisher(dispatch_mq, bundle.outbox)
    evaluation = EvalWorker(eval_mq)

    await gw.publish_batch(
        Component.GATEWAY,
        Topic.SPANS,
        [
            {
                "tenant_id": "t1",
                "trace_id": "0123456789abcdef0123456789abcdef",
                "span_id": "0123456789abcdef",
                "name": "chat",
                "timestamp": datetime.now(UTC).isoformat(),
                "status": "OK",
                "kind": "LLM",
                "response": "a sufficiently long answer for quality judge",
            }
        ],
    )
    assert await ingest.process_once() == 1
    assert await dispatch.publish_pending() == 1
    states = await evaluation.process_once()
    assert len(states) == 1
    assert states[0].tenant_id == "t1"
    assert states[0].final_score > 0
