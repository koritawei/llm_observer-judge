"""Pluggable MQ abstractions + memory / Redis Streams drivers (STEP-16)."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from llmobserverjudge.contracts.topics import Component, Topic, assert_can_publish

logger = logging.getLogger(__name__)

MQ_MSG_ID = "_mq_id"


@runtime_checkable
class MessageQueue(Protocol):
    async def publish(
        self, component: Component, topic: Topic, message: dict[str, Any]
    ) -> None: ...

    async def publish_batch(
        self, component: Component, topic: Topic, messages: list[dict[str, Any]]
    ) -> None: ...

    async def health_check(self) -> None: ...

    async def drain(self, topic: Topic, max_count: int = 100) -> list[dict[str, Any]]: ...

    async def depth(self, topic: Topic) -> int: ...

    async def consumer_lag(self, topic: Topic) -> int: ...

    async def ack(self, topic: Topic, message_ids: list[str]) -> None: ...

    async def reclaim_stale(
        self, topic: Topic, *, max_count: int = 50, min_idle_ms: int | None = None
    ) -> list[dict[str, Any]]: ...


@dataclass
class InMemoryMQ:
    """Dev/test driver. Production: Redis Streams (now) / RocketMQ (later)."""

    _queues: dict[str, deque[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    fail_health: bool = False
    # Simulated backlog ahead of consumer (for SamplingNode hysteresis)
    simulated_lag: dict[str, int] = field(default_factory=dict)

    async def publish(
        self, component: Component, topic: Topic, message: dict[str, Any]
    ) -> None:
        assert_can_publish(component, topic)
        async with self._lock:
            self._queues[topic.value].append(dict(message))

    async def publish_batch(
        self, component: Component, topic: Topic, messages: list[dict[str, Any]]
    ) -> None:
        assert_can_publish(component, topic)
        async with self._lock:
            self._queues[topic.value].extend(dict(m) for m in messages)

    async def drain(self, topic: Topic, max_count: int = 100) -> list[dict[str, Any]]:
        async with self._lock:
            q = self._queues[topic.value]
            out: list[dict[str, Any]] = []
            while q and len(out) < max_count:
                item = dict(q.popleft())
                item[MQ_MSG_ID] = item.get(MQ_MSG_ID) or f"mem-{uuid4().hex[:12]}"
                out.append(item)
            return out

    async def reclaim_stale(
        self, topic: Topic, *, max_count: int = 50, min_idle_ms: int | None = None
    ) -> list[dict[str, Any]]:
        _ = topic, max_count, min_idle_ms
        return []

    async def ack(self, topic: Topic, message_ids: list[str]) -> None:
        _ = topic, message_ids

    async def depth(self, topic: Topic) -> int:
        return len(self._queues[topic.value])

    async def consumer_lag(self, topic: Topic) -> int:
        if topic.value in self.simulated_lag:
            return self.simulated_lag[topic.value]
        return await self.depth(topic)

    async def health_check(self) -> None:
        if self.fail_health:
            raise RuntimeError("mq unavailable")


def create_mq(driver: str = "memory", **kwargs: Any) -> MessageQueue:
    """Factory: memory | redis | redis_streams | rocketmq (stub)."""
    name = (driver or "memory").lower().strip()
    if name in ("memory", "inmemory", "in_memory"):
        return InMemoryMQ(**kwargs)
    if name in ("redis", "redis_streams", "streams"):
        from llmobserverjudge.shared.redis_streams_mq import RedisStreamsMQ

        redis = kwargs.pop("redis", None)
        if redis is None:
            url = kwargs.pop("url", None) or os.environ.get("LLMJUDGE_REDIS_URL") or ""
            if not url:
                raise ValueError("redis MQ requires redis=… or LLMJUDGE_REDIS_URL / url=")
            from llmobserverjudge.benchmark.redis_signal import create_redis_client

            redis = create_redis_client(url)
        return RedisStreamsMQ(redis, **kwargs)
    if name in ("rocketmq", "rmq"):
        raise NotImplementedError(
            "RocketMQ driver not implemented yet; use LLMJUDGE_MQ=redis "
            "(Redis Streams) for multi-process, or memory for DevStack. "
            "See ADR-0003 / STEP-16."
        )
    raise ValueError(f"unsupported mq driver: {driver}")


def create_mq_from_env(
    *,
    driver: str | None = None,
    consumer_group: str | None = None,
    consumer_name: str | None = None,
    redis: Any | None = None,
) -> MessageQueue:
    """Build MQ from ``LLMJUDGE_MQ`` (default memory) + Redis URL / group."""
    chosen = (driver or os.environ.get("LLMJUDGE_MQ") or "memory").lower().strip()
    group = consumer_group or os.environ.get("LLMJUDGE_MQ_GROUP") or "llmj"
    cname = consumer_name or os.environ.get("LLMJUDGE_MQ_CONSUMER")
    if chosen in ("memory", "inmemory", "in_memory"):
        return create_mq("memory")
    if chosen in ("redis", "redis_streams", "streams"):
        from llmobserverjudge.shared.deployment import (
            redis_key_prefix,
            resolve_mq_claim_min_idle_ms,
        )

        kwargs: dict[str, Any] = {
            "consumer_group": group,
            "consumer_name": cname,
            "prefix": f"{redis_key_prefix()}stream:",
            "claim_min_idle_ms": resolve_mq_claim_min_idle_ms(),
        }
        if redis is not None:
            kwargs["redis"] = redis
        return create_mq("redis", **kwargs)
    if chosen in ("rocketmq", "rmq"):
        return create_mq("rocketmq")
    raise ValueError(f"unsupported LLMJUDGE_MQ={chosen!r}")
