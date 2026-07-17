"""Redis Streams MessageQueue driver (STEP-16 multi-process).

Each process configures ``consumer_group`` for the topics it drains.
Publishers (Gateway / Dispatch) only need Redis URL; drain uses XREADGROUP.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from llmobserverjudge.contracts.topics import Component, Topic, assert_can_publish

logger = logging.getLogger(__name__)

ResponseError: type[Exception]
try:
    from redis.exceptions import ResponseError as _ResponseError

    ResponseError = _ResponseError
except ImportError:  # pragma: no cover
    ResponseError = Exception  # type: ignore[misc, assignment]


class RedisStreamsMQ:
    """Stream per topic; consumer-group drain matches InMemoryMQ pop semantics."""

    def __init__(
        self,
        redis: Any,
        *,
        prefix: str = "llmj:stream:",
        consumer_group: str = "llmj",
        consumer_name: str | None = None,
        block_ms: int = 10,
    ) -> None:
        self.redis = redis
        self.prefix = prefix
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or f"c-{uuid4().hex[:8]}"
        self.block_ms = block_ms
        self._groups_ready: set[str] = set()

    def _stream(self, topic: Topic) -> str:
        return f"{self.prefix}{topic.value}"

    async def _ensure_group(self, topic: Topic) -> None:
        stream = self._stream(topic)
        if stream in self._groups_ready:
            return
        try:
            await self.redis.xgroup_create(
                stream, self.consumer_group, id="0", mkstream=True
            )
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        self._groups_ready.add(stream)

    async def publish(
        self, component: Component, topic: Topic, message: dict[str, Any]
    ) -> None:
        assert_can_publish(component, topic)
        # XADD creates the stream; consumer groups are created on first drain.
        await self.redis.xadd(
            self._stream(topic), {"payload": json.dumps(message, default=str)}
        )

    async def publish_batch(
        self, component: Component, topic: Topic, messages: list[dict[str, Any]]
    ) -> None:
        assert_can_publish(component, topic)
        if not messages:
            return
        stream = self._stream(topic)
        pipe = self.redis.pipeline(transaction=False)
        for message in messages:
            pipe.xadd(stream, {"payload": json.dumps(message, default=str)})
        await pipe.execute()

    async def drain(self, topic: Topic, max_count: int = 100) -> list[dict[str, Any]]:
        await self._ensure_group(topic)
        stream = self._stream(topic)
        raw = await self.redis.xreadgroup(
            groupname=self.consumer_group,
            consumername=self.consumer_name,
            streams={stream: ">"},
            count=max_count,
            block=self.block_ms,
        )
        if not raw:
            return []
        out: list[dict[str, Any]] = []
        ids: list[str] = []
        for _name, entries in raw:
            for msg_id, fields in entries:
                ids.append(msg_id)
                payload = fields.get("payload") or fields.get(b"payload")
                if isinstance(payload, bytes):
                    payload = payload.decode()
                out.append(json.loads(payload))
        if ids:
            await self.redis.xack(stream, self.consumer_group, *ids)
        return out

    async def depth(self, topic: Topic) -> int:
        return int(await self.redis.xlen(self._stream(topic)))

    async def consumer_lag(self, topic: Topic) -> int:
        """Approximate lag: pending for this group, else stream length."""
        stream = self._stream(topic)
        try:
            await self._ensure_group(topic)
            pending = await self.redis.xpending(stream, self.consumer_group)
            # redis-py: [count, min, max, consumers] or dict depending on version
            if isinstance(pending, dict):
                return int(pending.get("pending", 0) or 0)
            if isinstance(pending, (list, tuple)) and pending:
                return int(pending[0] or 0)
        except Exception:
            logger.debug("xpending_failed stream=%s", stream, exc_info=True)
        return await self.depth(topic)

    async def health_check(self) -> None:
        pong = await self.redis.ping()
        if not pong:
            raise RuntimeError("redis ping failed")

    async def aclose(self) -> None:
        close = getattr(self.redis, "aclose", None) or getattr(self.redis, "close", None)
        if close is not None:
            res = close()
            if hasattr(res, "__await__"):
                await res
