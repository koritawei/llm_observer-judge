"""Redis Streams MessageQueue driver (STEP-16 multi-process).

Claim/ACK semantics:
- ``drain`` / ``reclaim_stale`` deliver messages WITHOUT acknowledging.
- Callers must ``ack`` after successful processing (at-least-once).
- Stale pending entries are reclaimable via ``XAUTOCLAIM`` / ``XCLAIM``.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from llmobserverjudge.contracts.topics import Component, Topic, assert_can_publish

logger = logging.getLogger(__name__)

MQ_MSG_ID = "_mq_id"

ResponseError: type[Exception]
try:
    from redis.exceptions import ResponseError as _ResponseError

    ResponseError = _ResponseError
except ImportError:  # pragma: no cover
    ResponseError = Exception  # type: ignore[misc, assignment]


def pop_mq_id(message: dict[str, Any]) -> str | None:
    mid = message.pop(MQ_MSG_ID, None)
    return str(mid) if mid is not None else None


class RedisStreamsMQ:
    """Stream per topic; consumer-group claim matches at-least-once semantics."""

    def __init__(
        self,
        redis: Any,
        *,
        prefix: str = "llmj:stream:",
        consumer_group: str = "llmj",
        consumer_name: str | None = None,
        block_ms: int = 10,
        claim_min_idle_ms: int = 60_000,
    ) -> None:
        self.redis = redis
        self.prefix = prefix
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or f"c-{uuid4().hex[:8]}"
        self.block_ms = block_ms
        self.claim_min_idle_ms = claim_min_idle_ms
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

    def _decode_entries(
        self, entries: list[Any]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg_id, fields in entries:
            payload = fields.get("payload") or fields.get(b"payload")
            if isinstance(payload, bytes):
                payload = payload.decode()
            data = json.loads(payload)
            if not isinstance(data, dict):
                data = {"_payload": data}
            data[MQ_MSG_ID] = (
                msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
            )
            out.append(data)
        return out

    async def drain(self, topic: Topic, max_count: int = 100) -> list[dict[str, Any]]:
        """Claim new messages (``>``). Does NOT ack — caller must ``ack``."""
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
        for _name, entries in raw:
            out.extend(self._decode_entries(entries))
        return out

    async def reclaim_stale(
        self, topic: Topic, *, max_count: int = 50, min_idle_ms: int | None = None
    ) -> list[dict[str, Any]]:
        """Reclaim idle pending messages for this consumer group."""
        await self._ensure_group(topic)
        stream = self._stream(topic)
        idle = min_idle_ms if min_idle_ms is not None else self.claim_min_idle_ms
        try:
            # redis-py >=5: xautoclaim returns (next_id, messages, deleted)
            result = await self.redis.xautoclaim(
                stream,
                self.consumer_group,
                self.consumer_name,
                min_idle_time=idle,
                start_id="0-0",
                count=max_count,
            )
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                entries = result[1] or []
                return self._decode_entries(entries)
        except Exception:
            logger.debug("xautoclaim_failed stream=%s", stream, exc_info=True)
            try:
                pending = await self.redis.xpending_range(
                    stream,
                    self.consumer_group,
                    min="-",
                    max="+",
                    count=max_count,
                )
                ids = []
                for item in pending or []:
                    if isinstance(item, dict):
                        mid = item.get("message_id") or item.get("id")
                    elif isinstance(item, (list, tuple)) and item:
                        mid = item[0]
                    else:
                        continue
                    if mid:
                        ids.append(mid)
                if not ids:
                    return []
                claimed = await self.redis.xclaim(
                    stream,
                    self.consumer_group,
                    self.consumer_name,
                    idle,
                    ids,
                )
                return self._decode_entries(claimed or [])
            except Exception:
                logger.debug("xclaim_fallback_failed stream=%s", stream, exc_info=True)
        return []

    async def ack(self, topic: Topic, message_ids: list[str]) -> None:
        if not message_ids:
            return
        stream = self._stream(topic)
        await self.redis.xack(stream, self.consumer_group, *message_ids)

    async def depth(self, topic: Topic) -> int:
        return int(await self.redis.xlen(self._stream(topic)))

    async def consumer_lag(self, topic: Topic) -> int:
        stream = self._stream(topic)
        try:
            await self._ensure_group(topic)
            pending = await self.redis.xpending(stream, self.consumer_group)
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
