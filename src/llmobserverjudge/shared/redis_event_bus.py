"""Redis Pub/Sub pipeline event bus for multi-instance Dashboard WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from llmobserverjudge.dashboard.events import PipelineEvent, PipelineEventBus

logger = logging.getLogger(__name__)


class RedisPipelineEventBus(PipelineEventBus):
    """Fan-out via Redis channel; local queues still serve WebSocket subscribers."""

    def __init__(
        self,
        redis: Any,
        *,
        channel: str = "llmj:pipeline:events",
        history_key_prefix: str = "llmj:pipeline:hist:",
        history_maxlen: int = 200,
    ) -> None:
        super().__init__()
        self.redis = redis
        self.channel = channel
        self.history_key_prefix = history_key_prefix
        self.history_maxlen = history_maxlen
        self._listener_task: asyncio.Task[None] | None = None
        self._pubsub: Any | None = None

    async def start(self) -> None:
        if self._listener_task is not None:
            return
        self._pubsub = self.redis.pubsub()
        await self._pubsub.subscribe(self.channel)
        self._listener_task = asyncio.create_task(self._listen(), name="llmj-event-bus")

    async def stop(self) -> None:
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
        if self._pubsub is not None:
            try:
                await self._pubsub.unsubscribe(self.channel)
                await self._pubsub.aclose()
            except Exception:
                logger.debug("pubsub_close_failed", exc_info=True)
            self._pubsub = None

    async def _listen(self) -> None:
        assert self._pubsub is not None
        try:
            async for msg in self._pubsub.listen():
                if msg is None:
                    continue
                if msg.get("type") != "message":
                    continue
                data = msg.get("data")
                if isinstance(data, bytes):
                    data = data.decode()
                try:
                    payload = json.loads(data)
                    event = PipelineEvent(
                        trace_id=str(payload["trace_id"]),
                        tenant_id=str(payload.get("tenant_id") or ""),
                        node=str(payload.get("node") or ""),
                        status=str(payload.get("status") or ""),
                        duration_ms=payload.get("duration_ms"),
                        detail=dict(payload.get("detail") or {}),
                        ts=str(payload.get("ts") or ""),
                    )
                except Exception:
                    logger.debug("event_bus_decode_failed", exc_info=True)
                    continue
                await self._fanout_local(event, remote=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("event_bus_listen_failed")

    async def _fanout_local(self, event: PipelineEvent, *, remote: bool = False) -> None:
        async with self._lock:
            if not remote:
                hist = self._history[event.trace_id]
                hist.append(event)
                if len(hist) > 200:
                    del hist[:-200]
            targets = list(self._by_trace.get(event.trace_id, [])) + list(self._global)
        for q in targets:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def publish(self, event: PipelineEvent) -> None:
        payload = json.dumps(event.to_dict(), default=str)
        try:
            hist_key = f"{self.history_key_prefix}{event.trace_id}"
            await self.redis.rpush(hist_key, payload)
            await self.redis.ltrim(hist_key, -self.history_maxlen, -1)
            await self.redis.expire(hist_key, 86400)
            await self.redis.publish(self.channel, payload)
        except Exception:
            logger.warning("event_bus_publish_redis_failed; local only", exc_info=True)
            await super().publish(event)
            return
        await self._fanout_local(event, remote=False)

    def history(self, trace_id: str) -> list[PipelineEvent]:
        # sync API — return local cache; async hydrate via subscribe replay
        return list(self._history.get(trace_id, []))

    async def history_async(self, trace_id: str) -> list[PipelineEvent]:
        try:
            raw = await self.redis.lrange(
                f"{self.history_key_prefix}{trace_id}", 0, -1
            )
            out: list[PipelineEvent] = []
            for item in raw or []:
                if isinstance(item, bytes):
                    item = item.decode()
                d = json.loads(item)
                out.append(
                    PipelineEvent(
                        trace_id=str(d["trace_id"]),
                        tenant_id=str(d.get("tenant_id") or ""),
                        node=str(d.get("node") or ""),
                        status=str(d.get("status") or ""),
                        duration_ms=d.get("duration_ms"),
                        detail=dict(d.get("detail") or {}),
                        ts=str(d.get("ts") or ""),
                    )
                )
            return out
        except Exception:
            logger.debug("event_bus_history_failed", exc_info=True)
            return self.history(trace_id)

    async def subscribe(
        self, *, trace_id: str | None = None
    ) -> AsyncIterator[PipelineEvent]:
        q: asyncio.Queue[PipelineEvent | None] = asyncio.Queue(maxsize=256)
        async with self._lock:
            if trace_id:
                self._by_trace[trace_id].append(q)
            else:
                self._global.append(q)
        try:
            if trace_id:
                for ev in await self.history_async(trace_id):
                    yield ev
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
        finally:
            async with self._lock:
                if trace_id:
                    lst = self._by_trace.get(trace_id, [])
                    if q in lst:
                        lst.remove(q)
                elif q in self._global:
                    self._global.remove(q)


def create_event_bus_from_env() -> PipelineEventBus:
    import os

    from llmobserverjudge.benchmark.redis_signal import create_redis_client

    driver = (os.environ.get("LLMJUDGE_EVENT_BUS") or "").lower().strip()
    url = (os.environ.get("LLMJUDGE_REDIS_URL") or "").strip()
    if driver in ("memory", "local") or (not url and driver != "redis"):
        return PipelineEventBus()
    if not url:
        return PipelineEventBus()
    try:
        return RedisPipelineEventBus(create_redis_client(url))
    except Exception:
        logger.warning("event_bus_redis_unavailable; using memory", exc_info=True)
        return PipelineEventBus()


# keep defaultdict imported for type parity if subclass uses it
_ = defaultdict
