"""In-process pipeline event bus for Dashboard WebSocket (design §10.1)."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class PipelineEvent:
    trace_id: str
    tenant_id: str
    node: str
    status: str  # completed | running | failed | skipped | pending
    duration_ms: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "node": self.node,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
            "ts": self.ts,
        }


class PipelineEventBus:
    """Fan-out to per-trace queues + global subscribers."""

    def __init__(self) -> None:
        self._by_trace: dict[str, list[asyncio.Queue[PipelineEvent | None]]] = defaultdict(
            list
        )
        self._global: list[asyncio.Queue[PipelineEvent | None]] = []
        self._history: dict[str, list[PipelineEvent]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(self, event: PipelineEvent) -> None:
        async with self._lock:
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

    def history(self, trace_id: str) -> list[PipelineEvent]:
        return list(self._history.get(trace_id, []))

    async def subscribe(
        self, *, trace_id: str | None = None
    ) -> AsyncIterator[PipelineEvent]:
        q: asyncio.Queue[PipelineEvent | None] = asyncio.Queue(maxsize=256)
        async with self._lock:
            if trace_id:
                self._by_trace[trace_id].append(q)
                replay = list(self._history.get(trace_id, []))
            else:
                self._global.append(q)
                replay = []
        try:
            for ev in replay:
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
