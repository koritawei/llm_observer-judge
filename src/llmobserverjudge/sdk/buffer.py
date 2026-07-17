"""Bounded Ring Buffer — DropOldest (SDD-IG-10 / L-SYNC)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RingBufferMetrics:
    dropped_spans_total: int = 0
    sent_spans_total: int = 0
    buffer_size: int = 4096
    current_usage: int = 0


@dataclass
class BoundedRingBuffer:
    capacity: int = 4096
    _buffer: deque[Any] = field(init=False)
    metrics: RingBufferMetrics = field(init=False)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.capacity)
        self.metrics = RingBufferMetrics(buffer_size=self.capacity)

    def push(self, span: Any) -> None:
        """Sync push; never raises. Full buffer drops oldest."""
        old_len = len(self._buffer)
        self._buffer.append(span)
        if len(self._buffer) == self.capacity and old_len == self.capacity:
            self.metrics.dropped_spans_total += 1
        self.metrics.current_usage = len(self._buffer)

    def drain(self, max_count: int = 50) -> list[Any]:
        n = min(max_count, len(self._buffer))
        batch = [self._buffer.popleft() for _ in range(n)]
        self.metrics.current_usage = len(self._buffer)
        return batch

    def __len__(self) -> int:
        return len(self._buffer)
