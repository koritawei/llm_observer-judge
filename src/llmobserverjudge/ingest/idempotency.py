"""Ingest-side idempotency store (tenant_id:span_id)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class IdempotencyStore(Protocol):
    async def exists(self, key: str) -> bool: ...

    async def mark(self, key: str, *, ttl_seconds: int = 86400) -> None: ...


@dataclass
class InMemoryIdempotencyStore:
    _keys: set[str] = field(default_factory=set)

    async def exists(self, key: str) -> bool:
        return key in self._keys

    async def mark(self, key: str, *, ttl_seconds: int = 86400) -> None:
        _ = ttl_seconds  # TTL enforced by Redis in production
        self._keys.add(key)
