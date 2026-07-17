"""SQL / Redis / memory idempotency stores for ingest & eval."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.ingest.idempotency import InMemoryIdempotencyStore
from llmobserverjudge.shared.tables import IdempotencyKeyRow

logger = logging.getLogger(__name__)


class SqlIdempotencyStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def exists(self, key: str) -> bool:
        async with self._sf() as session:
            q = await session.execute(
                select(IdempotencyKeyRow).where(IdempotencyKeyRow.key == key)
            )
            return q.scalar_one_or_none() is not None

    async def mark(self, key: str, *, ttl_seconds: int = 86400) -> None:
        _ = ttl_seconds
        async with self._sf() as session:
            q = await session.execute(
                select(IdempotencyKeyRow).where(IdempotencyKeyRow.key == key)
            )
            if q.scalar_one_or_none() is None:
                session.add(IdempotencyKeyRow(key=key))
                await session.commit()

    async def try_mark(self, key: str, *, ttl_seconds: int = 86400) -> bool:
        """Return True if this caller acquired the key (first writer)."""
        _ = ttl_seconds
        if await self.exists(key):
            return False
        try:
            async with self._sf() as session:
                session.add(IdempotencyKeyRow(key=key))
                await session.commit()
            return True
        except Exception:
            logger.debug("idempotency_try_mark_race key=%s", key, exc_info=True)
            return False


class RedisIdempotencyStore:
    def __init__(self, redis: Any, *, prefix: str = "llmj:idem:") -> None:
        self.redis = redis
        self.prefix = prefix

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    async def exists(self, key: str) -> bool:
        return bool(await self.redis.exists(self._k(key)))

    async def mark(self, key: str, *, ttl_seconds: int = 86400) -> None:
        await self.redis.set(self._k(key), "1", ex=max(1, ttl_seconds))

    async def try_mark(self, key: str, *, ttl_seconds: int = 86400) -> bool:
        ok = await self.redis.set(
            self._k(key), "1", nx=True, ex=max(1, ttl_seconds)
        )
        return bool(ok)


class HybridIdempotencyStore:
    """Prefer Redis when available; fall back to SQL/memory."""

    def __init__(self, primary: Any, fallback: Any | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or InMemoryIdempotencyStore()

    async def exists(self, key: str) -> bool:
        try:
            return await self.primary.exists(key)
        except Exception:
            logger.debug("idempotency_exists_fallback", exc_info=True)
            return await self.fallback.exists(key)

    async def mark(self, key: str, *, ttl_seconds: int = 86400) -> None:
        try:
            await self.primary.mark(key, ttl_seconds=ttl_seconds)
        except Exception:
            logger.debug("idempotency_mark_fallback", exc_info=True)
            await self.fallback.mark(key, ttl_seconds=ttl_seconds)

    async def try_mark(self, key: str, *, ttl_seconds: int = 86400) -> bool:
        try_fn = getattr(self.primary, "try_mark", None)
        if try_fn is not None:
            try:
                return await try_fn(key, ttl_seconds=ttl_seconds)
            except Exception:
                logger.debug("idempotency_try_mark_fallback", exc_info=True)
        if await self.exists(key):
            return False
        await self.mark(key, ttl_seconds=ttl_seconds)
        return True
