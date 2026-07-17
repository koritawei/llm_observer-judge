"""SQL / Redis / memory idempotency stores for ingest & eval."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.ingest.idempotency import InMemoryIdempotencyStore
from llmobserverjudge.shared.deployment import redis_key_prefix, resolve_idempotency_ttl
from llmobserverjudge.shared.db import is_postgres_url, resolve_database_url
from llmobserverjudge.shared.tables import IdempotencyKeyRow

logger = logging.getLogger(__name__)


class SqlIdempotencyStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory
        self._postgres = is_postgres_url(resolve_database_url())

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

    async def clear(self, key: str) -> None:
        from sqlalchemy import delete

        async with self._sf() as session:
            await session.execute(delete(IdempotencyKeyRow).where(IdempotencyKeyRow.key == key))
            await session.commit()

    async def try_mark(self, key: str, *, ttl_seconds: int = 86400) -> bool:
        """Return True if this caller acquired the key (first writer)."""
        _ = ttl_seconds
        async with self._sf() as session:
            if self._postgres:
                stmt = (
                    pg_insert(IdempotencyKeyRow)
                    .values(key=key)
                    .on_conflict_do_nothing(index_elements=["key"])
                )
            else:
                stmt = sqlite_insert(IdempotencyKeyRow).values(key=key)
                stmt = stmt.prefix_with("OR IGNORE")
            result = await session.execute(stmt)
            await session.commit()
            return bool(getattr(result, "rowcount", 0))


class RedisIdempotencyStore:
    def __init__(self, redis: Any, *, prefix: str | None = None) -> None:
        self.redis = redis
        self.prefix = prefix or f"{redis_key_prefix()}idem:"

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    async def exists(self, key: str) -> bool:
        return bool(await self.redis.exists(self._k(key)))

    async def mark(self, key: str, *, ttl_seconds: int = 86400) -> None:
        ttl = ttl_seconds if ttl_seconds > 0 else resolve_idempotency_ttl()
        await self.redis.set(self._k(key), "1", ex=max(1, ttl))

    async def try_mark(self, key: str, *, ttl_seconds: int = 86400) -> bool:
        ttl = ttl_seconds if ttl_seconds > 0 else resolve_idempotency_ttl()
        ok = await self.redis.set(
            self._k(key), "1", nx=True, ex=max(1, ttl)
        )
        return bool(ok)

    async def clear(self, key: str) -> None:
        await self.redis.delete(self._k(key))


class HybridIdempotencyStore:
    """Prefer Redis when available; fall back to SQL/memory."""

    def __init__(self, primary: Any, fallback: Any | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or InMemoryIdempotencyStore()

    async def exists(self, key: str) -> bool:
        try:
            if await self.primary.exists(key):
                return True
        except Exception:
            logger.debug("idempotency_exists_primary_failed", exc_info=True)
        return await self.fallback.exists(key)

    async def mark(self, key: str, *, ttl_seconds: int = 86400) -> None:
        ttl = ttl_seconds if ttl_seconds > 0 else resolve_idempotency_ttl()
        try:
            await self.primary.mark(key, ttl_seconds=ttl)
        except Exception:
            logger.debug("idempotency_mark_fallback", exc_info=True)
        await self.fallback.mark(key, ttl_seconds=ttl)

    async def try_mark(self, key: str, *, ttl_seconds: int = 86400) -> bool:
        ttl = ttl_seconds if ttl_seconds > 0 else resolve_idempotency_ttl()
        try_fn = getattr(self.primary, "try_mark", None)
        if try_fn is not None:
            try:
                if await try_fn(key, ttl_seconds=ttl):
                    return True
            except Exception:
                logger.debug("idempotency_try_mark_fallback", exc_info=True)
        if await self.fallback.exists(key):
            return False
        fb_try = getattr(self.fallback, "try_mark", None)
        if fb_try is not None:
            return await fb_try(key, ttl_seconds=ttl)
        await self.fallback.mark(key, ttl_seconds=ttl)
        return True

    async def clear(self, key: str) -> None:
        clear_fn = getattr(self.primary, "clear", None)
        if clear_fn is not None:
            try:
                await clear_fn(key)
            except Exception:
                logger.debug("idempotency_clear_primary_failed", exc_info=True)
        fb_clear = getattr(self.fallback, "clear", None)
        if fb_clear is not None:
            await fb_clear(key)
