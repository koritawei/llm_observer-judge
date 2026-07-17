"""Batch completion signal with Redis Lua semantics (design §11.3).

Uses redis.asyncio when available; otherwise falls back to InMemoryBatchSignal.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from llmobserverjudge.benchmark.signal import BatchSignal, InMemoryBatchSignal

logger = logging.getLogger(__name__)

# Hash + completion signal TTL (design: batch signal ~7d; COMPLETE marker 1h)
BATCH_HASH_TTL_S = 7 * 24 * 3600
COMPLETE_SIGNAL_TTL_S = 3600
BENCHMARK_CLAIM_TTL_S = 86400

BATCH_COMPLETE_LUA = """
local key = KEYS[1]
local signal_key = KEYS[2]
local field = ARGV[1]
local signal_ttl = tonumber(ARGV[2]) or 3600
redis.call('HINCRBY', key, field, 1)
local completed = tonumber(redis.call('HGET', key, 'completed') or 0)
local failed = tonumber(redis.call('HGET', key, 'failed') or 0)
local dead = tonumber(redis.call('HGET', key, 'dead') or 0)
local total = tonumber(redis.call('HGET', key, 'total') or 0)
if completed + failed + dead >= total and total > 0 then
    if redis.call('SET', signal_key, '1', 'NX', 'EX', signal_ttl) then
        return 'COMPLETE'
    end
    return 'ALREADY_SIGNALED'
end
return 'PENDING'
"""


class RedisBatchSignal:
    """Redis-backed counters; requires redis.asyncio client."""

    def __init__(
        self,
        redis: Any,
        *,
        prefix: str = "llmj:batch:",
        hash_ttl_s: int = BATCH_HASH_TTL_S,
        signal_ttl_s: int = COMPLETE_SIGNAL_TTL_S,
        claim_ttl_s: int = BENCHMARK_CLAIM_TTL_S,
    ) -> None:
        self.redis = redis
        self.prefix = prefix
        self.hash_ttl_s = hash_ttl_s
        self.signal_ttl_s = signal_ttl_s
        self.claim_ttl_s = claim_ttl_s
        self._script = None

    def _keys(self, batch_job_id: str) -> tuple[str, str]:
        return f"{self.prefix}{batch_job_id}", f"{self.prefix}signal:{batch_job_id}"

    async def init_batch(self, batch_job_id: str, total: int) -> None:
        key, signal_key = self._keys(batch_job_id)
        await self.redis.delete(key, signal_key)
        await self.redis.hset(
            key,
            mapping={
                "total": total,
                "completed": 0,
                "failed": 0,
                "dead": 0,
            },
        )
        await self.redis.expire(key, self.hash_ttl_s)

    async def incr(self, batch_job_id: str, field: str) -> str:
        if field not in ("completed", "failed", "dead"):
            raise ValueError(f"invalid field: {field}")
        key, signal_key = self._keys(batch_job_id)
        if self._script is None:
            self._script = self.redis.register_script(BATCH_COMPLETE_LUA)
        result = await self._script(
            keys=[key, signal_key],
            args=[field, str(self.signal_ttl_s)],
        )
        if isinstance(result, bytes):
            result = result.decode()
        return str(result)

    async def try_claim_benchmark(self, batch_job_id: str) -> bool:
        k = f"{self.prefix}benchmark:{batch_job_id}"
        ok = await self.redis.set(k, "1", nx=True, ex=self.claim_ttl_s)
        return bool(ok)

    async def snapshot(self, batch_job_id: str) -> dict[str, int]:
        key, _ = self._keys(batch_job_id)
        data = await self.redis.hgetall(key)
        out: dict[str, int] = {}
        for k, v in data.items():
            kk = k.decode() if isinstance(k, bytes) else str(k)
            vv = v.decode() if isinstance(v, bytes) else v
            out[kk] = int(vv)
        return out

    async def aclose(self) -> None:
        close = getattr(self.redis, "aclose", None) or getattr(self.redis, "close", None)
        if close is not None:
            res = close()
            if hasattr(res, "__await__"):
                await res


def create_batch_signal(redis: Any | None = None) -> BatchSignal:
    if redis is None:
        return InMemoryBatchSignal()
    return RedisBatchSignal(redis)


def create_redis_client(url: str, **kwargs: Any) -> Any:
    """Create redis.asyncio client from URL. Raises ImportError if redis missing."""
    import redis.asyncio as aioredis

    return aioredis.from_url(url, decode_responses=True, **kwargs)


async def create_batch_signal_from_env(
    *,
    redis_url: str | None = None,
    ping: bool = True,
) -> BatchSignal:
    """Build signal from LLMJUDGE_REDIS_URL; missing/unreachable → memory + warning."""
    url = (redis_url if redis_url is not None else os.environ.get("LLMJUDGE_REDIS_URL") or "").strip()
    if not url:
        return InMemoryBatchSignal()
    try:
        client = create_redis_client(url)
    except ImportError:
        logger.warning("redis package not installed; using InMemoryBatchSignal")
        return InMemoryBatchSignal()
    if ping:
        try:
            await client.ping()
        except Exception as e:
            logger.warning("redis ping failed (%s); using InMemoryBatchSignal", e)
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is not None:
                res = close()
                if hasattr(res, "__await__"):
                    await res
            return InMemoryBatchSignal()
    return RedisBatchSignal(client)
