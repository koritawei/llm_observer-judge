"""Redis 列表暂存 — 流式 API 的 items 中间存储。

流式 API（POST /runs → POST /items × N → POST /finalize）需要一个
临时存储来累积 items，finalize 时一次性读出并入队 BullMQ。

使用 Redis 列表（RPUSH / LRANGE / LLEN / DEL）实现：
- 写入：RPUSH + EXPIRE（24h TTL 防泄漏）
- 读取：LRANGE 0 -1
- 计数：LLEN
- 清理：DEL
"""

import json
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from redis.asyncio import Redis

logger = get_logger("agent_eval.libs.job_queue.staging_buffer")

#: TTL for staging list keys (seconds) — 24h prevents unbounded growth.
_STAGING_TTL_S = 86400

#: Key prefix for Redis staging lists.
_KEY_PREFIX = "agent-eval:staging:"


def _staging_key(run_id: str) -> str:
    """Build Redis key for a run's staging list."""
    return f"{_KEY_PREFIX}{run_id}"


class StagingBuffer:
    """Redis-backed staging buffer for stream API items.

    Each item is stored as a JSON string in a Redis list keyed by run_id.
    The list has a 24h TTL to prevent unbounded growth from abandoned runs.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def close(self) -> None:
        """Close the underlying Redis connection (graceful shutdown)."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def append_items(
        self,
        run_id: str,
        items: list[dict[str, Any]],
    ) -> int:
        """Append items to the staging list for a run.

        Returns the number of items appended.
        """
        if not items:
            return 0
        key = _staging_key(run_id)
        values = [json.dumps(item, ensure_ascii=False, default=str) for item in items]
        await self._redis.rpush(key, *values)
        await self._redis.expire(key, _STAGING_TTL_S)
        logger.info(
            "staging.append run_id=%s count=%d",
            run_id,
            len(items),
        )
        return len(items)

    async def list_items(self, run_id: str) -> list[dict[str, Any]]:
        """Read all items from the staging list for a run."""
        key = _staging_key(run_id)
        raw_items = await self._redis.lrange(key, 0, -1)
        return [
            json.loads(raw) if isinstance(raw, (bytes, bytearray)) else json.loads(raw)
            for raw in raw_items
        ]

    async def count_items(self, run_id: str) -> int:
        """Count items in the staging list for a run."""
        key = _staging_key(run_id)
        return int(await self._redis.llen(key))

    async def clear(self, run_id: str) -> None:
        """Delete the staging list for a run."""
        key = _staging_key(run_id)
        await self._redis.delete(key)
        logger.info("staging.cleared run_id=%s", run_id)


__all__ = ["StagingBuffer"]
