"""queue_runtime — eval 共享 Redis 连接与 queue_prefix 持有。

供 ``eval_queue.inflight_slot``（槽位计数）、``eval_workers`` healer
锁、以及 ``benchmark.ieval`` query 去重等业务复用同一 Redis 连接，
避免 benchmark 反向依赖 eval_queue 基建包。

库层使用标准 ``logging``，不依赖服务层设施。
"""

from dataclasses import dataclass

import redis.asyncio as aioredis
from benchmark.libs.agenthost.observability.logger import get_logger
from redis.asyncio import Redis

logger = get_logger("agent_eval.libs.queue_runtime")


@dataclass
class _QueueRuntime:
    """队列运行时容器：Redis 连接 + BullMQ ``queue_prefix``。"""

    redis: Redis | None = None
    prefix: str = ""


_runtime = _QueueRuntime()


def init_redis(redis_url: str, *, prefix: str = "") -> Redis:
    """初始化模块级 Redis 客户端（``EvalQueueRegistry`` 启动时调用）。

    幂等：连接已存在则跳过；``prefix`` 非空时始终更新前缀记录。
    """
    if prefix:
        _runtime.prefix = prefix
    if _runtime.redis is not None:
        return _runtime.redis
    _runtime.redis = aioredis.from_url(redis_url, decode_responses=True)
    logger.info(
        "queue_runtime.initialized prefix=%s",
        _runtime.prefix or "(none)",
    )
    return _runtime.redis


async def close_redis() -> None:
    """关闭 Redis 连接（进程退出时调用）。"""
    if _runtime.redis is not None:
        await _runtime.redis.aclose()
        _runtime.redis = None
        _runtime.prefix = ""
        logger.info("queue_runtime.closed")


def get_redis() -> Redis:
    """返回已初始化的 Redis 客户端。"""
    if _runtime.redis is None:
        raise RuntimeError("queue_runtime not initialized. Call init_redis(redis_url) first.")
    return _runtime.redis


def get_prefix() -> str:
    """返回 ``init_redis`` 注入的队列前缀（YAML ``queue_prefix``）。"""
    return _runtime.prefix


__all__ = [
    "close_redis",
    "get_prefix",
    "get_redis",
    "init_redis",
]
