"""inflight_slot — Redis 计数信号量（设计文档 §6.5 D7）。

每节点每 run 的并发槽位管理，基于 Redis INCR/DECR + EXPIRE 实现。
**禁止**对计数键直接 DEL（D7），必须通过 ``release_slot`` 释放。

槽位键格式：
- 同步：``{prefix}:inflight:sync:{eval_type}:{node}:{run}``
- 异步：``{prefix}:inflight:{kind}:{node}:{run}``（``kind ∈ {judge, attr}``）
- 集群软顶：``{prefix}:inflight:cluster:{kind}``
- 槽位忙探测：``{prefix}:inflight:probe:{kind}:{node}:{run}``
  （按 kind 隔离，避免 judge/attr 共享探测配额）
"""

from dataclasses import dataclass

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.libs.queue_runtime import close_redis, get_prefix, get_redis, init_redis

logger = get_logger("agent_eval.eval_queue.inflight_slot")

# ── Lua 脚本 ──────────────────────────────────────────────────────────────

# INCR → 超限则 DECR 回滚并返回 0；否则 EXPIRE 并返回 1
_ACQUIRE_LUA = """
local current = redis.call('INCR', KEYS[1])
if current > tonumber(ARGV[1]) then
    redis.call('DECR', KEYS[1])
    return 0
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
return 1
"""

# DECR → ≤0 则 DEL（清理空键）；禁止直接 DEL
_RELEASE_LUA = """
local current = redis.call('DECR', KEYS[1])
if current <= 0 then
    redis.call('DEL', KEYS[1])
end
return current
"""

# 与 running 队列占槽数对齐：偏低则补回（TTL 过期），偏高则下降，相等则续期。
# 禁止无依据直接 DEL（expected<=0 且 current>0 时才 DEL）。
_CLAMP_LUA = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local expected = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
if current < expected then
    redis.call('SET', KEYS[1], expected, 'EX', ttl)
    return 0
end
if current == expected then
    if current > 0 then
        redis.call('EXPIRE', KEYS[1], ttl)
    end
    return 0
end
local excess = current - expected
if expected <= 0 then
    redis.call('DEL', KEYS[1])
else
    redis.call('SET', KEYS[1], expected, 'EX', ttl)
end
return excess
"""

# ── 模块级槽位运行时（由 init_slot_redis 初始化）────────────────────────


@dataclass
class _SlotRuntime:
    """槽位 Lua 脚本容器（Redis 连接见 :mod:`benchmark.libs.queue_runtime`）。

    键构造内聚在本模块，避免 inflight_slot → queue_factory 反向依赖成环。
    """

    acquire_script: object | None = None
    release_script: object | None = None
    clamp_script: object | None = None


_runtime = _SlotRuntime()


def init_slot_redis(redis_url: str, *, prefix: str = "") -> None:
    """初始化模块级 Redis 客户端与槽位键前缀（EvalQueueRegistry.__init__ 时调用）。

    幂等：Redis 已初始化则跳过；传入非空 ``prefix`` 时始终更新前缀记录。
    """
    redis_client = init_redis(redis_url, prefix=prefix)
    if _runtime.acquire_script is None:
        _runtime.acquire_script = redis_client.register_script(_ACQUIRE_LUA)
        _runtime.release_script = redis_client.register_script(_RELEASE_LUA)
        _runtime.clamp_script = redis_client.register_script(_CLAMP_LUA)
    logger.info(
        "inflight_slot.redis_initialized url=set prefix=%s",
        get_prefix() or "(none)",
    )


async def close_slot_redis() -> None:
    """关闭 Redis 连接（进程退出时调用）。"""
    _runtime.acquire_script = None
    _runtime.release_script = None
    _runtime.clamp_script = None
    await close_redis()
    logger.info("inflight_slot.redis_closed")


# ── 键构造（D7 / D13：judge 与 attr 分键）──────────────────────────────────

_ASYNC_SLOT_KINDS = frozenset({"judge", "attr"})


@dataclass(frozen=True)
class SlotKeySpec:
    """``build_slot_key`` 入参（FUNC-PARAMS-001）。"""

    kind: str
    node: str
    run_id: str
    eval_type: str | None = None


def build_slot_key(
    prefix: str,
    spec: "SlotKeySpec",
) -> str:
    """构造每节点每 run 计数槽位键。

    - ``kind ∈ {judge, attr}``：``{prefix}:inflight:{kind}:{node}:{run}``
    - 同步：``{prefix}:inflight:sync:{eval_type}:{node}:{run}``
    """
    kind = spec.kind
    node = spec.node
    run_id = spec.run_id
    eval_type = spec.eval_type
    if kind in _ASYNC_SLOT_KINDS:
        return f"{prefix}:inflight:{kind}:{node}:{run_id}"
    et = eval_type or kind
    return f"{prefix}:inflight:{kind}:{et}:{node}:{run_id}"


def build_probe_key(
    prefix: str,
    *,
    kind: str,
    node: str,
    run_id: str,
) -> str:
    """构造槽位忙探测限流键（按 kind 隔离 judge / attr）。"""
    return f"{prefix}:inflight:probe:{kind}:{node}:{run_id}"


def build_cluster_slot_key(kind: str) -> str:
    """构建集群软顶槽位键（全局计数，不含 node/run）。

    prefix 取 :mod:`benchmark.libs.queue_runtime` 注入的 ``queue_prefix``（与
    EvalQueueRegistry 队列前缀一致，由其 ``__init__`` 设置）。
    """
    prefix = get_prefix()
    if not prefix:
        raise RuntimeError(
            "inflight_slot prefix not initialized. "
            "Call init_slot_redis(redis_url, prefix=...) first."
        )
    return f"{prefix}:inflight:cluster:{kind}"


# ── 公开 API ──────────────────────────────────────────────────────────────


async def acquire_slot(key: str, *, limit: int, ttl_s: int) -> bool:
    """尝试获取一个槽位（D7）。

    INCR → 超限则 DECR 回滚返回 False；否则 EXPIRE 返回 True。

    Args:
        key: 槽位键（``{prefix}:inflight:sync:{eval_type}:{node}:{run}``）。
        limit: 该键允许的最大计数。
        ttl_s: 计数键 TTL（秒），防止崩溃泄漏。

    Returns:
        True 如果获得槽位；False 如果已满。
    """
    get_redis()
    assert _runtime.acquire_script is not None
    result = await _runtime.acquire_script(keys=[key], args=[limit, ttl_s])
    acquired = bool(int(result))
    if not acquired:
        logger.info("inflight_slot.acquired=False key=%s limit=%d", key, limit)
    else:
        logger.info("inflight_slot.acquired=True key=%s limit=%d", key, limit)
    return acquired


async def release_slot(key: str) -> None:
    """释放一个槽位（D7）。

    DECR → ≤0 则 DEL。**禁止**对计数键直接 DEL。

    Args:
        key: 槽位键。
    """
    get_redis()
    assert _runtime.release_script is not None
    try:
        await _runtime.release_script(keys=[key], args=[])
        logger.info("inflight_slot.released key=%s", key)
    except Exception:
        logger.warning("inflight_slot.release_failed key=%s", key, exc_info=True)


async def clamp_slot_count(key: str, *, expected: int, ttl_s: int) -> int:
    """把槽位计数对齐到 ``expected`` 并刷新 TTL（差额巡检）。

    - 计数偏低（key 过期 / 丢失）：SET 补回，避免 hold 期间 TTL 掉光后超发。
    - 计数偏高：降到 expected，避免泄漏导致 ``slot_busy`` 死循环。
    - 相等：仅 EXPIRE 续期。
    - ``expected <= 0`` 且仍有计数：DEL 键。

    Returns:
        降下来的差额；补回 / 续期 / 无需动返回 0。
    """
    get_redis()
    assert _runtime.clamp_script is not None
    expected_n = max(0, int(expected))
    ttl_n = max(1, int(ttl_s))
    try:
        excess = int(await _runtime.clamp_script(keys=[key], args=[expected_n, ttl_n]))
    except Exception:
        logger.warning(
            "inflight_slot.clamp_failed key=%s expected=%d",
            key,
            expected_n,
            exc_info=True,
        )
        return 0
    if excess > 0:
        logger.warning(
            "inflight_slot.clamped key=%s excess=%d expected=%d",
            key,
            excess,
            expected_n,
        )
    return excess


async def scan_slot_keys(pattern: str) -> list[str]:
    """SCAN 匹配槽位键（用于差额巡检发现孤儿计数）。"""
    r = get_redis()
    keys: list[str] = []
    async for key in r.scan_iter(match=pattern, count=100):
        keys.append(str(key))
    return keys


async def acquire_cluster_slot(
    kind: str,
    *,
    soft_cap: int,
    ttl_s: int,
) -> bool:
    """集群级软顶（D14）。

    ``soft_cap <= 0`` 则恒返回 True（禁用软顶）。
    否则对 ``{prefix}:inflight:cluster:{kind}`` 计数。

    Args:
        kind: ``"judge"`` / ``"attr"``。
        soft_cap: 集群级上限（≤0 禁用）。
        ttl_s: 计数键 TTL（秒）。

    Returns:
        True 如果获得槽位或软顶禁用。
    """
    if soft_cap <= 0:
        return True
    # cluster key 不含 node/run，全局计数；键构造内聚本模块（prefix 经
    # init_slot_redis 注入，勿反向依赖 queue_factory 取前缀）
    key = build_cluster_slot_key(kind)
    return await acquire_slot(key, limit=soft_cap, ttl_s=ttl_s)


async def release_cluster_slot(kind: str) -> None:
    """释放集群软顶槽位。"""
    try:
        get_redis()
    except RuntimeError:
        return
    key = build_cluster_slot_key(kind)
    await release_slot(key)


__all__ = [
    "acquire_cluster_slot",
    "acquire_slot",
    "build_cluster_slot_key",
    "build_probe_key",
    "build_slot_key",
    "clamp_slot_count",
    "close_slot_redis",
    "init_slot_redis",
    "release_cluster_slot",
    "release_slot",
    "scan_slot_keys",
]
