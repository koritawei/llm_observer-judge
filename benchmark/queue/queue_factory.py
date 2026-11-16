"""queue_factory — Redis 连接 + BullMQ Queue/Worker 工厂（设计文档 §6/§14）。

职责：
1. 通过 normandy-credential-sdk 或环境变量解析 Redis 连接 URL
2. 构建 6 个 BullMQ Queue（队列名 = ``{prefix}:bullmq:{name}``）
3. 提供 Worker 创建工厂函数

队列命名规则：
    逻辑前缀 ``{yaml_queue_prefix}:bullmq:{BULLMQ_KEYSPACE_GENERATION}``，
    Redis hash-tag 键 ``{{logic}:{queue_name}}``。
    裸泳道来自 YAML ``custom.bullmq.queue_prefix``（daily/pre/prod）或
    业务隔离用的 ``EVAL_QUEUE_PREFIX``（非保留字）；世代见
    :mod:`benchmark.queue.queue_prefix_policy`（递增即废弃历史键空间）。

Redis 连接策略（优先级递减）：
1. ``EVAL_QUEUE_REDIS_URL`` 环境变量直连（本地开发）
2. ``EVAL_QUEUE_REDIS_KVSTORE_INSTANCE_ID`` → normandy SDK 凭证 + endpoint
3. ``EVAL_QUEUE_REDIS_HOST`` + 密码环境变量（兼容旧配置）
"""

from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Queue

try:
    # normandy-credential-sdk：仅生产部署环境安装（kvstore 凭证 + endpoint），
    # 本地 / CI 缺包时置 None，resolve_redis_url 用到时再报错。
    from benchmark.libs.credential import factory, helpers, names
except ImportError:  # pragma: no cover - 取决于部署环境
    factory = helpers = names = None  # type: ignore[assignment]

from benchmark.settings.node_identity import (
    NodeIdentity,
    current_node_identity,
    match_node_identity,
)
from benchmark.settings.setting import Settings
from benchmark.libs.jobqueue.key_prefix import queue_key_prefix
from benchmark.queue.inflight_slot import (
    close_slot_redis,
    init_slot_redis,
)
from benchmark.queue.queue_prefix_policy import build_bullmq_logic_prefix

logger = get_logger("agent_eval.eval_queue.queue_factory")

# ── 队列名常量（设计文档 §6.1）──────────────────────────────────────────────

BENCHMARK_CASES = "benchmark-cases"
#: L2 进程隔离队列（多实例隔离设计 §12.7）：命中 L2 的 business 的
#: case job 走本队列，由 AGENT_EVAL_WORKER_PROFILE=l2 的独立进程
#: 消费，获得与默认队列的故障/资源物理隔离；job 载荷与默认队列同构。
BENCHMARK_CASES_L2 = "benchmark-cases-l2"
BENCHMARK_RUNNING = "benchmark-running"
EXTERNAL_WAIT = "external-wait"
EXTERNAL_RUNNING = "external-running"
ATTRIBUTION_WAIT = "attribution-wait"
ATTRIBUTION_RUNNING = "attribution-running"
CODE_ATTRIBUTION = "code-attribution"

ALL_QUEUE_NAMES = (
    BENCHMARK_CASES,
    BENCHMARK_CASES_L2,
    BENCHMARK_RUNNING,
    EXTERNAL_WAIT,
    EXTERNAL_RUNNING,
    ATTRIBUTION_WAIT,
    ATTRIBUTION_RUNNING,
    CODE_ATTRIBUTION,
)


def case_queue_name(business: str | None, l2_businesses: list[str]) -> str:
    """case job 路由：business 命中 L2 名单走隔离队列，其余默认队列。"""
    if business and business in l2_businesses:
        return BENCHMARK_CASES_L2
    return BENCHMARK_CASES


def _resolve_redis_url_from_normandy(instance_id: str, settings: Settings) -> str:
    """normandy SDK 凭据 → Redis URL（endpoint 优先显式 host/port 配置）。"""
    if factory is None:
        raise RuntimeError(
            "EVAL_QUEUE_REDIS_KVSTORE_INSTANCE_ID requires the normandy-credential-sdk package."
        )

    urn = names.of_aliyun_kvstore_instance_id(instance_id)
    provider = factory.get_default_credential_provider()
    cred = provider.get_credential(urn)

    # endpoint：优先用显式 host/port 配置，否则从 SDK 获取
    if settings.eval_queue_redis_host:
        endpoint = f"{settings.eval_queue_redis_host}:{settings.eval_queue_redis_port}"
    else:
        endpoint = helpers.get_endpoint(urn)
    endpoint = endpoint.removeprefix("redis://").removeprefix("rediss://")

    password = getattr(cred, "password", None)
    if not password:
        raise RuntimeError("Kvstore credential did not include a password.")
    username = getattr(cred, "username", None)

    if username:
        auth = f"{quote(str(username), safe='')}:{quote(str(password), safe='')}@"
    else:
        auth = f":{quote(str(password), safe='')}@"

    scheme = settings.eval_queue_redis_scheme
    db = settings.eval_queue_redis_db
    url = f"{scheme}://{auth}{endpoint}/{db}"
    logger.info(
        "eval_queue.redis_resolved source=normandy instance=%s endpoint=%s db=%d",
        instance_id,
        endpoint,
        db,
    )
    return url


def _resolve_redis_url_from_host(settings: Settings) -> str:
    """host + password 环境变量 → Redis URL（兼容旧配置）。"""
    host = settings.eval_queue_redis_host
    port = settings.eval_queue_redis_port
    db = settings.eval_queue_redis_db
    password = settings.eval_queue_redis_password
    username = settings.eval_queue_redis_username
    scheme = settings.eval_queue_redis_scheme

    if password:
        if username:
            auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
        else:
            auth = f":{quote(password, safe='')}@"
    else:
        auth = ""
    url = f"{scheme}://{auth}{host}:{port}/{db}"
    logger.info(
        "eval_queue.redis_resolved source=env host=%s port=%d db=%d",
        host,
        port,
        db,
    )
    return url


def resolve_redis_url(settings: Settings) -> str:
    """解析 Redis 连接 URL（normandy SDK → 环境变量 → fallback）。

    优先级：
    1. ``EVAL_QUEUE_REDIS_URL`` 直连（本地开发 / 调试）
    2. ``EVAL_QUEUE_REDIS_KVSTORE_INSTANCE_ID`` → normandy SDK
    3. ``EVAL_QUEUE_REDIS_HOST`` + password 环境变量
    """
    # 1. 直连 URL（最高优先级）
    if settings.eval_queue_redis_url:
        return settings.eval_queue_redis_url

    # 2. normandy SDK（生产推荐）
    instance_id = settings.eval_queue_redis_kvstore_instance_id
    if instance_id:
        return _resolve_redis_url_from_normandy(instance_id, settings)

    # 3. host + password 环境变量（兼容旧配置）
    if settings.eval_queue_redis_host:
        return _resolve_redis_url_from_host(settings)

    raise RuntimeError(
        "Redis connection not configured. Set EVAL_QUEUE_REDIS_URL, "
        "EVAL_QUEUE_REDIS_KVSTORE_INSTANCE_ID, or EVAL_QUEUE_REDIS_HOST."
    )


def _load_bullmq_cfg(yaml_path: Path) -> dict[str, Any]:
    """读 YAML ``custom.bullmq`` 段：文件缺失（本地开发可无配置）→ warning + 空回退；
    OSError（权限等）/ 坏 YAML → RuntimeError 带路径（静默回退更危险）。"""
    try:
        with yaml_path.open() as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("eval_queue.yaml_missing path=%s fallback=default", yaml_path)
        return {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"读取配置文件失败（权限/格式）: {yaml_path} — {exc!r}") from exc
    cfg = (data.get("custom") or {}).get("bullmq") or {}
    return cfg if isinstance(cfg, dict) else {}


def _select_node_override(
    overrides: object,
    identity: NodeIdentity,
) -> dict[str, Any] | None:
    """First ``node_slots`` entry whose match fields all equal ``identity``."""
    if not isinstance(overrides, list):
        return None
    for item in overrides:
        if not isinstance(item, dict):
            continue
        match = item.get("match")
        if not isinstance(match, dict):
            continue
        if match_node_identity(match, identity):
            return item
    return None


def load_queue_prefix_from_yaml(
    yaml_path: Path,
    *,
    identity: NodeIdentity | None = None,
) -> str:
    """从 ``conf/app.<env>.yaml`` 读取队列前缀。

    通用值在 ``custom.bullmq.queue_prefix``；``node_slots`` 匹配当前节点且
    写了 ``queue_prefix`` 时覆盖为该节点独占前缀。缺失时回退为空字符串。
    """
    cfg = _load_bullmq_cfg(yaml_path)
    generic = str(cfg.get("queue_prefix") or "")
    resolved = identity if identity is not None else current_node_identity()
    override = _select_node_override(cfg.get("node_slots"), resolved)
    if override is None:
        return generic
    pinned = override.get("queue_prefix")
    if pinned is None or str(pinned) == "":
        return generic
    prefix = str(pinned)
    logger.info(
        "eval_queue.node_prefix_applied hostname=%s prefix=%s",
        resolved.hostname,
        prefix,
    )
    return prefix


def _deep_merge_slots(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge node override onto generic slots; nested dicts are shallow-merged."""
    merged: dict[str, Any] = {}
    for key, value in base.items():
        merged[key] = dict(value) if isinstance(value, dict) else value
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = {**existing, **value}
        else:
            merged[key] = value
    return merged


def load_slots_from_yaml(
    yaml_path: Path,
    *,
    identity: NodeIdentity | None = None,
) -> dict[str, Any]:
    """从 ``conf/app.<env>.yaml`` 的 ``custom.bullmq.slots`` 读取槽位配置。

    若 ``node_slots`` 有条目且 hostname / host_ip / idc 全部匹配当前节点，
    则把该条目的 ``slots`` 合并到通用配置上。缺失时返回空字典。
    """
    cfg = _load_bullmq_cfg(yaml_path)
    slots = cfg.get("slots") or {}
    if not isinstance(slots, dict):
        slots = {}
    resolved = identity if identity is not None else current_node_identity()
    override = _select_node_override(cfg.get("node_slots"), resolved)
    extra = override.get("slots") if override is not None else None
    if not isinstance(extra, dict):
        return slots
    merged = _deep_merge_slots(slots, extra)
    logger.info(
        "eval_queue.node_slots_applied hostname=%s idc=%s host_ips=%s",
        resolved.hostname,
        resolved.idc,
        sorted(resolved.host_ips),
    )
    return merged


#: ``custom.bullmq.slots.sync`` int 字段 → Settings 属性（apply 用，缺省不动）
_SYNC_INT_SLOT_FIELDS: dict[str, str] = {
    "worker_concurrency": "eval_queue_worker_concurrency",
    "slots_per_run_per_node": "sync_slots_per_run_per_node",
    "slot_busy_delay_ms": "sync_slot_busy_delay_ms",
    "slot_heal_interval_s": "sync_slot_heal_interval_s",
    "retry_max_duration_ms": "sync_retry_max_duration_ms",
    "max_slot_probes_per_run": "max_slot_probes_per_run",
}

#: ``custom.bullmq.slots.external`` int 字段 → Settings 属性（apply 用，缺省不动）。
#: ``attr_callback_url``（str）与 ``attr_callback_timeout_ms``（fallback 语义）
#: 单独处理，不在此表。
_EXTERNAL_INT_SLOT_FIELDS: dict[str, str] = {
    "worker_concurrency": "external_worker_concurrency",
    "slots_per_run_per_node": "external_slots_per_run_per_node",
    "cluster_notification_soft_cap": "external_cluster_notification_soft_cap",
    "slot_busy_delay_ms": "external_slot_busy_delay_ms",
    "slot_heal_interval_s": "external_slot_heal_interval_s",
    "retry_max_duration_ms": "external_retry_max_duration_ms",
    "run_max_duration_ms": "external_run_max_duration_ms",
    "callback_concurrency": "external_callback_concurrency",
    "local_attr_concurrency": "external_local_attr_concurrency",
    "attr_worker_concurrency": "attr_worker_concurrency",
    "attr_slots_per_run_per_node": "attr_slots_per_run_per_node",
    "attr_cluster_soft_cap": "attr_cluster_soft_cap",
    "running_ttl_ms": "external_running_ttl_ms",
    "callback_timeout_ms": "external_callback_timeout_ms",
    "max_slot_probes_per_run": "max_slot_probes_per_run",
}


def _coerce_int_slot(value: object, key: str) -> int:
    """YAML slots int 字段强转；坏值（bool/None/非数字串）→ 带路径 ValueError。"""
    bad = f"custom.bullmq.{key}={value!r} 不是有效整数"
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(bad) from exc
    raise ValueError(bad)


def _apply_int_slot_fields(
    settings: Settings,
    cfg: dict[str, Any],
    fields: dict[str, str],
    section: str,
) -> None:
    """按映射把 YAML int 字段写入 Settings（缺省保持现状，坏值 fail-fast）。"""
    for yaml_key, attr in fields.items():
        if yaml_key in cfg:
            setattr(settings, attr, _coerce_int_slot(cfg[yaml_key], f"{section}.{yaml_key}"))


def apply_slots_to_settings(settings: Settings, slots: dict[str, Any]) -> None:
    """将 YAML slots 配置应用到 Settings 实例（原地修改）。

    仅覆盖出现字段（缺失保持默认）；int 坏值（字符串/布尔/None）抛带路径 ValueError。
    """
    sync_cfg = slots.get("sync") or {}
    _apply_int_slot_fields(settings, sync_cfg, _SYNC_INT_SLOT_FIELDS, "slots.sync")

    ext_cfg = slots.get("external") or {}
    _apply_int_slot_fields(settings, ext_cfg, _EXTERNAL_INT_SLOT_FIELDS, "slots.external")
    if "attr_callback_url" in ext_cfg:
        settings.attr_callback_url = str(ext_cfg["attr_callback_url"])
    if "attr_callback_timeout_ms" in ext_cfg:
        settings.attr_callback_timeout_ms = _coerce_int_slot(
            ext_cfg["attr_callback_timeout_ms"],
            "slots.external.attr_callback_timeout_ms",
        )
    elif "callback_timeout_ms" in ext_cfg:
        # 归因默认跟 Aone/judge 同一巡检间隔
        settings.attr_callback_timeout_ms = _coerce_int_slot(
            ext_cfg["callback_timeout_ms"],
            "slots.external.callback_timeout_ms",
        )


def _build_queue_prefix(settings: Settings) -> str:
    """构建逻辑前缀：``{yaml_prefix}:bullmq:{generation}``（不含 Redis hash tag）。

    ``settings.eval_queue_prefix`` 须已由 :func:`resolve_queue_prefix` 校验；
    禁止再回退 ``settings.environment``（避免未设前缀时撞上共享队列）。
    世代段使历史 ``{lane}:bullmq`` 键空间对新代码失效。
    """
    return build_bullmq_logic_prefix(settings.eval_queue_prefix)


class EvalQueueRegistry:
    """BullMQ Queue 注册表 — 创建并持有 6 个 Queue 实例。

    逻辑队列名 = ``dataset-cases`` 等短名；BullMQ Redis prefix =
    ``{env:bullmq:generation:queueName}``（带 hash tag）。共享同一 Redis URL。
    running 队列无 Worker，仅用于登记 inflight job（D9）。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._redis_url = resolve_redis_url(settings)
        self._prefix = _build_queue_prefix(settings)
        self._queues: dict[str, Queue] = {}
        self._closed = False

        for name in ALL_QUEUE_NAMES:
            opts = {
                "connection": self._redis_url,
                "prefix": queue_key_prefix(self._prefix, name),
            }
            # 逻辑名作 Queue.name；勿再拼 env 前缀，否则 key 双倍前缀且难对上 Worker
            self._queues[name] = Queue(name, opts)

        # 初始化 inflight_slot Redis 客户端（共享同一 Redis URL；集群软顶键
        # prefix 同时注入，键构造内聚在 inflight_slot，本模块不反向依赖）
        init_slot_redis(self._redis_url, prefix=self._prefix)

        logger.info(
            "eval_queue.manager_initiated prefix=%s bullmq_key_example=%s queues=%s redis=%s",
            self._prefix,
            queue_key_prefix(self._prefix, BENCHMARK_CASES),
            list(self._queues.keys()),
            "(set)" if self._redis_url else "(none)",
        )

    @property
    def prefix(self) -> str:
        """逻辑前缀（``{env}:bullmq:{generation}``），供 inflight 槽位键使用。"""
        return self._prefix

    @property
    def redis_url(self) -> str:
        """Redis 连接 URL。"""
        return self._redis_url

    def get_queue(self, name: str) -> Queue:
        """获取指定名称的 Queue。

        Args:
            name: 队列逻辑名（BENCHMARK_CASES / EXTERNAL_WAIT /
                  EXTERNAL_RUNNING）。
        """
        if name not in self._queues:
            raise KeyError(f"Unknown queue: {name}; available: {list(self._queues.keys())}")
        return self._queues[name]

    def build_worker_opts(
        self,
        *,
        queue_name: str,
        lock_duration_ms: int,
        stalled_interval_ms: int,
    ) -> dict[str, Any]:
        """构建 Worker 选项（须与对应 Queue 的 hash-tag prefix 一致）。

        ``maxStalledCount`` 默认 10（``eval_queue_max_stalled_count``）：进程
        重启 / 短暂卡死后 wait job 仍可被 stalled 机制重投（§9.2）；BullMQ
        默认 1 会在第二次 stall 后直接 failed+remove，与「崩溃后继续执行」冲突。
        """
        return {
            "connection": self._redis_url,
            "prefix": queue_key_prefix(self._prefix, queue_name),
            "lockDuration": lock_duration_ms,
            "stalledInterval": stalled_interval_ms,
            "maxStalledCount": self._settings.eval_queue_max_stalled_count,
        }

    async def close(self) -> None:
        """关闭所有 Queue 连接 + inflight_slot Redis。"""
        if self._closed:
            return
        self._closed = True
        for name, queue in self._queues.items():
            try:
                await queue.close()
            except Exception:
                logger.warning("eval_queue.close_failed name=%s", name, exc_info=True)
        # 关闭 inflight_slot Redis 连接
        await close_slot_redis()
        logger.info("eval_queue.manager_closed prefix=%s", self._prefix)


__all__ = [
    "BENCHMARK_CASES",
    "BENCHMARK_CASES_L2",
    "BENCHMARK_RUNNING",
    "case_queue_name",
    "EXTERNAL_RUNNING",
    "EXTERNAL_WAIT",
    "ATTRIBUTION_WAIT",
    "ATTRIBUTION_RUNNING",
    "CODE_ATTRIBUTION",
    "EvalQueueRegistry",
    "apply_slots_to_settings",
    "build_bullmq_logic_prefix",
    "queue_key_prefix",
    "load_queue_prefix_from_yaml",
    "load_slots_from_yaml",
    "resolve_redis_url",
]
