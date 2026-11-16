"""MQ 消费端配置。

三块职责：

1. :class:`EvalMQConfig` —— ``custom.mq_consumer`` 段的强类型模型（topic / group /
   namesrv / worker 并发等），由 :func:`load_mq_config` 从 ``conf/app.<env>.yaml``
   解析，运行时敏感值（namesrv）经 env 覆盖，不入仓。
2. :func:`to_backend_configuration` —— 把 ``EvalMQConfig`` 投影成 ``message_queue``
   的 :class:`Configuration`（backend=METAQ）。``env`` 非 daily/pre/prod 时映射为
   ``Env.LOCAL`` → QueueManager 自动改用 InMemoryBackend（本地 / 测试免依赖）。
3. :func:`should_start_consumer` —— 按 ``enabled`` 开关判定是否启动 consumer
   （多实例并发消费由 RocketMQ 消费组自动负载均衡，无需 IP 白名单去重）。

设计对齐 ``services/amap-agent/app/config/group_message_queue.py``。
"""

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from benchmark.libs.mq import Configuration

#: 兜底默认 namesrv：正常由 conf/app.<env>.yaml 的 ``namesrv_domain`` 提供（配置走 yaml），
#: 仅当 yaml 与 env(AGENT_EVAL_MQ_NAMESRV_DOMAIN) 均未配置时才回落到这里。
#: daily / pre 默认（标准 nameserver）。
_DEFAULT_NAMESRV = "http://jmenv.tbsite.net:8080/rocketmq/nsaddr?nofix=1"
#: prod 默认（上海单元）。
_PROD_NAMESRV = "http://jmenv.tbsite.net:8080/rocketmq/nsaddr-sh?nofix=1"

#: agent-eval Settings.env → message_queue Env 名称；未命中回落 "local"（InMemory）。
#: 用字符串而非 Env 枚举，避免模块导入期就依赖 message_queue（延迟到真正建 backend 时）。
_ENV_NAME_MAP: dict[str, str] = {
    "daily": "daily",
    "pre": "pre",
    "prod": "prod",
}


class ConsumerTopicConfig(BaseModel):
    """单个来源（source）的订阅：一个 topic + 可选独立消费组。

    路由按 **topic → source → converter**：``consumers`` 里的键即 source（对应
    ``consumers/<source>`` 的转换器/补充器），本模型描述该 source 订阅哪个 topic。
    """

    model_config = {"extra": "forbid"}

    #: 该来源订阅的 topic 名（跨环境不同，由 conf/app.<env>.yaml 提供）。
    topic: str = ""
    #: 消费组；留空则由 topic 派生 ``CID-<topic>``。
    consumer_group: str = ""
    #: 关掉某个来源的订阅而不删配置。
    enabled: bool = True

    def resolved_group(self) -> str:
        """生效消费组：显式配置优先，否则由 topic 派生 ``CID-<topic>``。"""

        return self.consumer_group or f"CID-{self.topic}"


class EvalMQConfig(BaseModel):
    """``custom.mq_consumer`` —— agent-eval 评测消息消费配置（多来源多 topic）。"""

    model_config = {"extra": "forbid"}

    enabled: bool = True
    #: source → 订阅配置（键即来源标识，对应 consumers/<source> 与协议转换器的 source）。
    consumers: dict[str, ConsumerTopicConfig] = Field(default_factory=dict)
    #: 为空时按 env 取默认 namesrv（见 :func:`resolve_namesrv_domain`）。
    namesrv_domain: str = ""
    #: RocketMQ tag 表达式，空串=订阅全量（保留字段，当前经 QueueManager 不下发 tag）。
    tag: str = ""
    #: MetaqConsumer C 线程 → async 的内部缓冲；满则 RECONSUME_LATER（对 broker 背压）。
    max_buffer_size: int = Field(default=1000, gt=0)
    poll_timeout_s: float = Field(default=1.0, gt=0)
    #: 单次评测尝试的超时；超时视为失败 → 重试（防单条挂死永不推进）。
    handler_timeout_s: float = Field(default=120.0, gt=0)
    #: 并发 worker 池大小（**固定，禁止无限增长**）：N 个 worker 并发跑评测提吞吐。
    worker_concurrency: int = Field(default=4, gt=0)
    #: worker 队列容量（**有界**）：满则入队阻塞 → 对上游 poll 背压；内存最多 queue+并发 条。
    worker_queue_size: int = Field(default=200, gt=0)
    #: 单个任务消费失败的最大重试次数；**0 = 无限重试直到成功**（池中每个任务的要求：
    #: 该 worker 处理成功后才推进自己的下一条）。>0 时重试耗尽则跳过（避免坏消息永久占用 worker）。
    retry_max_attempts: int = Field(default=0, ge=0)
    #: 指数退避的首次等待（毫秒）。
    retry_initial_backoff_ms: int = Field(default=500, gt=0)
    #: 退避上限（毫秒）：长时间失败时封顶，避免退避无限增长（资源可控）。
    retry_max_backoff_ms: int = Field(default=30000, gt=0)

    def active_consumers(self) -> list[tuple[str, ConsumerTopicConfig]]:
        """启用的订阅列表 ``(source, cfg)``，按 source 名排序（确定性）。"""

        return sorted(
            ((name, c) for name, c in self.consumers.items() if c.enabled),
            key=lambda item: item[0],
        )

    def manager_group(self) -> str:
        """QueueManager/backend 占位用的组名：取首个启用订阅的组，无则 service 级默认。"""

        active = self.active_consumers()
        return active[0][1].resolved_group() if active else "CID-agent-eval"


def load_mq_config(
    yaml_path: Path,
    *,
    namesrv_env: str = "",
    enabled_override: bool | None = None,
) -> EvalMQConfig:
    """从 ``conf/app.<env>.yaml`` 的 ``custom.mq_consumer`` 段构造配置。

    与 :func:`benchmark.settings.triage_loader.load_triage_configs` 一样用 ``yaml.safe_load``
    独立读取（不经 AppConfigManager schema 校验）。运行时敏感值经 env 覆盖：

    - ``namesrv_env`` (``AGENT_EVAL_MQ_NAMESRV_DOMAIN``) → ``namesrv_domain``
    - ``enabled_override`` (``AGENT_EVAL_MQ_ENABLED``) → ``enabled``

    ``custom.mq_consumer`` 缺失时返回全默认值。
    """

    with yaml_path.open() as f:
        data = yaml.safe_load(f) or {}
    raw = (data.get("custom") or {}).get("mq_consumer") or {}
    cfg = EvalMQConfig.model_validate(raw)

    updates: dict[str, object] = {}
    if namesrv_env:
        updates["namesrv_domain"] = namesrv_env
    if enabled_override is not None:
        updates["enabled"] = enabled_override
    return cfg.model_copy(update=updates) if updates else cfg


def resolve_namesrv_domain(cfg: EvalMQConfig, env: str) -> str:
    """返回生效的 namesrv：显式配置优先，否则按 env 取默认。"""

    if cfg.namesrv_domain:
        return cfg.namesrv_domain
    return _PROD_NAMESRV if env == "prod" else _DEFAULT_NAMESRV


def to_backend_configuration(cfg: EvalMQConfig, env: str) -> "Configuration":
    """把 :class:`EvalMQConfig` 投影为 ``message_queue`` 的 :class:`Configuration`。

    ``env`` 命中 daily/pre/prod → 对应 ``Env`` + MetaqBackend；否则 ``Env.LOCAL``
    → QueueManager 自动改用 InMemoryBackend（本地 / 单测免装 rocketmq）。
    """

    from benchmark.libs.mq import BackendType, Configuration, Env

    mq_env = Env(_ENV_NAME_MAP.get(env, "local"))
    return Configuration(
        env=mq_env,
        backend=BackendType.METAQ,
        service_name="agent-eval",
        extra={
            "namesrv_domain": resolve_namesrv_domain(cfg, env),
            "producer_group": cfg.manager_group(),
            "max_buffer_size": str(cfg.max_buffer_size),
            "ack_timeout": str(cfg.handler_timeout_s),
            # 并发消费（下游有界 worker 池并发处理）：不启用 orderly，避免与并发相悖；
            # 逐条必达由"每任务重试直到成功才提交"保证，而非依赖 broker 顺序投递。
            "orderly_consume": "false",
        },
    )


def should_start_consumer(cfg: EvalMQConfig) -> bool:
    """判定本进程是否应启动 consumer：``enabled`` 开启且至少有一个启用订阅。

    多实例并发消费由 RocketMQ 消费组（同 CID）自动负载均衡，无需 IP 白名单去重；
    本地/离线开发若不想连 broker，设 ``AGENT_EVAL_MQ_ENABLED=false`` 关闭即可。
    开启但未配置任何启用订阅（``consumers`` 为空）时不启动，避免空转的 worker 池。
    """

    return cfg.enabled and bool(cfg.active_consumers())


__all__ = [
    "ConsumerTopicConfig",
    "EvalMQConfig",
    "load_mq_config",
    "resolve_namesrv_domain",
    "should_start_consumer",
    "to_backend_configuration",
]
