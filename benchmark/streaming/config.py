"""Streaming Kafka 配置装载 — consumer 参数对象 / topic 过滤规则 / env 装载。

``KafkaConsumerConfig`` / ``TopicFilterConfig`` 迁移自 ``benchmark.judge.consumer``
与 ``benchmark.judge.service``（streaming 迁移 P1，纯移动）；env 名与语义不变
（§8 配置迁移映射：``JUDGE_KAFKA_*`` 作为所有 task 的 consumer 级默认）。
"""

import json
from dataclasses import dataclass, field

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.setting import settings

logger = get_logger("agent_eval.benchmark.streaming.config")


# ---------------------------------------------------------------------------
# TopicFilterConfig — per-topic filter rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopicFilterConfig:
    """Per-topic filter configuration.

    Allows different Kafka topics to apply different filtering rules.
    When a topic is not explicitly configured in ``topic_filter_rules``,
    the default config (record_types={"trace"}, skip_deleted=True) is used.

    Example::

        rules = {
            "langfuse.records": TopicFilterConfig(),  # default: trace only
            "langfuse.observations": TopicFilterConfig(
                record_types=frozenset({"observation"}),
            ),
        }
    """

    record_types: frozenset[str] = frozenset({"trace"})
    skip_deleted: bool = True


# ---------------------------------------------------------------------------
# KafkaConsumerConfig — grouped consumer parameters (安隐硬规: ≤7 params)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KafkaConsumerConfig:
    """Kafka consumer connection + behavior parameters.

    Groups all Kafka consumer settings into a single value object to keep
    :func:`create_streaming_consumer` parameter count ≤ 7 (安隐硬规).
    Built from ``settings.judge_kafka_*`` fields in
    :class:`~benchmark.streaming.runner.StreamingBenchmarkRunner`.
    """

    topics: list[str]
    group_id: str
    bootstrap_servers: str
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str = ""
    sasl_username: str = ""
    sasl_password: str = ""
    client_id: str = "agent-eval-judge"
    poll_timeout_s: float = 1.0
    auto_commit: bool = True
    max_retries: int = 0
    max_poll_interval_ms: int | None = None
    session_timeout_ms: int | None = None
    heartbeat_interval_ms: int | None = None


# ---------------------------------------------------------------------------
# StreamingKafkaConfig — env 装载（topics 为 P5 前冷启动兜底）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamingKafkaConfig:
    """Streaming runner 的运行参数（``settings.judge_kafka_*`` 同值装载）。

    ``topics`` 为冷启动兜底订阅集（P1 静态模式唯一来源）；P5 起 topics 以
    RemoteConfig ``agent-eval.benchmark.streaming`` 的 ``tasks[].topics`` 为准，
    本字段仅在 RemoteConfig 不可用时兜底（§8）。
    """

    consumer: KafkaConsumerConfig
    topics: list[str]
    concurrency: int
    evaluate_timeout_s: int
    max_msgs_per_min: float
    topic_filter_rules: dict[str, TopicFilterConfig] = field(default_factory=dict)
    #: shadow 对账模式（P2）：S2–S7 全量执行打 shadow 日志，S8 起短路
    #: （不调 LLM / 不落库 / 不归因）。
    shadow: bool = False
    #: shadow 期独立 consumer group（``<group_id>-shadow``），避免与旧链路
    #: 均摊分区导致两侧各只见一半消息（P2 / §11）。
    shadow_group_suffix: str = "-shadow"


def parse_topic_filter_rules(
    json_str: str,
) -> dict[str, TopicFilterConfig]:
    """Parse per-topic filter rules from JSON string.

    Input format::

        {
            "langfuse.records": {"record_types": ["trace"], "skip_deleted": true},
            "langfuse.observations": {"record_types": ["observation"]}
        }

    Returns an empty dict when *json_str* is empty or invalid.
    """
    if not json_str:
        return {}
    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(
            "Invalid JUDGE_KAFKA_TOPIC_FILTERS JSON, using defaults: %s",
            json_str[:200],
        )
        return {}

    rules: dict[str, TopicFilterConfig] = {}
    for topic, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        record_types = frozenset(cfg.get("record_types", ["trace"]))
        skip_deleted = cfg.get("skip_deleted", True)
        rules[topic] = TopicFilterConfig(
            record_types=record_types,
            skip_deleted=skip_deleted,
        )
    return rules


def load_streaming_kafka_config() -> StreamingKafkaConfig:
    """从 ``settings.judge_kafka_*`` 装载 streaming 运行参数（§8 同值接管）。

    Raises:
        RuntimeError: bootstrap servers / topic 未配置（与旧 JudgeService
            ``start`` 的 fail-fast 语义一致）。
    """
    if not settings.judge_kafka_bootstrap_servers:
        raise RuntimeError(
            "Kafka bootstrap servers not configured — set JUDGE_KAFKA_BOOTSTRAP_SERVERS"
        )
    topics = [t.strip() for t in settings.judge_kafka_topic.split(",") if t.strip()]
    if not topics:
        raise RuntimeError("No Kafka topic configured — set JUDGE_KAFKA_TOPIC")

    topic_filter_rules = parse_topic_filter_rules(settings.judge_kafka_topic_filters)
    if topic_filter_rules:
        logger.info(
            "Topic filter rules loaded: %s",
            {
                t: {"record_types": sorted(r.record_types), "skip_deleted": r.skip_deleted}
                for t, r in topic_filter_rules.items()
            },
        )

    consumer = KafkaConsumerConfig(
        topics=topics,
        group_id=settings.judge_kafka_group_id,
        bootstrap_servers=settings.judge_kafka_bootstrap_servers,
        security_protocol=settings.judge_kafka_security_protocol,
        sasl_mechanism=settings.judge_kafka_sasl_mechanism,
        sasl_username=settings.judge_kafka_sasl_username,
        sasl_password=settings.judge_kafka_sasl_password,
        client_id=settings.judge_kafka_client_id,
        poll_timeout_s=settings.judge_kafka_poll_timeout_s,
        auto_commit=settings.judge_kafka_auto_commit,
        max_retries=settings.judge_kafka_max_retries,
        max_poll_interval_ms=settings.judge_kafka_max_poll_interval_ms,
        session_timeout_ms=settings.judge_kafka_session_timeout_ms,
        heartbeat_interval_ms=settings.judge_kafka_heartbeat_interval_ms,
    )
    return StreamingKafkaConfig(
        consumer=consumer,
        topics=topics,
        concurrency=settings.judge_kafka_concurrency,
        evaluate_timeout_s=settings.judge_evaluate_timeout_s,
        max_msgs_per_min=settings.judge_kafka_max_msgs_per_min,
        topic_filter_rules=topic_filter_rules,
        shadow=settings.benchmark_streaming_shadow,
    )


__all__ = [
    "KafkaConsumerConfig",
    "StreamingKafkaConfig",
    "TopicFilterConfig",
    "load_streaming_kafka_config",
    "parse_topic_filter_rules",
]
