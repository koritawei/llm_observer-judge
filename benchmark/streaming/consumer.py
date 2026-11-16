"""Streaming trace consumer — consumes ``langfuse.records`` and runs the pipeline.

Consumes ``langfuse.records`` topic via ``libs/message_queue`` ``MQConsumer``
+ ``AioKafkaBackend``.  Only ``record_type=trace`` messages are processed;
``observation`` and ``score`` messages are silently skipped.

迁移自 ``benchmark.judge.consumer``（streaming 迁移 P1）：处理阶段顺序、跳过即
return 的短路点、日志字段（``Sampling hit`` / ``Judge funnel`` 等）逐位保留
——线上排障与对账的既有契约。与旧链路的差异仅在：

- 采样 / 属性过滤 / 测试 adiu 判定复用 ``benchmark.domain.trace_filters``
  （双链路共享，不再保留本地副本）；
- 补齐 ``filter_empty_io`` / ``filter_test_adiu`` / sidecar-ingest 排除 /
  120s 消息时间戳 holdback 四个阶段（顺序以 ``TraceEvaluator.__call__``
  实际顺序为准：去重在全部过滤之后，B1.4 deferred）；
- shadow 对账模式（``BENCHMARK_STREAMING_SHADOW``）：S2–S7 全量执行并打
  shadow 日志，S8 起短路（不调 LLM / 不落库 / 不归因）。

Architecture::

    langfuse.records topic
      → MQConsumer.run() (libs/message_queue, AIOKafkaConsumer)
      → JsonCodec.decode() → RecordStreamMessage
      → early commit_message (offset ack before judge)
      → 120s message-timestamp holdback (stop_event interruptible)
      → BenchmarkStreamingEvaluator.__call__()
          → filter record_type == "trace" (+ skip_deleted)
          → trace_to_observation() → ObservationForEval
          → agent_name resolve / config match (sidecar excluded)
          → content filters (empty IO / test adiu)
          → in-process dedup (after all filters) → deterministic sampling
          → evaluate per config (per-entry failures logged; fan-out continues;
            no Kafka retry after early commit)
"""

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger
from aiolimiter import AsyncLimiter
from cachetools import TTLCache
from benchmark.libs.mq import (
    AioKafkaBackend,
    BackendType,
    Configuration,
    ConsumedMessage,
    DefaultBackoffPerformer,
    MQConsumer,
    TopicDefinition,
)
from pydantic import ValidationError

from benchmark.constant import (
    CONSUMER_BACKOFF_INITIAL_MS,
    CONSUMER_BACKOFF_MAXIMUM_MS,
    RATE_LIMITER_TIME_PERIOD_S,
    TRACE_DEDUP_TTL_S,
    TRUNCATION_MARKER,
)
from benchmark.domain.agent_config import get_agent_names_by_project_id
from benchmark.domain.eval_config import (
    EvalConfigEntry,
    declared_agent_names,
    evaluator_sample_key,
    evaluator_slot_label,
    normalize_agent_name,
    retain_configs_requiring_name_when_multi,
)
from benchmark.domain.schemas import ScoreWritebackPolicy
from benchmark.domain.trace_filters import (
    matches_trace_filters,
    should_sample_deterministic,
    should_skip_test_adiu,
)
from benchmark.evaluators.base import EvaluatorRequest, EvaluatorResult
from benchmark.streaming.config import KafkaConsumerConfig, TopicFilterConfig
from benchmark.streaming.funnel import (
    _UTC_PLUS_8,
    StreamingFunnelCounters,
    _hour_key_cn,
    emit_funnel_log,
)
from benchmark.streaming.holdback import (
    kafka_message_holdback_s,
    wait_kafka_message_holdback,
)
from benchmark.streaming.metadata import (
    AGENT_NAME_METADATA_KEY,
    is_discard_agent_name,
    metadata_agent_name,
    observation_with_agent_name,
    set_metadata_agent_name,
)
from benchmark.streaming.schemas import (
    ObservationForEval,
    RecordStreamMessage,
    TraceRecord,
)
from benchmark.services.evaluation import (
    EvalRunOptions,
    judge_report,
    update_eval_report_score,
)
from benchmark.trace.tools import bind_sidecar_observations
from benchmark.utils.exceptions import EvaluatorError

logger = get_logger("agent_eval.benchmark.streaming.consumer")


# ---------------------------------------------------------------------------
# Protocols for injectable dependencies
# ---------------------------------------------------------------------------


class EvaluatorServiceProtocol(Protocol):
    async def evaluate(self, evaluator_name: str, request: Any, *, options: Any = None) -> Any: ...

    async def judge_and_record(
        self, evaluator_name: str, request: Any, *, options: Any = None
    ) -> Any: ...

    async def after_eval(
        self,
        evaluator_name: str,
        request: Any,
        result: Any,
        *,
        options: Any = None,
    ) -> None: ...


class StreamingJudgeProtocol(Protocol):
    """streaming judge 插件协议（§5.1 ``StreamingJudgeImpl``）。

    只接管**算分**环节；after_eval 落库与 triage 归因仍走统一层
    （``EvaluatorService.after_eval``），执行与落库零分叉（§2）。
    """

    async def evaluate(self, trace: Any, config: Any) -> Any: ...


class ConfigStoreProtocol(Protocol):
    def get_configs(self, project_id: str) -> list[EvalConfigEntry]: ...

    def get_config_file(self) -> Any: ...


class RequestConverter(Protocol):
    """Converts observation + config → EvaluatorRequest."""

    def __call__(self, observation: Any, config: EvalConfigEntry) -> Any: ...


# ---------------------------------------------------------------------------
# Kafka transport metadata → downstream request / trace_meta
# ---------------------------------------------------------------------------
# Agent identity helpers live in :mod:`benchmark.streaming.metadata`
# (shared with runner).


def extract_kafka_message_meta(
    msg: ConsumedMessage[Any],
) -> dict[str, Any]:
    """Extract Kafka transport fields from a consumed message.

    Nested under ``EvaluatorRequest.metadata['kafka']`` and copied into
    ``trace_meta_json.kafka`` by ``_build_trace_meta`` so evaluate / triage /
    DB can see topic position without re-reading the broker.
    """
    headers: dict[str, str] = {}
    for name, value in msg.headers or []:
        if isinstance(value, bytes):
            headers[str(name)] = value.decode("utf-8", errors="replace")
        elif value is not None:
            headers[str(name)] = str(value)
    payload: dict[str, Any] = {
        "topic": msg.topic,
        "partition": msg.partition,
        "offset": msg.offset,
    }
    if msg.key is not None:
        payload["key"] = msg.key
    if msg.timestamp_ms is not None and msg.timestamp_ms >= 0:
        payload["timestamp_ms"] = msg.timestamp_ms
    if headers:
        payload["headers"] = headers
    return payload


#: Metadata / trace_meta_json key for :func:`extract_kafka_message_meta`.
KAFKA_MESSAGE_META_KEY = "kafka"


def configs_matching_trace_name(
    configs: list[EvalConfigEntry],
    trace_name: str | None,
) -> list[EvalConfigEntry]:
    """``config.name`` 已配置且 ``trace.name`` 全字命中的条目。

    注册 benchmark ``agentName`` 只允许在此类条目上注入/覆盖——仅 tags
    命中或 name 过滤为空时，不得把 webUI 评测对象名烙进 trace。
    """
    if not trace_name:
        return []
    return [cfg for cfg in configs if cfg.name and trace_name in cfg.name]


def _kafka_configs_with_project(
    store: ConfigStoreProtocol,
) -> list[tuple[str, EvalConfigEntry]]:
    """快照中全部非 sidecar 条目及其所属 ``project_id``；快照不可用时返回空。"""
    try:
        config_file = store.get_config_file()
    except Exception:  # noqa: BLE001 — mock / 残缺 store 回退 project 查找
        return []
    raw = getattr(config_file, "configs", None)
    if not isinstance(raw, dict):
        return []
    out: list[tuple[str, EvalConfigEntry]] = []
    for project_key, entries in raw.items():
        if not isinstance(entries, list):
            continue
        owner = str(project_key)
        for cfg in entries:
            if isinstance(cfg, EvalConfigEntry) and not cfg.is_sidecar_ingest:
                out.append((owner, cfg))
    return out


def _all_kafka_configs(store: ConfigStoreProtocol) -> list[EvalConfigEntry]:
    """快照中全部非 sidecar 条目（跨 project）；快照不可用时返回空。"""
    return [cfg for _owner, cfg in _kafka_configs_with_project(store)]


def resolve_streaming_kafka_configs(
    store: ConfigStoreProtocol,
    *,
    project_id: str,
    trace_name: str | None,
) -> tuple[list[EvalConfigEntry], str, str | None]:
    """解析本条 Kafka trace 应对齐的评测条目。

    **name 优先**：任一已配置 ``name`` 过滤且与 ``trace.name`` 全字命中的
    非 sidecar 条目（**不限消息 project_id**）命中后，直接采用，并返回该
    条目所属登记 ``project_id``（调用方须用其**强制替换**消息原值）。
    无 name 命中时回退 ``get_configs(project_id)``，``resolved_project_id``
    仍为消息侧。

    返回 ``(configs, source, resolved_project_id)``；
    ``source`` 为 ``name`` / ``project`` / ``none``。
    """
    name_hits = [
        (owner, cfg)
        for owner, cfg in _kafka_configs_with_project(store)
        if cfg.name and trace_name and trace_name in cfg.name
    ]
    if name_hits:
        # 首个 name 命中的登记 project 为准；同 project 内按 slot 去重。
        # 其它 project 的同名命中丢弃，保证强制替换后仅一条 project 身份。
        resolved_project_id = name_hits[0][0]
        seen: set[str] = set()
        deduped: list[EvalConfigEntry] = []
        for owner, cfg in name_hits:
            if owner != resolved_project_id:
                continue
            key = evaluator_slot_label(cfg)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cfg)
        return deduped, "name", resolved_project_id

    project_cfgs = [cfg for cfg in store.get_configs(project_id) if not cfg.is_sidecar_ingest]
    if project_cfgs:
        return project_cfgs, "project", project_id
    return [], "none", None


def config_accepts_reported_agent(
    config: EvalConfigEntry,
    *,
    agent_name: str | None,
    agent_missing: bool,
) -> bool:
    """单条评测条目是否接受 SDK 上报的 ``agent_name``（S4e 按条目独立判定）。

    - 缺名 trace：本维度不拦，归属交 S3'/S8；
    - 条目未声明 ``agentName``：本维度不限；
    - 已声明：上报值须与声明归一后相等。

    同一份 Kafka 消息扇出到多条目时，**不得**用各条目声明的并集做全局门禁，
    否则「A 声明 amap-agent、B 未声明」会把本可进 B 的 trace 整条丢掉。
    """
    if agent_missing:
        return True
    declared = (config.agent_name or "").strip()
    if not declared:
        return True
    return normalize_agent_name(agent_name) == normalize_agent_name(declared)


def resolve_missing_agent_name(
    matched_configs: list[EvalConfigEntry],
    *,
    project_id: str,
    tags: list[str] | None,
    trace_name: str | None = None,
) -> tuple[str | None, str]:
    """metadata 缺 ``agent_name`` 时从 benchmark 配置回填归属。

    优先级（与 webUI 实时流登记一致：条目 ``agentName`` = 评测对象 /
    benchmark agent 名）：

    1. **仅** ``trace.name`` 命中条目 ``name`` 过滤时，用该条目声明的
       ``agentName``（唯一声明 → ``entry``；多声明 → ``multi_declared``，
       交 S8 按各条目 name 命中分别归属）
    2. 无 name 命中声明时：RemoteConfig registry 按 ``project_id`` 首个有效
       候选；或 tags 与 registry 交集恰好 1 个
    3. 仍无来源 → ``(None, "none")``，调用方丢弃

    返回 ``(agent_name | None, source)``；``source`` 供日志区分来源。
    """
    # 注册 agentName：必须 name 过滤命中，禁止仅靠 tags / 空 name 回填
    declared = declared_agent_names(
        configs_matching_trace_name(matched_configs, trace_name),
    )
    if len(declared) == 1:
        return next(iter(declared)), "entry"
    if len(declared) > 1:
        return None, "multi_declared"
    candidates = [
        name
        for name in get_agent_names_by_project_id(project_id)
        if not is_discard_agent_name(name)
    ]
    if not candidates:
        return None, "none"
    by_norm = {normalize_agent_name(name): name for name in candidates}
    hits = {
        by_norm[normalize_agent_name(tag)]
        for tag in (tags or [])
        if normalize_agent_name(tag) in by_norm
    }
    if len(hits) == 1:
        return hits.pop(), "registry_tag"
    return candidates[0], "registry"


def _observation_from_ingested(request: EvaluatorRequest) -> ObservationForEval:
    """把 sidecar 已组装的 request 收成 judge 插件需要的 observation。"""
    trace_id = request.trace_id or ""
    metadata = request.metadata if isinstance(request.metadata, dict) else {}
    return ObservationForEval(
        span_id=trace_id or "sidecar",
        trace_id=trace_id,
        project_id=request.project_id or "",
        type="TRACE",
        name=request.agent_name or "sidecar",
        environment="default",
        input=request.input,
        output=request.output,
        metadata=dict(metadata),
    )


def trace_to_observation(record: TraceRecord) -> ObservationForEval:
    """Convert a trace record to :class:`ObservationForEval`.

    Field mapping::

        TraceRecord.id   → ObservationForEval.span_id + trace_id
        TraceRecord.name → ObservationForEval.name + trace_name
        (all other fields map directly)

    Trace records have no observation-level fields (model, prompt, tool calls,
    etc.); those default to ``None`` / empty in :class:`ObservationForEval`.
    """
    _check_truncation(record.id, "input", record.input)
    _check_truncation(record.id, "output", record.output)

    # Trace records have no observation-level span ID — trace ID serves as both
    # span_id and trace_id. They are always equal for trace-sourced observations;
    # observation-sourced observations (from S3) would have distinct values.
    return ObservationForEval(
        span_id=record.id,
        trace_id=record.id,
        project_id=record.project_id,
        parent_span_id=None,
        type="TRACE",
        name=record.name or "",
        environment=record.environment,
        metadata=record.metadata,
        input=record.input,
        output=record.output,
        level="DEFAULT",
        trace_name=record.name,
        trace_timestamp=record.timestamp,
        user_id=record.user_id,
        session_id=record.session_id,
        tags=record.tags,
        release=record.release,
        version=record.version,
    )


def _check_truncation(record_id: str, field_name: str, value: Any) -> None:
    """Log a warning if a field contains the Langfuse truncation marker."""
    if isinstance(value, str) and TRUNCATION_MARKER in value:
        logger.warning(
            "Trace %s field '%s' was truncated by Langfuse "
            "(exceeded 1MB). Evaluation may be affected.",
            record_id,
            field_name,
        )


#: Max chars for sampling-hit payload previews in logs (input/output/metadata).
_SAMPLE_HIT_PREVIEW_MAX_CHARS = 2000


def _preview_for_log(value: Any, *, max_chars: int = _SAMPLE_HIT_PREVIEW_MAX_CHARS) -> str:
    """Serialize a value for structured logs; truncate oversized payloads."""
    if value is None:
        return "null"
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}…(truncated,len={len(text)})"


def _is_empty_io_value(value: Any) -> bool:
    """内容过滤判定：``None`` 或空白字符串视为空 IO（S5）。

    空 dict / list 不判空——保守不扩大拦截面，避免误杀合法空结构。
    """
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


# ---------------------------------------------------------------------------
# BenchmarkStreamingEvaluator — trace message pipeline (S2–S11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkStreamingEvaluatorDeps:
    """Injectable dependencies for :class:`BenchmarkStreamingEvaluator`.

    Groups the three required collaborators into a single value object to
    keep ``BenchmarkStreamingEvaluator.__init__`` parameter count within the
    governance threshold (FUNC-PARAMS-001, 安隐硬规).
    """

    config_store: ConfigStoreProtocol
    evaluator_service: EvaluatorServiceProtocol
    request_converter: RequestConverter


@dataclass(frozen=True)
class BenchmarkStreamingEvaluatorConfig:
    """Optional behaviour knobs for :class:`BenchmarkStreamingEvaluator`.

    Defaults preserve the historical behaviour: no evaluate timeout, no
    rate limiting, default per-topic filter (``record_types={"trace"}``,
    ``skip_deleted=True``).

    ``sampling_override``（§5.1 tasks[].sampling）：非空时替代 RemoteConfig eval
    config 的 ``sampling`` 传入 S7 判定（应急止血用途；不改变 sample_key，
    同 trace 判定仍确定）。``skip_triage``（attribution_class /
    attribution_execution=none）：S10 归因与派单关闭。``shadow``（P2）：S8 起
    短路，不调 LLM / 不落库 / 不归因。``judge_impl``（tasks[].judge_class
    登记的业务专用 judge 插件）：非空时 S8 改走插件算分（落库 / 归因仍
    走统一层）；``None`` = 现网 EvaluatorService 零代码路由。
    ``judge_execution`` / ``attribution_execution``：sync/async/none 组合。
    ``triage_classifier`` / ``attribution_agent_key``：任务级覆盖默认按
    evaluator_name 查表的 triage / attribution 路由。
    """

    evaluate_timeout_s: int = 0
    max_msgs_per_min: float = 0.0
    topic_filter_rules: dict[str, TopicFilterConfig] | None = None
    sampling_override: float | None = None
    skip_triage: bool = False
    shadow: bool = False
    judge_impl: StreamingJudgeProtocol | None = None
    judge_execution: str = "sync"
    triage_classifier: object | None = None
    attribution_agent_key: str | None = None
    attribution_execution: str = "sync"


class BenchmarkStreamingEvaluator:
    """Process trace messages through the streaming pipeline.

    Key design points:

    - Input is ``ConsumedMessage[RecordStreamMessage]``
    - Only ``record_type=trace`` messages are processed
    - Deleted traces (``is_deleted=1``) are silently skipped
    - sidecar-ingest eval config entries never reach Kafka sampling
    - Empty-IO records and test-adiu records are filtered **before** dedup
    - In-process dedup runs **after** all filters (B1.4 deferred) so an
      incomplete early record never burns the dedup slot
    - All evaluate errors (timeout, 429, network) are logged by the Kafka
      early-commit wrapper in :func:`create_streaming_consumer` (offset already
      committed); the evaluator itself may still raise so unit
      tests can assert the three-tier strategy when called directly
    - Sampling uses ``trace_id`` for hash, ensuring idempotent evaluation
    - ``after_eval`` still writes TraceRecord via ``EvaluatorService``

    Usage::

        evaluator = BenchmarkStreamingEvaluator(
            deps=BenchmarkStreamingEvaluatorDeps(
                config_store=store,
                evaluator_service=svc,
                request_converter=to_evaluator_request,
            ),
        )
        consumer = create_streaming_consumer(..., processor=evaluator)
        await consumer.start()
        await consumer.run()
    """

    def __init__(
        self,
        *,
        deps: BenchmarkStreamingEvaluatorDeps,
        config: BenchmarkStreamingEvaluatorConfig | None = None,
    ) -> None:
        self._config_store = deps.config_store
        self._evaluator_service = deps.evaluator_service
        self._request_converter = deps.request_converter
        cfg = config or BenchmarkStreamingEvaluatorConfig()
        self._evaluate_timeout_s = cfg.evaluate_timeout_s
        self._rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(max_rate=cfg.max_msgs_per_min, time_period=RATE_LIMITER_TIME_PERIOD_S)
            if cfg.max_msgs_per_min > 0
            else None
        )
        self._topic_filter_rules = cfg.topic_filter_rules or {}
        self._default_filter = TopicFilterConfig()
        self._sampling_override = cfg.sampling_override
        self._skip_triage = cfg.skip_triage
        self._shadow = cfg.shadow
        self._judge_impl = cfg.judge_impl
        self._judge_execution = cfg.judge_execution
        self._triage_classifier = cfg.triage_classifier
        self._attribution_agent_key = cfg.attribution_agent_key
        self._attribution_execution = cfg.attribution_execution
        # In-process trace_id dedup: Langfuse publishes multiple trace records
        # for the same trace_id throughout its lifecycle (create → update →
        # complete).  Without dedup the same trace is sampled and evaluated
        # multiple times.  Each evaluator instance owns its cache (per task);
        # caches are never shared across processes.
        #
        # maxsize is intentionally generous (1 000 000) — TTL is the real
        # eviction policy.  LRU only kicks in if we somehow accumulate >1M
        # unique trace_ids within the TTL window, at which point we'd have a
        # much bigger problem than dedup accuracy.
        self._seen_trace_ids: TTLCache[str, None] = TTLCache(
            maxsize=1_000_000, ttl=TRACE_DEDUP_TTL_S
        )
        #: 东八区整点漏斗计数；跨小时滚动打 WARNING。
        #: threading.Lock：多 MQConsumer 共享同一 evaluator 时保护计数/换桶。
        self._funnel = StreamingFunnelCounters()
        self._funnel_hour = _hour_key_cn()
        self._funnel_lock = threading.Lock()

    def _bump_funnel(self, field: str) -> None:
        """Atomically increment one funnel counter (safe under concurrency)."""
        with self._funnel_lock:
            setattr(self._funnel, field, getattr(self._funnel, field) + 1)

    def _roll_funnel_hour(self) -> None:
        """若已跨整点，把上一小时漏斗打 WARNING 并清零。"""
        with self._funnel_lock:
            hour = _hour_key_cn()
            if hour == self._funnel_hour:
                return
            prev_hour = self._funnel_hour
            prev = self._funnel
            self._funnel = StreamingFunnelCounters()
            self._funnel_hour = hour
        emit_funnel_log(prev_hour, prev)

    def flush_funnel(self) -> None:
        """Force-emit current hour bucket (tests / graceful shutdown)."""
        with self._funnel_lock:
            hour = self._funnel_hour
            prev = self._funnel
            self._funnel = StreamingFunnelCounters()
        emit_funnel_log(hour, prev)

    def note_stop_dropped(self, count: int) -> None:
        """实例 stop drain 超时丢弃 in-flight 消息时计数（§3.2 drain 交互）。"""
        if count > 0:
            with self._funnel_lock:
                self._funnel.stop_dropped += count

    async def __call__(self, msg: ConsumedMessage[RecordStreamMessage]) -> None:
        """Process a single message — streaming pipeline S2–S11.

        Raises on transient failures (LLM 429, network) when invoked
        directly.  The Kafka path wraps this handler with early
        ``commit_message`` + holdback and swallows post-commit errors so judge
        outcomes never block offset progress.
        """
        # 跨整点先刷上一小时漏斗（无流量时延后到下一条消息）。
        self._roll_funnel_hour()

        payload = msg.payload
        self._bump_funnel("msg_total")

        # -- S2: Topic-specific filter: record_type ------------------------
        topic_filter = self._topic_filter_rules.get(msg.topic, self._default_filter)
        if payload.record_type not in topic_filter.record_types:
            self._bump_funnel("skip_record_type")
            logger.debug(
                "Skipping message: topic=%s record_type=%s (allowed: %s), key=%s, offset=%d",
                msg.topic,
                payload.record_type,
                sorted(topic_filter.record_types),
                msg.key,
                msg.offset,
            )
            return

        # -- Parse trace record --------------------------------------------
        try:
            trace = TraceRecord.model_validate(payload.record)
        except ValidationError as exc:
            self._bump_funnel("skip_invalid")
            logger.error(
                "Invalid trace record (skipping): key=%s offset=%d error=%s",
                msg.key,
                msg.offset,
                exc,
            )
            return  # Skip malformed records (don't retry)

        if topic_filter.skip_deleted and trace.is_deleted:
            self._bump_funnel("skip_deleted")
            logger.debug(
                "Skipping deleted trace: topic=%s id=%s project_id=%s",
                msg.topic,
                trace.id,
                trace.project_id,
            )
            return

        # 过滤前：合法 trace 消息进入漏斗（含后续 filter / dedup / sample）。
        self._bump_funnel("received")

        # -- S2b: Rate limiting (aiolimiter) -------------------------------
        # aiolimiter.acquire() internally schedules token release via
        # loop.call_later(); no explicit release() call needed.
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

        observation = trace_to_observation(trace)
        trace_id = observation.trace_id
        project_id = observation.project_id

        # -- S3: Resolve agent_name from RemoteConfig agent registry ------------
        # When the upstream Agent SDK doesn't populate ``metadata.agent_name``,
        # look it up from the RemoteConfig ``config.agent.registry`` by ``project_id``.
        # This prevents ``metadata_missing: agent_name`` permanent errors for
        # traces that have a valid project_id but no agent_name in metadata.
        # Always stamp via model_copy so Observation.metadata updates stick.
        #
        # -- S3: Normalize agent_name（legacy 键归一） ----------------------
        # 缺名 trace 的归属注入后移到 S4c（S3'）：观测名已按条目 name 过滤
        # 命中后，优先注入**条目声明的 agentName**（webUI 传递的归属），
        # 条目未声明才回退 RemoteConfig registry 候选——避免 registry 的
        # project→agent 登记与线上事实矛盾时把错误归属提前烙进 trace。
        agent_raw = metadata_agent_name(observation.metadata)
        # True when trace 未上报可用 agent_name（空/unknown）——S4e 对缺名
        # trace 不校验，归属由 S3' 注入值 / S8 条目声明覆盖决定。
        agent_missing = is_discard_agent_name(agent_raw)
        agent_name_val = agent_raw if not agent_missing else None
        if not agent_missing:
            # Normalize legacy ``agentName`` → ``agent_name`` on a fresh copy.
            observation = observation_with_agent_name(observation, agent_raw)
            agent_name_val = agent_raw

        # -- S3b（已移除）：缺名 trace 不再在取条目前丢弃。归因依据改由
        # S3' 统一裁决——条目声明（单值）或 registry 候选任一可用即可评；
        # 无条目命中则 S4 自然拦截，不再重复预检。
        # 「Trace received」挪到 S3' 之后：缺名 trace 先按 benchmark 条目
        # agentName 回填，再打日志，避免误报 agent_name=<missing>。

        # -- S4: Local config lookup -----------------------------------------
        # name 优先：trace.name 命中任一条目 ``name`` 过滤后，跨 project 采用该
        # 条目；命中后**强制**将 project_id 替换为条目登记侧 project（丢弃
        # Kafka 原值），下游 judge / attribution / 落库一律用新值。
        # 无 name 命中再回退 get_configs(消息 project_id)。
        # sidecar-ingest 不进 Kafka 采样（§3.2 S4）。
        kafka_project_id = project_id
        configs, config_source, resolved_project_id = resolve_streaming_kafka_configs(
            self._config_store,
            project_id=project_id,
            trace_name=trace.name,
        )
        if not configs:
            self._bump_funnel("filter_no_config")
            project_cfgs = self._config_store.get_configs(project_id)
            if project_cfgs:
                logger.info(
                    "no kafka ingest configs, skipping: trace_id=%s project_id=%s",
                    trace_id,
                    project_id,
                )
            else:
                known_projects = sorted(self._config_store.get_config_file().configs.keys())
                logger.info(
                    "projectId not configured and no name match, skipping: "
                    "trace_id=%s project_id=%s trace_name=%s "
                    "configured_project_count=%d configured_project_ids_sample=%s",
                    trace_id,
                    project_id,
                    trace.name,
                    len(known_projects),
                    known_projects[:8],
                )
            return  # Skip (commit offset, no retry)

        if (
            config_source == "name"
            and resolved_project_id
            and resolved_project_id != kafka_project_id
        ):
            logger.info(
                "project_id forced from name-matched config: "
                "trace_id=%s kafka_project_id=%s resolved_project_id=%s trace_name=%s",
                trace_id,
                kafka_project_id,
                resolved_project_id,
                trace.name,
            )
            project_id = resolved_project_id
            observation = observation.model_copy(update={"project_id": resolved_project_id})
        elif resolved_project_id:
            project_id = resolved_project_id

        # project 回退路径：同 project 多条目时无 name 的条目强制剔除（防误吃流量）。
        # name 命中路径条目已带 name，retain 为 no-op。
        if config_source == "project":
            configs, dropped_no_name = retain_configs_requiring_name_when_multi(configs)
            if dropped_no_name:
                logger.info(
                    "multi-entry project: drop configs without name filter: "
                    "trace_id=%s project_id=%s dropped=%s kept=%d",
                    trace_id,
                    project_id,
                    [evaluator_slot_label(cfg) for cfg in dropped_no_name],
                    len(configs),
                )
            if not configs:
                self._bump_funnel("filter_attrs")
                logger.info(
                    "multi-entry project: no configs left after requiring name: "
                    "trace_id=%s project_id=%s",
                    trace_id,
                    project_id,
                )
                return

        logger.info(
            "Configs found: source=%s project_id=%s count=%d slots=%s "
            "trace_id=%s trace_name=%s kafka_project_id=%s",
            config_source,
            project_id,
            len(configs),
            [evaluator_slot_label(cfg) for cfg in configs],
            trace_id,
            trace.name,
            kafka_project_id,
        )

        # -- S4b: Environment filter ----------------------------------------
        # Keep only configs whose environment list is unset (all envs) or
        # contains the trace's environment.  If none match, skip the trace
        # entirely (commit offset, no retry).
        trace_env = trace.environment or "default"
        matched_configs = [
            cfg for cfg in configs if not cfg.environment or trace_env in cfg.environment
        ]
        if not matched_configs:
            self._bump_funnel("filter_env")
            logger.info(
                "Environment mismatch, skipping: trace_id=%s env=%s project_id=%s",
                trace_id,
                trace_env,
                project_id,
            )
            return  # Skip (no retry — environment doesn't match)
        configs = matched_configs

        # -- S4c: Trace attribute filters (name / tags / timestamp) ---------
        # Each config may optionally restrict which traces it applies to
        # by trace name, tags, or timestamp range.  Empty/None = pass all.
        matched_configs = [cfg for cfg in configs if matches_trace_filters(cfg, trace)]
        if not matched_configs:
            self._bump_funnel("filter_attrs")
            logger.info(
                "Trace attribute filters excluded all configs, skipping: "
                "trace_id=%s name=%s tags=%s timestamp=%s",
                trace_id,
                trace.name,
                trace.tags,
                trace.timestamp,
            )
            return  # Skip (no retry — trace doesn't match any config's filters)
        # S4e（按条目）：同一份 Kafka 消息复制扇出——各条目独立判定
        # agentName；不得用声明并集做全局门禁（否则 A 的声明会误杀 B）。
        per_entry = [
            cfg
            for cfg in matched_configs
            if config_accepts_reported_agent(
                cfg,
                agent_name=agent_name_val,
                agent_missing=agent_missing,
            )
        ]
        if not per_entry:
            self._bump_funnel("filter_agent_name")
            logger.info(
                "Skipping trace: no evaluator entry accepts reported agent: "
                "trace_id=%s project_id=%s reported_agent=%s slots=%s",
                trace_id,
                project_id,
                repr(agent_name_val) if agent_name_val else "<missing>",
                [evaluator_slot_label(cfg) for cfg in matched_configs],
            )
            return
        matched_configs = per_entry
        # S3'（缺名归属注入）：仅单条目命中时预注入（日志 / 下游可读）；
        # 多条目扇出时不预注入全局 agent_name，避免污染其他条目——各条目
        # 在 S8 按自身 name 命中烙入注册 agentName。
        if agent_missing and len(matched_configs) == 1:
            resolved, source = resolve_missing_agent_name(
                matched_configs,
                project_id=project_id,
                tags=list(trace.tags or []),
                trace_name=trace.name,
            )
            if source == "none":
                self._bump_funnel("filter_agent_name")
                logger.info(
                    "Skipping trace: agent_name missing without any source: "
                    "trace_id=%s project_id=%s slots=%s",
                    trace_id,
                    project_id,
                    [evaluator_slot_label(cfg) for cfg in matched_configs],
                )
                return
            if resolved is not None:
                observation = observation_with_agent_name(observation, resolved)
                agent_name_val = resolved
                logger.debug(
                    "agent_name filled from %s: trace_id=%s project_id=%s slot=%s name=%s",
                    source,
                    trace_id,
                    project_id,
                    evaluator_slot_label(matched_configs[0]),
                    observation.name,
                )
        elif agent_missing and len(matched_configs) > 1:
            logger.debug(
                "multi-entry fan-out: defer agent_name to per-config S8: "
                "trace_id=%s project_id=%s slots=%s",
                trace_id,
                project_id,
                [evaluator_slot_label(cfg) for cfg in matched_configs],
            )
        # Trace received：S4e 扇出之后；单条目缺名可能已回填。
        # 实例身份用 slots（茎@registrationId），不用 agent_name / internal。
        meta_keys = sorted(observation.metadata.keys()) if observation.metadata else []
        logger.info(
            "Trace received: trace_id=%s project_id=%s name=%s "
            "key=%s offset=%d environment=%s tags=%s "
            "metadata_keys=%s slots=%s fan_out_entries=%d",
            trace_id,
            project_id,
            observation.name,
            msg.key,
            msg.offset,
            observation.environment,
            list(trace.tags or []),
            meta_keys,
            [evaluator_slot_label(cfg) for cfg in matched_configs],
            len(matched_configs),
        )
        configs = matched_configs

        # -- S5: Content filters (empty IO / test adiu) ---------------------
        # Records with neither input nor output carry no signal for the judge;
        # test-adiu accounts (``*_test`` suffix) are internal traffic.  Both
        # are filtered **before** dedup so they never burn the dedup slot.
        if _is_empty_io_value(trace.input) and _is_empty_io_value(trace.output):
            self._bump_funnel("filter_empty_io")
            # info 级别（§4 保真：漏斗关键过滤点，线上 INFO 可见）；timestamp
            # 字段保留以排查乱序；AND 语义下 has_input/has_output 恒 False 不再输出。
            logger.info(
                "Skipping trace with empty input/output: trace_id=%s project_id=%s timestamp=%s",
                trace_id,
                project_id,
                trace.timestamp,
            )
            return
        if should_skip_test_adiu(user_id=trace.user_id, input_value=trace.input):
            self._bump_funnel("filter_test_adiu")
            logger.info(
                "Skipping test adiu trace: trace_id=%s project_id=%s user_id=%s",
                trace_id,
                project_id,
                trace.user_id,
            )
            return

        # -- S6 (B1.4 deferred): In-process trace dedup (LRU + TTL) ---------
        # Langfuse publishes create → update → complete for the same
        # trace_id.  Marking *before* filters would let an early incomplete
        # record (missing agent_name / wrong name / wrong env) burn the
        # slot so a later complete record never reaches sampling.
        # Only records that pass S3–S5 are eligible to claim the key;
        # subsequent updates within TTL are skipped (sampling stays
        # idempotent).  Key 用解析后的 project_id（name 命中后已强制替换）。
        dedup_key = f"{trace.id}:{project_id}"
        if dedup_key in self._seen_trace_ids:
            self._bump_funnel("dedup_skip")
            logger.debug(
                "Skipping duplicate trace (within TTL window): "
                "trace_id=%s project_id=%s topic=%s partition=%d offset=%d",
                trace.id,
                project_id,
                msg.topic,
                msg.partition,
                msg.offset,
            )
            return
        self._seen_trace_ids[dedup_key] = None

        # 过滤后且首次占坑：进入采样决策。
        self._bump_funnel("eligible")

        # -- S7: Independent sampling per config (deterministic SHA256) -----
        # Sampling key uses trace_id (not observation_id) for idempotency:
        # the same trace always produces the same sampling decision.
        # ``sampling_override``（tasks[].sampling）只替换 rate，不改 key。
        sampled_configs: list[EvalConfigEntry] = []
        for cfg in configs:
            sample_key = evaluator_sample_key(trace_id, cfg)
            rate = self._sampling_override if self._sampling_override is not None else cfg.sampling
            if should_sample_deterministic(rate, sample_key):
                sampled_configs.append(cfg)

        logger.info(
            "Sampling result: trace_id=%s total_configs=%d sampled=%d",
            trace_id,
            len(configs),
            len(sampled_configs),
        )

        if not sampled_configs:
            self._bump_funnel("sample_miss")
            logger.debug(
                "All configs sampled out: trace_id=%s",
                trace_id,
            )
            return  # All sampled out — nothing to do

        self._bump_funnel("sample_hit")

        # Sampling hit — dump trace payload details for ops / debugging.
        # 槽位标签 = 茎@registrationId，同茎多 prompt 可区分。
        sampled_evaluators = [evaluator_slot_label(cfg) for cfg in sampled_configs]
        sampled_rates = {evaluator_slot_label(cfg): cfg.sampling for cfg in sampled_configs}
        if self._shadow:
            # P2 shadow 对账：S2–S7 全量执行，S8 起短路（不调 LLM / 不落库 /
            # 不归因）。日志字段与正式「Sampling hit」同构，便于逐 trace 比对
            # eligible / sampled 集合。
            logger.info(
                "Shadow sampling hit: trace_id=%s project_id=%s slots=%s name=%s "
                "environment=%s tags=%s user_id=%s session_id=%s "
                "sampling_rates=%s "
                "topic=%s partition=%d offset=%d "
                "input=%s output=%s metadata=%s",
                trace_id,
                project_id,
                sampled_evaluators,
                observation.name,
                observation.environment,
                list(observation.tags or []),
                observation.user_id,
                observation.session_id,
                sampled_rates,
                msg.topic,
                msg.partition,
                msg.offset,
                _preview_for_log(observation.input),
                _preview_for_log(observation.output),
                _preview_for_log(observation.metadata),
            )
            return

        logger.info(
            "Sampling hit: trace_id=%s project_id=%s slots=%s name=%s "
            "environment=%s tags=%s user_id=%s session_id=%s "
            "sampling_rates=%s "
            "topic=%s partition=%d offset=%d "
            "input=%s output=%s metadata=%s",
            trace_id,
            project_id,
            sampled_evaluators,
            observation.name,
            observation.environment,
            list(observation.tags or []),
            observation.user_id,
            observation.session_id,
            sampled_rates,
            msg.topic,
            msg.partition,
            msg.offset,
            _preview_for_log(observation.input),
            _preview_for_log(observation.output),
            _preview_for_log(observation.metadata),
        )

        # -- S8: Evaluate each sampled config independently -----------------
        # Error strategy（early-commit 后 Kafka 不再重试整条消息）：
        #   1. TimeoutError → 本条目跳过，继续下一条目。
        #   2. EvaluatorError code < 500 → 永久失败，本条目跳过。
        #   3. 其余异常（429 / 5xx / 网络）→ 记 transient，仍 continue 下一
        #      条目：不得 raise 中断扇出（否则同 project 其它 agent 永久丢评；
        #      offset 已提交，raise 也被 wrapper 吞掉，无重试价值）。
        for config in sampled_configs:
            request = self._request_converter(observation, config)
            # Carry Kafka transport metadata + force agent_name into
            # request.metadata (Judge LLM prompts / attribution often read
            # metadata, while converters may only set Request.agent_name).
            # EvaluatorRequest is frozen — always use model_copy.
            #
            # 扇出副本：每条目独立 request；agentName 仅在本 entry 的 name
            # 全字命中时覆盖。多条目互不污染 observation / metadata。
            meta = dict(request.metadata) if request.metadata else {}
            meta[KAFKA_MESSAGE_META_KEY] = extract_kafka_message_meta(msg)
            top_agent = request.agent_name
            declared = (config.agent_name or "").strip()
            if declared and config.name and observation.name and observation.name in config.name:
                top_agent = declared
            updates: dict[str, Any] = {"metadata": meta}
            if (
                isinstance(top_agent, str)
                and top_agent.strip()
                and not is_discard_agent_name(top_agent)
            ):
                stamped = top_agent.strip()
                set_metadata_agent_name(meta, stamped)
                updates["agent_name"] = stamped
            request = request.model_copy(update=updates)
            try:
                await self._evaluate_one(config, observation, request)
            except TimeoutError:
                # Permanent for this message — LLM is slow, skip this config.
                self._bump_funnel("eval_fail_timeout")
                logger.warning(
                    "evaluate timeout for trace_id=%s slot=%s after %ds, skipping",
                    trace_id,
                    evaluator_slot_label(config),
                    self._evaluate_timeout_s,
                )
                continue  # Skip to next config
            except EvaluatorError as exc:
                if exc.code < 500:
                    # Permanent error (400-level): metadata_missing,
                    # model_not_found, evaluator_prompt_not_found,
                    # trace_not_found.  The message is immutable —
                    # retrying won't fix missing metadata or config mismatch.
                    self._bump_funnel("eval_fail_permanent")
                    logger.warning(
                        "evaluate permanent failure for trace_id=%s "
                        "slot=%s code=%s error_type=%s: %s, skipping",
                        trace_id,
                        evaluator_slot_label(config),
                        exc.code,
                        exc.error_type,
                        exc.message,
                    )
                    continue  # Skip to next config
                # Transient (500+)：early-commit 下无 Kafka 重试；continue 保扇出。
                self._bump_funnel("eval_fail_transient")
                logger.warning(
                    "evaluate transient failure for trace_id=%s "
                    "slot=%s code=%s error_type=%s: %s, "
                    "skipping config (fan-out continues)",
                    trace_id,
                    evaluator_slot_label(config),
                    exc.code,
                    exc.error_type,
                    exc.message,
                )
                continue
            except Exception as exc:
                # Transient (429/网络)：同上，不中断同消息其它 benchmark 条目。
                self._bump_funnel("eval_fail_transient")
                logger.warning(
                    "evaluate failed for trace_id=%s slot=%s: %s, "
                    "skipping config (fan-out continues)",
                    trace_id,
                    evaluator_slot_label(config),
                    exc,
                )
                continue

            self._bump_funnel("eval_ok")
            logger.info(
                "Config evaluated: trace_id=%s slot=%s",
                trace_id,
                evaluator_slot_label(config),
            )

        # -- S9/S10: after_eval 落库 + 同步 triage 归因（EvaluatorService 内部）

    async def _evaluate_one(
        self,
        config: EvalConfigEntry,
        observation: ObservationForEval,
        request: Any,
    ) -> Any:
        """单 config 评测调用（含可选超时包装）。

        无覆盖路径的调用签名与旧链路逐位一致（不传 ``options``）；
        ``skip_triage``（attribution_class=no_attribution）时经
        :class:`EvalRunOptions` 关闭 S10 同步归因与派单；条目配置
        ``scoreWriteback`` 时随 options 透传给 ``EvaluatorService``
        （judge 分写回原业务 Langfuse trace，对齐旧链路）。
        """
        if self._evaluate_timeout_s > 0:
            return await asyncio.wait_for(
                self._invoke_evaluate(config, observation, request),
                timeout=self._evaluate_timeout_s,
            )
        return await self._invoke_evaluate(config, observation, request)

    def _eval_run_options(
        self,
        *,
        skip_triage: bool,
        score_writeback: ScoreWritebackPolicy | None = None,
    ) -> EvalRunOptions | None:
        """构造 after_eval / evaluate 共用选项；无覆盖时返回 ``None``（旧链路签名）。

        ``score_writeback`` 为条目级 RemoteConfig 配置（``evaluators[].scoreWriteback``），
        非 ``None`` 即视为覆盖（对齐旧链路 ``app/judge/consumer.py`` 的透传），
        保证条目未配置时旧链路默认签名不变。
        """
        has_override = (
            skip_triage
            or score_writeback is not None
            or self._triage_classifier is not None
            or self._attribution_agent_key is not None
            or self._attribution_execution not in (None, "sync")
        )
        if not has_override:
            return None
        return EvalRunOptions(
            skip_triage=skip_triage,
            score_writeback=score_writeback,
            triage_classifier=self._triage_classifier,  # type: ignore[arg-type]
            attribution_agent_key=self._attribution_agent_key,
            attribution_execution=self._attribution_execution,  # type: ignore[arg-type]
        )

    async def _invoke_evaluate(
        self,
        config: EvalConfigEntry,
        observation: ObservationForEval,
        request: Any,
    ) -> Any:
        if self._judge_execution == "async":
            # async judge：提交后立即返回，不落库 / 不归因（回调路径另行接入）。
            if self._judge_impl is None or not callable(getattr(self._judge_impl, "submit", None)):
                raise RuntimeError(
                    "judge_execution=async requires judge_impl.submit(); "
                    f"got impl={type(self._judge_impl)!r}",
                )
            job_id = await self._judge_impl.submit(observation, config)  # type: ignore[attr-defined]
            logger.info(
                "streaming.async_judge_submitted slot=%s job_id=%s",
                evaluator_slot_label(config),
                job_id,
            )
            return None

        # 条目级 attributionExecution=none（注册关闭归因时登记）同样跳过归因
        skip = (
            self._skip_triage
            or self._attribution_execution == "none"
            or config.attribution_execution == "none"
        )
        options = self._eval_run_options(
            skip_triage=skip,
            score_writeback=config.score_writeback,
        )

        if self._judge_impl is not None and callable(getattr(self._judge_impl, "evaluate", None)):
            # 业务专用 judge 插件（tasks[].judge_class）：只接管算分；
            # after_eval 落库与 triage 归因仍走统一层（§2 / §5.1）。
            result = await self._judge_impl.evaluate(observation, config)
            await self._evaluator_service.after_eval(
                config.evaluator_name,
                request,
                result,
                options=options,
            )
            return result
        if options is not None:
            return await self._evaluator_service.evaluate(
                config.evaluator_name,
                request,
                options=options,
            )
        return await self._evaluator_service.evaluate(config.evaluator_name, request)

    async def judge_ingested(
        self,
        config: EvalConfigEntry,
        request: EvaluatorRequest,
        observations: list[Any],
    ) -> Any:
        """评测 sidecar 已读回的 trace。

        不调用 ``EvaluatorService.evaluate``。Langfuse 根 span 与
        ``evaluate`` 共用 :func:`judge_report`。算分走
        ``judge_and_record``，或已登记的 streaming judge 插件加 ``after_eval``。
        """
        with (
            bind_sidecar_observations(observations),
            judge_report(config.evaluator_name, request) as root,
        ):
            if self._uses_judge_plugin():
                result = await self._invoke_evaluate(
                    config,
                    _observation_from_ingested(request),
                    request,
                )
            else:
                result = await self._evaluator_service.judge_and_record(
                    config.evaluator_name,
                    request,
                    options=EvalRunOptions(
                        resolve_trace=False,
                        sidecar_observations=observations,
                    ),
                )
            if isinstance(result, EvaluatorResult):
                update_eval_report_score(root, result)
            return result

    def _uses_judge_plugin(self) -> bool:
        """任务级 judge 插件接管算分；缺省走 ``judge_and_record``。"""
        if self._judge_impl is None:
            return False
        if self._judge_execution == "async":
            return True
        return callable(getattr(self._judge_impl, "evaluate", None))


# ---------------------------------------------------------------------------
# Factory — creates a configured MQConsumer
# ---------------------------------------------------------------------------


def create_streaming_consumer(
    *,
    config: KafkaConsumerConfig,
    processor: BenchmarkStreamingEvaluator,
    stop_event: asyncio.Event | None = None,
) -> MQConsumer:
    """Create a :class:`MQConsumer` configured for Langfuse trace consumption.

    Offset policy: as soon as a decoded message is handed to the processor
    wrapper, ``commit_message`` runs **before** judge evaluation.  Judge
    failures are logged and swallowed — they must not block Kafka progress
    or trigger MQ backoff retry of the same offset.

    Pipeline order inside the wrapper: ① early commit → ② 120s message
    timestamp holdback（``stop_event`` 可中断，实例 stop drain 时打断等待）→
    ③ processor。early commit 发生在 holdback 之前，因此 stop 时处于
    holdback 等待或处理中的消息 offset 已提交、重启不会重放（§3.2）。

    Args:
        config: :class:`KafkaConsumerConfig` with all Kafka consumer settings.
        processor: :class:`BenchmarkStreamingEvaluator` instance to handle messages.
        stop_event: optional :class:`asyncio.Event` that interrupts the
            holdback wait when the owning task instance stops (drain).

    Returns:
        Configured but not started consumer.  Call ``await consumer.start()``
        then ``await consumer.run()`` to begin consuming.
    """
    servers = [s.strip() for s in config.bootstrap_servers.split(",") if s.strip()]
    if not servers:
        raise ValueError("No bootstrap servers configured — set JUDGE_KAFKA_BOOTSTRAP_SERVERS")

    config_kwargs: dict[str, Any] = {
        "backend": BackendType.KAFKA,
        "bootstrap_servers": servers,
        "security_protocol": config.security_protocol,
        "client_id": config.client_id,
    }
    if config.sasl_mechanism and config.sasl_username and config.sasl_password:
        config_kwargs["sasl_mechanism"] = config.sasl_mechanism
        config_kwargs["sasl_plain_username"] = config.sasl_username
        config_kwargs["sasl_plain_password"] = config.sasl_password

    mq_config = Configuration(
        **config_kwargs,
        max_poll_interval_ms=config.max_poll_interval_ms,
        session_timeout_ms=config.session_timeout_ms,
        heartbeat_interval_ms=config.heartbeat_interval_ms,
    )
    backend = AioKafkaBackend(mq_config)

    backoff = DefaultBackoffPerformer(
        maximum_retries=config.max_retries if config.max_retries > 0 else 0,
        initial_backoff_ms=CONSUMER_BACKOFF_INITIAL_MS,
        maximum_backoff_ms=CONSUMER_BACKOFF_MAXIMUM_MS,
    )

    consumer = MQConsumer(
        backend=backend,
        group_id=config.group_id,
        poll_timeout_s=config.poll_timeout_s,
        auto_commit=config.auto_commit,
        backoff=backoff,
    )

    async def _early_commit_processor(
        msg: ConsumedMessage[RecordStreamMessage],
    ) -> None:
        # Ack first — judge latency / failure must not hold the offset.
        await consumer.commit_message(msg)
        # 120s message-timestamp holdback (interruptible by stop_event)。
        # 日志保留（§4 保真：消费延迟排障的关键日志）。
        delay_s = kafka_message_holdback_s(msg.timestamp_ms)
        if delay_s > 0:
            logger.info(
                "kafka consume holdback: delay_s=%.1f topic=%s partition=%d "
                "offset=%d timestamp_ms=%s",
                delay_s,
                msg.topic,
                msg.partition,
                msg.offset,
                msg.timestamp_ms,
            )
        await wait_kafka_message_holdback(msg.timestamp_ms, stop_event)
        try:
            await processor(msg)
        except Exception:
            logger.exception(
                "judge failed after offset commit (no kafka retry): "
                "topic=%s partition=%d offset=%d key=%s",
                msg.topic,
                msg.partition,
                msg.offset,
                msg.key,
            )

    for topic_name in config.topics:
        topic_def = TopicDefinition(
            name=topic_name,
            payload_type=RecordStreamMessage,
        )
        consumer.subscribe(topic_def, _early_commit_processor)

    # When max_retries > 0, register an error handler that commits the offset
    # so poison-pill messages are discarded instead of blocking the consumer.
    # (Judge failures are already swallowed after early commit; this mainly
    # covers commit_message / decode-adjacent handler failures.)
    if config.max_retries > 0 and config.auto_commit:

        async def _on_retry_exhausted(e: Exception) -> None:
            logger.error(
                "Message discarded after %d retries, committing offset: %s",
                config.max_retries,
                e,
            )
            await consumer.commit()

        consumer.on_error(_on_retry_exhausted)

    logger.info(
        "Consumer created: topics=%s group_id=%s brokers=%d "
        "security=%s auto_commit=%s early_commit=true max_retries=%s "
        "max_poll_interval_ms=%s session_timeout_ms=%s "
        "heartbeat_interval_ms=%s",
        config.topics,
        config.group_id,
        len(servers),
        config.security_protocol,
        config.auto_commit,
        config.max_retries if config.max_retries > 0 else "∞",
        config.max_poll_interval_ms,
        config.session_timeout_ms,
        config.heartbeat_interval_ms,
    )

    return consumer


__all__ = [
    "AGENT_NAME_METADATA_KEY",
    "KAFKA_MESSAGE_META_KEY",
    "BenchmarkStreamingEvaluator",
    "BenchmarkStreamingEvaluatorConfig",
    "BenchmarkStreamingEvaluatorDeps",
    "KafkaConsumerConfig",
    "TopicFilterConfig",
    "_UTC_PLUS_8",
    "_hour_key_cn",
    "configs_matching_trace_name",
    "create_streaming_consumer",
    "extract_kafka_message_meta",
    "kafka_message_holdback_s",
    "resolve_streaming_kafka_configs",
    "trace_to_observation",
    "wait_kafka_message_holdback",
]
