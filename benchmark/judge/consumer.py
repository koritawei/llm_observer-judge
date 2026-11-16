"""Trace record consumer — consumes ``langfuse.records`` and runs B2→B6 pipeline.

Consumes ``langfuse.records`` topic via ``libs/message_queue`` ``MQConsumer``
+ ``AioKafkaBackend``.  Only ``record_type=trace`` messages are processed;
``observation`` and ``score`` messages are silently skipped.

The trace record is converted to :class:`ObservationForEval`,
then the existing B2→B6 pipeline runs:

    B2: Config lookup (RemoteConfig, by project_id)
    B3: Sampling (SHA256, deterministic, keyed by trace_id for idempotency)
    B4: Data already in memory (no S3 download needed)
    B5: Evaluate (LLM judge, multi-config)
    B6: after_eval writes TraceRecord (job_execution status update skipped)

Architecture:

    langfuse.records topic
      → MQConsumer.run() (libs/message_queue, AIOKafkaConsumer)
      → JsonCodec.decode() → RecordStreamMessage
      → early commit_message (offset ack before judge)
      → 120s Kafka timestamp holdback (blocks next poll)
      → TraceEvaluator.__call__()
          → filter record_type == "trace"
          → trace_to_observation() → ObservationForEval
          → B2→B6 pipeline (failures logged; no Kafka retry)
"""

import asyncio
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
from benchmark.domain.agent_config import get_agent_name_by_project_id
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.domain.trace_filters import (
    matches_trace_filters as _matches_trace_filters,
)
from benchmark.domain.trace_filters import (
    should_sample_deterministic as _should_sample_deterministic,
)
from benchmark.domain.trace_filters import (
    should_skip_test_adiu as _should_skip_test_adiu,
)
from benchmark.judge.message_holdback import (
    kafka_message_holdback_s,
    wait_kafka_message_holdback,
)
from benchmark.judge.metadata import (
    AGENT_NAME_METADATA_KEY,
    is_discard_agent_name,
    metadata_agent_name,
    observation_with_agent_name,
    set_metadata_agent_name,
)
from benchmark.judge.schemas import (
    ObservationForEval,
    RecordStreamMessage,
    TraceRecord,
)
from benchmark.streaming.consumer import resolve_streaming_kafka_configs
from benchmark.utils.exceptions import EvaluatorError

logger = get_logger("agent_eval.judge.consumer")

#: 漏斗对账时区（东八区整点）；与业务落库 stamp 一致。
_UTC_PLUS_8 = timezone(timedelta(hours=8))


# ---------------------------------------------------------------------------
# Hourly funnel counters (filter / sampling stages; eval 侧自行观测)
# ---------------------------------------------------------------------------


@dataclass
class _JudgeFunnelCounters:
    """进程内按小时累计的 Judge 漏斗计数。

    对账（消息粒度）::

        msg_total = skip_record_type + skip_invalid + skip_deleted + received
        received = dedup_skip + filter_* + sample_miss + sample_hit
        eligible = sample_miss + sample_hit

    ``eval_*`` 按 **config 调用** 计（一条 sample_hit 可对应多次 evaluate）。
    """

    msg_total: int = 0
    skip_record_type: int = 0
    skip_invalid: int = 0
    skip_deleted: int = 0
    received: int = 0
    dedup_skip: int = 0
    filter_agent_name: int = 0
    filter_no_config: int = 0
    filter_env: int = 0
    filter_attrs: int = 0
    filter_empty_io: int = 0
    filter_test_adiu: int = 0
    eligible: int = 0
    sample_miss: int = 0
    sample_hit: int = 0
    eval_ok: int = 0
    eval_fail_timeout: int = 0
    eval_fail_permanent: int = 0
    eval_fail_transient: int = 0


def _hour_key_cn(now: datetime | None = None) -> str:
    """Return ``YYYY-MM-DDTHH+08:00`` for funnel hour buckets."""
    dt = now or datetime.now(_UTC_PLUS_8)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC_PLUS_8)
    else:
        dt = dt.astimezone(_UTC_PLUS_8)
    return dt.strftime("%Y-%m-%dT%H+08:00")


# ---------------------------------------------------------------------------
# Protocols for injectable dependencies
# ---------------------------------------------------------------------------


class EvaluatorServiceProtocol(Protocol):
    async def evaluate(self, evaluator_name: str, request: Any) -> Any: ...


class ConfigStoreProtocol(Protocol):
    def get_configs(self, project_id: str) -> list[EvalConfigEntry]: ...

    def get_config_file(self) -> object: ...


class RequestConverter(Protocol):
    """Converts observation + config → EvaluatorRequest."""

    def __call__(self, observation: Any, config: EvalConfigEntry) -> Any: ...


# ---------------------------------------------------------------------------
def _is_empty_io(value: object) -> bool:
    """True when Kafka trace ``input`` / ``output`` has no evaluable payload.

    Empty: ``None``, blank string, ``{}``, ``[]``. Trace ``timestamp`` is not IO.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple)):
        return len(value) == 0
    return False


# ---------------------------------------------------------------------------
# Kafka transport metadata → downstream request / trace_meta
# ---------------------------------------------------------------------------
# Agent identity helpers live in :mod:`benchmark.judge.metadata` (shared with service).


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
# TraceEvaluator — B2→B6 pipeline for trace messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceEvaluatorDeps:
    """Injectable dependencies for :class:`TraceEvaluator`.

    Groups the three required collaborators into a single value object to
    keep ``TraceEvaluator.__init__`` parameter count within the governance
    threshold (FUNC-PARAMS-001, 安隐硬规).
    """

    config_store: ConfigStoreProtocol
    evaluator_service: EvaluatorServiceProtocol
    request_converter: RequestConverter


@dataclass(frozen=True)
class TraceEvaluatorConfig:
    """Optional behaviour knobs for :class:`TraceEvaluator`.

    Defaults preserve the historical behaviour: no evaluate timeout, no
    rate limiting, default per-topic filter (``record_types={"trace"}``,
    ``skip_deleted=True``).
    """

    evaluate_timeout_s: int = 0
    max_msgs_per_min: float = 0.0
    topic_filter_rules: dict[str, TopicFilterConfig] | None = None


class TraceEvaluator:
    """Process trace messages through the B2→B6 pipeline.

    Key design points:

    - Input is ``ConsumedMessage[RecordStreamMessage]``
    - Only ``record_type=trace`` messages are processed
    - Deleted traces (``is_deleted=1``) are silently skipped
    - B4 (S3 download) is skipped — data is already in the message
    - B6 (job_execution status update) is skipped — no jobExecutionId
    - All evaluate errors (timeout, 429, network) are logged by the Kafka
      early-commit wrapper in :func:`create_consumer` (offset already
      committed); :class:`TraceEvaluator` itself may still raise so unit
      tests can assert the three-tier strategy when called directly
    - Sampling uses ``trace_id`` for hash, ensuring idempotent evaluation
    - ``after_eval`` still writes TraceRecord via ``EvaluatorService``

    Usage::

        evaluator = TraceEvaluator(
            deps=TraceEvaluatorDeps(
                config_store=store,
                evaluator_service=svc,
                request_converter=to_evaluator_request,
            ),
        )
        consumer = create_consumer(..., processor=evaluator)
        await consumer.start()
        await consumer.run()
    """

    def __init__(
        self,
        *,
        deps: TraceEvaluatorDeps,
        config: TraceEvaluatorConfig | None = None,
    ) -> None:
        self._config_store = deps.config_store
        self._evaluator_service = deps.evaluator_service
        self._request_converter = deps.request_converter
        cfg = config or TraceEvaluatorConfig()
        self._evaluate_timeout_s = cfg.evaluate_timeout_s
        self._rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(max_rate=cfg.max_msgs_per_min, time_period=RATE_LIMITER_TIME_PERIOD_S)
            if cfg.max_msgs_per_min > 0
            else None
        )
        self._topic_filter_rules = cfg.topic_filter_rules or {}
        self._default_filter = TopicFilterConfig()
        # In-process trace_id dedup: Langfuse publishes multiple trace records
        # for the same trace_id throughout its lifecycle (create → update →
        # complete).  Without dedup the same trace is sampled and evaluated
        # multiple times.  Shared across all MQConsumer instances in this
        # process (JudgeService creates a single TraceEvaluator for all).
        #
        # maxsize is intentionally generous (1 000 000) — TTL is the real
        # eviction policy.  LRU only kicks in if we somehow accumulate >1M
        # unique trace_ids within the TTL window, at which point we'd have a
        # much bigger problem than dedup accuracy.
        self._seen_trace_ids: TTLCache[str, None] = TTLCache(
            maxsize=1_000_000, ttl=TRACE_DEDUP_TTL_S
        )
        #: 东八区整点漏斗计数；跨小时滚动打 WARNING。
        #: threading.Lock：多 MQConsumer 共享同一 TraceEvaluator 时保护计数/换桶。
        self._funnel = _JudgeFunnelCounters()
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
            self._funnel = _JudgeFunnelCounters()
            self._funnel_hour = hour
        self._emit_funnel(prev_hour, prev)

    def _emit_funnel(self, hour: str, counters: _JudgeFunnelCounters) -> None:
        """Emit one WARNING line for ops hour-bucket reconciliation.

        线上 INFO 常被过滤看不到，故用 WARNING 保证漏斗对账可见。
        """
        c = counters
        logger.warning(
            "Judge funnel: hour=%s msg_total=%d skip_record_type=%d "
            "skip_invalid=%d skip_deleted=%d received=%d dedup_skip=%d "
            "filter_agent_name=%d filter_no_config=%d filter_env=%d "
            "filter_attrs=%d filter_empty_io=%d filter_test_adiu=%d "
            "eligible=%d sample_miss=%d "
            "sample_hit=%d "
            "eval_ok=%d eval_fail_timeout=%d eval_fail_permanent=%d "
            "eval_fail_transient=%d",
            hour,
            c.msg_total,
            c.skip_record_type,
            c.skip_invalid,
            c.skip_deleted,
            c.received,
            c.dedup_skip,
            c.filter_agent_name,
            c.filter_no_config,
            c.filter_env,
            c.filter_attrs,
            c.filter_empty_io,
            c.filter_test_adiu,
            c.eligible,
            c.sample_miss,
            c.sample_hit,
            c.eval_ok,
            c.eval_fail_timeout,
            c.eval_fail_permanent,
            c.eval_fail_transient,
        )

    def flush_funnel(self) -> None:
        """Force-emit current hour bucket (tests / graceful shutdown)."""
        with self._funnel_lock:
            hour = self._funnel_hour
            prev = self._funnel
            self._funnel = _JudgeFunnelCounters()
        self._emit_funnel(hour, prev)

    async def __call__(self, msg: ConsumedMessage[RecordStreamMessage]) -> None:
        """Process a single message — B2→B6 pipeline.

        Raises on transient failures (LLM 429, network) when invoked
        directly.  The Kafka path wraps this handler with early
        ``commit_message`` and swallows post-commit errors so judge
        outcomes never block offset progress.
        """
        # 跨整点先刷上一小时漏斗（无流量时延后到下一条消息）。
        self._roll_funnel_hour()

        payload = msg.payload
        self._bump_funnel("msg_total")

        # -- Topic-specific filter: record_type ----------------------------
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

        # -- Rate limiting (aiolimiter) -----------------------------------
        # aiolimiter.acquire() internally schedules token release via
        # loop.call_later(); no explicit release() call needed.
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

        observation = trace_to_observation(trace)
        trace_id = observation.trace_id
        project_id = observation.project_id

        # -- B1.5: Resolve agent_name from RemoteConfig agent registry ----------
        # When the upstream Agent SDK doesn't populate ``metadata.agent_name``,
        # look it up from the RemoteConfig ``config.agent.registry`` by ``project_id``.
        # This prevents ``metadata_missing: agent_name`` permanent errors for
        # traces that have a valid project_id but no agent_name in metadata.
        # Always stamp via model_copy so Observation.metadata updates stick.
        agent_raw = metadata_agent_name(observation.metadata)
        if is_discard_agent_name(agent_raw):
            resolved = get_agent_name_by_project_id(project_id)
            if resolved and not is_discard_agent_name(resolved):
                observation = observation_with_agent_name(observation, resolved)
                agent_raw = resolved
                logger.info(
                    "agent_name resolved from registry: trace_id=%s project_id=%s agent_name=%s",
                    trace_id,
                    project_id,
                    resolved,
                )
        elif isinstance(agent_raw, str) and agent_raw.strip():
            # Normalize legacy ``agentName`` → ``agent_name`` on a fresh copy.
            observation = observation_with_agent_name(observation, agent_raw)

        # Log metadata keys + agent_name for diagnostics: the most common
        # B5 permanent failure is metadata_missing: agent_name.
        meta_keys = sorted(observation.metadata.keys()) if observation.metadata else []
        agent_name_val = metadata_agent_name(observation.metadata)
        logger.info(
            "Trace received: trace_id=%s project_id=%s name=%s "
            "key=%s offset=%d environment=%s "
            "metadata_keys=%s agent_name=%s",
            trace_id,
            project_id,
            observation.name,
            msg.key,
            msg.offset,
            observation.environment,
            meta_keys,
            repr(agent_name_val) if agent_name_val else "<missing>",
        )

        # -- B1.6: Discard unknown / missing agent_name --------------------
        # Upstream may omit agent_name or set the sentinel "unknown"; empty
        # names also surface as workflow_name=agent-eval.unknown via
        # build_eval_trace_name.  Skip before RemoteConfig eval-config lookup /
        # LLM judge to avoid burning quota on unattributable traces.
        if is_discard_agent_name(agent_name_val):
            self._bump_funnel("filter_agent_name")
            logger.info(
                "Skipping trace: agent_name unknown/missing: "
                "trace_id=%s project_id=%s agent_name=%s",
                trace_id,
                project_id,
                repr(agent_name_val) if agent_name_val else "<missing>",
            )
            return

        # -- B2: Local config lookup（与 streaming 同口径：name 优先 + 强制替换）
        # sidecar ingest entries skip Kafka sampling; they are extracted by
        # Path C only. name 命中后丢弃 Kafka project_id，改用登记侧。
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
                logger.info(
                    "projectId not configured and no name match, skipping: "
                    "trace_id=%s project_id=%s trace_name=%s",
                    trace_id,
                    project_id,
                    trace.name,
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

        logger.info(
            "Configs found: source=%s project_id=%s count=%d "
            "trace_id=%s trace_name=%s kafka_project_id=%s",
            config_source,
            project_id,
            len(configs),
            trace_id,
            trace.name,
            kafka_project_id,
        )

        # -- B2b: Environment filter ----------------------------------------
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

        # -- B2c: Trace attribute filters (name / tags / timestamp) --------
        # Each config may optionally restrict which traces it applies to
        # by trace name, tags, or timestamp range.  Empty/None = pass all.
        matched_configs = [cfg for cfg in configs if _matches_trace_filters(cfg, trace)]
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
        configs = matched_configs

        # -- B2d: Incomplete IO (Langfuse create / mid-update) --------------
        # Create events often have agent_name + timestamp but empty input/output.
        # Occupying the dedup slot here would skip the later complete record.
        has_input = not _is_empty_io(trace.input)
        has_output = not _is_empty_io(trace.output)
        if not has_input or not has_output:
            self._bump_funnel("filter_empty_io")
            logger.info(
                "Skipping incomplete trace (empty input/output): "
                "trace_id=%s project_id=%s has_input=%s has_output=%s timestamp=%s",
                trace_id,
                project_id,
                has_input,
                has_output,
                trace.timestamp,
            )
            return

        # -- B2e: Test adiu (suffix ``_test``) -----------------------------
        if _should_skip_test_adiu(user_id=observation.user_id, input_value=observation.input):
            self._bump_funnel("filter_test_adiu")
            logger.info(
                "Skipping test adiu: trace_id=%s project_id=%s user_id=%s",
                trace_id,
                project_id,
                observation.user_id,
            )
            return

        # -- B1.4 (deferred): In-process trace dedup (LRU + TTL) ----------
        # Langfuse publishes create → update → complete for the same
        # trace_id.  Marking *before* filters would let an early incomplete
        # record (missing agent_name / empty IO / wrong name / wrong env)
        # burn the slot so a later complete record never reaches sampling.
        # Only records that pass B1.6–B2e are eligible to claim the key;
        # subsequent updates within TTL are skipped (sampling stays
        # idempotent).  Key is ``trace_id:project_id``.
        # Key 用解析后的 project_id（name 命中后已强制替换，不保留 Kafka 原值）。
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

        # -- B3: Independent sampling per config (deterministic SHA256) ----
        # Sampling key uses trace_id (not observation_id) for idempotency:
        # the same trace always produces the same sampling decision.
        sampled_configs: list[EvalConfigEntry] = []
        for cfg in configs:
            sample_key = f"{trace_id}:{cfg.evaluator_name}"
            if _should_sample_deterministic(cfg.sampling, sample_key):
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
        sampled_evaluators = [cfg.evaluator_name for cfg in sampled_configs]
        sampled_rates = {cfg.evaluator_name: cfg.sampling for cfg in sampled_configs}
        logger.info(
            "Sampling hit: trace_id=%s project_id=%s agent_name=%s name=%s "
            "environment=%s tags=%s user_id=%s session_id=%s "
            "evaluators=%s sampling_rates=%s "
            "topic=%s partition=%d offset=%d "
            "input=%s output=%s metadata=%s",
            trace_id,
            project_id,
            repr(agent_name_val) if agent_name_val else "<missing>",
            observation.name,
            observation.environment,
            list(observation.tags or []),
            observation.user_id,
            observation.session_id,
            sampled_evaluators,
            sampled_rates,
            msg.topic,
            msg.partition,
            msg.offset,
            _preview_for_log(observation.input),
            _preview_for_log(observation.output),
            _preview_for_log(observation.metadata),
        )

        # -- B5: Evaluate each sampled config independently ----------------
        # B4 (S3 download) is skipped — data is already in memory.
        #
        # Error strategy (domain-driven, three tiers):
        #   1. TimeoutError → permanent for this message: LLM is slow, retrying
        #      won't help.  Skip this config, continue to the next.
        #   2. EvaluatorError with code < 500 (metadata_missing,
        #      model_not_found, evaluator_prompt_not_found, trace_not_found)
        #      → permanent: the message is immutable, retrying won't
        #      add missing metadata or fix a config mismatch.  Skip this
        #      config, continue to the next.
        #   3. Other exceptions (429, 5xx LLM, network) → transient: raise to
        #      let MQConsumer backoff retry the entire message.
        for config in sampled_configs:
            request = self._request_converter(observation, config)
            # Carry Kafka transport metadata + force agent_name into
            # request.metadata (Judge LLM prompts / attribution often read
            # metadata, while converters may only set Request.agent_name).
            # EvaluatorRequest is frozen — always use model_copy.
            meta = dict(request.metadata) if request.metadata else {}
            meta[KAFKA_MESSAGE_META_KEY] = extract_kafka_message_meta(msg)
            top_agent = request.agent_name
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
                if self._evaluate_timeout_s > 0:
                    await asyncio.wait_for(
                        self._evaluator_service.evaluate(config.evaluator_name, request),
                        timeout=self._evaluate_timeout_s,
                    )
                else:
                    await self._evaluator_service.evaluate(config.evaluator_name, request)
            except TimeoutError:
                # Permanent for this message — LLM is slow, skip this config.
                self._bump_funnel("eval_fail_timeout")
                logger.warning(
                    "evaluate timeout for trace_id=%s evaluator_name=%s after %ds, skipping",
                    trace_id,
                    config.evaluator_name,
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
                        "evaluator_name=%s code=%s error_type=%s: %s, skipping",
                        trace_id,
                        config.evaluator_name,
                        exc.code,
                        exc.error_type,
                        exc.message,
                    )
                    continue  # Skip to next config
                # Transient error (500+): llm_failed, llm_unparseable,
                # evaluation_failed — let MQConsumer backoff retry.
                # Kafka early-commit path still swallows; count before raise.
                self._bump_funnel("eval_fail_transient")
                logger.warning(
                    "evaluate transient failure for trace_id=%s "
                    "evaluator_name=%s code=%s error_type=%s: %s",
                    trace_id,
                    config.evaluator_name,
                    exc.code,
                    exc.error_type,
                    exc.message,
                )
                raise  # Re-raise for backoff retry
            except Exception as exc:
                # Transient error (LLM 429/5xx, network) — let MQConsumer
                # backoff handle retry by re-raising.
                self._bump_funnel("eval_fail_transient")
                logger.warning(
                    "evaluate failed for trace_id=%s evaluator_name=%s: %s",
                    trace_id,
                    config.evaluator_name,
                    exc,
                )
                raise  # Re-raise for backoff retry

            self._bump_funnel("eval_ok")
            logger.info(
                "Config evaluated: trace_id=%s evaluator_name=%s",
                trace_id,
                config.evaluator_name,
            )

        # -- B6: job_execution status update skipped -----------------------
        # No jobExecutionId in trace messages.  Score is already persisted by
        # ``EvaluatorService.after_eval`` (writes to TraceRecord).


# ---------------------------------------------------------------------------
# KafkaConsumerConfig — grouped consumer parameters (安隐硬规: ≤7 params)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KafkaConsumerConfig:
    """Kafka consumer connection + behavior parameters.

    Groups all Kafka consumer settings into a single value object to keep
    :func:`create_consumer` parameter count ≤ 7 (安隐硬规).
    Built from ``settings.judge_kafka_*`` fields in :class:`JudgeService`.
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
# Factory — creates a configured MQConsumer
# ---------------------------------------------------------------------------


def create_consumer(
    *,
    config: KafkaConsumerConfig,
    processor: TraceEvaluator,
) -> MQConsumer:
    """Create a :class:`MQConsumer` configured for Langfuse trace consumption.

    Offset policy: as soon as a decoded message is handed to the processor
    wrapper, ``commit_message`` runs **before** judge evaluation.  If the
    Kafka record timestamp is younger than 120s, the handler waits out the
    remainder (blocking the next poll).  Judge failures are logged and
    swallowed — they must not block Kafka progress or trigger MQ backoff
    retry of the same offset.

    Args:
        config: :class:`KafkaConsumerConfig` with all Kafka consumer settings.
        processor: :class:`TraceEvaluator` instance to handle messages.

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
        await wait_kafka_message_holdback(
            msg.timestamp_ms,
            getattr(consumer, "_stop_event", None),
        )
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
    "KafkaConsumerConfig",
    "TopicFilterConfig",
    "TraceEvaluator",
    "TraceEvaluatorConfig",
    "TraceEvaluatorDeps",
    "create_consumer",
    "extract_kafka_message_meta",
    "trace_to_observation",
]
