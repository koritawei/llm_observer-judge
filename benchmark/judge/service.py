"""Judge Service — Kafka consumer for LLM-as-a-Judge evaluation.

This module provides:
    - :func:`to_evaluator_request` — observation → EvaluatorRequest conversion
    - :class:`JudgeService` — wires EvaluatorService, EvalConfigStore,
      and Kafka ``MQConsumer`` together.

The JudgeService is instantiated in ``benchmark.service.main`` lifespan and conditionally
starts Kafka consumers when ``JUDGE_SERVICE_ENABLED=true``.

.. versionchanged:: 0.5.0
    Replaced BullMQ ``llm-as-a-judge-execution-queue`` consumption with
    Kafka ``langfuse.records`` topic consumption.  Uses ``libs/message_queue``
    ``MQConsumer`` + ``AioKafkaBackend``.
"""

import asyncio
import json
import re
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.mq import MQConsumer

from benchmark.settings.setting import is_judge_service_enabled_runtime, settings
from benchmark.domain.eval_config import EvalConfigEntry, EvalConfigStore, VariableMapping
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.judge.consumer import (
    KafkaConsumerConfig,
    TopicFilterConfig,
    TraceEvaluator,
    TraceEvaluatorConfig,
    TraceEvaluatorDeps,
    create_consumer,
)
from benchmark.judge.metadata import (
    AGENT_NAME_METADATA_KEY,
    metadata_agent_name,
    set_metadata_agent_name,
)
from benchmark.judge.schemas import ObservationForEval
from benchmark.services.evaluation import EvaluatorService

logger = get_logger("agent_eval.judge.service")


# ---------------------------------------------------------------------------
# observation → EvaluatorRequest conversion
# ---------------------------------------------------------------------------


def to_evaluator_request(
    observation: ObservationForEval,
    config: EvalConfigEntry,
) -> EvaluatorRequest:
    """Convert S3 observation + eval config → :class:`EvaluatorRequest`.

    Field mapping (see architecture doc §6.2):

    +-------------------+-----------------------------------+
    | EvaluatorRequest  | Source                            |
    +===================+===================================+
    | ``input``         | ``observation.input``             |
    | ``output``        | ``observation.output``            |
    | ``trace_id``      | ``observation.trace_id``          |
    | ``trace_timestamp`` | ``observation.trace_timestamp``  |
    | ``agent_name``    | ``observation.metadata['agent_name']`` |
    |                   | (legacy ``agentName``; force-stamp)    |
    | ``project_id``    | ``observation.project_id``        |
    | ``metadata``      | ``observation.metadata`` (copy) + |
    |                   | forced ``agent_name``             |
    | ``variables``     | ``config.variable_mapping`` →     |
    |                   | 按 variableMapping 提取            |
    +-------------------+-----------------------------------+

    ``evaluator_name`` is **not** a field of ``EvaluatorRequest`` — it is
    passed as a separate argument to ``EvaluatorService.evaluate()`` and
    comes from ``config.evaluator_name``.
    """
    metadata = dict(observation.metadata) if observation.metadata else {}
    # Stamp kafka source for trace_meta_json filtering (Judge Kafka Path B).
    from benchmark.domain.enums import (  # noqa: PLC0415
        PIPELINE_SOURCE_METADATA_KEY,
        PipelineSource,
    )

    metadata[PIPELINE_SOURCE_METADATA_KEY] = PipelineSource.KAFKA.value
    # Normalize identity key to snake_case before judge / attribution.
    resolved_agent = metadata_agent_name(metadata)
    agent_name: str | None = None
    if isinstance(resolved_agent, str) and resolved_agent.strip():
        agent_name = resolved_agent.strip()
        set_metadata_agent_name(metadata, agent_name)

    # 对齐 Langfuse：按 variable_mapping 从 observation 提取数据填充 variables
    variables = _build_variables_from_mapping(observation, config.variable_mapping)

    request = EvaluatorRequest(
        input=observation.input,
        output=observation.output,
        metadata=metadata,
        variables=variables,
        trace_id=observation.trace_id,
        trace_timestamp=observation.trace_timestamp,
        agent_name=agent_name,
        project_id=observation.project_id,
        # Kafka consumer already has the Langfuse primary trace ID (record.id);
        # skip before_eval metadata resolution that would fail with trace_not_found.
        skip_trace_resolve=True,
        prompt_text=config.prompt,
        model_config_override=config.model,
    )

    # Force agent_name into metadata even when only present on Request top-level
    # (some converters / callers set agent_name without writing metadata).
    # EvaluatorRequest is frozen — use model_copy so the stamp is reliable.
    if isinstance(request.agent_name, str) and request.agent_name.strip():
        stamped_name = request.agent_name.strip()
        stamped_meta = dict(request.metadata) if request.metadata else {}
        if stamped_meta.get(AGENT_NAME_METADATA_KEY) != stamped_name:
            set_metadata_agent_name(stamped_meta, stamped_name)
            request = request.model_copy(
                update={"metadata": stamped_meta, "agent_name": stamped_name},
            )
        elif request.agent_name != stamped_name:
            request = request.model_copy(update={"agent_name": stamped_name})
    return request


# ---------------------------------------------------------------------------
# variable_mapping helpers — Langfuse-aligned observation field extraction
# ---------------------------------------------------------------------------


def _build_variables_from_mapping(
    observation: ObservationForEval,
    variable_mapping: list[VariableMapping],
) -> dict[str, Any]:
    """对齐 Langfuse ``variableMapping``：按映射列表从 observation/trace 提取数据。

    对每条 mapping：
      1. 按 ``selected_column_id`` 从 observation 取原始值；
      2. 若有 ``json_selector``（JSONPath），从原始值中提取子值；
      3. 以 ``template_variable`` 为 key 填入返回 dict。

    mapping 列表为空时返回空 dict（evaluator 回退到硬编码解析）。
    """
    variables: dict[str, Any] = {}
    for mapping in variable_mapping:
        value = _extract_field(observation, mapping.selected_column_id)
        if mapping.json_selector and value is not None:
            value = _apply_json_selector(value, mapping.json_selector)
        variables[mapping.template_variable] = value
    return variables


def _extract_field(observation: ObservationForEval, selected_column_id: str) -> Any:
    """按 ``selected_column_id`` 从 observation 提取字段值。

    支持的字段名对齐 Langfuse ``ObservationForEval`` schema：
    - observation 级别: input / output / metadata / tool_calls
    - trace 级别: trace_name / user_id / session_id / tags / name / type / environment
    """
    field_map = {
        "input": observation.input,
        "output": observation.output,
        "metadata": observation.metadata,
        "tool_calls": observation.tool_calls,
        "trace_name": observation.trace_name,
        "user_id": observation.user_id,
        "session_id": observation.session_id,
        "tags": observation.tags,
        "name": observation.name,
        "type": observation.type,
        "environment": observation.environment,
    }
    return field_map.get(selected_column_id)


def _apply_json_selector(value: Any, json_selector: str) -> Any:
    """简化 JSONPath 实现：支持 ``$.key.path[0].subkey`` 语法。

    对齐 Langfuse ``jsonSelector`` 语义——从复杂 JSON 中按路径提取子值。
    支持点号访问 (``.key``)、索引访问 (``[0]``)、通配符 (``[*]``)。
    不依赖第三方 JSONPath 库，仅覆盖 Langfuse 常见用法。
    """
    if not json_selector.startswith("$"):
        return value

    path = json_selector[1:]  # strip leading $
    current: Any = value
    tokens = re.findall(r"\.(\w+)|\[(\d+)\]|(\[\*\])", path)
    for key, idx, wildcard in tokens:
        if wildcard == "[*]":
            if not isinstance(current, list):
                return None
            # [*] — map remaining path over each element
            remaining_start = path.index("[*]") + 3
            remaining = path[remaining_start:]
            if remaining:
                return [_apply_json_selector(item, "$" + remaining) for item in current]
            return current
        elif key:
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key, None)
            else:
                return None
        elif idx:
            if isinstance(current, (list, tuple)):
                i = int(idx)
                current = current[i] if i < len(current) else None
            else:
                return None
        if current is None:
            return None
    return current


def _parse_topic_filter_rules(
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


# ---------------------------------------------------------------------------
# JudgeService — dependency wiring + lifecycle
# ---------------------------------------------------------------------------


class JudgeService:
    """Wires all Judge Service dependencies and manages Kafka consumer lifecycle.

    Usage in ``benchmark.service.main`` lifespan::

        judge = JudgeService(evaluator_service=svc)
        if settings.judge_service_enabled:
            await judge.start()
        # ... on shutdown:
        await judge.stop()
    """

    def __init__(
        self,
        *,
        evaluator_service: EvaluatorService,
        config_store: EvalConfigStore | None = None,
    ) -> None:
        self._evaluator_service = evaluator_service
        self._config_store = config_store or EvalConfigStore()
        self._consumers: list[MQConsumer] = []
        self._run_tasks: list[asyncio.Task[None]] = []
        self._trace_evaluator: TraceEvaluator | None = None

    async def start(self) -> None:
        """Create and start Kafka consumers for trace record topics.

        Spawns ``judge_kafka_concurrency`` :class:`MQConsumer` instances,
        each running its own poll loop as an asyncio task.  All consumers
        share the same ``group_id``; Kafka auto-assigns partitions across
        them.
        """
        if self._consumers:
            logger.warning("JudgeService already started; skipping")
            return

        if not settings.judge_kafka_bootstrap_servers:
            raise RuntimeError(
                "Kafka bootstrap servers not configured — set JUDGE_KAFKA_BOOTSTRAP_SERVERS"
            )

        topics = [t.strip() for t in settings.judge_kafka_topic.split(",") if t.strip()]
        if not topics:
            raise RuntimeError("No Kafka topic configured — set JUDGE_KAFKA_TOPIC")

        topic_filter_rules = _parse_topic_filter_rules(settings.judge_kafka_topic_filters)
        if topic_filter_rules:
            logger.info(
                "Topic filter rules loaded: %s",
                {
                    t: {"record_types": sorted(r.record_types), "skip_deleted": r.skip_deleted}
                    for t, r in topic_filter_rules.items()
                },
            )

        evaluator = TraceEvaluator(
            deps=TraceEvaluatorDeps(
                config_store=self._config_store,
                evaluator_service=self._evaluator_service,
                request_converter=to_evaluator_request,
            ),
            config=TraceEvaluatorConfig(
                evaluate_timeout_s=settings.judge_evaluate_timeout_s,
                max_msgs_per_min=settings.judge_kafka_max_msgs_per_min,
                topic_filter_rules=topic_filter_rules,
            ),
        )
        self._trace_evaluator = evaluator

        kafka_config = KafkaConsumerConfig(
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

        for _ in range(settings.judge_kafka_concurrency):
            consumer = create_consumer(
                config=kafka_config,
                processor=evaluator,
            )
            await consumer.start()
            self._consumers.append(consumer)

        for i, consumer in enumerate(self._consumers):
            task = asyncio.create_task(consumer.run(), name=f"judge-kafka-consumer-{i}")
            self._run_tasks.append(task)

        logger.info(
            "Kafka consumers started: topics=%s group_id=%s concurrency=%d",
            topics,
            settings.judge_kafka_group_id,
            len(self._consumers),
        )

    async def stop(self, timeout: float = 30.0) -> None:
        """Gracefully stop all Kafka consumers.

        Signals each consumer's run loop to stop, waits up to *timeout*
        seconds for in-flight messages to finish, then closes all backend
        connections.
        """
        if not self._consumers:
            return

        consumer_count = len(self._consumers)
        logger.info(
            "Stopping Kafka consumers (count=%d, timeout=%.1fs)...",
            consumer_count,
            timeout,
        )

        # Signal all consumers to stop their run loops
        for consumer in self._consumers:
            consumer.stop()

        # Wait for run tasks to complete (in-flight messages drain)
        if self._run_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._run_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.warning(
                    "Kafka consumers did not stop within %.1fs; cancelling",
                    timeout,
                )
                for task in self._run_tasks:
                    task.cancel()

        # Close all backend connections
        await asyncio.gather(
            *[c.close() for c in self._consumers],
            return_exceptions=True,
        )

        self._consumers = []
        self._run_tasks = []

        # 停服时刷出当前未满小时的漏斗，避免重启丢对账。
        if self._trace_evaluator is not None:
            self._trace_evaluator.flush_funnel()
            self._trace_evaluator = None

        logger.info("Kafka consumers stopped (count=%d)", consumer_count)

    @property
    def is_running(self) -> bool:
        """Whether any Kafka consumer is running."""
        return any(c.is_running for c in self._consumers)

    def pause(self) -> None:
        """Pause all consumers (stop consuming, finish in-flight)."""
        for c in self._consumers:
            c.stop()

    def resume(self) -> None:
        """Resume all consumers after pause.

        Only starts new run loops for consumers that are not already running,
        preventing duplicate tasks.
        """
        self._run_tasks = [t for t in self._run_tasks if not t.done()]
        for i, c in enumerate(self._consumers):
            if not c.is_running:
                task = asyncio.create_task(c.run(), name=f"judge-kafka-consumer-{i}")
                self._run_tasks.append(task)

    async def watch_switch(self, *, interval: int = 30) -> None:
        """Background task: periodically re-check JUDGE_SERVICE_ENABLED.

        When the switch flips to ``false`` the consumer is **paused** (in-flight
        messages drain, no new messages consumed). When it flips back to
        ``true`` the consumer is **resumed**.
        """
        was_enabled = True  # consumer was started, so initial state is enabled
        while True:
            await asyncio.sleep(interval)
            enabled = is_judge_service_enabled_runtime()

            if enabled and not was_enabled:
                self.resume()
                was_enabled = True
                logger.info("Judge Service resumed (JUDGE_SERVICE_ENABLED=true)")
            elif not enabled and was_enabled:
                self.pause()
                was_enabled = False
                logger.info("Judge Service paused (JUDGE_SERVICE_ENABLED=false)")


__all__ = ["JudgeService", "to_evaluator_request"]
