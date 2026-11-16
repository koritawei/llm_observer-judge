"""StreamingBenchmarkRunner — streaming 形态的生命周期与实例管理（取代 JudgeService）。

职责：

- ``to_evaluator_request`` — observation → EvaluatorRequest 转换
  （迁移自 ``benchmark.judge.service``，逐位保留）；
- 任务实例组管理：按 :class:`StreamingTaskSpec` 创建 / 停止 per-topic
  consumer 实例组（§6.3 topic 粒度 diff），登记进
  :class:`~benchmark.streaming.registry.StreamingTaskRegistry`；
- 实例 stop **drain** 语义（§3.2）：置 stop_event 中断 holdback →
  等待 in-flight 处理完成（上限 ``timeout``）→ close；超时取消的消息按
  现状语义丢失并计数（``stop_dropped``）；
- 启动 sidecar MetaQ 消费（``ingest=sidecar``）。Kafka reconciler 仅在
  ``kafka_enabled``（``BENCHMARK_STREAMING_ENABLED``）为真时挂载，默认不启动。
"""

import asyncio
from dataclasses import replace
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.mq import MQConsumer

from benchmark.settings.setting import Settings
from benchmark.domain.eval_config import EvalConfigStore
from benchmark.runs.common.execution_classes import (
    StreamingJudgeImpl,
    load_execution_instance,
    resolve_streaming_attribution_class,
    resolve_streaming_judge_class,
    resolve_streaming_triage_class,
)
from benchmark.streaming.config import (
    StreamingKafkaConfig,
    load_streaming_kafka_config,
)
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorConfig,
    BenchmarkStreamingEvaluatorDeps,
    create_streaming_consumer,
)
from benchmark.streaming.reconciler import StreamingTopicReconciler
from benchmark.streaming.registry import RunningTask, StreamingTaskRegistry
from benchmark.streaming.request import to_evaluator_request
from benchmark.streaming.tasks import StreamingTaskSpec
from benchmark.services.evaluation import EvaluatorService
from benchmark.ingest.sidecar.lifecycle import start_sidecar_ingest, stop_sidecar_ingest

logger = get_logger("agent_eval.benchmark.streaming.runner")

#: 实例 stop drain 默认超时（与现状 ``JudgeService.stop`` 60s 口径一致）。
DEFAULT_STOP_TIMEOUT_S = 60.0

# ---------------------------------------------------------------------------
# StreamingBenchmarkRunner — lifecycle + task instance management
# ---------------------------------------------------------------------------


class StreamingBenchmarkRunner:
    """Streaming 形态 runner：持有 sidecar 消费，评测走 benchmark judge。

    ``start`` 启动 ``ingest=sidecar`` 的 MetaQ 消费。Kafka 由
    ``kafka_enabled`` 决定，对应环境变量 ``BENCHMARK_STREAMING_ENABLED``，
    默认 false 不创建 consumer。
    """

    def __init__(
        self,
        *,
        evaluator_service: EvaluatorService,
        config_store: EvalConfigStore | None = None,
        kafka_config: StreamingKafkaConfig | None = None,
        settings: Settings | None = None,
        kafka_enabled: bool = False,
    ) -> None:
        self._evaluator_service = evaluator_service
        self._config_store = config_store or EvalConfigStore()
        self._kafka_config = kafka_config
        self._settings = settings
        self._registry = StreamingTaskRegistry()
        self._reconciler_task: asyncio.Task[None] | None = None
        self._reconciler: Any = None
        self._sidecar_service: Any = None
        # 默认关闭。main 传入 settings.benchmark_streaming_enabled。
        self._kafka_enabled = kafka_enabled

    @property
    def registry(self) -> StreamingTaskRegistry:
        """任务注册表（reconciler 对账 / 管理 API 双视图数据源）。"""
        return self._registry

    @property
    def kafka_config(self) -> StreamingKafkaConfig:
        """已装载的 Kafka 运行参数（``start`` 前调用会现场装载）。"""
        if self._kafka_config is None:
            self._kafka_config = load_streaming_kafka_config()
        return self._kafka_config

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def sidecar_service(self) -> Any:
        """当前 sidecar MetaQ 服务；未启动或已停止时为 ``None``。"""
        return self._sidecar_service

    async def start(self) -> None:
        """启动 sidecar。Kafka 仅在 ``kafka_enabled`` 时对账并创建 consumer。"""
        await self._start_sidecar()
        if not self._kafka_enabled:
            logger.info(
                "benchmark kafka consumption disabled; BENCHMARK_STREAMING_ENABLED=false",
            )
            return
        try:
            await self._start_kafka()
        except Exception:
            logger.exception("benchmark kafka consumption failed to start")

    async def _start_sidecar(self) -> None:
        """启动 ingest=sidecar 的 MetaQ 消费。无 settings 时跳过。"""
        if self._settings is None:
            logger.info("sidecar ingest not started: runner has no settings")
            return
        self._sidecar_service = await start_sidecar_ingest(
            benchmark_judge=self.build_sidecar_judge(),
            settings=self._settings,
        )

    async def _start_kafka(self) -> None:
        """冷启动对账并挂载 reconciler，由声明配置创建 Kafka consumer。"""
        kafka_config = self.kafka_config
        self._reconciler = StreamingTopicReconciler(
            runner=self,
            kafka_config=kafka_config,
        )
        await self._reconciler.reconcile_once()
        self._reconciler_task = asyncio.create_task(
            self._reconciler.run(),
            name="benchmark-streaming-reconciler",
        )
        logger.info(
            "Benchmark kafka consumption started: tasks=%s shadow=%s",
            self._registry.names(),
            kafka_config.shadow,
        )

    def build_sidecar_judge(self) -> BenchmarkStreamingEvaluator:
        """sidecar 命中后使用的 benchmark judge（无 Kafka 采样覆盖）。"""
        return BenchmarkStreamingEvaluator(
            deps=BenchmarkStreamingEvaluatorDeps(
                config_store=self._config_store,
                evaluator_service=self._evaluator_service,
                request_converter=to_evaluator_request,
            ),
            config=BenchmarkStreamingEvaluatorConfig(),
        )

    async def stop(self, timeout: float = DEFAULT_STOP_TIMEOUT_S) -> None:
        """停 sidecar，再停 Kafka reconciler 并 drain 已登记的任务实例。"""
        await stop_sidecar_ingest(self._sidecar_service)
        self._sidecar_service = None
        if self._reconciler_task is not None:
            self._reconciler_task.cancel()
            try:
                await self._reconciler_task
            except asyncio.CancelledError:
                pass
            self._reconciler_task = None
        for name in self._registry.names():
            await self.stop_task(name, timeout=timeout, remove=True)
        logger.info("Benchmark streaming stopped")

    @property
    def is_running(self) -> bool:
        """Whether any task instance is running."""
        return any(task.is_running for task in self._registry.list())

    # ── task instance primitives (reconciler / 管理 API 调用) ────────────

    async def start_task(self, spec: StreamingTaskSpec) -> RunningTask:
        """按 spec 创建任务实例组并登记（幂等：已 running 直接返回现状）。

        ``kafka_enabled`` 为 False 时拒绝，避免在开关关闭时创建 consumer。
        """
        if not self._kafka_enabled:
            raise RuntimeError(
                "kafka consumption is disabled (BENCHMARK_STREAMING_ENABLED=false); "
                f"refusing to start task {spec.name}",
            )
        existing = self._registry.get(spec.name)
        if existing is not None and existing.is_running:
            return existing
        evaluator = self.build_evaluator(spec)
        task = RunningTask(spec=spec, evaluator=evaluator, runtime_state="running")
        self._registry.upsert(task)
        for topic in spec.topics:
            await self._start_topic_instances(task, topic)
        logger.info(
            "streaming task started: name=%s topics=%s group_id=%s instances=%d shadow=%s",
            spec.name,
            spec.topics,
            spec.group_id,
            task.instance_count,
            spec.shadow,
        )
        return task

    async def stop_task(
        self,
        name: str,
        *,
        timeout: float = DEFAULT_STOP_TIMEOUT_S,
        remove: bool = True,
        mark: StreamingTaskSpec | None = None,
    ) -> RunningTask | None:
        """停止任务实例组（drain）；``remove=False`` 时保留注册并标记意向。

        ``mark`` 传入最新 spec 以同步注册表中的声明态（paused / stopped
        场景 spec 本身也要更新）。
        """
        task = self._registry.get(name)
        if task is None:
            return None
        await self._drain_task(task, timeout=timeout)
        if remove:
            self._registry.remove(name)
        else:
            task.spec = mark or task.spec
        return task

    async def pause_task(self, name: str, *, timeout: float = DEFAULT_STOP_TIMEOUT_S) -> bool:
        """内存级暂停：实例 stop（drain）→ 注册表标记 paused（§6.2）。

        committed offset 停在暂停点；恢复后从该点追平消费（暂停窗口数据
        会被补消费，与 fan-out 共享点位模型的「不补」语义不同）。
        """
        task = await self.stop_task(name, timeout=timeout, remove=False)
        if task is None:
            return False
        self._registry.mark_state(name, "paused")
        logger.info("streaming task paused: name=%s", name)
        return True

    async def resume_task(self, name: str) -> bool:
        """内存级恢复：按注册表中的 spec 重建实例组（committed offset 接续）。"""
        if not self._kafka_enabled:
            logger.info("kafka consumption disabled; not resuming task name=%s", name)
            return False
        task = self._registry.get(name)
        if task is None:
            return False
        await self.start_task(replace(task.spec, state="running"))
        self._registry.mark_state(name, "running")
        logger.info("streaming task resumed: name=%s", name)
        return True

    async def add_topics(self, name: str, topics: list[str]) -> None:
        """topic 粒度 diff（§6.3）：对 task 增补订阅实例（未变更不重建）。"""
        task = self._registry.get(name)
        if task is None or task.runtime_state != "running":
            return
        for topic in topics:
            if topic not in task.consumers:
                await self._start_topic_instances(task, topic)
        logger.info("streaming task topics added: name=%s topics=%s", name, topics)

    async def remove_topics(
        self,
        name: str,
        topics: list[str],
        *,
        timeout: float = DEFAULT_STOP_TIMEOUT_S,
    ) -> None:
        """topic 粒度 diff（§6.3）：对应实例 stop（drain）→ 订阅移除。"""
        task = self._registry.get(name)
        if task is None:
            return
        for topic in topics:
            await self._drain_topic_instances(task, topic, timeout=timeout)
        logger.info("streaming task topics removed: name=%s topics=%s", name, topics)

    # ── internals ────────────────────────────────────────────────────────

    def build_evaluator(self, spec: StreamingTaskSpec) -> BenchmarkStreamingEvaluator:
        """按任务 spec 构造消息处理器（sampling / triage / shadow / judge 插件生效）。

        公共方法：reconciler 登记非 running 新条目时需预构造 evaluator。
        """
        kafka_config = self.kafka_config
        triage_classifier = self._resolve_triage_classifier(spec)
        attribution_agent_key = self._resolve_attribution_agent_key(spec)
        return BenchmarkStreamingEvaluator(
            deps=BenchmarkStreamingEvaluatorDeps(
                config_store=self._config_store,
                evaluator_service=self._evaluator_service,
                request_converter=to_evaluator_request,
            ),
            config=BenchmarkStreamingEvaluatorConfig(
                evaluate_timeout_s=kafka_config.evaluate_timeout_s,
                max_msgs_per_min=kafka_config.max_msgs_per_min,
                topic_filter_rules=kafka_config.topic_filter_rules,
                sampling_override=spec.sampling,
                skip_triage=spec.no_attribution,
                shadow=spec.shadow,
                judge_impl=self._resolve_judge_impl(spec),
                judge_execution=spec.judge_execution,
                triage_classifier=triage_classifier,
                attribution_agent_key=attribution_agent_key,
                attribution_execution=spec.attribution_execution,
            ),
        )

    @staticmethod
    def _resolve_judge_impl(spec: StreamingTaskSpec) -> StreamingJudgeImpl | None:
        """按 ``tasks[].judge_class`` 解析业务专用 judge 插件（§5.1）。

        ``null`` / ``evaluator_service_judge`` → ``None``（现网 EvaluatorService
        零代码路由）；登记的业务类（实现 ``evaluate`` / ``submit``）→ 加载实例。
        """
        if not spec.judge_class:
            return None
        resolved = resolve_streaming_judge_class(spec.judge_class)
        if resolved.target is None:
            return None
        return load_execution_instance(resolved.class_ref, what="streaming judge_class")  # type: ignore[return-value]

    @staticmethod
    def _resolve_triage_classifier(spec: StreamingTaskSpec) -> object | None:
        """按 ``tasks[].triage_class`` 解析分类器实例；``null`` → 现网按名查表。"""
        if not spec.triage_class:
            return None
        resolved = resolve_streaming_triage_class(spec.triage_class)
        plugin = load_execution_instance(resolved.class_ref, what="streaming triage_class")
        build = getattr(plugin, "build_classifier", None)
        if not callable(build):
            return None
        return build()

    @staticmethod
    def _resolve_attribution_agent_key(spec: StreamingTaskSpec) -> str | None:
        """按 ``tasks[].attribution_class`` 解析 attribution agent_key 覆盖。"""
        if not spec.attribution_class or spec.attribution_class == "no_attribution":
            return None
        resolved = resolve_streaming_attribution_class(spec.attribution_class)
        if resolved.target is None:
            return None
        plugin = load_execution_instance(
            resolved.class_ref,
            what="streaming attribution_class",
        )
        key = getattr(plugin, "agent_key", None)
        return key if isinstance(key, str) and key else None

    async def _start_topic_instances(self, task: RunningTask, topic: str) -> None:
        """为单个 topic 创建 ``concurrency`` 个 consumer 实例并启动 run 任务。

        per-topic 独立实例组是 §6.3 元素级 diff 的基础；同 group 的订阅集
        变更由 broker 原生 rebalance 均摊分区，in-flight 重复由 upsert
        幂等吸收。
        """
        kafka_config = self.kafka_config
        stop_event = asyncio.Event()
        task.stop_events[topic] = stop_event
        instances: list[MQConsumer] = []
        for _ in range(kafka_config.concurrency):
            consumer = create_streaming_consumer(
                config=replace(
                    kafka_config.consumer,
                    topics=[topic],
                    group_id=task.spec.group_id,
                ),
                processor=task.evaluator,
                stop_event=stop_event,
            )
            await consumer.start()
            instances.append(consumer)
        task.consumers[topic] = instances
        task.run_tasks[topic] = [
            asyncio.create_task(
                consumer.run(),
                name=f"streaming-{task.spec.name}-{topic}-{i}",
            )
            for i, consumer in enumerate(instances)
        ]

    async def _drain_topic_instances(
        self,
        task: RunningTask,
        topic: str,
        *,
        timeout: float,
    ) -> None:
        """停止单 topic 实例组：stop_event 中断 holdback → drain → close。"""
        consumers = task.consumers.pop(topic, [])
        run_tasks = task.run_tasks.pop(topic, [])
        stop_event = task.stop_events.pop(topic, None)
        if stop_event is not None:
            stop_event.set()
        for consumer in consumers:
            consumer.stop()
        dropped = 0
        if run_tasks:
            # asyncio.wait 不取消 pending：超时后 pending 即「drain 未完成」集合。
            _, pending = await asyncio.wait(run_tasks, timeout=timeout)
            dropped = len(pending)
            if pending:
                logger.warning(
                    "streaming topic instances did not stop within %.1fs; cancelling: "
                    "task=%s topic=%s dropped=%d",
                    timeout,
                    task.spec.name,
                    topic,
                    dropped,
                )
                for run_task in pending:
                    run_task.cancel()
        await asyncio.gather(
            *[consumer.close() for consumer in consumers],
            return_exceptions=True,
        )
        # early-commit 语义下超时丢弃的消息不重放，仅计数（§3.2）。
        task.evaluator.note_stop_dropped(dropped)

    async def _drain_task(self, task: RunningTask, *, timeout: float) -> None:
        """drain 整个任务实例组（逐 topic）并刷出漏斗。"""
        for topic in list(task.consumers):
            await self._drain_topic_instances(task, topic, timeout=timeout)
        # 停服时刷出当前未满小时的漏斗，避免重启丢对账。
        task.evaluator.flush_funnel()


__all__ = [
    "DEFAULT_STOP_TIMEOUT_S",
    "StreamingBenchmarkRunner",
    "to_evaluator_request",
]
