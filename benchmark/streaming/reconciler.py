"""StreamingTopicReconciler — RemoteConfig 声明配置 vs 运行态周期对账（§6）。

收敛模型（§6.1）：任何时刻系统目标 **实际运行态 ≡ RemoteConfig 声明态**；
reconciler 周期对账并执行 diff，全部操作幂等；进程重启后按 RemoteConfig
冷启动收敛一次。声明式（RemoteConfig）为唯一真相源；管理 API 只做内存级
加速与查询，不落持久态（§6.5）。

保护语义（§11）：

- **解析失败保持现状**：RemoteConfig 配置解析失败（JSON / schema）时不回退
  空集，保持上一次成功配置对应的运行态并 error 告警；
- **空集收敛二次确认**：``enabled=false`` 或 ``tasks=[]`` 时，仅
  ``allow_empty_tasks=true`` 显式配置才允许停掉全部实例，否则保持现状
  + error 告警；
- **env 冷启动兜底**（§8）：RemoteConfig 从未发布过 streaming 配置（且无
  last-good）时，冷启动以 env ``JUDGE_KAFKA_TOPIC`` 静态任务兜底，
  声明式配置一旦可用即接管。
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.streaming_config import (
    StreamingConfig,
    StreamingTaskConfig,
    fetch_streaming_config,
)
from benchmark.runs.common.execution_classes import (
    ExecutionClassResolutionError,
    effective_streaming_attribution_execution,
    effective_streaming_judge_execution,
    resolve_streaming_attribution_class,
    resolve_streaming_judge_class,
    resolve_streaming_triage_class,
    validate_streaming_execution_combo,
)
from benchmark.runs.common.rubric_classes import (
    RubricClassResolutionError,
    resolve_rubric_class,
)
from benchmark.streaming.config import StreamingKafkaConfig
from benchmark.streaming.persistence import sync_streaming_tasks
from benchmark.streaming.registry import RunningTask
from benchmark.streaming.tasks import (
    ENV_FALLBACK_TASK_NAME,
    StreamingTaskSpec,
    StreamingTaskState,
)

logger = get_logger("agent_eval.benchmark.streaming.reconciler")

#: 合法任务状态（§6.2）；domain 层宽容归一，reconciler 严格校验拒绝。
_TASK_STATES: tuple[StreamingTaskState, ...] = ("running", "paused", "stopped")

#: reconciler 对账周期默认值与下限（§5.2 ``reconcile_interval_s`` int ≥ 10，
#: 默认 30s；高频增删 topic 的 rebalance 风暴由单次对账内合并 diff 缓解）。
DEFAULT_RECONCILE_INTERVAL_S = 30
MIN_RECONCILE_INTERVAL_S = 10


class StreamingTopicReconciler:
    """声明式配置 vs 运行态的周期对账循环（§6）。

    Args:
        runner: :class:`StreamingBenchmarkRunner` 实例（任务实例原语提供者）。
        kafka_config: env 装载的 Kafka 运行参数（consumer 级默认 + 兜底 topics）。
        fetch_config: RemoteConfig 配置拉取函数（可注入以便测试）。
    """

    def __init__(
        self,
        *,
        runner: "StreamingBenchmarkRunnerProtocol",
        kafka_config: StreamingKafkaConfig,
        fetch_config: "FetchStreamingConfig | None" = None,
    ) -> None:
        self._runner = runner
        self._kafka_config = kafka_config
        self._fetch_config = fetch_config or fetch_streaming_config
        self._interval_s = DEFAULT_RECONCILE_INTERVAL_S
        self._last_reconcile_at: datetime | None = None

    @property
    def interval_s(self) -> int:
        """当前对账周期（随最近一次成功解析的 RemoteConfig 配置更新）。"""
        return self._interval_s

    @property
    def last_reconcile_at(self) -> datetime | None:
        """最近一次成功执行对账的时间（UTC；管理 API 双视图展示用）。"""
        return self._last_reconcile_at

    async def run(self) -> None:
        """周期对账循环；单轮异常不中断后续对账（保持现状）。"""
        while True:
            await asyncio.sleep(self._interval_s)
            try:
                await self.reconcile_once()
            except Exception:
                logger.exception("streaming reconcile iteration failed; keeping current state")

    async def reconcile_once(self) -> None:
        """执行一轮对账：fetch → 校验 → diff（全部幂等）。"""
        cfg = self._fetch_config()
        if cfg is None:
            # 无声明式配置（未发布 / 解析失败且无 last-good）：冷启动用 env
            # 兜底任务；已有任务在跑则保持现状（§8 / §11）。
            if not self._runner.registry.names():
                await self._start_env_fallback()
            return
        self._interval_s = max(cfg.reconcile_interval_s, MIN_RECONCILE_INTERVAL_S)
        desired = self._desired_specs(cfg)
        if desired is None:
            # 空集保护触发：保持现状 + error 已打。
            return
        # env 兜底任务与声明式配置互斥：声明式接管前先撤掉兜底。
        await self._retire_env_fallback(desired)
        await self._apply_diff(desired)
        self._last_reconcile_at = datetime.now(UTC)
        # 声明态镜像落库（benchmark_kafka_tasks；fail-soft，多节点 upsert 幂等）。
        await sync_streaming_tasks(desired, source="remote_config", now=self._last_reconcile_at)

    # ── env 兜底（§8）──────────────────────────────────────────────────

    async def _start_env_fallback(self) -> None:
        """冷启动 env 静态兜底任务（RemoteConfig 未配置时的 P1 形态）。"""
        spec = StreamingTaskSpec(
            name=ENV_FALLBACK_TASK_NAME,
            topics=list(self._kafka_config.topics),
            group_id=self._effective_group_id(None),
            state="running",
            shadow=self._kafka_config.shadow,
        )
        await self._runner.start_task(spec)
        logger.info(
            "streaming env fallback task started (RemoteConfig config absent): "
            "name=%s topics=%s group_id=%s",
            spec.name,
            spec.topics,
            spec.group_id,
        )
        # env 兜底任务同样落库（source=env 标记，供审计区分）。
        await sync_streaming_tasks({spec.name: spec}, source="env")

    async def _retire_env_fallback(self, desired: dict[str, StreamingTaskSpec]) -> None:
        """声明式配置可用后撤销 env 兜底任务（同名声明任务除外）。"""
        if ENV_FALLBACK_TASK_NAME in desired:
            return
        if self._runner.registry.get(ENV_FALLBACK_TASK_NAME) is not None:
            await self._runner.stop_task(ENV_FALLBACK_TASK_NAME, remove=True)
            logger.info("streaming env fallback task retired (RemoteConfig config active)")

    # ── 声明态校验（§5.2）───────────────────────────────────────────────

    def _desired_specs(self, cfg: StreamingConfig) -> dict[str, StreamingTaskSpec] | None:
        """把 RemoteConfig 配置归一为任务 spec 集；返回 ``None`` = 保持现状。

        空集保护（§11）：``enabled=false`` 或 ``tasks=[]`` 时需
        ``allow_empty_tasks=true`` 二次确认，否则保持现状 + error 告警。
        条目级非法（校验失败）只跳过该条目并 error，不影响其他任务。
        """
        if not cfg.enabled or not cfg.tasks:
            if cfg.allow_empty_tasks:
                return {}
            logger.error(
                "streaming config converges to empty task set but "
                "allow_empty_tasks is false (enabled=%s tasks=%d); "
                "keeping current state",
                cfg.enabled,
                len(cfg.tasks),
            )
            return None

        specs: dict[str, StreamingTaskSpec] = {}
        null_group_owner: str | None = None
        running_topic_groups: set[tuple[str, str]] = set()
        for entry in cfg.tasks:
            spec = self._validate_task(
                entry,
                null_group_owner=null_group_owner,
                running_topic_groups=running_topic_groups,
            )
            if spec is None:
                continue
            if entry.group_id is None:
                null_group_owner = spec.name
            if spec.state == "running":
                for topic in spec.topics:
                    running_topic_groups.add((spec.group_id, topic))
            if spec.name in specs:
                logger.error(
                    "duplicate streaming task name %r; keeping first, skipping later",
                    spec.name,
                )
                continue
            specs[spec.name] = spec
        return specs

    def _validate_task(
        self,
        entry: StreamingTaskConfig,
        *,
        null_group_owner: str | None,
        running_topic_groups: set[tuple[str, str]],
    ) -> StreamingTaskSpec | None:
        """单条目校验（§5.2 fail-fast）：非法返回 ``None`` 并 error 日志。"""
        name = entry.name.strip()
        if not name:
            logger.error("streaming task with empty name skipped")
            return None
        topics = [t.strip() for t in entry.topics if t.strip()]
        if not topics:
            logger.error("streaming task %r has empty topics, skipped", name)
            return None
        state = entry.state.strip().lower()
        if state not in _TASK_STATES:
            logger.error(
                "streaming task %r state %r invalid (expected one of %s), skipped",
                name,
                entry.state,
                _TASK_STATES,
            )
            return None
        if entry.group_id is None and null_group_owner is not None:
            logger.error(
                "streaming task %r uses null group_id but task %r already does "
                "(at most one migration task may inherit JUDGE_KAFKA_GROUP_ID), skipped",
                name,
                null_group_owner,
            )
            return None
        group_id = self._effective_group_id(entry.group_id)
        if entry.state == "running":
            conflicts = sorted(
                f"{gid}/{topic}"
                for gid, topic in running_topic_groups
                if gid == group_id and topic in topics
            )
            if conflicts:
                logger.error(
                    "streaming task %r subscribes topics already consumed by "
                    "another running task in group %r: %s, skipped",
                    name,
                    group_id,
                    conflicts,
                )
                return None
        if entry.sampling is not None and not (0.0 < entry.sampling <= 1.0):
            logger.error(
                "streaming task %r sampling %r out of (0, 1], skipped "
                "(use state=paused instead of sampling=0)",
                name,
                entry.sampling,
            )
            return None
        try:
            judge_class_execution = None
            if entry.judge_class:
                judge_class_execution = resolve_streaming_judge_class(entry.judge_class).execution
            if entry.triage_class:
                resolve_streaming_triage_class(entry.triage_class)
            attr_class_execution = None
            if entry.attribution_class:
                attr_class_execution = resolve_streaming_attribution_class(
                    entry.attribution_class,
                ).execution
            if entry.rubric_class:
                resolve_rubric_class(entry.rubric_class)
            judge_execution = effective_streaming_judge_execution(
                class_execution=judge_class_execution,
                override=entry.judge_execution,
            )
            attribution_execution = effective_streaming_attribution_execution(
                class_execution=attr_class_execution,
                override=entry.attribution_execution,
                attribution_class=entry.attribution_class,
            )
            validate_streaming_execution_combo(
                judge_execution=judge_execution,
                attribution_execution=attribution_execution,
                judge_class=entry.judge_class,
                attribution_class=entry.attribution_class,
            )
        except (ExecutionClassResolutionError, RubricClassResolutionError) as exc:
            logger.error("streaming task %r class alias invalid: %s, skipped", name, exc)
            return None
        return StreamingTaskSpec(
            name=name,
            topics=topics,
            group_id=group_id,
            state=cast(StreamingTaskState, state),
            judge_class=entry.judge_class,
            judge_execution=judge_execution,
            triage_class=entry.triage_class,
            attribution_class=entry.attribution_class,
            attribution_execution=attribution_execution,
            rubric_class=entry.rubric_class,
            sampling=entry.sampling,
            daily_report_enabled=entry.daily_report.enabled,
            shadow=self._kafka_config.shadow,
        )

    def _effective_group_id(self, group_id: str | None) -> str:
        """解析任务消费组：``null`` → ``JUDGE_KAFKA_GROUP_ID`` 同值接管；
        shadow 模式强制独立 group（``<group>-shadow``，P2 / §11）。"""
        base = (group_id or "").strip() or self._kafka_config.consumer.group_id
        if self._kafka_config.shadow:
            return f"{base}{self._kafka_config.shadow_group_suffix}"
        return base

    # ── diff 执行（§6.2 / §6.3）────────────────────────────────────────

    async def _apply_diff(self, desired: dict[str, StreamingTaskSpec]) -> None:
        """声明态 vs 运行态 diff：删除 / 新增 / 状态迁移 / topics 元素级增删。"""
        current = {task.spec.name: task for task in self._runner.registry.list()}
        for name in sorted(set(current) - set(desired)):
            # 删除 = 条目移除：实例 stop + 注册移除（terminal）。
            await self._runner.stop_task(name, remove=True)
            logger.info("streaming task deleted: name=%s", name)
        for name, spec in desired.items():
            await self._apply_task(current.get(name), spec)

    async def _apply_task(
        self,
        task: RunningTask | None,
        spec: StreamingTaskSpec,
    ) -> None:
        """单任务收敛：新增 / 暂停 / 停止 / 恢复 / topics diff / 执行参数重建。"""
        if task is None:
            if spec.state == "running":
                await self._runner.start_task(spec)
            else:
                # 新条目即 paused / stopped：登记意向但不启动实例。
                self._runner.registry.upsert(
                    RunningTask(
                        spec=spec,
                        evaluator=self._runner.build_evaluator(spec),
                        runtime_state=spec.state,
                    ),
                )
            return

        old = task.spec
        if old == spec and task.runtime_state == spec.state:
            return  # 无变更

        if spec.state != "running":
            # running → paused / stopped：实例 stop（drain）→ 注册标记意向。
            await self._runner.stop_task(spec.name, remove=False, mark=spec)
            self._runner.registry.mark_state(spec.name, spec.state)
            logger.info(
                "streaming task %s: name=%s",
                "paused" if spec.state == "paused" else "stopped",
                spec.name,
            )
            return

        if not task.is_running:
            # paused / stopped → running：按原配置重建实例组，offset 接续。
            await self._runner.start_task(spec)
            return

        non_topic_exec_changed = (
            old.group_id != spec.group_id
            or old.judge_class != spec.judge_class
            or old.judge_execution != spec.judge_execution
            or old.triage_class != spec.triage_class
            or old.attribution_class != spec.attribution_class
            or old.attribution_execution != spec.attribution_execution
            or old.sampling != spec.sampling
            or old.shadow != spec.shadow
        )
        if non_topic_exec_changed:
            # 执行参数变更（group / judge / 归因 / 采样 / shadow）→ 重建实例组。
            await self._runner.stop_task(spec.name, remove=True)
            await self._runner.start_task(spec)
            return

        if old.topics != spec.topics:
            # topic 粒度 diff（§6.3）：元素级增删，未变更 topic 实例不重建。
            added = [t for t in spec.topics if t not in old.topics]
            removed = [t for t in old.topics if t not in spec.topics]
            await self._runner.remove_topics(spec.name, removed)
            await self._runner.add_topics(spec.name, added)
        # rubric_class / daily_report 等声明态字段变更运行无感，仅同步 spec。
        task.spec = spec


# ── 类型别名（避免运行时循环 import）─────────────────────────────────────

if TYPE_CHECKING:
    from benchmark.streaming.consumer import BenchmarkStreamingEvaluator
    from benchmark.streaming.registry import StreamingTaskRegistry


class StreamingBenchmarkRunnerProtocol(Protocol):
    """reconciler 依赖的 runner 原语（结构型，避免循环 import）。"""

    @property
    def registry(self) -> "StreamingTaskRegistry": ...

    async def start_task(self, spec: StreamingTaskSpec) -> RunningTask: ...

    async def stop_task(
        self,
        name: str,
        *,
        timeout: float = ...,
        remove: bool = ...,
        mark: StreamingTaskSpec | None = ...,
    ) -> RunningTask | None: ...

    async def add_topics(self, name: str, topics: list[str]) -> None: ...

    async def remove_topics(
        self,
        name: str,
        topics: list[str],
        *,
        timeout: float = ...,
    ) -> None: ...

    def build_evaluator(self, spec: StreamingTaskSpec) -> "BenchmarkStreamingEvaluator": ...


FetchStreamingConfig = Callable[[], "StreamingConfig | None"]


__all__ = [
    "DEFAULT_RECONCILE_INTERVAL_S",
    "MIN_RECONCILE_INTERVAL_S",
    "StreamingTopicReconciler",
]
