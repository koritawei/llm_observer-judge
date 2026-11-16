"""TriageAttributionService — evaluator 跑完后的 attribution 协调。

``run`` 串行三步: before_hook → AgentManager.get(evaluator_name).call → after_hook。
切面可由业务方通过 :meth:`set_before` / :meth:`set_after` 替换。
"""

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from benchmark.libs.agenthost.agents.manager import AgentManager
from benchmark.libs.agenthost.agents.request import AgentRequest, AgentResponse
from benchmark.libs.agenthost.observability.logger import get_logger
from langfuse import Langfuse
from pydantic import ValidationError

from benchmark.records.enums import TriageLevel
from benchmark.domain.notification_dispatch_resolve import (
    AgentNotificationSnapshot,
    load_agent_notification_context,
    resolve_effective_notification_targets,
    resolve_notification_root_cause_groups,
)
from benchmark.domain.eval_config import persist_evaluator_name
from benchmark.domain.owner_mapping import OwnerMapping, resolve_owner_route
from benchmark.evaluators.prompt_stages import (
    attribution_bundle_refs,
    dump_ref,
    sanitize_evaluator_request_for_attribution,
)
from benchmark.services._stage_key import STAGE_ATTRIBUTION, stage_agent_key
from benchmark.attribution.badcase.publisher import NotificationConfig
from benchmark.notification.aggregate import publish_root_cause_groups
from benchmark.notification.registry import resolve_notification_publisher
from benchmark.attribution.badcase.dispatch_policy import (
    resolve_dispatch_policy,
    should_dispatch_notification,
)
from benchmark.attribution.badcase.schemas import (
    NotificationDispatchResult,
    NotificationDispatchStatus,
    AttributionResult,
)
from business.evaluators.example.grading import grade_fused_score
from benchmark.services.trace_record_service import (
    NotificationTaskEnqueue,
    AttributionUpdate,
    DispatchUpdate,
    TraceRecordKey,
    TraceRecordService,
)
from benchmark.trace.resolver import TraceResolver
from benchmark.trace.schema import ResolvedTrace
from benchmark.utils.trace_score import AttributionScoreWrite, write_attribution_score
from benchmark.utils.llm_json import parse_and_validate_llm_json

from .classifier import TriageClassifier
from .triage import TriageConfig, TriageDecision, triage_decide

logger = get_logger(__name__)

_TRIAGE_LEVEL_VALUES: frozenset[str] = frozenset(m.value for m in TriageLevel)


# ---------------------------------------------------------------------------
# 切面协议
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeforeRunInput:
    """前置切面输入（FUNC-PARAMS-001 参数收敛）。

    ``evaluator_request`` / ``evaluator_result`` 为 agentgraph 信封快照，
    多 evaluator 形态不一，协议边界处保持 ``Any``。
    ``classifier_override``：streaming ``tasks[].triage_class`` 覆盖默认
    按 ``evaluator_name`` 查表。
    """

    evaluator_name: str
    evaluator_request: Any
    evaluator_result: Any
    resolved_route: "Mapping[str, Any] | None" = None
    classifier_override: TriageClassifier | None = None


@dataclass(frozen=True)
class AfterRunInput:
    """后置切面输入（FUNC-PARAMS-001 参数收敛）。"""

    evaluator_name: str
    evaluator_request: Any
    attribution_response: AgentResponse
    attribution: AttributionResult | None
    resolved_route: "Mapping[str, Any] | None" = None
    #: 融合分。有值时经 :func:`grade_fused_score` 判级，仅 POOR ∪ FAIL 派单；
    #: 缺失则放行。分制由 ``score_scale_max`` / ``pass_threshold`` 决定；
    #: 二者皆空时分数 >2 按 0–10 推断（与 grade_fused_score 一致）。
    fused_score: float | None = None
    #: evaluator ``extra.sub_score_scale`` 上界（如 2 / 10）。
    score_scale_max: float | None = None
    #: 兼容尚无 ``score_scale_max`` 的 fused 流程（>2 视为 0–10 分制）。
    pass_threshold: float | None = None


@runtime_checkable
class BeforeRunHook(Protocol):
    """前置切面 — run 起步前调; 返回 :class:`BeforeRunDecision` 决定是否继续。"""

    async def __call__(
        self,
        input: BeforeRunInput,
    ) -> "BeforeRunDecision": ...


@runtime_checkable
class AfterRunHook(Protocol):
    """后置切面 — attribution agent 跑完调; 返回 :class:`TriageOutcome`。"""

    async def __call__(
        self,
        input: AfterRunInput,
    ) -> "TriageOutcome": ...


@dataclass(frozen=True)
class BeforeRunDecision:
    """前置切面的对外回报 — ``should_proceed=False`` 时 run 立即返回。"""

    should_proceed: bool
    reason: str | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# 内置切面实现
# ---------------------------------------------------------------------------


class TriageBeforeRunHook:
    """默认前置切面: 按 evaluator_name 查 classifiers 表,调 classifier.classify()。

    classifier 表由 :func:`benchmark.attribution.classifiers_registry.triage_classifiers`
    在 lifespan 启动时加载,传入本 hook。
    """

    def __init__(
        self,
        *,
        classifiers: dict[str, TriageClassifier],
        triage_config: "TriageConfig | None" = None,
    ) -> None:
        self._classifiers = classifiers
        self._config = triage_config

    async def __call__(
        self,
        input: BeforeRunInput,
    ) -> BeforeRunDecision:
        evaluator_name = input.evaluator_name
        from benchmark.runs.common.dynamic_rubric import (  # noqa: PLC0415
            resolve_dynamic_rubric,
            should_trigger_with_optional_threshold,
        )

        dynamic = resolve_dynamic_rubric(input.evaluator_result)
        if dynamic.attribution_threshold is not None:
            score = dynamic.fused_score
            if score is None:
                score = getattr(input.evaluator_result, "score", None)
            should_attribute = should_trigger_with_optional_threshold(
                fused_score=score,
                attribution_threshold=dynamic.attribution_threshold,
            )
            return BeforeRunDecision(
                should_proceed=bool(should_attribute),
                reason="dynamic_attribution_threshold",
                metadata={
                    "computed_fused_score": dynamic.fused_score,
                    "attribution_threshold": dynamic.attribution_threshold,
                    "dynamic_rubric_fallback_reason": dynamic.fallback_reason,
                },
            )
        classifier = input.classifier_override or self._classifiers.get(evaluator_name)
        if classifier is None:
            logger.info(
                "before_hook.no_classifier evaluator=%s",
                evaluator_name,
            )
            return BeforeRunDecision(
                should_proceed=False,
                reason=f"no_classifier_for_evaluator: {evaluator_name}",
            )

        decision = await triage_decide(
            evaluator_name=evaluator_name,
            evaluator_request=input.evaluator_request,
            evaluator_result=input.evaluator_result,
            classifier=classifier,
            config=self._config or TriageConfig(),
        )
        return BeforeRunDecision(
            should_proceed=decision.should_attribute,
            reason=decision.reason,
            metadata={
                "base_level": decision.base_level,
                "final_level": decision.final_level,
            },
        )
        # 注: should_proceed=False 只短路后续 attribution agent + aone 派发,
        #    不影响 metadata.final_level → triage_level 标量落库(见 TriageAttributionService.run)。


@dataclass(frozen=True)
class _DispatchSkipGate:
    """``_dispatch_skip_reason`` 入参（FUNC-PARAMS-001）。"""

    input: AfterRunInput
    attribution: AttributionResult
    snapshot: AgentNotificationSnapshot | None
    is_safety: bool


class NotificationAfterRunHook:
    """默认后置切面: 委托 :func:`create_aone_task`。

    安全类问题（safety_violation / medical_boundary_violation）单 case 即时出单，
    **不受**融合分等级门槛约束。profile 配了 ``root_cause_groups`` 时，非安全类
    先进入缓冲，由 :meth:`flush_groups` 按根因各发一条给对应人群。
    未配人群且 ``aggregate_non_safety`` 为真时，非安全类不出单。
    """

    #: 安全类 root_cause_code：单 case 即时出 Aone 工单，不等聚合。
    SAFETY_ROOT_CAUSE_CODES: frozenset[str] = frozenset(
        {
            "safety_violation",
            "medical_boundary_violation",
        }
    )

    def __init__(self, *, notification_config: NotificationConfig, aggregate_non_safety: bool = False) -> None:
        self._config = notification_config
        self._aggregate_non_safety = aggregate_non_safety
        self._pending: list[tuple[str, AttributionResult]] = []

    async def __call__(
        self,
        input: AfterRunInput,
    ) -> "TriageOutcome":
        evaluator_name = input.evaluator_name
        attribution = input.attribution
        if attribution is None:
            detail = _response_failure_detail(input.attribution_response)
            logger.warning(
                "after_run.no_attribution evaluator=%s detail=%s",
                evaluator_name,
                detail,
            )
            return TriageOutcome(
                status="attribution_failed",
                error=(f"AttributionResult not extractable from agent response ({detail})."),
            )

        is_safety = attribution.root_cause_code in self.SAFETY_ROOT_CAUSE_CODES
        agent_name, snapshot = await self._load_agent_notification(input)
        skip_reason = await self._dispatch_skip_reason(
            _DispatchSkipGate(
                input=input,
                attribution=attribution,
                snapshot=snapshot,
                is_safety=is_safety,
            ),
        )
        if skip_reason is not None:
            logger.info(
                "after_run.dispatch_skipped evaluator=%s reason=%s root_cause=%s",
                evaluator_name,
                skip_reason,
                attribution.root_cause,
            )
            return TriageOutcome(
                status="skipped",
                reason=skip_reason,
                attribution=attribution,
                notification_task=NotificationDispatchResult(
                    status=NotificationDispatchStatus.SKIPPED,
                    message=skip_reason,
                ),
            )

        if not is_safety and self._root_cause_groups(agent_name):
            self._pending.append((agent_name, attribution))
            logger.info(
                "after_run.root_cause_group_pending evaluator=%s root_cause_code=%s",
                evaluator_name,
                attribution.root_cause_code,
            )
            return TriageOutcome(
                status="skipped",
                reason="root_cause_group_pending",
                attribution=attribution,
                notification_task=NotificationDispatchResult(
                    status=NotificationDispatchStatus.SKIPPED,
                    message="root_cause_group_pending",
                ),
            )

        if not is_safety and self._aggregate_non_safety:
            logger.info(
                "after_run.non_safety_aggregate_only evaluator=%s root_cause_code=%s",
                evaluator_name,
                attribution.root_cause_code,
            )
            return TriageOutcome(
                status="skipped",
                reason="non_safety_aggregate_only",
                attribution=attribution,
                notification_task=NotificationDispatchResult(
                    status=NotificationDispatchStatus.SKIPPED,
                    message="non_safety_aggregate_via_cluster_report",
                ),
            )

        # should_dispatch=True：默认单 case 即时出单。
        notification_config = self._resolve_notification_config(
            input,
            agent_name=agent_name,
            snapshot=snapshot,
        )
        try:
            publisher = resolve_notification_publisher(agent_name)
            dispatch = await publisher.create_task(
                notification_config=notification_config,
                attribution_result=attribution,
                resolved_route=input.resolved_route,
            )
        except Exception as exc:
            logger.exception("after_run.aone_call_failed evaluator=%s", evaluator_name)
            return TriageOutcome(
                status="dispatch_failed",
                attribution=attribution,
                error=str(exc),
            )
        if dispatch.status.value in ("failed", "blocked"):
            return TriageOutcome(
                status="dispatch_failed",
                attribution=attribution,
                notification_task=dispatch,
                error=dispatch.message,
            )
        return TriageOutcome(
            status="created",
            attribution=attribution,
            notification_task=dispatch,
        )

    async def _dispatch_skip_reason(self, gate: _DispatchSkipGate) -> str | None:
        """注册关闭 / agent 拒派 / 候选根因 / 等级门槛。None 表示继续建单。

        有融合分时按 :func:`grade_fused_score`（0–2 / 0–10 等）仅 POOR ∪ FAIL
        放行；分数缺失则 fail-open；分制字段皆空且分数 >2 时按 0–10 推断。
        **安全类**（``SAFETY_ROOT_CAUSE_CODES``）不受分数门槛约束，与「单 case
        即时出单」一致。``snapshot`` 由调用方一次性加载后传入。
        """
        attribution = gate.attribution
        input = gate.input
        # 派单目标统一 RemoteConfig；注册文档 notification_dispatch_enabled 不再拦截。
        if attribution.candidate_root_cause is not None:
            return "candidate_root_cause"
        if not attribution.should_dispatch:
            return "agent_should_dispatch_false"
        if gate.is_safety:
            return None
        score = input.fused_score
        grade = (
            grade_fused_score(
                score,
                score_scale_max=input.score_scale_max,
                pass_threshold=input.pass_threshold,
            )
            if score is not None
            else None
        )
        policy = resolve_dispatch_policy(input.evaluator_name)
        if await should_dispatch_notification(policy, grade=grade, score=score):
            return None
        logger.info(
            "after_run.dispatch_skipped_by_grade evaluator=%s score=%s grade=%s "
            "score_scale_max=%s pass_threshold=%s",
            input.evaluator_name,
            score,
            grade,
            input.score_scale_max,
            input.pass_threshold,
        )
        return "score_threshold"

    async def _load_agent_notification(
        self,
        input: AfterRunInput,
    ) -> tuple[str, AgentNotificationSnapshot | None]:
        """从 evaluator_request.metadata 解析注册派单视图。"""
        return await load_agent_notification_context(_request_notification_config_view(input.evaluator_request))

    def _resolve_notification_config(
        self,
        input: AfterRunInput,
        *,
        agent_name: str,
        snapshot: AgentNotificationSnapshot | None,
    ) -> NotificationConfig:
        """用注册文档 / RemoteConfig 覆盖装配期 AoneConfig 的目标字段。"""
        name = agent_name.strip()
        if not name:
            name = str(getattr(input.evaluator_request, "agent_name", "") or "").strip()
        project, assignee, participants, _enabled = resolve_effective_notification_targets(
            name,
            snapshot,
            remote_config_key_override=(self._config.dispatch_profile or None),
        )
        updates: dict[str, object] = {}
        if project:
            updates["project"] = project
        if assignee:
            updates["assignee"] = assignee
        if participants:
            updates["participants"] = list(participants)
        if not updates:
            return self._config
        return self._config.model_copy(update=updates)

    def _root_cause_groups(self, agent_name: str) -> dict[str, tuple[str, ...]]:
        return resolve_notification_root_cause_groups(
            agent_name,
            remote_config_key_override=(self._config.dispatch_profile or None),
        )

    async def flush_groups(self) -> list[NotificationDispatchResult]:
        """把缓冲中的非安全归因按根因各发一条通知，然后清空缓冲。"""

        pending = self._pending
        self._pending = []
        by_agent: dict[str, list[AttributionResult]] = {}
        for agent_name, attribution in pending:
            by_agent.setdefault(agent_name, []).append(attribution)
        dispatched: list[NotificationDispatchResult] = []
        for agent_name, cases in by_agent.items():
            project, assignee, participants, _enabled = resolve_effective_notification_targets(
                agent_name,
                remote_config_key_override=(self._config.dispatch_profile or None),
            )
            config = self._config
            if project:
                config = config.model_copy(update={"project": project})
            default_members = tuple(
                member for member in (assignee, *participants) if str(member).strip()
            )
            dispatched.extend(await publish_root_cause_groups(
                cases,
                groups=self._root_cause_groups(agent_name),
                default_members=default_members,
                notification_config=config,
                publisher=resolve_notification_publisher(agent_name),
            ))
        return dispatched


# ---------------------------------------------------------------------------
# 对外结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriageOutcome:
    """TriageAttributionService.run 的对外回报。

    status:
      - "skipped": before 切面决定不归因
      - "created": attribution 跑完且 after 切面成功落地
      - "attribution_failed": attribution agent 跑失败
      - "dispatch_failed": attribution 已落地, after 切面失败
    """

    status: str
    reason: str | None = None
    attribution: AttributionResult | None = None
    notification_task: NotificationDispatchResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class TriageAssembly:
    """TriageAttributionService 装配原料（``with_defaults`` 的参数收敛）。

    由装配层（main.py / 集成测试）从配置构造：分类器与 triage 配置
    装配默认 TriageBeforeRunHook，aone 配置装配默认 AoneAfterRunHook。
    """

    agent_manager: AgentManager
    classifiers: dict[str, TriageClassifier]
    triage_config: TriageConfig
    notification_config: NotificationConfig
    trace_record_service: TraceRecordService
    trace_resolver: TraceResolver | None = None


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


HookFactory = Callable[["TriageAttributionService"], BeforeRunHook | AfterRunHook]


@dataclass(frozen=True)
class TriageServiceDeps:
    """TriageAttributionService 的依赖收敛（FUNC-PARAMS-001）。"""

    agent_manager: AgentManager
    before_hook: BeforeRunHook
    after_hook: AfterRunHook
    trace_record_service: TraceRecordService
    trace_resolver: TraceResolver | None = None


@dataclass(frozen=True)
class TriageRunInput:
    """TriageAttributionService.run 的输入收敛（FUNC-PARAMS-001）。

    ``evaluator_request`` / ``evaluator_result`` 为 agentgraph 信封快照，
    多 evaluator 形态不一，协议边界处保持 ``Any``。

    streaming 任务可通过 ``triage_classifier`` / ``attribution_agent_key`` /
    ``attribution_execution`` 覆盖默认「按 evaluator_name 查表」路由，
    并选择 sync / async / none 归因执行方式。
    """

    evaluator_name: str
    evaluator_request: Any
    evaluator_result: Any
    trace_client: Langfuse | None = None
    triage_classifier: TriageClassifier | None = None
    attribution_agent_key: str | None = None
    attribution_execution: str = "sync"


@dataclass(frozen=True)
class _AttributionWriteInput:
    """_write_attribution 的落库入参收敛（FUNC-PARAMS-001）。"""

    record_id: str
    agent_name: str
    attribution: AttributionResult
    original_triage_level: str | None = None


@dataclass(frozen=True)
class _AttributionPipelineCtx:
    """归因 agent.call → 落库 → after_hook 共用上下文（FUNC-PARAMS-001）。

    ``evaluator_request`` / ``evaluator_result`` 为 agentgraph 信封快照，
    多 evaluator 形态不一，协议边界处保持 ``Any``。
    """

    evaluator_name: str
    evaluator_request: Any  # noqa: ANN401 — agentgraph 请求快照（多 evaluator 信封）
    evaluator_result: Any  # noqa: ANN401 — 评测结果信封由各 evaluator 自定
    agent_key: str
    record_id: str | None
    agent_name: str
    final_level: str | None
    trace_client: Langfuse | None
    trace_id: str | None


class TriageAttributionService:
    """协调 before → attribution agent → after。

    切面可通过 :meth:`set_before` / :meth:`set_after` 运行时替换。
    默认切面: :class:`TriageBeforeRunHook` (triage) + :class:`AoneAfterRunHook` (aone)。

    写库责任: before 决策后写 ``triage_level`` 标量；attribution 产出后
    写 ``root_cause`` / ``owner_group`` 标量 + ``attribution_detail_json`` bucket；
    after 派发后写 ``notification_dispatch_json`` + enqueue ``notification_job_json``。
    """

    def __init__(
        self,
        deps: TriageServiceDeps,
    ) -> None:
        self._agent_manager = deps.agent_manager
        self._before_hook = deps.before_hook
        self._after_hook = deps.after_hook
        self._trace_record_service = deps.trace_record_service
        self._trace_resolver = deps.trace_resolver

    @classmethod
    def with_defaults(
        cls,
        *,
        assembly: TriageAssembly,
    ) -> "TriageAttributionService":
        """便利工厂: 装配默认 TriageBeforeRunHook + AoneAfterRunHook。"""

        return cls(
            TriageServiceDeps(
                agent_manager=assembly.agent_manager,
                before_hook=TriageBeforeRunHook(
                    classifiers=assembly.classifiers,
                    triage_config=assembly.triage_config,
                ),
                after_hook=NotificationAfterRunHook(notification_config=assembly.notification_config),
                trace_record_service=assembly.trace_record_service,
                trace_resolver=assembly.trace_resolver,
            ),
        )

    def set_before(self, hook: BeforeRunHook) -> None:
        """运行时替换前置切面。"""

        self._before_hook = hook

    @property
    def after_hook(self) -> AfterRunHook:
        """当前后置切面（只读）。装配层用于包装装饰器。"""
        return self._after_hook

    def set_after(self, hook: AfterRunHook) -> None:
        """运行时替换后置切面。"""

        self._after_hook = hook

    async def flush_notification_groups(self) -> list[NotificationDispatchResult]:
        """按根因把缓冲中的通知发给对应人群。切面没有缓冲时返回空列表。"""

        flush = getattr(self._after_hook, "flush_groups", None)
        if flush is None:
            return []
        return await flush()

    async def run(  # noqa: PLR0911, PLR0912, PLR0915 — before→agent→after 三步编排入口，分支即步骤链
        self,
        input: TriageRunInput,
    ) -> TriageOutcome:
        """evaluator after_eval 钩子入口 — 串行三步: before → agent → after。

        写库点:
          - before 决策后调 :meth:`_write_triage_level`
          - agent attribution 产出后调 :meth:`_write_attribution`
          - after 派发后调 :meth:`_write_dispatch_and_enqueue`
        """

        evaluator_name = input.evaluator_name
        evaluator_request = input.evaluator_request
        evaluator_result = input.evaluator_result
        trace_client = input.trace_client
        trace_id = getattr(evaluator_request, "trace_id", None)
        project_id = getattr(evaluator_request, "project_id", None)
        agent_name = getattr(evaluator_request, "agent_name", None)
        missing = [
            k
            for k, v in [
                ("trace_id", trace_id),
                ("project_id", project_id),
                ("agent_name", agent_name),
            ]
            if not v
        ]
        if missing:
            logger.error(
                "triage.metadata_missing evaluator=%s missing=%s",
                evaluator_name,
                missing,
            )
            return TriageOutcome(
                status="attribution_failed",
                error=f"metadata_missing: {', '.join(missing)}",
            )

        # Look up trace record by business keys; record_id is None if not found.
        # 落库键与 ScoreWrite 一致：有 registrationId 用 stem@reg，否则用茎。
        record_evaluator_name = persist_evaluator_name(
            evaluator_name,
            getattr(evaluator_request, "metadata", None),
        )
        record_id: str | None = None
        try:
            record = await self._trace_record_service.get_by_trace_id(
                TraceRecordKey(
                    trace_id=trace_id,
                    project_id=project_id,
                    agent_name=agent_name,
                    evaluator_name=record_evaluator_name,
                ),
            )
            if record is not None:
                record_id = str(record.id)
        except Exception:
            record_id = None

        try:
            decision = await self._before_hook(
                BeforeRunInput(
                    evaluator_name=evaluator_name,
                    evaluator_request=evaluator_request,
                    evaluator_result=evaluator_result,
                    classifier_override=input.triage_classifier,
                ),
            )
        except Exception as exc:
            logger.exception("run.before_failed evaluator=%s", evaluator_name)
            return TriageOutcome(status="skipped", reason=f"before_failed: {exc}")

        final_level = (decision.metadata or {}).get("final_level")

        from benchmark.labeling.context import (  # noqa: PLC0415
            clear_triage_judge_context,
            set_triage_judge_context,
        )

        set_triage_judge_context(
            evaluator_result=evaluator_result,
            triage_level=final_level,
        )

        background_scheduled = False
        try:
            # 1) level 落库: classifier 给了合法 level 就直写,与归因决策正交。
            if record_id is not None and final_level:
                await self._write_triage_level(
                    record_id=record_id,
                    final_level=final_level,
                )

            # 2) 归因 + 派发按 should_proceed 决策;非 badcase 走 skipped 分支,
            #    不再影响上面已写的 triage_level 标量。
            if not decision.should_proceed:
                logger.info(
                    "run.skipped evaluator=%s reason=%s",
                    evaluator_name,
                    decision.reason,
                )
                return TriageOutcome(
                    status="skipped",
                    reason=decision.reason,
                )

            agent_key = input.attribution_agent_key or stage_agent_key(
                STAGE_ATTRIBUTION,
                evaluator_name,
            )
            attribution_execution = (input.attribution_execution or "sync").strip().lower()
            if attribution_execution == "none":
                logger.info(
                    "run.attribution_skipped_by_execution evaluator=%s",
                    evaluator_name,
                )
                return TriageOutcome(status="skipped", reason="attribution_execution_none")

            # 受理标记: 放行后、agent 跑完前落 attribution_status=pending,
            # 使聚合 progress 能区分“未触发/进行中/已完成”三态。
            # merge 式局部更新,不影响 judge_json / status 标量列。
            if record_id is not None:
                try:
                    await self._trace_record_service.mark_attribution_pending(
                        record_id=record_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "run.mark_pending_failed record_id=%s err=%s",
                        record_id,
                        exc,
                    )

            if attribution_execution == "async":
                # 消息路径提交后立即返回；agent + after 在后台完成。
                # 显式 async 配置（tasks[].attribution_execution / async 别名）
                # 才走此路径；默认 sync 仍同槽位 await，保 SLA。
                pipeline_ctx = _AttributionPipelineCtx(
                    evaluator_name=evaluator_name,
                    evaluator_request=evaluator_request,
                    evaluator_result=evaluator_result,
                    agent_key=agent_key,
                    record_id=record_id,
                    agent_name=str(agent_name),
                    final_level=final_level,
                    trace_client=trace_client,
                    trace_id=trace_id,
                )
                asyncio.create_task(
                    self._run_attribution_pipeline_background(pipeline_ctx),
                    name=f"streaming-async-attr-{trace_id}",
                )
                background_scheduled = True
                logger.info(
                    "run.async_attribution_scheduled evaluator=%s trace_id=%s agent_key=%s",
                    evaluator_name,
                    trace_id,
                    agent_key,
                )
                return TriageOutcome(
                    status="created",
                    reason="async_attribution_scheduled",
                )

            try:
                agent = self._agent_manager.get(agent_key)
                agent_request = _build_attribution_request(
                    evaluator_request,
                    evaluator_result,
                )
                attribution_response = await agent.call(agent_request)
            except KeyError:
                logger.warning(
                    "run.no_agent evaluator=%s agent_key=%s",
                    evaluator_name,
                    agent_key,
                )
                return TriageOutcome(
                    status="attribution_failed",
                    error=f"Unknown attribution agent {agent_key!r}.",
                )
            except Exception as exc:
                logger.exception("run.agent_call_failed evaluator=%s", evaluator_name)
                return TriageOutcome(status="attribution_failed", error=str(exc))

            return await self._finish_attribution(
                _AttributionPipelineCtx(
                    evaluator_name=evaluator_name,
                    evaluator_request=evaluator_request,
                    evaluator_result=evaluator_result,
                    agent_key=agent_key,
                    record_id=record_id,
                    agent_name=str(agent_name),
                    final_level=final_level,
                    trace_client=trace_client,
                    trace_id=trace_id,
                ),
                attribution_response,
            )
        finally:
            try:
                if not background_scheduled:
                    await self._close_pending_attribution(record_id)
            finally:
                clear_triage_judge_context()

    async def _run_attribution_pipeline_background(
        self,
        ctx: _AttributionPipelineCtx,
    ) -> None:
        """async attribution 后台任务：失败只打日志，不回传消息路径。"""
        try:
            agent = self._agent_manager.get(ctx.agent_key)
            agent_request = _build_attribution_request(
                ctx.evaluator_request,
                ctx.evaluator_result,
            )
            attribution_response = await agent.call(agent_request)
            await self._finish_attribution(ctx, attribution_response)
        except Exception:
            logger.exception(
                "run.async_attribution_background_failed evaluator=%s trace_id=%s",
                ctx.evaluator_name,
                ctx.trace_id,
            )

        finally:
            await self._close_pending_attribution(ctx.record_id)

    async def _close_pending_attribution(self, record_id: str | None) -> None:
        if record_id is None:
            return
        try:
            await self._trace_record_service.mark_attribution_failed(record_id=record_id)
        except Exception:
            logger.warning(
                "run.mark_attribution_failed_failed record_id=%s", record_id, exc_info=True
            )

    async def _finish_attribution(
        self,
        ctx: _AttributionPipelineCtx,
        attribution_response: AgentResponse,
    ) -> TriageOutcome:
        """sync / async 共用的归因落库 + after_hook + 派单 enqueue。"""
        evaluator_name = ctx.evaluator_name
        evaluator_request = ctx.evaluator_request
        evaluator_result = ctx.evaluator_result
        record_id = ctx.record_id
        agent_name = ctx.agent_name
        final_level = ctx.final_level
        trace_client = ctx.trace_client
        trace_id = ctx.trace_id
        attribution = _extract_attribution_result(attribution_response)
        # 归因推翻 judge：强制拒派 Aone（即使 agent 忘了设 should_dispatch）。
        if (
            attribution is not None
            and attribution.corrected_triage_level is not None
            and attribution.should_dispatch
        ):
            attribution = attribution.model_copy(update={"should_dispatch": False})
        resolved_route: Mapping[str, Any] | None = None
        if record_id is not None and attribution is not None:
            resolved_route = await self._write_attribution(
                _AttributionWriteInput(
                    record_id=record_id,
                    agent_name=agent_name,
                    attribution=attribution,
                    original_triage_level=final_level,
                ),
            )
            _meta = getattr(evaluator_request, "metadata", None) or {}
            if _meta.get("dataset_item_id") or _meta.get("dataset_name"):
                logger.info(
                    "triage.langfuse_attribution_will_write trace_id=%s evaluator=%s "
                    "dataset_item_id=%s",
                    trace_id,
                    evaluator_name,
                    _meta.get("dataset_item_id"),
                )
                await write_attribution_score(
                    AttributionScoreWrite(
                        trace_id=trace_id,
                        evaluator_name=evaluator_name,
                        attribution=attribution,
                        original_score=getattr(evaluator_result, "score", None),
                        client=trace_client,
                    ),
                )
            else:
                logger.info(
                    "triage.langfuse_attribution_skipped trace_id=%s evaluator=%s "
                    "reason=non_dataset",
                    trace_id,
                    evaluator_name,
                )

        try:
            scale_max, pass_threshold = _score_grade_params(evaluator_result)
            outcome = await self._after_hook(
                AfterRunInput(
                    evaluator_name=evaluator_name,
                    evaluator_request=evaluator_request,
                    attribution_response=attribution_response,
                    attribution=attribution,
                    resolved_route=resolved_route,
                    fused_score=_coerce_fused_score(getattr(evaluator_result, "score", None)),
                    score_scale_max=scale_max,
                    pass_threshold=pass_threshold,
                ),
            )
        except Exception as exc:
            logger.exception("run.after_failed evaluator=%s", evaluator_name)
            return TriageOutcome(
                status="dispatch_failed",
                attribution=attribution,
                error=str(exc),
            )
        if record_id is not None and outcome.notification_task is not None:
            await self._write_dispatch_and_enqueue(
                record_id=record_id,
                trace_id=trace_id,
                dispatch=outcome.notification_task,
            )
        return outcome

    async def _resolve_raw_trace(
        self,
        keys: tuple[str, str, str, str] | None,
    ) -> ResolvedTrace | None:
        """Fetch Langfuse trace by business trace_id; API callers never pass raw_trace."""

        if keys is None or self._trace_resolver is None:
            return None
        trace_id, project_id, _, _ = keys
        try:
            resolved = await self._trace_resolver.resolve(trace_id, project_id)
        except Exception as exc:
            logger.warning(
                "run.resolve_raw_trace_failed trace=%s project=%s error=%s",
                trace_id,
                project_id,
                exc,
            )
            return None
        return resolved

    async def _write_triage_level(
        self,
        *,
        record_id: str,
        final_level: str,
    ) -> None:
        try:
            if final_level not in _TRIAGE_LEVEL_VALUES:
                logger.warning(
                    "run.write_triage_level_skipped record_id=%s final_level=%r not in TriageLevel",
                    record_id,
                    final_level,
                )
                return
            updated = await self._trace_record_service.update_triage_level(
                record_id=record_id,
                triage_level=TriageLevel(final_level),
            )
            if updated is None:
                logger.warning(
                    "run.write_triage_level_skipped record_id=%s "
                    "final_level=%r reason=record_not_found",
                    record_id,
                    final_level,
                )
        except Exception as exc:
            logger.warning("run.write_triage_level_failed record_id=%s err=%s", record_id, exc)

    async def _write_attribution(
        self,
        input: _AttributionWriteInput,
    ) -> "Mapping[str, Any] | None":
        record_id = input.record_id
        agent_name = input.agent_name
        attribution = input.attribution
        original_triage_level = input.original_triage_level
        try:
            owner_group: str | None = None
            mapping: OwnerMapping | None = None
            # `other:` 候选不是稳定根因，禁止回退使用自由文本 root_cause 解析 owner。
            # 只有正式 root_cause_code 或无候选的历史结果才允许走兼容回退。
            primary_cause = (
                None
                if attribution.candidate_root_cause is not None
                else attribution.root_cause_code or attribution.root_cause
            )
            if primary_cause:
                mapping = resolve_owner_route(
                    agent_name=agent_name,
                    primary_cause=primary_cause,
                )
                if mapping is not None:
                    owner_group = mapping.owner_group
            # attribution_raw 优先用模型原文；缺失时落完整 AttributionResult 快照，
            # 避免 bucket 只有分类摘要、推理/证据只能去 aone_job 翻。
            attribution_raw = attribution.raw_response or attribution.model_dump_json()
            # 归因结论优先：推翻 judge 时标量写 ATTRIBUTION_OVERRIDDEN，
            # 与 bucket 同一次 UPDATE 落库（拆成两条写会留下不一致行）。
            overridden = (
                TriageLevel.ATTRIBUTION_OVERRIDDEN
                if attribution.corrected_triage_level is not None
                else None
            )
            updated = await self._trace_record_service.update_attribution(
                record_id=record_id,
                update=AttributionUpdate(
                    root_cause=attribution.root_cause,
                    owner_domain=attribution.category,
                    owner_group=owner_group,
                    confidence=attribution.confidence,
                    attribution_raw=attribution_raw,
                    evidence_raw=json.dumps(
                        attribution.evidence_refs,
                        ensure_ascii=False,
                    ),
                    reasoning_raw=attribution.reasoning,
                    action_hint=attribution.action_hint,
                    needs_human_review=attribution.needs_human_review,
                    corrected_triage_level=attribution.corrected_triage_level,
                    original_triage_level=original_triage_level,
                    triage_level=overridden,
                    # 聚合报告消费的结构化字段（F7/F8）：category 独立键消除
                    # owner_domain 的语义错位；evidence_details 序列化为 JSON-safe dict。
                    category=attribution.category,
                    root_cause_code=attribution.root_cause_code,
                    all_root_cause_codes=attribution.all_root_cause_codes or None,
                    conclusion_type=attribution.conclusion_type,
                    evidence_details=[
                        item.model_dump(mode="json") for item in attribution.evidence_details
                    ]
                    or None,
                    candidate_root_cause=(
                        attribution.candidate_root_cause.model_dump(mode="json")
                        if attribution.candidate_root_cause is not None
                        else None
                    ),
                ),
            )
            if updated is None:
                logger.warning(
                    "run.write_attribution_skipped record_id=%s reason=record_not_found",
                    record_id,
                )
            return mapping
        except Exception as exc:
            logger.warning("run.write_attribution_failed record_id=%s err=%s", record_id, exc)
            return None

    async def _write_dispatch_and_enqueue(
        self,
        *,
        record_id: str,
        trace_id: str,
        dispatch: NotificationDispatchResult,
    ) -> None:
        try:
            skip_reason: str | None = None
            if dispatch.status == NotificationDispatchStatus.SKIPPED:
                skip_reason = dispatch.message or "skipped"
            created = dispatch.status == NotificationDispatchStatus.CREATED
            assignee = None
            if created and dispatch.payload is not None:
                text = str(dispatch.payload.assignee or "").strip()
                assignee = text or None
            await self._trace_record_service.update_dispatch(
                record_id=record_id,
                update=DispatchUpdate(
                    dispatch_ref=dispatch.workitem_id,
                    dispatch_url=(dispatch.workitem_url or None) if created else None,
                    dispatch_assigned_to=assignee,
                    dispatch_title=dispatch.message or None if skip_reason is None else None,
                    dispatched_at=datetime.now(UTC),
                    dispatch_skip_reason=skip_reason,
                ),
            )
            if dispatch.payload is not None:
                await self._trace_record_service.enqueue_notification_task(
                    record_id=record_id,
                    task=NotificationTaskEnqueue(
                        idempotency_key=dispatch.workitem_id or f"aone_{trace_id}",
                        payload=dispatch.payload.model_dump(mode="json"),
                    ),
                )
        except Exception as exc:
            logger.warning("run.write_dispatch_failed record_id=%s err=%s", record_id, exc)


def _coerce_fused_score(value: object) -> float | None:
    """evaluator 分 → float。缺失或非数字返回 None（派单等级门槛放行）。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


_SCORE_SCALE_LENGTH = 2


def _score_grade_params(evaluator_result: object) -> tuple[float | None, float | None]:
    """从 evaluator 结果取 ``(score_scale_max, pass_threshold)`` 供派单判级。

    优先 ``extra.sub_score_scale`` 上界（example 等显式分制）；再读
    ``pass_threshold`` / ``passThreshold`` 作 grade_fused_score 兼容回退。
    """
    extra = getattr(evaluator_result, "extra", None)
    if not isinstance(extra, Mapping):
        return None, None
    scale_max: float | None = None
    raw_scale = extra.get("sub_score_scale")
    if isinstance(raw_scale, (list, tuple)) and len(raw_scale) == _SCORE_SCALE_LENGTH:
        high = raw_scale[1]
        if not isinstance(high, bool) and isinstance(high, (int, float)):
            scale_max = float(high)
    pass_threshold: float | None = None
    for key in ("pass_threshold", "passThreshold"):
        raw = extra.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        pass_threshold = float(raw)
        break
    return scale_max, pass_threshold


def _request_notification_config_view(evaluator_request: object) -> dict[str, Any]:
    """evaluator_request → ``load_agent_aone_context`` 可用的 config 视图。"""
    meta = getattr(evaluator_request, "metadata", None)
    if not isinstance(meta, Mapping):
        meta = {}
    view: dict[str, Any] = {}
    reg_id = meta.get("agent_registration_id") or meta.get("agentRegistrationId")
    if isinstance(reg_id, str) and reg_id.strip():
        view["agent_registration_id"] = reg_id.strip()
    snap = meta.get("agent_registration") or meta.get("agentRegistration")
    if isinstance(snap, Mapping):
        view["agent_registration"] = dict(snap)
    elif "agent_registration_id" not in view:
        name = getattr(evaluator_request, "agent_name", None) or meta.get("agent_name")
        if isinstance(name, str) and name.strip():
            view["agent_registration"] = {"name": name.strip()}
    return view


def _build_attribution_request(
    evaluator_request: Any,  # noqa: ANN401 — agentgraph 请求快照（多 evaluator 信封形态的协议边界）
    evaluator_result: Any,  # noqa: ANN401 — 同上，评测结果信封由各 evaluator 自定
) -> AgentRequest:
    """从 evaluator request/result 构造 attribution agent 的 AgentRequest。

    调用方 :meth:`run` 已守卫 trace_id / project_id;此处不再兜底。
    Langfuse 取数由 attribution/judge 经 ``trace`` tool 按需完成。

    Prompt 阶段隔离：剥离 judge 的 ``prompt_text`` / ``judge_prompt_text`` /
    judge-stage bundle，只把 attribution 阶段 refs 顶到 metadata，避免
    打分模板被归因当 system prompt 使用。
    """
    trace_id = getattr(evaluator_request, "trace_id", None) or getattr(
        evaluator_request, "traceid", None
    )
    session_id = (
        getattr(evaluator_request, "session_id", None)
        or getattr(evaluator_request, "sessionid", None)
        or trace_id
    )
    project_id = getattr(evaluator_request, "project_id", None)
    er_dump = sanitize_evaluator_request_for_attribution(
        _safe_model_dump(evaluator_request),
    )
    metadata: dict[str, Any] = {
        "evaluator_request": er_dump,
        "evaluator_result": _safe_model_dump(evaluator_result),
        "trace_id": trace_id,
        "project_id": project_id,
    }
    # 仅 attribution-stage refs 顶到 metadata；judge refs 已在 er_dump 中剥离。
    prompt_ref, skill_ref = attribution_bundle_refs(evaluator_request)
    if prompt_ref is not None:
        metadata["prompt_bundle_ref"] = dump_ref(prompt_ref)
    if skill_ref is not None:
        metadata["skill_bundle_ref"] = dump_ref(skill_ref)

    return AgentRequest(
        traceid=trace_id,
        session_id=session_id,
        messages=[],
        metadata=metadata,
    )


def _extract_attribution_result(
    response: AgentResponse,
) -> AttributionResult | None:
    """从 AgentResponse 抽出 AttributionResult; 解析失败返 None。

    先读 ``metadata["attribution_result"]``（agent 成功与降级路径都会写入结构化
    副本），再回退 ``content_parts`` 文本解析——避免字符串序列化抖动被误判为
    "not extractable"。
    """

    metadata = response.metadata if isinstance(response.metadata, dict) else {}
    structured = metadata.get("attribution_result")
    if isinstance(structured, dict):
        try:
            return AttributionResult.model_validate(structured)
        except ValidationError:
            pass
    try:
        parts = response.content_parts or []
        for part in parts:
            data = getattr(part, "data", None)
            if isinstance(data, str) and data.strip():
                parsed = parse_and_validate_llm_json(data, AttributionResult)
                if parsed is not None:
                    return parsed
            elif isinstance(data, dict):
                try:
                    return AttributionResult.model_validate(data)
                except ValidationError:
                    continue
    except Exception:
        return None
    return None


def _response_failure_detail(response: AgentResponse) -> str:
    """AgentResponse 的失败元信息——服务层用它把真因带出去。"""

    metadata = response.metadata if isinstance(response.metadata, dict) else {}
    status = getattr(response.status, "value", response.status)
    parts = [f"status={status}"]
    error_code = metadata.get("error_code")
    if error_code:
        parts.append(f"error_code={error_code}")
    error = metadata.get("error")
    if error:
        parts.append(f"error={error}")
    return " ".join(parts)


def _safe_model_dump(obj: Any) -> dict[str, Any] | None:  # noqa: ANN401 — pydantic/非 pydantic 退化的鸭子类型输入
    """pydantic model 优先用 model_dump, 其它类型退化为空 dict。"""

    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return None
    return None


__all__ = [
    "TriageAttributionService",
    "TriageOutcome",
    "BeforeRunHook",
    "AfterRunHook",
    "BeforeRunDecision",
    "TriageBeforeRunHook",
    "NotificationAfterRunHook",
    "TriageConfig",
    "TriageDecision",
]
