"""公共评测 Runner：固定九节点标准 Graph 与阶段编排。

公共运行时只负责机制（方案 §6.1）：

1. 固定逻辑节点与拓扑——``STAGE_NODES`` 九阶段闭集，业务不得增删节点；
2. 节点异常隔离——阶段抛出的未预期异常收敛为 ``error`` / ``error_code`` 通道，
   不向调用方泄漏裸异常；可选超时由 ``stage_timeouts`` 显式声明；
3. 显式 ``skipped`` 语义——没有实际工作的可选阶段必须返回 :func:`skipped` 及原因，
   由 Runner 记入 ``stage_report``，而不是由业务删除节点；
4. 耗时观测——每个阶段的执行状态、原因与耗时写入 ``stage_report``，保持可回放；
5. 信封身份贯通——``resolve_profile`` 阶段写入 :class:`EnvelopeIdentity`
   （Profile 身份 + config_digest + 场景），供聚合与归因锚定可复现性。

公共层不出现任何业务分支：阶段实际逻辑全部由业务插件经
:class:`EvaluationPipelineHooks` 注入，Runner 只做编排、隔离与观测。
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict

from benchmark.pipeline.profiles import ProfileSpec, compute_config_digest
from benchmark.utils.exceptions import EVALUATION_FAILED_CODE

#: 与各业务 state 的错误通道字面量保持一致（``VehicleAgentState`` / ``VuiAgentState``）。
ERROR = "error"
ERROR_CODE = "error_code"
#: Runner 新增的观测通道：阶段执行报告与信封身份。
STAGE_REPORT = "stage_report"
ENVELOPE_IDENTITY = "envelope_identity"

#: 标准九节点闭集（方案 §6.1）：拓扑固定，业务只能通过 Hooks 决定各节点做什么。
STAGE_NODES: tuple[str, ...] = (
    "build_context",
    "resolve_profile",
    "collect_evidence",
    "run_checkers",
    "plan_judges",
    "run_judges",
    "collect_metrics",
    "aggregate_results",
    "cluster_results",
    "emit_envelope",
)

#: skipped 标记键：阶段处理器返回后由 Runner 弹出并记入 stage_report，不进入业务通道。
_STAGE_SKIPPED = "__stage_skipped_reason__"

StageStatus = Literal["executed", "skipped", "failed"]


def skipped(reason: str) -> dict[str, Any]:
    """可选阶段没有实际工作时的返回值：Runner 记录 ``skipped`` 及原因。

    方案 §4.1：没有实际工作的可选节点必须返回带原因的 ``skipped``，
    不得由业务自行删除节点，保证节点轨迹跨业务一致。
    """

    return {_STAGE_SKIPPED: reason}


class EnvelopeIdentity(BaseModel):
    """信封身份：Profile 身份 + 场景 + config_digest，评测/归因/报表的可复现锚点。"""

    model_config = ConfigDict(frozen=True)

    profile_id: str
    profile_version: str
    plugin: str
    catalog_version: str
    config_digest: str
    scenario: str = ""


def envelope_identity_from_profile(profile: ProfileSpec, scenario: str = "") -> EnvelopeIdentity:
    """从已校验 Profile 生成信封身份（``config_digest`` 为最终装配的指纹）。"""

    return EnvelopeIdentity(
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        plugin=profile.plugin,
        catalog_version=profile.catalog_version,
        config_digest=compute_config_digest(profile),
        scenario=scenario,
    )


StageHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class EvaluationPipelineHooks:
    """业务插件注入公共 Runner 的九阶段实现。

    每个 Handler 接收当前完整 state、返回 state patch（与既有节点形状一致）；
    无实际工作的可选阶段返回 :func:`skipped`。``stage_timeouts`` 为可选的每阶段
    超时（秒），未配置即不启用，保持现有行为。
    """

    build_context: StageHandler
    resolve_profile: StageHandler
    collect_evidence: StageHandler
    run_checkers: StageHandler
    plan_judges: StageHandler
    run_judges: StageHandler
    collect_metrics: StageHandler
    aggregate_results: StageHandler
    cluster_results: StageHandler
    emit_envelope: StageHandler
    stage_timeouts: dict[str, float] = field(default_factory=dict)


def build_evaluation_graph(
    state_schema: object, hooks: EvaluationPipelineHooks
) -> CompiledStateGraph:
    """按固定九节点拓扑编译业务评测图。

    拓扑：``START → build_context → … → emit_envelope → END`` 线性推进；
    任一阶段写入 ``error`` 后直接短路到 END，与既有业务图的条件路由语义一致。
    """

    builder = StateGraph(state_schema)  # type: ignore[arg-type]
    for stage in STAGE_NODES:
        builder.add_node(stage, _stage_node(stage, getattr(hooks, stage), hooks.stage_timeouts))
    builder.add_edge(START, STAGE_NODES[0])
    for index, stage in enumerate(STAGE_NODES[:-1]):
        builder.add_conditional_edges(stage, _next_or_end(STAGE_NODES[index + 1]))
    builder.add_edge(STAGE_NODES[-1], END)
    return builder.compile()


def _next_or_end(next_stage: str) -> Callable[[dict[str, Any]], str]:
    def _route(state: dict[str, Any]) -> str:
        return END if state.get(ERROR) else next_stage

    return _route


def _stage_node(
    stage: str, handler: StageHandler, timeouts: dict[str, float]
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    timeout = timeouts.get(stage)

    async def _node(state: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            patch = (
                await handler(state)
                if timeout is None
                else await asyncio.wait_for(handler(state), timeout)
            )
        except TimeoutError:
            return _finish(
                state,
                _record(stage, "failed", f"stage timeout after {timeout}s", started),
                {
                    ERROR: f"{stage} 阶段执行超时",
                    ERROR_CODE: EVALUATION_FAILED_CODE,
                },
            )
        except Exception as exc:  # 节点异常隔离：收敛为错误通道，不泄漏裸异常
            return _finish(
                state,
                _record(stage, "failed", str(exc), started),
                {
                    ERROR: f"{stage} 阶段执行异常: {exc}",
                    ERROR_CODE: EVALUATION_FAILED_CODE,
                },
            )
        skip_reason = patch.pop(_STAGE_SKIPPED, None) if isinstance(patch, dict) else None
        if skip_reason is not None:
            return _finish(state, _record(stage, "skipped", skip_reason, started), patch)
        error = patch.get(ERROR) if isinstance(patch, dict) else None
        status: StageStatus = "failed" if error else "executed"
        return _finish(state, _record(stage, status, str(error or ""), started), patch)

    return _node


def _record(stage: str, status: StageStatus, reason: str, started: float) -> dict[str, Any]:
    record: dict[str, Any] = {
        "status": status,
        "duration_ms": round((time.monotonic() - started) * 1000, 2),
    }
    if reason:
        record["reason"] = reason
    return {stage: record}


def _finish(state: dict[str, Any], record: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    report = dict(state.get(STAGE_REPORT) or {})
    report.update(record)
    return {**patch, STAGE_REPORT: report}


__all__ = [
    "ENVELOPE_IDENTITY",
    "ERROR",
    "ERROR_CODE",
    "STAGE_NODES",
    "STAGE_REPORT",
    "EnvelopeIdentity",
    "EvaluationPipelineHooks",
    "StageHandler",
    "StageStatus",
    "build_evaluation_graph",
    "envelope_identity_from_profile",
    "skipped",
]
