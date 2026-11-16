"""Benchmark HTTP API endpoints.

Endpoints:
    GET    /v1/benchmarks                        — list all registered benchmark definitions
    POST   /v1/benchmarks/{name}/run             — accept async run (202 + run_id)
    POST   /v1/benchmarks/{name}/run-pipeline    — accept async pipeline benchmark (202 + run_id)
    POST   /v1/benchmarks/run-inline             — inline dataset run (202 + run_id)
    POST   /v1/benchmarks/manual-runs          — manual external rerun (202 + distinct run_id)
    POST   /v1/benchmarks/runs                   — create stream run (201, ACCEPTING)
    POST   /v1/benchmarks/runs/{run_id}/items    — append stream items (multi-call)
    POST   /v1/benchmarks/runs/{run_id}/finalize — finish ingest, start evaluation
    GET    /v1/benchmarks/runs                  — page-shaped run list (no benchmark name required)
    GET    /v1/benchmarks/run-counts            — run counts grouped by benchmark name
    GET    /v1/benchmarks/{name}/runs            — list run history for a benchmark
    GET    /v1/benchmarks/runs/{run_id}          — get single run details
    DELETE /v1/benchmarks/runs/{id}              — soft-delete one run (deleted_at mark)
    GET    /v1/benchmarks/runs/{run_id}/report   — get HTML report (302 redirect or inline)
    GET    /v1/benchmarks/runs/{run_id}/metrics  — get aggregated metrics JSON

Async-only: POST returns 202 immediately; work runs in-process via
``asyncio.create_task``. Clients may disconnect after submit. Poll
``GET /runs/{run_id}`` optionally.

**Distributed execution**: ``POST /{name}/run`` and ``POST /run-inline``
spawn BullMQ Workers — case jobs are distributed via BullMQ queues,
no PG row locks. Multiple nodes can execute the same run concurrently.

Incomplete runs (drain cancel, heartbeat timeout after crash/deploy) are
marked ``failed`` via ``fail_inflight_runs_for_owner`` / ``recover_stale_runs``.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, TypeVar
from uuid import UUID

from benchmark.libs.agenthost.observability.logger import get_logger
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, JsonValue, ValidationError

from benchmark.routes._guards import (
    raise_if_benchmark_disabled,
    raise_if_eval_queue_degraded,
)
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun
from benchmark.records.models.runtime_config_record import (
    KIND_EVALUATOR,
    compose_evaluator_key_tail,
    evaluator_record_key,
    project_scope_key,
)
from benchmark.domain.agent_config import get_agent_config
from benchmark.domain.benchmark_console import (
    AgentDocument,
    ConsoleError,
    DatasetDocument,
    check_customization_policy,
    find_dataset_document_by_id,
    find_dataset_document_by_name,
    list_agent_documents,
    resolve_agent_registration,
    save_dataset,
    start_sampling,
)
from benchmark.domain.benchmark_console import (
    registration_snapshot as build_registration_snapshot,
)
from benchmark.domain.eval_config import ingest_conflict_message
from benchmark.domain.eval_plugin_config import is_benchmark_component_enabled
from benchmark.domain.report_storage_config import get_default_download_prefix
from benchmark.domain.runtime_config_defaults import stamp_eval_entry
from benchmark.domain.runtime_config_store import (
    current_eval_config_file,
    refresh_eval_config_snapshot,
    retire_sibling_evaluator_entries,
    upsert_runtime_config,
)
from benchmark.domain.trace_report_config import TraceReportSeed, ensure_default_trace_report
from benchmark.evaluators.customization_resolution import bundle_refs_from_snapshot
from benchmark.libs.jobqueue.staging_buffer import StagingBuffer
from benchmark.runs.common.agent_capabilities import (
    CapabilityBindingView,
    capabilities_or_default,
    customization_policy_for_name,
    customization_policy_view,
    declared_capabilities,
    default_batch_capabilities,
)
from benchmark.runs.common.agent_catalog import (
    EvaluatorComponents,
    canonical_agent_name,
    configured_evaluators,
    evaluator_capabilities,
)
from benchmark.runs.common.config_helpers import resolve_oss_prefix
from benchmark.runs.common.dynamic_component import (
    ensure_dynamic_benchmark_component,
)
from benchmark.runs.common.oss_date_macro import (
    expand_date_macros,
    has_date_macro,
    object_key_for_dated_prefix,
    partition_date_from_path,
    strip_oss_scheme,
    validate_date_macros,
)
from benchmark.runs.common.parse_interceptor import (
    CaseIdentityScope,
    normalize_case_identity,
)
from benchmark.runs.common.profile_resolver import (
    ProfileResolutionError,
    ProfileResolver,
    ResolvedProfile,
)
from benchmark.runs.common.rubric_base import load_rubric_class_ref
from benchmark.runs.common.rubric_classes import (
    RUBRIC_CLASS_REFS,
    resolve_rubric_class,
)
from benchmark.runs.common.submit_pipeline import (
    SubmitPipelineSpec,
    submit_prepared_items,
)
DailyDataLoader = None
from benchmark.runs.manual_run_id import (
    ManualRunClaimScope,
    ManualRunIdExhaustedError,
    ManualRunIdTakenError,
    ManualRunNamespace,
    claim_manual_run_id,
    list_manual_run_ids,
    resolve_external_data_date,
    validate_idempotency_key,
)
from benchmark.runs.registry import BENCHMARK_SERVICES_CLASSES
from benchmark.runs.run_lifecycle import ListRunsQuery
from benchmark.runs.run_service import (
    BenchmarkRunSpec,
    BenchmarkService,
    ParseFailedStubWrite,
)
from benchmark.runs.runner import BenchmarkRunner, FetchItemsSpec, PipelineRunExec
from benchmark.runs.schema import (
    FUSED_CRITERION_ID,
    AgentCapabilityView,
    AttributionBatchCallback,
    AttributionCallback,
    AttributionProfileConfig,
    BenchmarkDefinition,
    BenchmarkRuntimeProfile,
    ExternalBenchmarkRequest,
    ExternalJudgeItemResult,
    ExternalRunAcceptedResponse,
    JudgeCallback,
    JudgeProfileConfig,
    PipelineBenchmarkRequest,
    TriageMode,
)
from benchmark.reporting.framework.base import (
    BENCHMARK_METADATA_ATTR,
    BenchmarkComponentMetadata,
    BenchmarkCustomizationPolicy,
)
from benchmark.queue import BENCHMARK_CASES, EvalQueueRegistry
from benchmark.workers.finalize_trigger import tick_and_maybe_finalize

if TYPE_CHECKING:
    from benchmark.libs.agenthost.app import AppRuntime

    from benchmark.reporting.framework.base import BaseBenchmarkServices
    from benchmark.workers.external_runner import ExternalBenchmarkRunner

from benchmark.workers.external_runner import profile_from_external_request

logger = get_logger("agent_eval.api.benchmark")

router = APIRouter(prefix="/v1/benchmarks", tags=["benchmark"])


#: BullMQ EvalQueueRegistry 依赖；init 失败时返回 503 而非 500。
def _require_eval_queue_manager(request: Request) -> EvalQueueRegistry:
    manager = getattr(request.app.state, "eval_queue_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="BullMQ eval queue not available (init failed or not configured)",
        )
    return manager


EvalQueueRegistryDep = Annotated[
    EvalQueueRegistry,
    Depends(_require_eval_queue_manager),
]


# ════════════════════════════════════════════════════════════════════════════
# Request / Response models
# ════════════════════════════════════════════════════════════════════════════


class BenchmarkRunRequest(BaseModel):
    """POST /v1/benchmarks/{name}/run body.

    Defaults match ``BenchmarkConfig``; lifespan may override via
    ``app.state.benchmark_config`` when the request omits fields.
    """

    sample_size: int | None = Field(
        default=None,
        ge=1,
        description="Max items to evaluate; null = use benchmark default",
    )
    sample_seed: int = Field(default=42, description="Random seed for sampling")
    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        description="Max concurrent evaluations per criterion; null = YAML default",
    )
    parallel_evaluators: bool | None = Field(
        default=None,
        description="Run all criteria in parallel; null = YAML default",
    )
    triage_mode: TriageMode = Field(
        default=TriageMode.FUSED,
        description="When to trigger triage: fused / per_eval / none",
    )
    generate_report: bool = Field(
        default=True,
        description="Generate HTML report via webdev-agent",
    )
    dataset_name: str | None = Field(
        default=None,
        description="Override Langfuse dataset name; null = use benchmark default",
    )
    project_id: str | None = Field(
        default=None,
        description="Override Langfuse project ID; null = use benchmark default",
    )
    # ── 配置化运行时可选扩展（§5.1；缺省 = meta.rubric + 现网行为，完全兼容）──
    rubric_ref: str | None = Field(
        default=None,
        description="Rubric 类引用 module:Class（缺省 = component meta.rubric）",
    )
    parse_interceptor_ref: str | None = Field(
        default=None,
        description="数据 parse 拦截器引用 module:Class（缺省 = Rubric 类挂载值）",
    )
    judge: JudgeProfileConfig | None = Field(
        default=None,
        description="judge 配置化（execution + target）；缺省 = 本地 sync 评测",
    )
    attribution: AttributionProfileConfig | None = Field(
        default=None,
        description="归因配置化；缺省 = none（push 现网无 per-item 归因）",
    )


class InlineDatasetItem(BaseModel):
    """单条评测数据（对标 Langfuse ``DatasetItem`` 的最小属性集）。"""

    id: str = Field(description="Case 唯一标识 → trace 记录 dataset_item_id")
    input: JsonValue = Field(description="用户输入 / query")
    output: JsonValue = Field(
        default=None,
        description="Agent 实际输出（待评回复）→ EvaluatorRequest.output",
    )
    expected_output: JsonValue = Field(
        default=None,
        description="Ground truth / 参考答案 → EvaluatorRequest.expected_output",
    )
    source_trace_id: str | None = Field(
        default=None,
        description="关联的 Langfuse trace ID（可选）",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="业务元数据（react_rounds / turns 等）",
    )


class BenchmarkInlineRunRequest(BaseModel):
    """POST /v1/benchmarks/run-inline body.

    调用方必须显式指定 benchmark_services / dataset_name / agent_name /
    project_id，不依赖 URL 路径参数或 @benchmark_component 默认值。
    """

    # ── 必填：benchmark services 选择 + 数据标识 ──────────────────────
    benchmark_services: str = Field(
        description=(
            "Benchmark services 注册名称，决定调用哪个 "
            "BaseBenchmarkServices 子类（rubric + aggregation + "
            "_build_evaluator_request 逻辑）。"
            "对应 @benchmark_component(name=...) 的 name。"
        ),
    )
    dataset_name: str = Field(
        description=(
            "数据集名称标识（调用方自定义）。"
            "写入 benchmark_runs.dataset_name，用于区分不同批次的内联评测。"
            "不用于 Langfuse fetch（内联模式跳过）。"
        ),
    )
    agent_name: str = Field(
        description=(
            "被评测的 Agent / 服务名称。"
            "写入 benchmark_runs.service_name 和 trace_pipeline_records.agent_name。"
            "覆盖 @benchmark_component 中的 service_name 默认值。"
        ),
    )
    project_id: str = Field(
        description=(
            "Langfuse 项目 ID。用于 evaluator LLM 调用的 trace 上报凭证。"
            "写入 benchmark_runs.project_id。"
        ),
    )

    # ── 必填：评测数据 ────────────────────────────────────────────────
    items: list[InlineDatasetItem] = Field(
        min_length=1,
        description="Inline dataset items（跳过 Langfuse fetch）",
    )

    # ── 可选：执行参数 ────────────────────────────────────────────────
    max_concurrency: int | None = Field(default=None, ge=1)
    parallel_evaluators: bool | None = Field(default=None)
    triage_mode: TriageMode = Field(default=TriageMode.FUSED)
    generate_report: bool = Field(default=True)

    # ── 可选：配置化运行时（§5.1；缺省 = meta.rubric + 现网行为）───────
    rubric_ref: str | None = Field(
        default=None,
        description="Rubric 类引用 module:Class（缺省 = component meta.rubric）",
    )
    parse_interceptor_ref: str | None = Field(
        default=None,
        description="数据 parse 拦截器引用 module:Class（缺省 = Rubric 类挂载值）",
    )
    judge: JudgeProfileConfig | None = Field(
        default=None,
        description="judge 配置化（execution + target）；缺省 = 本地 sync 评测",
    )
    attribution: AttributionProfileConfig | None = Field(
        default=None,
        description="归因配置化；缺省 = none（push 现网无 per-item 归因）",
    )


class BenchmarkStreamRunRequest(BaseModel):
    """POST /v1/benchmarks/runs body — 流式 ingest 创建 run（ACCEPTING）。"""

    # ── 必填：与内联接口一致的身份参数 ────────────────────────────────
    benchmark_services: str = Field(
        description="Benchmark services 注册名称（同 run-inline）",
    )
    dataset_name: str = Field(
        description="数据集名称标识（调用方自定义）",
    )
    agent_name: str = Field(
        description="被评测的 Agent / 服务名称",
    )
    project_id: str = Field(
        description="Langfuse 项目 ID",
    )

    # ── 可选：执行参数（finalize 时从 config_json 恢复）────────────────
    max_concurrency: int | None = Field(default=None, ge=1)
    parallel_evaluators: bool | None = Field(default=None)
    triage_mode: TriageMode = Field(default=TriageMode.FUSED)
    generate_report: bool = Field(default=True)


class StreamItemsRequest(BaseModel):
    """POST /v1/benchmarks/runs/{run_id}/items body（与 InlineDatasetItem 同构）。"""

    items: list[InlineDatasetItem] = Field(min_length=1, max_length=100)


# ════════════════════════════════════════════════════════════════════════════
# Dependencies
# ════════════════════════════════════════════════════════════════════════════


def get_benchmark_service(request: Request) -> BenchmarkService:
    """Resolve lifespan-scoped BenchmarkService from app.state."""
    service = getattr(request.app.state, "benchmark_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="BenchmarkService not initialized")
    return service


def get_benchmark_runner(request: Request) -> BenchmarkRunner:
    """Resolve lifespan-scoped BenchmarkRunner from app.state."""
    runner = getattr(request.app.state, "benchmark_runner", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="BenchmarkRunner not initialized")
    return runner


def get_runtime(request: Request) -> "AppRuntime":
    """Resolve the agentgraph AppRuntime from app.state."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not initialized")
    return runtime


BenchmarkServiceDep = Annotated[BenchmarkService, Depends(get_benchmark_service)]
BenchmarkRunnerDep = Annotated[BenchmarkRunner, Depends(get_benchmark_runner)]
RuntimeDep = Annotated["AppRuntime", Depends(get_runtime)]


@dataclass
class BenchmarkRunDeps:
    """``POST /{name}/run`` 聚合依赖（create_run → fetch → enqueue 链路）。"""

    benchmark_service: BenchmarkService
    runner: BenchmarkRunner
    runtime: "AppRuntime"
    queue_manager: EvalQueueRegistry


def get_benchmark_run_deps(
    benchmark_service: BenchmarkServiceDep,
    runner: BenchmarkRunnerDep,
    runtime: RuntimeDep,
    queue_manager: EvalQueueRegistryDep,
) -> BenchmarkRunDeps:
    """FastAPI 子依赖聚合：run 路由单参注入，签名保持 ≤5。"""
    return BenchmarkRunDeps(
        benchmark_service=benchmark_service,
        runner=runner,
        runtime=runtime,
        queue_manager=queue_manager,
    )


BenchmarkRunDepsDep = Annotated[BenchmarkRunDeps, Depends(get_benchmark_run_deps)]


def _get_staging_buffer(request: Request) -> StagingBuffer:
    """Resolve lifespan-scoped StagingBuffer from app.state."""
    buf = getattr(request.app.state, "staging_buffer", None)
    if buf is None:
        raise HTTPException(
            status_code=503,
            detail="staging buffer not wired (lifespan)",
        )
    return buf


StagingBufferDep = Annotated[StagingBuffer, Depends(_get_staging_buffer)]


# ════════════════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════════════════


@router.get("/rubric-classes")
async def list_rubric_classes() -> list[dict[str, Any]]:
    """只读列出 ``RUBRIC_CLASS_REFS`` 白名单（D2，供 RemoteConfig 配置填写核对）。

    返回 ``[{alias, class_ref, criteria[], aggregation_method, pass_threshold}]``；
    ``component_rubric`` 未挂 Rubric 类（criteria 由组件元数据承载），
    criteria 返回空列表。新 Rubric 类接入流程 = 登记别名 + CI 校验。
    """
    from benchmark.plugins.class_loader import load_plugin_class  # noqa: PLC0415

    entries: list[dict[str, Any]] = []
    for alias, class_ref in sorted(RUBRIC_CLASS_REFS.items()):
        entry: dict[str, Any] = {
            "alias": alias,
            "class_ref": class_ref,
            "criteria": [],
            "aggregation_method": None,
            "pass_threshold": None,
        }
        spec = resolve_rubric_class(alias)
        if spec.class_ref is not None:
            cls = load_plugin_class(spec.class_ref, what="rubric_class")
            entry["criteria"] = [
                {
                    "id": c.id,
                    "name": c.name,
                    "weight": c.weight,
                    "evaluator_ref": c.evaluator_ref,
                    "description": c.description,
                }
                for c in cls.criteria()
            ]
            entry["aggregation_method"] = cls.aggregation_method
            entry["pass_threshold"] = cls.pass_threshold
        entries.append(entry)
    return entries


async def _assemble_agent_definitions() -> list[tuple[BenchmarkDefinition, str]]:
    """组装统一 agent 视图（每行带来源标记，供接口回退与 DB 同步共用）。

    - 静态源：``BENCHMARK_SERVICES_CLASSES``（组件开关放行）→
      :func:`_definition_from_metadata`，来源 ``static``；
    - 动态源：``list_agent_documents()`` → :func:`_definition_from_document`，
      来源 ``dynamic``；静态/动态同名时静态优先；
    - evaluator 源：全量已配置 evaluator 按 agent name 聚合，未被覆盖的
      → :func:`_definition_from_evaluator`，来源 ``evaluator``。
    """
    definitions: dict[str, BenchmarkDefinition] = {}
    sources: dict[str, str] = {}
    for cls in BENCHMARK_SERVICES_CLASSES:
        meta = cls.get_metadata()
        if not is_benchmark_component_enabled(meta.name):
            continue
        definitions[meta.name] = _definition_from_metadata(meta, cls)
        sources[meta.name] = "static"
    for document in await list_agent_documents():
        if document.name in definitions:
            continue
        definitions[document.name] = _definition_from_document(document)
        sources[document.name] = "dynamic"
    covered = {item.agent_name for item in definitions.values()}
    for agent_name, components in configured_evaluators().items():
        if agent_name in covered or agent_name in definitions:
            continue
        definitions[agent_name] = _definition_from_evaluator(agent_name, components)
        sources[agent_name] = "evaluator"
    return [
        (definitions[name], sources[name])
        for name in sorted(definitions, key=lambda item: definitions[item].name)
    ]


def sync_component_ownership_from_registry() -> None:
    """把静态 ``BENCHMARK_SERVICES_CLASSES`` 的 ``agent_scopes`` 刷入 L1 overlay。

    与开关无关：未启用组件也声明归属，避免「关掉开关后白名单空洞」；
    目录列表仍由 :func:`is_benchmark_component_enabled` 另行过滤。
    """
    from benchmark.domain.agent_scope_ownership import (  # noqa: PLC0415
        refresh_component_scope_overlay,
    )

    declarations = [
        (cls.get_metadata().name, cls.get_metadata().agent_scopes)
        for cls in BENCHMARK_SERVICES_CLASSES
    ]
    refresh_component_scope_overlay(declarations)


async def sync_agent_view_catalog() -> None:
    """把当前组装结果物化进 DB（启动 / 页面保存后调用；失败只记日志）。"""
    from benchmark.domain.agent_view_store import sync_agent_views  # noqa: PLC0415

    sync_component_ownership_from_registry()
    try:
        await sync_agent_views(await _assemble_agent_definitions())
    except Exception:
        logger.exception("agent view catalog sync failed")


@router.get("", response_model=list[BenchmarkDefinition])
async def list_benchmarks(
    agent_scope: str | None = Query(default=None, alias="agentScope"),
) -> list[BenchmarkDefinition]:
    """List benchmark agents as a unified view (agent-form exposure).

    优先读 DB 物化行（``kind=agent_view``，启动与页面保存时同步）；
    无行时回退内存组装（组装入口见 :func:`_assemble_agent_definitions`）。
    每行 ``agent_name`` 为对外统一名（benchmark name / evaluator_name
    归一，规则在 ``agent_catalog.canonical_agent_name``）；``name``
    保留执行侧 benchmark name 供触发兼容。

    可选 ``agentScope``：按 L1 归属白名单过滤；缺省返回全量（运维脚本兼容）。
    """
    from benchmark.domain.agent_scope_ownership import get_ownership_policy  # noqa: PLC0415
    from benchmark.domain.agent_view_store import list_agent_view_definitions  # noqa: PLC0415

    stored = await list_agent_view_definitions()
    definitions = stored or [definition for definition, _ in await _assemble_agent_definitions()]
    scope = agent_scope.strip() if isinstance(agent_scope, str) else ""
    if not scope:
        return definitions
    policy = get_ownership_policy()
    return [item for item in definitions if policy.allows(scope, item.name)]


#: 缺省 Rubric 稳定别名（与 ``RUBRIC_CLASS_REFS`` / 页面 document 口径一致）
_COMPONENT_RUBRIC_ALIAS = "component_rubric"
_DEFAULT_RUBRIC_DESCRIPTION = "未声明独立 Rubric；使用组件默认评分口径。"


def _capability_views(bindings: list[CapabilityBindingView]) -> list[AgentCapabilityView]:
    """能力绑定 dataclass → API 视图模型，统一补齐缺省 report。"""
    return [
        AgentCapabilityView(
            pipeline=item.pipeline,
            kind=item.kind,
            execution=item.execution,
            provider=item.provider,
            description=item.description,
            alias=item.alias,
            class_ref=item.class_ref,
            note=item.note,
        )
        for item in declared_capabilities(bindings)
    ]


def _evaluator_refs(rubric: list[Any]) -> list[str]:
    """criteria 的 evaluator_ref 去重列表（实时流与 judge 组件的路由键）。"""
    refs: list[str] = []
    for criterion in rubric:
        ref = (getattr(criterion, "evaluator_ref", None) or "").strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _rubric_description(
    rubric_cls: type | None,
    criteria: list[dict[str, Any]],
    aggregation_method: str,
) -> str:
    """动态 agent 的 rubric 评分标准说明：优先 rubric 类 docstring 首段，
    无类可加载时按 criteria 维度名与聚合方式拼装。"""
    doc = (getattr(rubric_cls, "__doc__", None) or "").strip()
    if doc:
        return doc.split("\n\n")[0].strip()
    if criteria:
        names = "、".join(str(c.get("name")) for c in criteria)
        return f"{len(criteria)} 个维度加权（{names}），聚合方式 {aggregation_method}"
    return ""


def _criteria_views(rubric: list[Any]) -> list[dict[str, Any]]:
    """rubric criteria → 目录视图 dict（静态与动态 agent 共用）。"""
    return [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "weight": c.weight,
            "evaluator_ref": c.evaluator_ref,
        }
        for c in rubric
    ]


def _definition_from_metadata(
    meta: BenchmarkComponentMetadata,
    services_cls: type | None = None,
) -> BenchmarkDefinition:
    """静态组件元数据 → BenchmarkDefinition（能力链读组件类声明，可含双链）。"""
    declared = getattr(services_cls, "benchmark_capabilities", None) if services_cls else None
    capabilities = capabilities_or_default(declared)
    # 静态 agent 级描述：实现类 benchmark_description 兑底页面展示。
    agent_description = (
        (getattr(services_cls, "benchmark_description", "") if services_cls else "")
        or _rubric_description(services_cls, [], "")
        or None
    )
    rubric_description = (
        (getattr(services_cls, "benchmark_rubric_description", "") if services_cls else "")
        or _rubric_description(
            None,
            _criteria_views(list(meta.rubric)),
            meta.aggregation.method,
        )
        or _DEFAULT_RUBRIC_DESCRIPTION
    )
    return BenchmarkDefinition(
        name=meta.name,
        agent_name=canonical_agent_name(meta.name, meta.service_name),
        service_name=meta.service_name,
        dataset_name=meta.dataset.dataset_name,
        project_id=meta.dataset.project_id,
        criteria=_criteria_views(list(meta.rubric)),
        aggregation_method=meta.aggregation.method,
        attribution_services=meta.attribution_services,
        description=agent_description,
        evaluators=_evaluator_refs(meta.rubric),
        rubric_class=meta.rubric_class or _COMPONENT_RUBRIC_ALIAS,
        rubric_description=rubric_description,
        capabilities=_capability_views(capabilities),
        # 定制策略统一走框架登记表（登记 → 组件自声明 → 默认），与保存/触发
        # 校验同源；不能只读 meta.customization，否则框架登记的 policy
        # （如 example_benchmark 的 prompt/skill 槽位）在目录里不可见。
        customization=customization_policy_view(
            customization_policy_for_name(meta.name), capabilities
        ),
    )


def _definition_from_document(document: AgentDocument) -> BenchmarkDefinition:
    """页面注册文档 → BenchmarkDefinition（rubric 取文档 rubricClass，加载失败容错为空）。"""
    criteria: list[dict[str, Any]] = []
    aggregation_method = "weighted_avg"
    rubric_cls = None
    rubric: list[Any] = []
    rubric_class = (document.rubric_class or "").strip()
    if rubric_class:
        try:
            rubric, aggregation, rubric_cls = load_rubric_class_ref(rubric_class)
        except ValueError:
            # 目录展示容错：rubricClass 指向的类被重构移除时降级为空 rubric
            # （触发评测时 fail-fast 400，不阻断 agent 目录）。
            logger.warning(
                "benchmark agent %s rubric_class %s unavailable, listed with empty rubric",
                document.name,
                rubric_class,
            )
        else:
            criteria = _criteria_views(rubric)
            aggregation_method = aggregation.method
    return BenchmarkDefinition(
        name=document.name,
        agent_name=canonical_agent_name(document.name, document.name),
        service_name=document.name,
        dataset_name="",
        project_id="",
        criteria=criteria,
        aggregation_method=aggregation_method,
        attribution_services=[],
        description=document.description or None,
        owner=document.owner,
        registered_at=document.registered_at,
        evaluators=_evaluator_refs(rubric),
        rubric_class=rubric_class or _COMPONENT_RUBRIC_ALIAS,
        rubric_description=(
            _rubric_description(rubric_cls, criteria, aggregation_method)
            or _DEFAULT_RUBRIC_DESCRIPTION
        ),
        # 页面注册的动态 agent 承接离线数据集评测 → 批量链；实时流 agent
        # 由服务端内置（静态注册），不混挂两条链。
        capabilities=_capability_views(default_batch_capabilities(rubric_cls)),
        # 动态 agent 的定制策略：同名静态组件存在则继承其 policy，
        # 否则四格全关（纯自定义名不支持业务定制）。
        customization=customization_policy_view(
            customization_policy_for_name(document.name),
            default_batch_capabilities(rubric_cls),
        ),
    )


def _definition_from_evaluator(
    agent_name: str,
    components: EvaluatorComponents,
) -> BenchmarkDefinition:
    """已配置 evaluator（无对应 benchmark）→ agent 行（只有 evaluator 能力）。"""
    return BenchmarkDefinition(
        name=agent_name,
        agent_name=agent_name,
        service_name=agent_name,
        dataset_name="",
        project_id="",
        criteria=[],
        aggregation_method="",
        attribution_services=[agent_name] if components.has_attribution else [],
        description=components.judge_description or f"{agent_name} 实时流评测 agent",
        evaluators=[agent_name],
        rubric_class=_COMPONENT_RUBRIC_ALIAS,
        rubric_description=_DEFAULT_RUBRIC_DESCRIPTION,
        capabilities=_capability_views(evaluator_capabilities(agent_name, components)),
        # evaluator-only agent 无 prompt/skill 定制，但归因开关默认放行：
        # 注册关闭后实时流按 attribution_execution=none 跳过归因。
        customization=customization_policy_view(
            BenchmarkCustomizationPolicy(),
            evaluator_capabilities(agent_name, components),
        ),
    )


_PayloadT = TypeVar("_PayloadT", bound=BaseModel)


def _apply_benchmark_config_defaults(
    payload: _PayloadT,
    request: Request,
) -> _PayloadT:
    """Fill null request fields from ``app.state.benchmark_config``.

    适用于所有带 ``max_concurrency`` / ``parallel_evaluators`` 可选字段的
    benchmark 请求模型（duck-typed）。
    """
    cfg = getattr(request.app.state, "benchmark_config", None)
    updates: dict[str, Any] = {}
    if getattr(payload, "max_concurrency", None) is None:
        updates["max_concurrency"] = getattr(cfg, "default_max_concurrency", None) or 1
    if payload.parallel_evaluators is None:
        updates["parallel_evaluators"] = bool(
            getattr(cfg, "default_parallel_evaluators", True),
        )
    if not updates:
        return payload
    return payload.model_copy(update=updates)


async def _resolve_benchmark_services(
    runtime: "AppRuntime",
    name: str,
) -> "BaseBenchmarkServices":
    """按注册名解析 benchmark 组件；静态未命中时回落页面注册文档动态注册。

    解析顺序（与“页面配置为准、RemoteConfig 组件配置下线”的演进方向一致）：
    1. ``agent_manager`` 静态注册命中且组件开关放行 → 直接返回；
    2. 静态被开关拦 / 未命中：按页面注册的 benchmark-agent 文档动态注册
       （业务自定义名，rubric 取文档 rubricClass）；文档未注册该名时若静态
       实例存在仍返回静态（兼容仅关开关、未页面注册的过渡期）；
    3. 静态与文档均未命中 → 404（带静态可用列表，仅提示不影响动态名）。
    """
    if name in runtime.agent_manager.list():
        if is_benchmark_component_enabled(name):
            benchmark_services = runtime.agent_manager.get(name)
            if hasattr(benchmark_services, BENCHMARK_METADATA_ATTR):
                return benchmark_services
        else:
            # 静态名被开关拦：若页面也未注册该名，过渡期仍允许返回静态实例
            # （开关只对“未注册名”拒新，已注册组件保持可触发）。
            benchmark_services = runtime.agent_manager.get(name)
            if hasattr(benchmark_services, BENCHMARK_METADATA_ATTR):
                return benchmark_services
    try:
        return await ensure_dynamic_benchmark_component(name, runtime)
    except KeyError:
        available = [
            cls.get_metadata().name
            for cls in BENCHMARK_SERVICES_CLASSES
            if is_benchmark_component_enabled(cls.get_metadata().name)
        ]
        raise HTTPException(
            status_code=404,
            detail=(f"Benchmark services not found: {name}. Available: {available}"),
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _rubric_config_json(meta: BenchmarkComponentMetadata) -> dict[str, Any]:
    """rubric / aggregation 的 config_json 片段（各创建端点共享）。"""
    return {
        "rubric": [
            {
                "id": c.id,
                "name": c.name,
                "weight": c.weight,
                "evaluator_ref": c.evaluator_ref,
                "description": c.description,
            }
            for c in meta.rubric
        ],
        "aggregation": {
            "method": meta.aggregation.method,
            "pass_threshold": meta.aggregation.pass_threshold,
            "fusion_fn_ref": meta.aggregation.fusion_fn_ref,
        },
    }


def _item_field(item: object, name: str, default: object = None) -> object:
    """Get a field from an item that may be a dict or object."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


# ════════════════════════════════════════════════════════════════════════════
# 配置化运行时（§5.2 ProfileResolver 接入）
# ════════════════════════════════════════════════════════════════════════════


def _get_profile_resolver(request: Request) -> ProfileResolver:
    """Resolve lifespan 装配的 ProfileResolver（external runner 持有；缺省无 factory）。"""
    runner = getattr(request.app.state, "external_benchmark_runner", None)
    resolver = getattr(runner, "profile_resolver", None) if runner is not None else None
    if resolver is not None:
        return resolver
    return ProfileResolver(client_factory=None)


def _resolve_push_profile(
    request: Request,
    *,
    benchmark_name: str,
    meta: BenchmarkComponentMetadata,
    payload: BenchmarkRunRequest | BenchmarkInlineRunRequest,
) -> ResolvedProfile:
    """push 入口 profile 解析（非法引用 → 400，不进流水线）。

    push 缺省归因 = ``none``（现网 push 无 per-item 归因，不传新字段行为不变）；
    pull 场景的缺省（example sync）由 pull 侧 v1 映射承担，两条入口解耦。
    judge 缺省 = ``sync``（现网 push 恒为本地同步评测；缺省推导的
    async 语义只属于 pull 入口，不能翻转到 push）。
    """
    profile = BenchmarkRuntimeProfile(
        benchmark_name=benchmark_name,
        rubric_ref=payload.rubric_ref,
        parse_interceptor_ref=payload.parse_interceptor_ref,
        judge=payload.judge or JudgeProfileConfig(execution="sync"),
        attribution=payload.attribution
        or AttributionProfileConfig(execution="none", provider="none"),
    )
    try:
        return _get_profile_resolver(request).resolve(profile, meta)
    except ProfileResolutionError as exc:
        logger.warning(
            "profile.resolve_failed benchmark=%s reason=%s err=%s",
            benchmark_name,
            exc.reason,
            exc,
        )
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "reason": exc.reason},
        ) from exc


def _push_uses_external_count_model(resolved: ResolvedProfile) -> bool:
    """judge / 归因任一 async → external 计数器模型（§8.2 finalize 路由同源）。"""
    return resolved.judge.execution == "async" or resolved.attribution.execution == "async"


@dataclass(frozen=True)
class _PreparedPushRun:
    """push 入口 profile 解析 + 配置组装产物（PLR0915 收敛，两入口共用）。"""

    resolved: ResolvedProfile
    config_json: dict[str, Any]
    external_count_model: bool
    is_fused: bool


@dataclass(frozen=True)
class _PushRunPrepare:
    """``_prepare_push_run`` 入参（FUNC-PARAMS-001）。"""

    benchmark_name: str
    meta: BenchmarkComponentMetadata
    payload: "BenchmarkRunRequest | BenchmarkInlineRunRequest"
    extra_config: dict[str, Any]


@dataclass(frozen=True)
class _EmptySubmitFinalize:
    """``_tick_finalize_for_empty_submit`` 入参（FUNC-PARAMS-001）。"""

    run_id: str
    enqueued: int
    total_raw: int
    external_count_model: bool


def _prepare_push_run(
    request: Request,
    prepare: _PushRunPrepare,
) -> _PreparedPushRun:
    """push 两入口共用的 profile 解析 + config_json 组装（§5.1/§5.2）。

    - Resolver 解析（非法引用 → 400，不进流水线）；
    - per_eval + async 组合 fail-fast（per-criterion 分支不产生
      judge_received / 归因路由，external 计数模型会永不收敛）；
    - config_json = Resolver 快照 + 入口自定义键 + 统一队列标记（§8.3）。
    """
    benchmark_name = prepare.benchmark_name
    meta = prepare.meta
    payload = prepare.payload
    extra_config = prepare.extra_config
    resolved = _resolve_push_profile(
        request,
        benchmark_name=benchmark_name,
        meta=meta,
        payload=payload,
    )
    snapshot = resolved.config_snapshot()
    external_count_model = _push_uses_external_count_model(resolved)
    is_fused = payload.triage_mode == TriageMode.FUSED
    if not is_fused and external_count_model:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "triage_mode=per_eval 与 async judge/归因组合暂不支持"
                "（仅 fused 模式支持 async 执行）",
                "reason": "invalid_mode_combination",
            },
        )
    config_json: dict[str, Any] = {
        "rubric": snapshot["rubric"],
        "aggregation": snapshot["aggregation"],
        "rubric_ref": resolved.rubric_ref,
        "parse_interceptor_ref": resolved.parse_interceptor_ref,
        # 评分规则链接（Rubric 类 rule_link_url）：报告槽位填充来源
        "rule_link_url": resolved.rule_link_url,
        "judge": snapshot["judge"],
        "attribution": snapshot["attribution"],
        # §8.3 统一队列标记：push case 恒驻留 benchmark-cases/running（回调释槽路由依据）
        "queue_pair": "benchmark",
        **extra_config,
    }
    return _PreparedPushRun(
        resolved=resolved,
        config_json=config_json,
        external_count_model=external_count_model,
        is_fused=is_fused,
    )


def _to_raw_item(
    item: object,
    run_id: str,
    seq: int,
    *,
    seen_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    """push 侧 raw item 归一（全量 KV 语义，§7.1）：请求体 / Langfuse fetch 原样搬运。

    身份归一（D5）：``case_id > id > task_id`` 列优先（与 OSS loader /
    dataset ingest 共用 :func:`normalize_case_identity`）；``item_id`` 恒为
    ``{run_id}:{seq}``（技术定位键），不再沿用业务 ``id`` / ``item_id`` 列。
    """
    identity = normalize_case_identity(
        {
            "case_id": _item_field(item, "case_id"),
            "id": _item_field(item, "id"),
            "task_id": _item_field(item, "task_id"),
        },
        CaseIdentityScope(run_id=run_id, seq=seq, seen_case_ids=seen_case_ids),
    )
    return {
        "id": identity.id,
        "item_id": identity.item_id,
        "input": _item_field(item, "input", None),
        "output": _item_field(item, "output", None),
        "expected_output": _item_field(item, "expected_output", None),
        "source_trace_id": _item_field(item, "source_trace_id", None),
        "metadata": _item_field(item, "metadata", None) or {},
        "case_id": identity.case_id,
    }


def _raw_items_with_identity(
    items: list[Any],
    run_id: str,
) -> list[dict[str, Any]]:
    """批量 raw item 归一（D5）：run 内 case_id 唯一性校验逐行生效。"""
    seen_case_ids: set[str] = set()
    return [
        _to_raw_item(item, run_id, seq, seen_case_ids=seen_case_ids)
        for seq, item in enumerate(items)
    ]


async def _tick_finalize_for_empty_submit(
    benchmark_service: BenchmarkService,
    finalize: _EmptySubmitFinalize,
) -> None:
    """全量 parse 失败（enqueued=0）的 finalize 补偿（§7.6/§10）。

    parse 失败 item 已代位推进计数（judge_received / failed_items），
    但无 case 入队 → 无 worker / 回调触发 finalize；此处按计数模型
    补一次 tick，避免 run 永远停在 RUNNING。
    """
    run_id = finalize.run_id
    enqueued = finalize.enqueued
    total_raw = finalize.total_raw
    external_count_model = finalize.external_count_model
    if enqueued != 0 or total_raw == 0:
        return
    logger.warning(
        "benchmark.submit_all_parse_failed run_id=%s total=%d — trigger finalize",
        run_id,
        total_raw,
    )
    await tick_and_maybe_finalize(
        run_id,
        "external" if external_count_model else "benchmark",
        benchmark_run_service=benchmark_service,
    )


@dataclass(frozen=True)
class _ParseFailurePlan:
    """push parse 失败 item 的行模型 / 计数器模式参数（§7.6）。"""

    criteria: list[dict[str, Any]]
    is_fused: bool
    external_count_model: bool


@dataclass(frozen=True)
class _PushItemParseFailed:
    """``_mark_push_item_parse_failed`` 入参（FUNC-PARAMS-001）。"""

    run_id: str
    item: dict[str, Any]
    reason: str
    plan: _ParseFailurePlan


@dataclass(frozen=True)
class _PushRunBgSubmit:
    """``_submit_push_run_bg`` 入参（FUNC-PARAMS-001）。"""

    run_id: str
    name: str
    payload: BenchmarkRunRequest
    meta: BenchmarkComponentMetadata
    prepared: _PreparedPushRun


async def _mark_push_item_parse_failed(
    benchmark_service: BenchmarkService,
    failure: _PushItemParseFailed,
) -> None:
    """push 路径 parse 失败 item：eval_failed stub 落库 + 完成计数推进（§7.6）。

    - benchmark 进度列模型：按行模型写 failed 行 + ``increment_progress(failed_delta)``
      （fused = 1 行；per_eval = 每 criterion 一行）；
    - external 计数器模型（judge/归因 async）：``cas_advance_external_judge_item``
      单事务推进（``judge_received_count`` 自动 +1）。
    """
    run_id = failure.run_id
    item = failure.item
    reason = failure.reason
    plan = failure.plan
    item_id = str(item.get("item_id") or item.get("id") or "")
    if plan.external_count_model:
        await benchmark_service.cas_advance_external_judge_item(
            result=ExternalJudgeItemResult(
                run_id=run_id,
                item_id=item_id,
                failed=True,
                score=0.0,
                judge_json={"error": reason, "reason": reason, "stage": "parse"},
                error_message=reason,
                item_input=item.get("input") if isinstance(item.get("input"), dict) else None,
                item_output=item.get("output") if isinstance(item.get("output"), dict) else None,
            ),
        )
        return
    # 进度列模型：写库下沉 service 层（API 层不直接操作 data_model）
    row_ids = [FUSED_CRITERION_ID] if plan.is_fused else [str(c["id"]) for c in plan.criteria]
    await benchmark_service.write_parse_failed_stubs(
        ParseFailedStubWrite(
            run_id=run_id,
            item_id=item_id,
            reason=reason,
            criterion_ids=row_ids,
            item=item,
        ),
    )


@dataclass(frozen=True)
class BenchmarkCaseEnqueueSpec:
    """benchmark case 入队规格（payload 构建 / BullMQ 入队共用，§6.2）。

    承载 run 维度参数（criteria / 服务与数据集标识 / 分流配置）；
    item 本身逐个传递（Langfuse DatasetItem / pydantic model / dict 均可，
    经 ``_item_field`` 统一访问）。
    """

    run_id: str
    criteria: list[dict[str, Any]]
    benchmark_services_name: str = ""
    service_name: str = ""
    dataset_name: str = ""
    project_id: str | None = None
    triage_mode: str = "fused"
    generate_report: bool = True


def _build_benchmark_case_payload(
    item: object,
    spec: BenchmarkCaseEnqueueSpec,
) -> dict[str, Any]:
    """从 case item 构建 BullMQ job payload（对齐设计文档 §6.2）。

    Items may be Langfuse DatasetItem objects, pydantic models, or dicts
    (from StagingBuffer). Uses ``_item_field`` for uniform access.
    """
    evaluator_names = [c["evaluator_ref"] for c in spec.criteria]
    item_id = str(_item_field(item, "id", "") or _item_field(item, "item_id", "") or "")
    return {
        "eval_type": "benchmark",
        "run_id": spec.run_id,
        "retry_count": 0,
        "first_enqueued_at": int(time.time() * 1000),
        "item_id": item_id,
        "input": _item_field(item, "input", None),
        "output": _item_field(item, "output", None),
        "expected_output": _item_field(item, "expected_output", None),
        "source_trace_id": _item_field(item, "source_trace_id", None),
        "metadata": _item_field(item, "metadata", None) or {},
        "case_id": _item_field(item, "case_id", None) or item_id,
        "evaluator_names": evaluator_names,
        "criteria": spec.criteria,
        "benchmark_services": spec.benchmark_services_name,
        "service_name": spec.service_name,
        "dataset_name": spec.dataset_name,
        "agent_name": spec.service_name,
        "project_id": spec.project_id,
        "triage_mode": spec.triage_mode,
        "generate_report": spec.generate_report,
    }


async def _enqueue_benchmark_cases(
    queue_manager: EvalQueueRegistry,
    items: list[Any],
    spec: BenchmarkCaseEnqueueSpec,
) -> int:
    """逐个 case 入队 ``benchmark-cases`` BullMQ 队列。

    v2.1: ``attempts=1`` + payload ``retry_count=0``（D4/D17）。
    """
    queue = queue_manager.get_queue(BENCHMARK_CASES)
    job_opts = {
        "attempts": 1,
        "removeOnComplete": True,
        "removeOnFail": True,
    }
    count = 0
    for item in items:
        item_id = str(_item_field(item, "id", "") or _item_field(item, "item_id", "") or "")
        payload = _build_benchmark_case_payload(item, spec)
        await queue.add(
            "case",
            payload,
            {"jobId": f"{spec.run_id}:{item_id}", **job_opts},
        )
        count += 1
    return count


async def _submit_push_run_bg(
    deps: BenchmarkRunDeps,
    submit: _PushRunBgSubmit,
) -> None:
    """``POST /{name}/run`` 后台任务：fetch Langfuse items → 统一提交管线入队。

    PLR0915 收敛：自 ``run_benchmark`` 提取为模块级函数（嵌套闭包语句
    会被 ruff 计入外层函数）。
    """
    run_id = submit.run_id
    name = submit.name
    payload = submit.payload
    meta = submit.meta
    prepared = submit.prepared
    benchmark_service = deps.benchmark_service
    queue_manager = deps.queue_manager
    runner = deps.runner
    resolved = prepared.resolved
    config_json = prepared.config_json
    external_count_model = prepared.external_count_model
    is_fused = prepared.is_fused
    effective_dataset_name = payload.dataset_name or meta.dataset.dataset_name
    effective_project_id = payload.project_id or meta.dataset.project_id
    criteria = [{"id": c.id, "evaluator_ref": c.evaluator_ref} for c in resolved.rubric]

    try:
        logger.info(
            "benchmark.job_started run_id=%s name=%s",
            run_id,
            meta.name,
        )
        # 1. Fetch Langfuse items (in-memory, no PG staging)
        items = await runner.fetch_items(
            FetchItemsSpec(
                dataset_name=effective_dataset_name,
                project_id=effective_project_id,
                sample_size=payload.sample_size or meta.dataset.sample_size,
                sample_seed=payload.sample_seed,
            ),
        )
        # 2. Update to RUNNING + 落定 total_samples/total_evaluators
        await benchmark_service.update_run_status(
            run_id,
            BenchmarkStatus.RUNNING,
            total_samples=len(items),
            total_evaluators=len(resolved.rubric),
            started_at=datetime.now(UTC),
        )
        # external 计数器模型（judge/归因 async）：expected 对齐 raw 总数，
        # parse 失败 item 经失败 stub 代位推进 received（计数收敛）
        if external_count_model:
            await benchmark_service.merge_config_json(
                run_id,
                {"judge_expected_count": len(items)},
            )
        # 3. 统一提交管线（§7.2 拦截点必经）→ Enqueue cases via BullMQ
        spec = BenchmarkCaseEnqueueSpec(
            run_id=run_id,
            criteria=criteria,
            benchmark_services_name=name,
            service_name=meta.service_name,
            dataset_name=effective_dataset_name,
            project_id=effective_project_id,
            triage_mode=payload.triage_mode.value,
            generate_report=payload.generate_report,
        )
        summary = await submit_prepared_items(
            SubmitPipelineSpec(
                run_id=run_id,
                resolved=resolved,
                raw_items=_raw_items_with_identity(items, run_id),
                enqueue_items=lambda parsed: _enqueue_benchmark_cases(
                    queue_manager,
                    parsed,
                    spec,
                ),
                mark_item_failed=lambda item, reason: _mark_push_item_parse_failed(
                    benchmark_service,
                    _PushItemParseFailed(
                        run_id=run_id,
                        item=item,
                        reason=reason,
                        plan=_ParseFailurePlan(
                            criteria=criteria,
                            is_fused=is_fused,
                            external_count_model=external_count_model,
                        ),
                    ),
                ),
                snapshot=config_json,
            ),
        )
        # 全量 parse 失败（enqueued=0）→ 无 case/回调触发 finalize，补偿一次
        await _tick_finalize_for_empty_submit(
            benchmark_service,
            _EmptySubmitFinalize(
                run_id=run_id,
                enqueued=summary.enqueued,
                total_raw=summary.total_raw,
                external_count_model=external_count_model,
            ),
        )
    except asyncio.CancelledError:
        logger.warning("benchmark.job_cancelled run_id=%s", run_id)
        try:
            await benchmark_service.fail_run(
                run_id=run_id,
                reason="shutdown: task cancelled",
            )
        except Exception:
            logger.exception(
                "benchmark.cancel_fail_run_failed run_id=%s",
                run_id,
            )
        raise
    except Exception as exc:
        logger.exception("benchmark.async_run_failed run_id=%s", run_id)
        try:
            await benchmark_service.fail_run(
                run_id=run_id,
                reason=f"async run failed: {exc}",
            )
        except Exception:
            logger.exception(
                "benchmark.async_fail_run_failed run_id=%s",
                run_id,
            )


@router.post("/{name}/run", response_model=None)
async def run_benchmark(
    name: str,
    payload: BenchmarkRunRequest,
    request: Request,
    deps: BenchmarkRunDepsDep,
) -> JSONResponse:
    """Accept a benchmark run asynchronously (always 202, BullMQ).

    Persists ``benchmark_runs``, schedules background task to fetch Langfuse
    items → enqueue cases via BullMQ ``benchmark-cases`` queue.
    Returns ``run_id`` immediately.
    """
    raise_if_benchmark_disabled()
    payload = _apply_benchmark_config_defaults(payload, request)
    benchmark_service = deps.benchmark_service
    runtime = deps.runtime

    # 静态注册优先；未命中时按页面注册的 benchmark-agent 文档动态注册
    # （业务自定义名称，rubric 取文档 rubricClass）。
    try:
        benchmark_services = await ensure_dynamic_benchmark_component(name, runtime)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark services not found: {name}",
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not hasattr(benchmark_services, BENCHMARK_METADATA_ATTR):
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not a benchmark services component",
        )

    meta = benchmark_services.get_metadata()
    effective_dataset_name = payload.dataset_name or meta.dataset.dataset_name
    effective_project_id = payload.project_id or meta.dataset.project_id

    # 配置化运行时（§5.1/§5.2）：profile 解析 + config_json 组装（非法引用 → 400）
    prepared = _prepare_push_run(
        request,
        _PushRunPrepare(
            benchmark_name=name,
            meta=meta,
            payload=payload,
            extra_config={
                "sample_size": payload.sample_size or meta.dataset.sample_size,
                "sample_seed": payload.sample_seed,
                "benchmark_services": name,
                "max_concurrency": payload.max_concurrency,
                "parallel_evaluators": payload.parallel_evaluators,
                "triage_mode": payload.triage_mode.value,
                "generate_report": payload.generate_report,
            },
        ),
    )
    resolved = prepared.resolved
    config_json = prepared.config_json

    try:
        run_id = await benchmark_service.create_run(
            BenchmarkRunSpec(
                benchmark_name=meta.name,
                service_name=meta.service_name,
                dataset_name=effective_dataset_name,
                project_id=effective_project_id,
                total_evaluators=len(resolved.rubric),
                config_json=config_json,
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "hint": (
                    "Ensure PG has benchmark_runs (sql/schema.sql) and "
                    "column service_name (not business_agent_name)."
                ),
            },
        ) from exc

    asyncio.create_task(
        _submit_push_run_bg(
            deps,
            _PushRunBgSubmit(
                run_id=run_id,
                name=name,
                payload=payload,
                meta=meta,
                prepared=prepared,
            ),
        ),
    )

    poll_url = f"/v1/benchmarks/runs/{run_id}"
    return JSONResponse(
        status_code=202,
        content={
            "run_id": run_id,
            "benchmark_name": meta.name,
            "status": "initializing",
            "poll_url": poll_url,
            "message": (
                "Benchmark accepted (BullMQ: fetch → enqueue). "
                f"Client may disconnect; agent-eval continues in background. "
                f"Optional poll: GET {poll_url}"
            ),
        },
    )


@router.post("/{name}/run-pipeline", response_model=None)
async def run_pipeline_benchmark(
    name: str,
    payload: PipelineBenchmarkRequest,
    benchmark_service: BenchmarkServiceDep,
    runner: BenchmarkRunnerDep,
    runtime: RuntimeDep,
) -> JSONResponse:
    """Accept a pipeline benchmark run asynchronously (always 202).

    Scans Kafka-sourced ``trace_pipeline_records`` by date range and
    aggregates existing judge scores — no LLM, no Langfuse dataset fetch.
    """
    raise_if_benchmark_disabled()
    if payload.date_end <= payload.date_start:
        raise HTTPException(
            status_code=400,
            detail="date_end must be after date_start",
        )

    # 与 manual-runs 同语义：允许页面注册文档动态注册的组件名。
    try:
        benchmark_services = await ensure_dynamic_benchmark_component(name, runtime)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark services not found: {name}",
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not hasattr(benchmark_services, BENCHMARK_METADATA_ATTR):
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not a benchmark services component",
        )

    meta = benchmark_services.get_metadata()

    try:
        run_id = await benchmark_service.create_run(
            BenchmarkRunSpec(
                benchmark_name=meta.name,
                service_name=meta.service_name,
                dataset_name="(pipeline-kafka)",
                project_id=payload.project_id or meta.dataset.project_id,
                total_evaluators=len(meta.rubric),
                config_json={
                    "date_start": payload.date_start.isoformat(),
                    "date_end": payload.date_end.isoformat(),
                    "sample_size": payload.sample_size,
                    "sample_seed": payload.sample_seed,
                    "project_id": payload.project_id,
                    "source": "pipeline-kafka",
                    "rubric": [
                        {
                            "id": c.id,
                            "name": c.name,
                            "weight": c.weight,
                            "evaluator_ref": c.evaluator_ref,
                        }
                        for c in meta.rubric
                    ],
                    "aggregation": {
                        "method": meta.aggregation.method,
                        "pass_threshold": meta.aggregation.pass_threshold,
                    },
                },
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "hint": (
                    "Ensure PG has benchmark_runs (sql/schema.sql) and "
                    "column service_name (not business_agent_name)."
                ),
            },
        ) from exc

    async def _run_pipeline_bg() -> None:
        try:
            logger.info(
                "pipeline_benchmark.job_started run_id=%s name=%s",
                run_id,
                meta.name,
            )
            await runner.run_pipeline_benchmark(
                PipelineRunExec(
                    benchmark_services=benchmark_services,
                    run_id=run_id,
                    request=payload,
                    generate_report=payload.generate_report,
                ),
            )
        except asyncio.CancelledError:
            logger.warning("pipeline_benchmark.job_cancelled run_id=%s", run_id)
            try:
                await benchmark_service.fail_run(
                    run_id=run_id,
                    reason="shutdown: task cancelled",
                )
            except Exception:
                logger.exception(
                    "pipeline_benchmark.cancel_fail_run_failed run_id=%s",
                    run_id,
                )
            raise
        except Exception as exc:
            logger.exception("pipeline_benchmark.async_run_failed run_id=%s", run_id)
            try:
                await benchmark_service.fail_run(
                    run_id=run_id,
                    reason=f"async pipeline run failed: {exc}",
                )
            except Exception:
                logger.exception(
                    "pipeline_benchmark.async_fail_run_failed run_id=%s",
                    run_id,
                )

    asyncio.create_task(_run_pipeline_bg())

    poll_url = f"/v1/benchmarks/runs/{run_id}"
    return JSONResponse(
        status_code=202,
        content={
            "run_id": run_id,
            "benchmark_name": meta.name,
            "status": "initializing",
            "poll_url": poll_url,
            "message": (
                "Pipeline benchmark accepted (async). Scanning Kafka records "
                f"{payload.date_start.isoformat()}.."
                f"{payload.date_end.isoformat()}. "
                f"Optional poll: GET {poll_url}"
            ),
        },
    )


# ════════════════════════════════════════════════════════════════════════════
# Inline dataset endpoints (POST body carries items; skip Langfuse fetch)
# ════════════════════════════════════════════════════════════════════════════


@router.post("/run-inline", response_model=None)
async def run_benchmark_inline(
    payload: BenchmarkInlineRunRequest,
    request: Request,
    benchmark_service: BenchmarkServiceDep,
    runtime: RuntimeDep,
    queue_manager: EvalQueueRegistryDep,
) -> JSONResponse:
    """Accept inline dataset items and run benchmark (always 202, BullMQ).

    内联数据集：items 随请求体传入，跳过 Langfuse fetch / sampling。
    BullMQ 分发：update RUNNING → enqueue cases（items 直接随 payload 传递）。
    """
    raise_if_benchmark_disabled()
    payload = _apply_benchmark_config_defaults(payload, request)
    benchmark_services = await _resolve_benchmark_services(
        runtime,
        payload.benchmark_services,
    )
    meta = benchmark_services.get_metadata()

    # 配置化运行时（§5.1/§5.2）：profile 解析 + config_json 组装（非法引用 → 400）
    prepared = _prepare_push_run(
        request,
        _PushRunPrepare(
            benchmark_name=payload.benchmark_services,
            meta=meta,
            payload=payload,
            extra_config={
                "source": "inline",
                "item_count": len(payload.items),
                "benchmark_services": payload.benchmark_services,
                "max_concurrency": payload.max_concurrency,
                "parallel_evaluators": payload.parallel_evaluators,
                "triage_mode": payload.triage_mode.value,
                "generate_report": payload.generate_report,
            },
        ),
    )
    resolved = prepared.resolved
    config_json = prepared.config_json
    external_count_model = prepared.external_count_model
    is_fused = prepared.is_fused

    try:
        run_id = await benchmark_service.create_run(
            BenchmarkRunSpec(
                benchmark_name=meta.name,
                service_name=payload.agent_name,
                dataset_name=payload.dataset_name,
                project_id=payload.project_id,
                total_evaluators=len(resolved.rubric),
                config_json=config_json,
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "hint": (
                    "Ensure PG has benchmark_runs (sql/schema.sql) and "
                    "column service_name (not business_agent_name)."
                ),
            },
        ) from exc

    criteria = [{"id": c.id, "evaluator_ref": c.evaluator_ref} for c in resolved.rubric]

    # Update to RUNNING + enqueue cases (items passed directly from payload)
    item_count = len(payload.items)
    await benchmark_service.update_run_status(
        run_id,
        BenchmarkStatus.RUNNING,
        total_samples=item_count,
        total_evaluators=len(resolved.rubric),
        started_at=datetime.now(UTC),
    )

    # external 计数器模型（judge/归因 async）：expected 对齐 raw 总数
    if external_count_model:
        await benchmark_service.merge_config_json(
            run_id,
            {"judge_expected_count": item_count},
        )

    # 统一提交管线（§7.2 拦截点必经）→ Enqueue cases via BullMQ
    spec = BenchmarkCaseEnqueueSpec(
        run_id=run_id,
        criteria=criteria,
        benchmark_services_name=payload.benchmark_services,
        service_name=payload.agent_name,
        dataset_name=payload.dataset_name,
        project_id=payload.project_id,
        triage_mode=payload.triage_mode.value,
        generate_report=payload.generate_report,
    )
    summary = await submit_prepared_items(
        SubmitPipelineSpec(
            run_id=run_id,
            resolved=resolved,
            raw_items=_raw_items_with_identity(payload.items, run_id),
            enqueue_items=lambda parsed: _enqueue_benchmark_cases(
                queue_manager,
                parsed,
                spec,
            ),
            mark_item_failed=lambda item, reason: _mark_push_item_parse_failed(
                benchmark_service,
                _PushItemParseFailed(
                    run_id=run_id,
                    item=item,
                    reason=reason,
                    plan=_ParseFailurePlan(
                        criteria=criteria,
                        is_fused=is_fused,
                        external_count_model=external_count_model,
                    ),
                ),
            ),
            snapshot=config_json,
        ),
    )
    # 全量 parse 失败（enqueued=0）→ 无 case/回调触发 finalize，补偿一次
    await _tick_finalize_for_empty_submit(
        benchmark_service,
        _EmptySubmitFinalize(
            run_id=run_id,
            enqueued=summary.enqueued,
            total_raw=summary.total_raw,
            external_count_model=external_count_model,
        ),
    )

    logger.info(
        "benchmark.inline_enqueued run_id=%s name=%s items=%d",
        run_id,
        meta.name,
        item_count,
    )

    poll_url = f"/v1/benchmarks/runs/{run_id}"
    return JSONResponse(
        status_code=202,
        content={
            "run_id": run_id,
            "benchmark_name": meta.name,
            "service_name": payload.agent_name,
            "dataset_name": payload.dataset_name,
            "status": BenchmarkStatus.RUNNING.value,
            "total_evaluators": len(meta.rubric),
            "item_count": len(payload.items),
            "poll_url": poll_url,
        },
    )


# ════════════════════════════════════════════════════════════════════════════
# Stream dataset endpoints (ingest-then-evaluate; multi-node workers)
# ════════════════════════════════════════════════════════════════════════════


@router.post("/runs", response_model=None, status_code=201)
async def create_stream_run(
    payload: BenchmarkStreamRunRequest,
    request: Request,
    benchmark_service: BenchmarkServiceDep,
    runtime: RuntimeDep,
) -> JSONResponse:
    """Create a benchmark run in ACCEPTING status (data ingestion phase).

    执行参数（max_concurrency / triage_mode / generate_report 等）随
    ``config_json`` 持久化，finalize 时恢复。run 在 ``ACCEPTING`` 期间
    通过 append items 维持心跳；中断 ingest 的残留由 ``recover_stale_runs``
    兜底标记 FAILED。
    """
    raise_if_benchmark_disabled()
    payload = _apply_benchmark_config_defaults(payload, request)
    benchmark_services = await _resolve_benchmark_services(
        runtime,
        payload.benchmark_services,
    )
    meta = benchmark_services.get_metadata()

    try:
        run_id = await benchmark_service.create_run(
            BenchmarkRunSpec(
                benchmark_name=meta.name,
                service_name=payload.agent_name,
                dataset_name=payload.dataset_name,
                project_id=payload.project_id,
                total_evaluators=len(meta.rubric),
                config_json={
                    "source": "stream",
                    "benchmark_services": payload.benchmark_services,
                    "max_concurrency": payload.max_concurrency,
                    "parallel_evaluators": payload.parallel_evaluators,
                    "triage_mode": payload.triage_mode.value,
                    "generate_report": payload.generate_report,
                    **_rubric_config_json(meta),
                },
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "hint": (
                    "Ensure PG has benchmark_runs (sql/schema.sql) and "
                    "column service_name (not business_agent_name)."
                ),
            },
        ) from exc
    await benchmark_service.update_run_status(run_id, BenchmarkStatus.ACCEPTING)

    return JSONResponse(
        status_code=201,
        content={
            "run_id": run_id,
            "benchmark_name": meta.name,
            "status": BenchmarkStatus.ACCEPTING.value,
            "append_url": f"/v1/benchmarks/runs/{run_id}/items",
            "finalize_url": f"/v1/benchmarks/runs/{run_id}/finalize",
        },
    )


@router.post("/runs/{run_id}/items", response_model=None)
async def append_stream_items(
    run_id: str,
    payload: StreamItemsRequest,
    benchmark_service: BenchmarkServiceDep,
    staging_buffer: StagingBufferDep,
) -> JSONResponse:
    """Append items to a run in ACCEPTING status (callable multiple times)."""
    run = await benchmark_service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    if run.status != BenchmarkStatus.ACCEPTING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Run is {run.status}, not accepting items",
        )

    item_dicts = [item.model_dump() for item in payload.items]
    count = await staging_buffer.append_items(run_id, item_dicts)
    # ingest 活性：持续 append 视为心跳，防止被 recover_stale_runs 误杀
    await benchmark_service.heartbeat(run_id)
    return JSONResponse(
        status_code=200,
        content={"run_id": run_id, "accepted": count},
    )


@router.post("/runs/{run_id}/finalize", response_model=None)
async def finalize_stream_run(
    run_id: str,
    benchmark_service: BenchmarkServiceDep,
    staging_buffer: StagingBufferDep,
    runtime: RuntimeDep,
    queue_manager: EvalQueueRegistryDep,
) -> JSONResponse:
    """Signal all items sent; CAS to RUNNING and enqueue via BullMQ.

    ``ACCEPTING → RUNNING`` 条件 UPDATE 防并发 finalize；随后从 Redis
    StagingBuffer 读出 items → 逐个入队 ``benchmark-cases`` BullMQ 队列。
    """
    run = await benchmark_service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    if run.status != BenchmarkStatus.ACCEPTING.value:
        raise HTTPException(
            status_code=409,
            detail=f"Run is {run.status}, already finalized",
        )

    config = run.config_json or {}
    services_name = config.get("benchmark_services")
    if not services_name:
        raise HTTPException(
            status_code=409,
            detail="Run was not created via stream API (missing benchmark_services)",
        )
    benchmark_services = await _resolve_benchmark_services(runtime, services_name)
    meta = benchmark_services.get_metadata()

    item_count = await staging_buffer.count_items(run_id)
    if item_count == 0:
        await benchmark_service.fail_run(
            run_id=run_id,
            reason="finalized with no items",
        )
        raise HTTPException(status_code=422, detail="No items to evaluate")

    # CAS: ACCEPTING → RUNNING + total_samples/total_evaluators in one UPDATE
    updated = await benchmark_service.cas_finalize_run(
        run_id=run_id,
        total_samples=item_count,
        total_evaluators=len(meta.rubric),
    )
    if updated == 0:
        raise HTTPException(status_code=409, detail="Already finalized")

    criteria = [{"id": c.id, "evaluator_ref": c.evaluator_ref} for c in meta.rubric]

    # Read staged items from Redis → enqueue via BullMQ
    items = await staging_buffer.list_items(run_id)
    await _enqueue_benchmark_cases(
        queue_manager,
        items,
        BenchmarkCaseEnqueueSpec(
            run_id=run_id,
            criteria=criteria,
            benchmark_services_name=services_name,
            service_name=meta.service_name,
            dataset_name=run.dataset_name,
            project_id=run.project_id,
            triage_mode=config.get("triage_mode") or TriageMode.FUSED.value,
            generate_report=bool(config.get("generate_report", True)),
        ),
    )

    logger.info(
        "benchmark.stream_enqueued run_id=%s name=%s items=%d",
        run_id,
        meta.name,
        item_count,
    )

    return JSONResponse(
        status_code=200,
        content={
            "run_id": run_id,
            "status": BenchmarkStatus.RUNNING.value,
            "total_items": item_count,
            "total_evaluators": len(meta.rubric),
            "poll_url": f"/v1/benchmarks/runs/{run_id}",
        },
    )


@router.get("/runs")
async def list_page_runs(
    benchmark_service: BenchmarkServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    # 勿用 BaseModel + Field(alias=) 包进 Query()：Pydantic 2.13 对
    # ``str | None = Field(alias=...)`` 在 OpenAPI 生成时会抛
    # UnsupportedFieldAttributeWarning，且 alias 实际失效。
    benchmark_name: Annotated[str | None, Query(alias="benchmarkName")] = None,
    task_id: Annotated[str | None, Query(alias="taskId")] = None,
) -> list[dict[str, Any]]:
    """列出 run，字段对齐评测任务列表。不要求先知道注册名。"""
    from benchmark.domain.benchmark_run_page import page_run_summary  # noqa: PLC0415

    runs = await benchmark_service.list_runs(
        ListRunsQuery(
            benchmark_name=benchmark_name,
            task_id=task_id,
            limit=limit,
            offset=offset,
        )
    )
    return [page_run_summary(run) for run in runs]


@router.get("/run-counts")
async def list_run_counts(
    benchmark_service: BenchmarkServiceDep,
) -> list[dict[str, Any]]:
    """按 benchmark 名返回已记录的运行次数。"""
    return await benchmark_service.count_runs_by_benchmark()


@router.get("/{name}/runs")
async def list_runs(
    name: str,
    benchmark_service: BenchmarkServiceDep,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List run history for a specific benchmark."""
    runs = await benchmark_service.list_runs(
        ListRunsQuery(benchmark_name=name, limit=limit, offset=offset)
    )
    from benchmark.domain.eval_config import logical_benchmark_run_name  # noqa: PLC0415

    return [
        {
            "run_id": r.run_id,
            "benchmark_name": logical_benchmark_run_name(
                column_name=r.benchmark_name,
                config=r.config_json if isinstance(r.config_json, dict) else {},
            )
            or r.benchmark_name,
            "service_name": r.service_name,
            "status": r.status,
            "final_score": r.final_score,
            "total_samples": r.total_samples,
            "total_evaluators": r.total_evaluators,
            "completed_evaluators": r.completed_evaluators,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "duration_ms": r.duration_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]


@router.get("/runs/latest")
async def get_latest_run(
    benchmark_service: BenchmarkServiceDep,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Get the most recent benchmark run (latest created_at).

    If ``run_id`` is provided, fetch that specific run instead.
    Returns a lightweight summary: dataset_name, status,
    config_json, and report_url from report_meta_json.
    """
    if run_id is not None:
        run = await benchmark_service.get_run(run_id)
    else:
        runs = await benchmark_service.list_runs(ListRunsQuery(limit=1))
        run = runs[0] if runs else None

    if run is None:
        raise HTTPException(status_code=404, detail="No benchmark runs found")

    report_meta = run.report_meta_json if isinstance(run.report_meta_json, dict) else {}
    return {
        "run_id": run.run_id,
        "dataset_name": run.dataset_name,
        "status": run.status,
        "config_json": run.config_json,
        "report_url": report_meta.get("report_url"),
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    benchmark_service: BenchmarkServiceDep,
) -> dict[str, Any]:
    """Get single run details (criterion stats live in ``metrics_json``).

    ``progress`` 聚合分布式执行进度：run 级计数器（evaluated_items /
    failed_items / total_samples）+ per-criterion 样本统计
    （``metrics_json.criterion_metrics`` 增量可见），实时查 PG。
    """
    run = await benchmark_service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    from benchmark.domain.benchmark_run_page import page_run_summary  # noqa: PLC0415
    from benchmark.domain.eval_config import logical_benchmark_run_name  # noqa: PLC0415

    run_cfg = run.config_json if isinstance(run.config_json, dict) else {}
    return {
        "run_id": run.run_id,
        "benchmark_name": logical_benchmark_run_name(
            column_name=run.benchmark_name,
            config=run_cfg,
        )
        or run.benchmark_name,
        "page": page_run_summary(run),
        "service_name": run.service_name,
        "dataset_name": run.dataset_name,
        "project_id": run.project_id,
        "status": run.status,
        "final_score": run.final_score,
        "total_samples": run.total_samples,
        "total_evaluators": run.total_evaluators,
        "completed_evaluators": run.completed_evaluators,
        "progress": await benchmark_service.build_run_progress(run),
        "metrics_json": run.metrics_json,
        "config_json": run.config_json,
        "report_meta_json": run.report_meta_json,
        "error_message": run.error_message,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "duration_ms": run.duration_ms,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.delete("/runs/{run_pk}", status_code=204)
async def delete_run(
    run_pk: UUID,
    benchmark_service: BenchmarkServiceDep,
) -> None:
    """软删单次评测：按行主键 id 给该行打 deleted_at 标记。

    不物理删行、不动 pipeline 明细；标记后列表/计数/日报回填/详情
    查询全部跳过该行。行不存在或已删除 → 404。
    """
    if not await benchmark_service.delete_run(run_pk):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_pk}")


@router.get("/runs/{run_id}/report")
async def get_report(
    run_id: str,
    benchmark_service: BenchmarkServiceDep,
) -> RedirectResponse:
    """Get HTML report — redirects to OSS URL if available."""
    run = await benchmark_service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    report_meta = run.report_meta_json or {}
    report_url = report_meta.get("report_url")

    if not report_url:
        raise HTTPException(
            status_code=404,
            detail="No report available for this run",
        )

    return RedirectResponse(url=report_url, status_code=302)


@router.get("/runs/{run_id}/metrics")
async def get_metrics(
    run_id: str,
    benchmark_service: BenchmarkServiceDep,
) -> dict[str, Any]:
    """Get aggregated metrics JSON for a run."""
    run = await benchmark_service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    if run.metrics_json is None:
        raise HTTPException(
            status_code=404,
            detail="Metrics not yet available (run may still be in progress)",
        )

    return {
        "run_id": run.run_id,
        "final_score": run.final_score,
        "metrics": run.metrics_json,
    }


# ════════════════════════════════════════════════════════════════════════════
# Shutdown helper
# ════════════════════════════════════════════════════════════════════════════


async def drain_pending_benchmark_tasks(
    *,
    benchmark_service: BenchmarkService | None = None,
) -> None:
    """Mark in-flight benchmark runs as failed during app shutdown.

    BullMQ Workers are closed by ``main.py`` lifespan; this function handles
    PG-side cleanup so that runs left in ``RUNNING`` by a crashed process are
    not stuck forever.
    """
    if benchmark_service is not None:
        try:
            n = await benchmark_service.fail_inflight_runs_for_owner(
                reason="shutdown: process stopping",
            )
            if n:
                logger.warning(
                    "shutdown: marked %d in-flight benchmark runs failed",
                    n,
                )
        except Exception:
            logger.exception("shutdown: fail_inflight_runs_for_owner failed")


# ════════════════════════════════════════════════════════════════════════════
# External trigger endpoints (§3 / §4)
# ════════════════════════════════════════════════════════════════════════════


def get_external_runner(request: Request) -> "ExternalBenchmarkRunner":
    """Resolve lifespan-scoped ExternalBenchmarkRunner from app.state."""
    runner = getattr(request.app.state, "external_benchmark_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="ExternalBenchmarkRunner not initialized",
        )
    return runner


ExternalRunnerDep = Annotated["ExternalBenchmarkRunner", Depends(get_external_runner)]


@dataclass
class ManualRunDeps:
    """``POST /manual-runs`` 聚合依赖（占号 + 受理链路共用的三服务）。"""

    benchmark_service: BenchmarkService
    runtime: "AppRuntime"
    external_runner: "ExternalBenchmarkRunner"


def get_manual_run_deps(
    benchmark_service: BenchmarkServiceDep,
    runtime: RuntimeDep,
    external_runner: ExternalRunnerDep,
) -> ManualRunDeps:
    """FastAPI 子依赖聚合：run 路由单参注入，签名保持 ≤4。"""
    return ManualRunDeps(
        benchmark_service=benchmark_service,
        runtime=runtime,
        external_runner=external_runner,
    )


ManualRunDepsDep = Annotated[ManualRunDeps, Depends(get_manual_run_deps)]


@dataclass(frozen=True)
class AcceptExternalRunInput:
    """``_accept_external_run`` 参数对象（FUNC-PARAMS-001 收敛）。"""

    benchmark_service: BenchmarkService
    runtime: "AppRuntime"
    external_runner: "ExternalBenchmarkRunner"
    assigned_run_id: str | None = None
    data_date: str | None = None
    reuse_existing: bool = True
    #: 克隆继承的注册快照（非空时受理期不再重新解析注册）
    registration_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ManualRunAccept:
    """手动入口交给共享受理流程的上下文。run_id 由分配器另传。"""

    payload: ExternalBenchmarkRequest
    benchmark_service: BenchmarkService
    runtime: "AppRuntime"
    external_runner: "ExternalBenchmarkRunner"
    data_date: str
    reuse_existing: bool
    #: 克隆继承的注册快照（非空时受理期不再重新解析注册）
    registration_snapshot: dict[str, Any] | None = None


async def _accept_assigned_external_run(
    run_id: str,
    ctx: _ManualRunAccept,
) -> ExternalRunAcceptedResponse | JSONResponse:
    """用已分配的 run_id 走与 /run-external 相同的受理流程。"""
    return await _accept_external_run(
        ctx.payload,
        AcceptExternalRunInput(
            benchmark_service=ctx.benchmark_service,
            runtime=ctx.runtime,
            external_runner=ctx.external_runner,
            assigned_run_id=run_id,
            data_date=ctx.data_date,
            reuse_existing=ctx.reuse_existing,
            registration_snapshot=ctx.registration_snapshot,
        ),
    )


def _coerce_external_request(
    payload: "ExternalBenchmarkRequest | BenchmarkRuntimeProfile",
) -> ExternalBenchmarkRequest:
    """v2 与手动入口共用：旧 profile 契约归一成扁平请求。"""
    if isinstance(payload, BenchmarkRuntimeProfile):
        logger.warning(
            "run-external.legacy_v2_profile benchmark=%s — "
            "profile 原生契约已弃用，请迁移类别名契约（judge_class 等）",
            payload.benchmark_name,
        )
        return ExternalBenchmarkRequest(
            benchmark_name=payload.benchmark_name,
            date=payload.data_date,
            run_id_suffix=payload.run_id_suffix,
            oss_prefix=payload.oss_prefix,
            dataset_name=payload.dataset_name,
            project_id=payload.project_id,
            agent_registration_id=payload.agent_registration_id,
            generate_report=payload.generate_report,
            aggregation=payload.aggregation,
            mode=payload.mode,
            limit=payload.limit,
            rubric_ref=payload.rubric_ref,
            parse_interceptor_ref=payload.parse_interceptor_ref,
            judge=payload.judge,
            attribution=payload.attribution,
        )
    return payload


_RUN_ID_DATE_RE = re.compile(r"-(\d{8})(?:-|$)")


async def _apply_source_run_clone(
    payload: "ExternalBenchmarkRequest",
) -> tuple["ExternalBenchmarkRequest", str, dict[str, Any] | None]:
    """再次触发克隆：benchmark/dataset/OSS/执行类配置以源 run 快照为准。

    源 run 的 ``config_json`` 存有首次 init 落盘的 ``data_date`` /
    ``oss_prefix`` / ``oss_object_key`` / judge·attribution·rubric 类与
    generate_report；克隆仅分配新 run_id，其余原样重放——活文档与系统
    时钟的漂移（改 ossPrefix、重算今天日期拼错分区）不进克隆请求。
    请求显式给出的字段优先，缺省才回填快照。克隆源按行主键
    ``source_id`` 定位，已软删的 run 不可克隆（视为不存在）。
    """
    try:
        source_pk = UUID(str(payload.source_id))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"source run not found: {payload.source_id}",
        ) from exc
    source = await BenchmarkRun.filter(
        id=source_pk,
        deleted_at__isnull=True,
    ).first()
    if source is None:
        raise HTTPException(
            status_code=404,
            detail=f"source run not found: {payload.source_id}",
        )
    cfg = source.config_json if isinstance(source.config_json, dict) else {}
    from benchmark.domain.eval_config import run_matches_benchmark_name  # noqa: PLC0415

    if not run_matches_benchmark_name(
        column_name=source.benchmark_name,
        config=cfg,
        query_name=payload.benchmark_name,
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"source run benchmark mismatch: {source.benchmark_name} "
                f"!= {payload.benchmark_name}"
            ),
        )
    date = str(cfg.get("data_date") or "")
    if not date:
        matched = _RUN_ID_DATE_RE.search(source.run_id)
        date = matched.group(1) if matched else ""
    if not date:
        raise HTTPException(
            status_code=400,
            detail=f"source run has no data_date snapshot: {source.run_id}",
        )
    updates: dict[str, Any] = {"date": date}
    for field in (
        "oss_prefix",
        "oss_object_key",
        "judge_class",
        "attribution_class",
        "rubric_class",
    ):
        if getattr(payload, field) is None and cfg.get(field):
            updates[field] = cfg[field]
    if isinstance(cfg.get("generate_report"), bool):
        updates["generate_report"] = cfg["generate_report"]
    if not payload.dataset_name and source.dataset_name:
        updates["dataset_name"] = source.dataset_name
    # 归属继承：克隆出的新 run 仍归原任务（页面按任务隔离不断链）
    if not payload.eval_task_id and cfg.get("eval_task_id"):
        updates["eval_task_id"] = cfg["eval_task_id"]
    if not payload.dataset_id and cfg.get("dataset_id"):
        updates["dataset_id"] = cfg["dataset_id"]
    # 注册快照继承：克隆默认复现原 run 的定制版本与归因开关（即使注册
    # 已删除）；仅显式传 agent_registration_id 时才在受理期重新解析。
    snapshot = cfg.get("agent_registration")
    return (
        payload.model_copy(update=updates),
        date,
        snapshot if isinstance(snapshot, dict) else None,
    )


def _pin_dated_oss_object_key(payload: "ExternalBenchmarkRequest") -> None:
    """prefix 末段已是 ``yyyymmdd`` 时补定 object key，避免 loader 再拼日期。

    不要求末段等于本次 ``data_date``：页面预展开 ``${yyyymmdd-1}`` 后的偏移日、
    失败 init 快照里的字面分区日同样是数据目录。
    """
    if payload.oss_object_key or not payload.oss_prefix:
        return
    pinned = object_key_for_dated_prefix(
        payload.oss_prefix,
        file_name=DailyDataLoader.FILE_NAME,
    )
    if pinned:
        payload.oss_object_key = pinned


def _expand_request_oss_date_macro(
    payload: "ExternalBenchmarkRequest",
    *,
    base_date: str,
) -> None:
    """请求 ``oss_prefix`` 含日期宏时按 ``base_date`` 展开一次并钉死 object key。

    Raises:
        ValueError: 宏语法非法。
    """
    prefix = (payload.oss_prefix or "").strip()
    if not prefix or not has_date_macro(prefix):
        return
    validate_date_macros(prefix)
    payload.oss_prefix = expand_date_macros(prefix, base_date=base_date)
    payload.oss_object_key = object_key_for_dated_prefix(
        payload.oss_prefix,
        file_name=DailyDataLoader.FILE_NAME,
    )


@dataclass(frozen=True)
class ExternalRunIdentity:
    """受理期解析出的数据日、run_id 与归一 OSS 前缀。"""

    date: str
    run_id: str
    effective_oss_prefix: str


def _resolve_external_run_identity(
    payload: ExternalBenchmarkRequest,
    dataset_document: "DatasetDocument | None",
    *,
    assigned_run_id: str | None,
    data_date: str | None,
) -> ExternalRunIdentity:
    """显式日/当天为宏基准 → 单次展开 → 再按字面分区日写回（未占号）。

    宏展开的 ``base_date`` **只**取显式 ``date`` / ``data_date`` / 当天，
    绝不用路径里已写出的字面分区日。否则任意偏移宏都会被套两次：

    - ``${yyyymmdd-1}``：20261111 → 20261110 → **20261109**
    - ``${yyyymmdd+1}``：20261111 → 20261112 → **20261113**
    - ``${yyyymmdd}``：字面路径若再被文档宏按错误基准重算，分区日会被改写

    含宏路径不在 ``resolve_external_data_date`` 内预展开。已占号入口锁定
    ``data_date`` / ``assigned_run_id``（调用方须事先按显式日完成一次展开）。
    """
    locked = assigned_run_id is not None and bool(data_date)
    # 宏基准：忽略 storage_path，避免把「已偏移分区日」当成再展开的基准
    expand_base = resolve_external_data_date(
        str(data_date) if locked else (data_date or payload.date),
    )
    effective_oss_prefix = _resolve_oss_target(
        payload,
        dataset_document,
        base_date=expand_base,
    )
    if locked:
        return ExternalRunIdentity(
            date=str(data_date),
            run_id=str(assigned_run_id),
            effective_oss_prefix=effective_oss_prefix,
        )
    date = resolve_external_data_date(
        expand_base,
        storage_path=payload.oss_object_key or effective_oss_prefix,
    )
    if payload.run_id_suffix:
        run_id = f"{payload.benchmark_name}-{date}-{payload.run_id_suffix}"
    else:
        run_id = f"{payload.benchmark_name}-{date}"
    return ExternalRunIdentity(
        date=date,
        run_id=run_id,
        effective_oss_prefix=effective_oss_prefix,
    )


@router.post("/manual-runs", response_model=ExternalRunAcceptedResponse, status_code=202)
async def run_manual_benchmark(
    payload: "ExternalBenchmarkRequest | BenchmarkRuntimeProfile",
    request: Request,
    deps: ManualRunDepsDep,
    idempotency_key: Annotated[str | None, Header()] = None,
) -> ExternalRunAcceptedResponse | JSONResponse:
    """业务手动多次触发同一外部 benchmark。

    请求体、校验、占位 run、后台 ``initialize_and_submit`` 与
    ``POST /run-external/v2`` 相同。唯一差别是 run_id：定时到点的
    ``{benchmark_name}-{yyyymmdd}``（及 RemoteConfig 后缀）不占用；立即任务分配
    ``-m{n}``，周期任务手动再次触发分配 ``-s{n}``。调用方不传 ``run_id_suffix``。
    202 返回的 ``run_id`` 用 ``GET /v1/benchmarks/runs/{run_id}`` 查询。

    ``Idempotency-Key`` 放在头里，是为了重试去重，不写进评测请求体，
    这样 body 与 ``/run-external/v2`` 保持同一份契约。
    """
    raise_if_benchmark_disabled()
    # 队列栈降级（worker 未启动）时拒收：降级态受理的 run 永远不会被消费
    raise_if_eval_queue_degraded(request)
    payload = _coerce_external_request(payload)
    if payload.run_id_suffix:
        raise HTTPException(
            status_code=422,
            detail="manual-runs assigns run_id; omit run_id_suffix",
        )
    try:
        key = validate_idempotency_key(idempotency_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    registration_snapshot: dict[str, Any] | None = None
    if payload.source_id:
        payload, date, registration_snapshot = await _apply_source_run_clone(payload)
    else:
        # 宏只在占号前按显式日/当天展开一次，再取字面分区日占号；
        # 避免 accept 内以已偏移日为基准二次展开。
        expand_base = resolve_external_data_date(payload.date)
        try:
            _expand_request_oss_date_macro(payload, base_date=expand_base)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        date = resolve_external_data_date(
            payload.date,
            storage_path=payload.oss_object_key or payload.oss_prefix,
        )
    manual_namespace: ManualRunNamespace = payload.manual_run_namespace or "m"
    if payload.manual_run_namespace is None and payload.eval_task_id:
        from benchmark.domain.benchmark_console import find_task_document  # noqa: PLC0415

        task = await find_task_document(payload.eval_task_id)
        if task is not None and task.trigger != "immediate":
            manual_namespace = "s"
    _pin_dated_oss_object_key(payload)
    accept = _ManualRunAccept(
        payload=payload,
        benchmark_service=deps.benchmark_service,
        runtime=deps.runtime,
        external_runner=deps.external_runner,
        data_date=date,
        reuse_existing=key is not None,
        registration_snapshot=registration_snapshot,
    )
    try:
        return await claim_manual_run_id(
            ManualRunClaimScope(
                benchmark_name=payload.benchmark_name,
                date=date,
                idempotency_key=key,
                namespace=manual_namespace,
            ),
            existing_run_ids=partial(
                list_manual_run_ids,
                payload.benchmark_name,
                date,
                namespace=manual_namespace,
            ),
            claim=partial(_accept_assigned_external_run, ctx=accept),
        )
    except ManualRunIdExhaustedError as exc:
        raise HTTPException(
            status_code=503,
            detail="failed to allocate a distinct manual run_id",
        ) from exc


@router.post("/run-external", response_model=ExternalRunAcceptedResponse, status_code=202)
async def run_external_benchmark(
    payload: ExternalBenchmarkRequest,
    request: Request,
    response: Response,
    deps: ManualRunDepsDep,
) -> ExternalRunAcceptedResponse | JSONResponse:
    """手动补跑 / 重跑指定日期的外部触发 benchmark（§3，**v1 扁平契约**）。

    .. deprecated:: §9
        v1 保留一个版本周期；新调用方请用 profile 原生契约
        ``POST /v1/benchmarks/run-external/v2``。
    """
    raise_if_benchmark_disabled()
    # 队列栈降级（worker 未启动）时拒收：降级态受理的 run 永远不会被消费
    raise_if_eval_queue_degraded(request)
    # v1 兼容标记（§9）：响应头告知弃用，行为与现网完全一致
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = '</v1/benchmarks/run-external/v2>; rel="successor-version"'
    return await _accept_external_run(
        payload,
        AcceptExternalRunInput(
            benchmark_service=deps.benchmark_service,
            runtime=deps.runtime,
            external_runner=deps.external_runner,
        ),
    )


@router.post("/run-external/v2", response_model=ExternalRunAcceptedResponse, status_code=202)
async def run_external_benchmark_v2(
    payload: "ExternalBenchmarkRequest | BenchmarkRuntimeProfile",
    request: Request,
    deps: ManualRunDepsDep,
) -> ExternalRunAcceptedResponse | JSONResponse:
    """执行类别名契约的外部触发 benchmark（pull 模式 v2）。

    Judge / Attribution 仅接受服务端注册的稳定类别名，不接受
    ``execution/provider/target`` 组合。

    迁移窗口（§9）：同时接受旧 profile 原生契约（``data_date`` /
    ``rubric_ref`` / 结构化 ``judge`` / ``attribution``），归一化为 v1
    兼容请求走同一 Resolver（deprecated，一个版本周期后移除）。
    """
    raise_if_benchmark_disabled()
    # 队列栈降级（worker 未启动）时拒收：降级态受理的 run 永远不会被消费
    raise_if_eval_queue_degraded(request)
    payload = _coerce_external_request(payload)
    return await _accept_external_run(
        payload,
        AcceptExternalRunInput(
            benchmark_service=deps.benchmark_service,
            runtime=deps.runtime,
            external_runner=deps.external_runner,
        ),
    )


def _realtime_eval_raw(
    evaluator_name: str,
    document: "DatasetDocument",
) -> dict[str, Any]:
    """数据集采样条件 + 页面级覆盖 → stamp 入参（与 EvalEntryWrite 同构）。"""
    sampling = document.sampling_ratio if document.sampling_ratio is not None else 1.0
    conditions = document.trace_sampling
    raw: dict[str, Any] = {
        "evaluator_name": evaluator_name,
        "sampling": sampling,
        "ingest": conditions.ingest if conditions is not None else "kafka",
    }
    if conditions is not None:
        if conditions.name:
            raw["name"] = conditions.name
        if conditions.tags:
            raw["tags"] = conditions.tags
        if conditions.environment:
            raw["environment"] = conditions.environment
        if conditions.variable_mapping:
            raw["variable_mapping"] = [
                item.model_dump(by_alias=True) for item in conditions.variable_mapping
            ]
    # 页面级可选覆盖：agentName（流级归属，须命中 registry 候选集）、
    # prompt、evaluatorModel（LLM 覆盖；apiKeyEnv 只存环境变量名）。
    if document.agent_name:
        raw["agent_name"] = document.agent_name
    if document.prompt:
        raw["prompt"] = document.prompt
    if document.evaluator_model is not None and not document.evaluator_model.is_empty:
        raw["model"] = document.evaluator_model.model_dump(by_alias=True, exclude_none=True)
    return raw


async def _activate_realtime_eval(
    payload: ExternalBenchmarkRequest,
    *,
    meta: "BenchmarkComponentMetadata",
    document: "DatasetDocument",
) -> JSONResponse:
    """实时采样数据集：激活实时评测流程，不创建批量 run。

    数据集 ``source=realtime`` 表示线上 trace 实时采样，正确执行链是
    Kafka 实时消费链（streaming consumer 按 evaluator 配置采样评测，
    结果落 ``trace_pipeline_records``），而非批量回放（OSS 拉取 +
    BullMQ case jobs + Aone judge 回调）。本函数把数据集的采样条件
    登记为一条 ``kind=evaluator`` 配置（与页面评测配置同表同构）并刷新
    快照，实时链随后接管；数据集状态置为 ``sampling``。
    """
    refs = _evaluator_refs(meta.rubric)
    if not refs:
        raise HTTPException(
            status_code=400,
            detail=(
                f"benchmark {payload.benchmark_name} 的 rubric 未声明 evaluator，无法路由实时评测"
            ),
        )
    if len(refs) > 1:
        raise HTTPException(
            status_code=400,
            detail=(f"实时评测需单一 evaluator，benchmark {payload.benchmark_name} 声明了 {refs}"),
        )
    evaluator_name = refs[0]

    # project_id：请求入参 > 数据集文档 projectId（页面采样表单配置）>
    # RemoteConfig agent registry 按 service_name 反查（恰好一个 project 时可用，
    # 多个/未登记需在前两者显式指定）。
    project_id = (payload.project_id or "").strip() or (document.project_id or "").strip()
    if not project_id:
        agent_cfg = get_agent_config(meta.service_name)
        projects = agent_cfg.projects if agent_cfg is not None else []
        if len(projects) == 1:
            project_id = projects[0]
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"无法确定实时评测的 project_id（service={meta.service_name} "
                    f"registry projects={projects or '未登记'}），请在请求中传 "
                    "project_id 或在数据集采样配置中填写 projectId"
                ),
            )

    # 数据集采样条件 + 页面级覆盖 → evaluator 配置（环境级字段未携带时
    # 由 stamp_eval_entry 盖入）。
    raw = _realtime_eval_raw(evaluator_name, document)
    # 实时流实例：优先文档已有值；否则用数据集 id（创建时含 ms）保证
    # 同过滤/仅采样率不同的多登记互不覆盖。
    stream_instance_id = (document.stream_instance_id or "").strip() or document.id.strip()
    raw["stream_instance_id"] = stream_instance_id
    # 业务定制：解析触发注册（judge refs / 归因关闭）写入条目，无注册不变。
    await _apply_realtime_registration(payload, raw)
    try:
        stamped = stamp_eval_entry(raw)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    agent_name_part = str(stamped.get("agentName", ""))
    registration_id = str(raw.get("agent_registration_id") or "").strip() or None
    stamped["streamInstanceId"] = stream_instance_id
    record_key = evaluator_record_key(
        project_id,
        evaluator_name,
        agent_name_part,
        compose_evaluator_key_tail(registration_id or "", stream_instance_id),
    )
    ingest_part = str(stamped.get("ingest") or raw.get("ingest") or "kafka")
    conflict = ingest_conflict_message(
        current_eval_config_file().configs.get(project_id, []),
        evaluator_name=evaluator_name,
        ingest=ingest_part,
    )
    if conflict is not None:
        raise HTTPException(status_code=400, detail=conflict)
    # 同流实例重复提交：只清该 stream 槽内其它 agentName 变体。
    # 不同 stream / registration 并行各评，互不 retire。
    keep_tail = compose_evaluator_key_tail(registration_id or "", stream_instance_id)
    await retire_sibling_evaluator_entries(
        project_id,
        evaluator_name,
        agent_name_part,
        keep_registration_id=keep_tail,
    )
    await upsert_runtime_config(
        kind=KIND_EVALUATOR,
        record_key=record_key,
        payload=stamped,
        scope_key=project_scope_key(project_id),
    )
    # 立即刷新快照（不等 30s 定时任务），实时链尽快接管。
    await refresh_eval_config_snapshot()
    # 自动登记 T+1 报表项：命名空间含 ``__ts__{streamInstanceId}``，
    # 同日多次 / 仅采样率不同的多实时流互不串报。
    judge_cv_raw = raw.get("judge_content_version")
    judge_cv = (
        str(judge_cv_raw).strip()
        if judge_cv_raw is not None and str(judge_cv_raw).strip()
        else None
    )
    report_ns = await ensure_default_trace_report(
        TraceReportSeed(
            name=document.name,
            benchmark_name=payload.benchmark_name,
            source=str(raw["ingest"]),
            agent_name=agent_name_part or None,
            project_id=project_id,
            agent_registration_id=registration_id,
            stream_instance_id=stream_instance_id,
            judge_content_version=judge_cv,
        )
    )
    # evaluator_name / stream_instance_id 持久化进数据集文档，便于停采样
    # 精确定位条目；顺带完成 ready→sampling 状态切换。
    updated = document.model_copy(
        update={
            "evaluator_name": evaluator_name,
            "stream_instance_id": stream_instance_id,
        },
    )
    if document.status == "ready":
        updated = start_sampling(updated)
    await save_dataset(updated)

    logger.info(
        "run-external.realtime_activated benchmark=%s dataset=%s "
        "registration_id=%s stream=%s slot=%s project=%s ingest=%s "
        "sampling=%s record_key=%s report_ns=%s",
        payload.benchmark_name,
        payload.dataset_name,
        registration_id or "-",
        stream_instance_id,
        (
            f"{evaluator_name}@{registration_id}@{stream_instance_id}"
            if registration_id
            else f"{evaluator_name}@{stream_instance_id}"
        ),
        project_id,
        raw["ingest"],
        raw["sampling"],
        record_key,
        report_ns or "-",
    )
    return JSONResponse(
        status_code=202,
        content={
            "run_id": "",
            "benchmark_name": payload.benchmark_name,
            "dataset_name": payload.dataset_name,
            "status": "sampling",
            "total_samples": 0,
            "mode": "realtime",
            "evaluator_name": evaluator_name,
            "project_id": project_id,
            "message": (
                "realtime dataset: streaming eval activated "
                f"(evaluator={evaluator_name}, project={project_id}); "
                "no batch run created; results land in trace_pipeline_records"
            ),
        },
    )


async def _find_dataset_document(
    payload: ExternalBenchmarkRequest,
) -> "DatasetDocument | None":
    """按请求反查数据集文档：带 dataset_id 时按 id 精确命中优先。

    同名数据集可有多条文档（同文件多次上传），按名查只能取首命中，
    同名改内容重传时会拼到另一条文档的 OSS 路径；未带 id 的存量
    调用方仍按名查。
    """
    if payload.dataset_id:
        document = await find_dataset_document_by_id(payload.dataset_id)
        if document is not None:
            return document
    if payload.dataset_name:
        return await find_dataset_document_by_name(payload.dataset_name)
    return None


def _apply_dataset_oss_path(
    payload: ExternalBenchmarkRequest,
    dataset_document: "DatasetDocument | None",
) -> None:
    """已上传数据集 OSS 路径拼装（触发入口统一收口，服务端从 DB 拉取）。

    数据集文档只存 ``oss_prefix`` + ``oss_partition_date``（上传落盘
    分区）+ ``oss_upload_id``（每次上传唯一批次段），触发时在此拼出
    完整 object key 写入 payload：
    ``{prefix}/{oss_partition_date}/{oss_upload_id}/benchmark_final_results.csv``。
    文档带分区日期即以文档为准（覆盖请求侧任何 oss_object_key），
    不按触发日重拼分区、不保留冻结整串。存量文档无 upload_id 时按
    旧布局 ``{prefix}/{date}/{file}`` 拼（其文件就躺在旧布局）。
    管道日产型数据集文档无分区日期，不在此处理（按评测日期取当日
    分区，loader 侧拼装）。
    """
    if dataset_document is None or not dataset_document.oss_partition_date:
        return
    prefix = (dataset_document.oss_prefix or payload.oss_prefix or "").strip()
    if not prefix:
        return
    key_path = DailyDataLoader._strip_oss_scheme(prefix).rstrip("/")
    partition = f"{key_path}/{dataset_document.oss_partition_date}"
    if dataset_document.oss_upload_id:
        partition = f"{partition}/{dataset_document.oss_upload_id}"
    payload.oss_object_key = f"{partition}/{DailyDataLoader.FILE_NAME}"
    if dataset_document.oss_prefix:
        payload.oss_prefix = dataset_document.oss_prefix


def _apply_dataset_oss_date_macro(
    payload: ExternalBenchmarkRequest,
    dataset_document: "DatasetDocument | None",
    *,
    base_date: str,
) -> bool:
    """OSS 前缀日期宏展开（``${yyyymmdd±n}``），返回是否展开。

    - 有 ``source=oss`` 数据集文档：CSV/JSON 上传与实时采样、带
      ``oss_partition_date`` 的上传快照不参与。请求仍带宏时以请求为准。
    - 请求已是字面 ``yyyymmdd`` 分区而文档仍含宏：**信任请求字面路径**，
      不再用文档宏按 ``base_date`` 重算。否则页面/占号前已展开后，若
      ``base_date`` 又是该字面日，``${yyyymmdd±n}`` 会二次偏移
      （``-1``→再早一天，``+1``→再晚一天）；``${yyyymmdd}`` 也会被错误
      基准改写分区。
    - 请求无字面分区、文档含宏时，才回退文档宏按 ``base_date`` 展开。
    - 无数据集文档（API 直调 / RemoteConfig 调度只带 ``oss_prefix``）：仍展开
      请求侧宏，避免字面 ``${yyyymmdd}`` 进 loader。

    宏按调度日/显式日 ``base_date`` 展开后即数据文件所在目录：调用方
    据此把 object key 定死为 ``{展开前缀}/benchmark_final_results.csv``，
    不再追加评测日期分区（宏本身已是日期分区）。

    Raises:
        HTTPException: 宏语法非法（422，与 oss_prefix 归一校验同口径）。
    """
    req_prefix = (payload.oss_prefix or "").strip()
    if dataset_document is None:
        prefix = req_prefix
    elif dataset_document.source != "oss" or dataset_document.oss_partition_date:
        return False
    else:
        doc_prefix = (dataset_document.oss_prefix or "").strip()
        if has_date_macro(doc_prefix) and not has_date_macro(req_prefix):
            # 请求已带字面分区日：禁止文档宏重算（防二次偏移）
            if partition_date_from_path(req_prefix) is not None:
                prefix = req_prefix
            else:
                prefix = doc_prefix
        else:
            prefix = req_prefix or doc_prefix
    if not prefix:
        return False
    payload.oss_prefix = prefix
    try:
        # 非法宏不放过：即使不含合法宏，`${...}` 也不能原样透传到 OSS
        validate_date_macros(prefix)
        if not has_date_macro(prefix):
            return False
        payload.oss_prefix = expand_date_macros(prefix, base_date=base_date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info(
        "dataset.oss_date_macro_expanded dataset_id=%s date=%s prefix=%s",
        getattr(dataset_document, "id", None),
        base_date,
        payload.oss_prefix,
    )
    return True


def _macro_oss_object_key(effective_prefix: str) -> str:
    """宏展开后的数据文件 key：展开前缀即目录，不再追加评测日期分区。"""
    key_path = strip_oss_scheme(effective_prefix).rstrip("/")
    return f"{key_path}/{DailyDataLoader.FILE_NAME}"


def _resolve_oss_target(
    payload: ExternalBenchmarkRequest,
    dataset_document: "DatasetDocument | None",
    *,
    base_date: str,
) -> str:
    """归一本次 run 的 OSS 前缀；已是日期分区的前缀定死 object key。

    Returns:
        归一后的 OSS prefix（前缀即完整路径，不再拼接公共前缀）。

    Raises:
        HTTPException: 前缀或日期宏非法（422）。
    """
    macro_expanded = _apply_dataset_oss_date_macro(
        payload,
        dataset_document,
        base_date=base_date,
    )
    try:
        effective = resolve_oss_prefix(
            payload.oss_prefix,
            default_prefix=get_default_download_prefix(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if macro_expanded:
        # 宏即日期分区：数据文件直接落在展开前缀下，不再追加评测日期
        payload.oss_object_key = _macro_oss_object_key(effective)
    elif not payload.oss_object_key:
        # 文档无宏但请求/快照已带任意 yyyymmdd 尾段（立即预展开、偏移日等）
        pinned = object_key_for_dated_prefix(
            effective,
            file_name=DailyDataLoader.FILE_NAME,
        )
        if pinned:
            payload.oss_object_key = pinned
    return effective


async def _ensure_snapshot_bundles(snapshot: dict[str, Any]) -> None:
    """受理期按 commit 拉取快照中的 bundle（失败 503、run 不创建）。

    归因已关闭（``attribution_enabled=false``）时跳过 attribution stage：
    即使快照里存有归因定制配置也不拉取、不执行，拉取失败更不阻塞 run。
    """
    customization = snapshot.get("customization") or {}
    stages = ("judge", "attribution")
    if not snapshot.get("attribution_enabled", True):
        stages = ("judge",)
    refs = [
        customization.get(stage) for stage in stages if isinstance(customization.get(stage), dict)
    ]
    if not refs:
        return
    from benchmark.customization.bundle_store import get_bundle_store  # noqa: PLC0415
    from benchmark.customization.errors import CustomizationError  # noqa: PLC0415

    store = get_bundle_store()
    for ref in refs:
        try:
            await store.ensure(ref)
        except CustomizationError as exc:
            raise HTTPException(
                status_code=503,
                detail={"message": exc.detail, "reason": "customization_fetch_failed"},
            ) from exc


async def _apply_realtime_registration(
    payload: ExternalBenchmarkRequest,
    raw: dict[str, Any],
) -> None:
    """实时流登记业务定制：注册 → judge refs / 归因关闭写入条目 raw（stamp 前）。

    无注册 / 无定制时 raw 不变（行为与存量一致）；带定制先按 commit 预物化
    bundle（失败 503 且不登记条目）；policy 已回收 409；归因关闭写
    ``attribution_execution=none``，消费侧据此跳过 triage/归因。
    """
    try:
        document = await resolve_agent_registration(
            payload.benchmark_name,
            payload.agent_registration_id,
        )
        if document is not None:
            check_customization_policy(
                document,
                status_code=409,
                code="customization_no_longer_allowed",
            )
    except ConsoleError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    if document is None:
        return
    snapshot = build_registration_snapshot(document)
    raw["agent_registration_id"] = snapshot["id"]
    # 多实例归属键（设计 §12.2）：写 entry 独立字段，随请求 metadata
    # 透传落记录；不写 agentName——那会污染 trace 的 agent 过滤/注入语义。
    if snapshot.get("internal_agent_name"):
        raw["internal_agent_name"] = snapshot["internal_agent_name"]
    prompt_ref, skill_ref = bundle_refs_from_snapshot(snapshot, "judge")
    if prompt_ref is not None:
        raw["prompt_bundle_ref"] = prompt_ref.model_dump()
    if skill_ref is not None:
        raw["skill_bundle_ref"] = skill_ref.model_dump()
    # T+1 出报 / 调用次数版本隔离：judge content_version 直写 raw，
    # 供 ensure_default_trace_report 采样过滤（与落库 lineage 对齐）。
    if document.customization is not None and document.customization.judge is not None:
        raw["judge_content_version"] = str(document.customization.judge.content_version)
    if not snapshot.get("attribution_enabled", True):
        raw["attribution_execution"] = "none"
    await _ensure_snapshot_bundles(snapshot)
    logger.info(
        "customization.realtime_registered benchmark=%s registration=%s attribution_enabled=%s",
        payload.benchmark_name,
        snapshot.get("id"),
        snapshot.get("attribution_enabled", True),
    )


async def _resolve_run_registration(
    payload: ExternalBenchmarkRequest,
    registration_snapshot: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, ExternalBenchmarkRequest]:
    """解析本次 run 的注册：返回快照与（可能被归因开关改写的）payload。

    克隆继承的快照优先（不重新解析，注册已删也可复现）；否则按显式 id /
    唯一 / 默认 / 存量规则解析。attribution 关闭时强制 no_attribution
    （优先级高于请求与默认值），attribution_only 模式直接 400。
    """
    snapshot = registration_snapshot
    if snapshot is None:
        try:
            document = await resolve_agent_registration(
                payload.benchmark_name,
                payload.agent_registration_id,
            )
            if document is not None:
                check_customization_policy(
                    document,
                    status_code=409,
                    code="customization_no_longer_allowed",
                )
        except ConsoleError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        if document is not None:
            snapshot = build_registration_snapshot(document)
    if snapshot is None:
        return None, payload
    if not snapshot.get("attribution_enabled", True):
        if payload.mode == "attribution_only":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "注册已关闭 attribution，attribution_only 模式不可用",
                    "reason": "attribution_disabled",
                },
            )
        payload = payload.model_copy(update={"attribution_class": "no_attribution"})
    await _ensure_snapshot_bundles(snapshot)
    customization = snapshot.get("customization") or {}
    logger.info(
        "customization.resolved benchmark=%s registration=%s version=%s "
        "attribution_enabled=%s judge_version=%s attribution_version=%s",
        payload.benchmark_name,
        snapshot.get("id"),
        snapshot.get("version"),
        snapshot.get("attribution_enabled", True),
        (customization.get("judge") or {}).get("content_version"),
        (customization.get("attribution") or {}).get("content_version"),
    )
    return snapshot, payload


async def _accept_external_run(  # noqa: PLR0915 — 批量/实时分流 + 落库编排入口
    payload: ExternalBenchmarkRequest,
    accept: AcceptExternalRunInput,
) -> ExternalRunAcceptedResponse | JSONResponse:
    """run-external 共享流程（v1 扁平契约 / v2 profile 契约汇流，§9）。

    202 立即返回 run_id；后台异步执行 OSS 拉取 + judge 提交。
    旧入口同日重跑需传 run_id_suffix。``assigned_run_id`` 由手动入口传入，
    此时不再套用 ``{name}-{date}`` 公式。``reuse_existing=False`` 时占用
    冲突抛 ``ManualRunIdTakenError``，由调用方换号重试。
    """
    benchmark_service = accept.benchmark_service
    runtime = accept.runtime
    external_runner = accept.external_runner
    assigned_run_id = accept.assigned_run_id
    data_date = accept.data_date
    reuse_existing = accept.reuse_existing
    registration_snapshot = accept.registration_snapshot

    # 1. 解析 benchmark_services
    benchmark_services = await _resolve_benchmark_services(runtime, payload.benchmark_name)
    meta = benchmark_services.get_metadata()

    # 1.2 实时采样数据集分流：source=realtime 时激活实时评测流程
    #     （登记 evaluator 配置，Kafka 实时链接管），不创建批量 run。
    #     未注册 / 非实时来源的数据集保持批量回放，行为不变。
    dataset_document: DatasetDocument | None = None
    if payload.dataset_name or payload.dataset_id:
        dataset_document = await _find_dataset_document(payload)
        if dataset_document is not None and dataset_document.source == "realtime":
            return await _activate_realtime_eval(
                payload,
                meta=meta,
                document=dataset_document,
            )
        # 1.3 已上传数据集 OSS 路径拼装（见 _apply_dataset_oss_path）
        _apply_dataset_oss_path(payload, dataset_document)

    # 1.4 注册解析：版本级选择 → 快照；attribution 关闭强制 no_attribution；
    #     带定制 ref 时受理期先拉 bundle（失败 503、run 不创建）。
    agent_snapshot, payload = await _resolve_run_registration(
        payload,
        registration_snapshot,
    )

    # 1.5 配置化运行时预校验（§5.2，400 fail-fast；initialize 内二次解析幂等）
    try:
        external_runner.profile_resolver.resolve(
            profile_from_external_request(payload),
            meta,
        )
    except ProfileResolutionError as exc:
        logger.warning(
            "profile.resolve_failed benchmark=%s reason=%s err=%s",
            payload.benchmark_name,
            exc.reason,
            exc,
        )
        raise HTTPException(
            status_code=400,
            detail={"message": str(exc), "reason": exc.reason},
        ) from exc

    # 2. 数据日 + run_id + OSS 前缀（宏只展开一次，见
    #    ``_resolve_external_run_identity``）
    identity = _resolve_external_run_identity(
        payload,
        dataset_document,
        assigned_run_id=assigned_run_id,
        data_date=data_date,
    )
    date = identity.date
    run_id = identity.run_id
    effective_oss_prefix = identity.effective_oss_prefix

    # 归一化 effective 字段（前缀即完整路径，不拼公共前缀；
    # 任何形态拒绝 ``..`` 段与空段 → 422）
    payload.effective_date = date
    payload.effective_oss_prefix = effective_oss_prefix

    # 3. 检查 run 是否已存在
    existing = await BenchmarkRun.filter(run_id=run_id).first()
    if existing is not None:
        if not reuse_existing:
            raise ManualRunIdTakenError(run_id)
        if existing.status == "failed":
            return JSONResponse(
                status_code=409,
                content={
                    "detail": (
                        f"run_id={run_id} already exists with status=failed. "
                        "DELETE the run and retry, or pass run_id_suffix."
                    ),
                },
            )
        # 非终态或已完成 → 幂等跳过
        return ExternalRunAcceptedResponse(
            id=str(existing.id),
            run_id=run_id,
            benchmark_name=payload.benchmark_name,
            dataset_name=payload.dataset_name,
            status=existing.status,
            total_samples=existing.total_samples,
            message="run already exists (idempotent)",
        )

    # 3.5 实时采样数据集已在 1.2 分流（不创建批量 run），走到这里的
    # 均为批量回放：OSS 拉取 + BullMQ case jobs + Aone judge 回调。

    # 4. 创建占位 run：列 benchmark_name 优先 registrationId（查表/报告/
    #    日志与实时流 evaluator_name 同口径）；逻辑名仍进 config。
    from benchmark.domain.eval_config import stored_benchmark_run_name  # noqa: PLC0415

    registration_id = (
        (str(agent_snapshot.get("id") or "").strip() if isinstance(agent_snapshot, dict) else "")
        or (payload.agent_registration_id or "").strip()
        or None
    )
    created = await benchmark_service.create_placeholder_run(
        run_id,
        BenchmarkRunSpec(
            benchmark_name=stored_benchmark_run_name(
                logical_name=payload.benchmark_name,
                registration_id=registration_id,
            ),
            service_name=meta.service_name,
            dataset_name=payload.dataset_name or "",
            config_json={
                "benchmark_services": payload.benchmark_name,
                "judge_class": payload.judge_class,
                "attribution_class": payload.attribution_class,
                "rubric_class": payload.rubric_class,
                "generate_report": payload.generate_report,
                # 数据定位快照占位即落：init 失败的 run 也可被再次触发克隆
                "data_date": payload.effective_date,
                "oss_prefix": payload.effective_oss_prefix,
                "oss_object_key": payload.oss_object_key,
                # 归属快照：页面报告按任务隔离只认 eval_task_id（产生时
                # 定死，克隆继承）；dataset_id 供同名数据集溯源。
                "dataset_id": payload.dataset_id,
                "eval_task_id": payload.eval_task_id,
                # 注册快照：定制版本与归因开关在受理时定死（克隆继承），
                # worker/归因/报表一律读快照不回查注册文档。
                **({"agent_registration": agent_snapshot} if agent_snapshot is not None else {}),
            },
        ),
    )
    if not created:
        if not reuse_existing:
            raise ManualRunIdTakenError(run_id)
        raced = await BenchmarkRun.filter(run_id=run_id).first()
        return ExternalRunAcceptedResponse(
            id=str(raced.id) if raced is not None else None,
            run_id=run_id,
            benchmark_name=payload.benchmark_name,
            dataset_name=payload.dataset_name,
            status="initializing",
            message="run already exists (idempotent)",
        )

    # 5. 后台执行 initialize_and_submit
    async def _bg_init() -> None:
        try:
            await external_runner.initialize_and_submit(
                run_id=run_id,
                payload=payload,
                benchmark_services=benchmark_services,
            )
        except Exception as exc:
            logger.exception("run-external.bg_init_failed run_id=%s", run_id)
            await benchmark_service.fail_run(
                run_id=run_id,
                reason=f"external init failed: {exc}",
            )

    asyncio.create_task(_bg_init(), name=f"external-init-{run_id}")

    created_row = await BenchmarkRun.filter(run_id=run_id).first()
    return ExternalRunAcceptedResponse(
        id=str(created_row.id) if created_row is not None else None,
        run_id=run_id,
        benchmark_name=payload.benchmark_name,
        dataset_name=payload.dataset_name,
        status="initializing",
        total_samples=0,
        message="accepted; OSS fetch + judge submission running in background",
    )


@router.post("/callback")
async def benchmark_callback(
    payload: dict[str, Any],
    external_runner: ExternalRunnerDep,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """统一回调端点（judge / attribution / attribution_batch，§4）。

    数据解析成功后立即返回 ``accepted``，下游处理（归因、聚合等）
    通过 ``BackgroundTasks`` 在 HTTP 响应发送后同步执行。
    根据 ``type`` 字段路由到对应的回调处理器。
    """
    cb_type = payload.get("type", "")
    logger.info(
        "callback.raw_received type=%s keys=%s payload=%s",
        cb_type,
        list(payload.keys()),
        json.dumps(payload, ensure_ascii=False),
    )
    if cb_type == "judge":
        callback = JudgeCallback.model_validate(payload)
    elif cb_type == "attribution":
        callback = AttributionCallback.model_validate(payload)
    elif cb_type == "attribution_batch":
        callback = AttributionBatchCallback.model_validate(payload)
    else:
        logger.warning("callback.unknown_type type=%s", cb_type)
        raise HTTPException(
            status_code=400,
            detail=f"unknown callback type: {cb_type}",
        )

    logger.info(
        "callback.parsed type=%s run_id=%s item_id=%s job_id=%s",
        cb_type,
        getattr(callback, "run_id", ""),
        getattr(callback, "item_id", ""),
        getattr(callback, "job_id", ""),
    )

    # 解析成功后立即返回；下游处理（handle_callback）通过 BackgroundTasks
    # 在 HTTP 响应发送后执行，不阻塞回调方。
    background_tasks.add_task(external_runner.handle_callback, callback)
    return {"status": "accepted"}
