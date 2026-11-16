"""Evaluator service layer — scheduling, before/after hooks, agent round-trip.

Thin business layer over the native agentgraph runtime:

- :class:`EvaluatorService` — select (via ``runtime.agent_manager``) → before_eval →
  eval → after_eval; in/out are the domain types :class:`EvaluatorRequest` /
  :class:`EvaluatorResult` (the OpenAI ↔ domain conversion lives in the API layer,
  :mod:`benchmark.routes.schema`).
- :func:`get_evaluator_service` / :data:`EvaluatorServiceDep` — the FastAPI DI seam
  that resolves the lifespan-scoped service (routes inject it, never read app.state).
- :func:`build_app_runtime` — load ``conf/app.<env>.yaml`` via the standard
  agentgraph ``AppConfigManager`` and build the ``AppRuntime`` that装配 + selects
  evaluators from the ``agents`` / ``models`` / ``prompts`` sections.

The request/verdict cross the agent boundary as JSON in the message / response
content (see ``encode_eval_json`` / ``decode_eval_text`` in :mod:`benchmark.evaluators.base`),
not via metadata. This layer holds no registry of its own — evaluators are装配
natively by AppRuntime.
"""

import asyncio
import json
import time
import uuid
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

from benchmark.libs.agenthost.agents.base import BaseAgent
from benchmark.libs.agenthost.agents.request import (
    AgentRequest,
    AgentResponse,
    ChatMessage,
    ResponseStatus,
)
from benchmark.libs.agenthost.app import AppConfigManager, AppRuntime
from benchmark.libs.agenthost.app.config.plugin import register_section_extension
from benchmark.libs.agenthost.init import update_request_context
from benchmark.libs.agenthost.observability.logger import get_logger
from fastapi import Depends, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse import Langfuse
from pydantic import BaseModel, ConfigDict, JsonValue

from benchmark.routes.schema import (
    ChatCompletion,
    ChatCompletionRequest,
    ToolCall,
    ToolCallFunction,
    build_proxy_chat_completion,
    validate_evaluator_request,
)
from benchmark.settings.setting import settings
from benchmark.domain.enums import (
    PIPELINE_SOURCE_METADATA_KEY,
    PipelineSource,
    format_dataset_pipeline_source,
    is_dataset_pipeline_source,
)
from benchmark.domain.eval_config import evaluator_slot_from_request, persist_evaluator_name
from benchmark.domain.schemas import ScoreWritebackPolicy
from benchmark.evaluators.base import (
    EvaluatorRequest,
    EvaluatorResult,
    decode_eval_text,
    encode_eval_json,
)
from benchmark.service.bootstrap import AgentEvalRuntime
from benchmark.services._stage_key import STAGE_JUDGE, stage_agent_key
from benchmark.runs.common.instance_limiter import get_instance_limiter
from benchmark.runs.common.trace_url_builder import build_direct_trace_url

SEARCH_EVAL_BENCHMARKS: frozenset[str] = frozenset()
from benchmark.services.metrics import TraceMetricCollector, unknown_observations
from benchmark.services.source_score_client import (
    ProjectScoreClientProvider,
    SourceScoreClientProvider,
)
from benchmark.services.trace_record_service import TraceRecordService
from benchmark.trace.http_backend import TraceHttpError
from benchmark.trace.tools import (
    TraceToolExt,
    TraceToolProvider,
    bind_sidecar_observations,
)
from benchmark.services.trace_writer import DailyTraceRecordWriter, ScoreWrite
from benchmark.attribution.service.service import TriageRunInput
from benchmark.ingest.sidecar.observations import _SIDECAR_OBSERVATIONS_KEY
from benchmark.utils.exceptions import (
    EVALUATION_FAILED_CODE,
    INCOMPLETE_TRACE_IO_CODE,
    TRACE_UNAVAILABLE_CODE,
    LLM_FAILED_CODE,
    LLM_SCHEMA_INVALID_CODE,
    LLM_UNPARSEABLE_CODE,
    METADATA_MISSING_CODE,
    MODEL_NOT_FOUND_CODE,
    PROMPT_NOT_FOUND_CODE,
    TRACE_NOT_FOUND_CODE,
    EvaluatorError,
    evaluator_status,
)
from benchmark.utils.trace_report import (
    eval_report_root,
    finish_report_child_span,
    report_child_scope,
    start_report_child_span,
)
from benchmark.utils.trace_score import JudgeScoreWrite, write_judge_score
from benchmark.utils.trace_naming import build_eval_trace_name

if TYPE_CHECKING:
    from benchmark.services.trace_writer import TraceRecordWriter
    from benchmark.attribution.service.service import TriageAttributionService

logger = get_logger("agent_eval.evaluator")


def judge_report(
    evaluator_name: str,
    request: EvaluatorRequest,
    *,
    enabled: bool = True,
) -> AbstractContextManager[Any]:
    """打开评测 Langfuse 根 span，并先写上 workflow 名。

    ``enabled=False`` 时仍更新请求上下文，但不建 span（benchmark 批量）。
    HTTP、洛书 MQ、sidecar 共用这一处，避免两套根 span 字段。
    """
    trace_name = build_eval_trace_name(request.agent_name)
    update_request_context(workflow_name=trace_name, agent_name=request.agent_name)
    if not enabled:
        return nullcontext()
    return eval_report_root(
        name=trace_name,
        input_value={
            "evaluator_name": evaluator_name,
            "trace_id": request.trace_id,
            "project_id": request.project_id,
            "agent_name": request.agent_name,
            "stage": STAGE_JUDGE,
        },
        metadata={"evaluator_name": evaluator_name, "stage": STAGE_JUDGE},
    )


class _ReportRoot(Protocol):
    """Langfuse 根 span 的最小写回面（``eval_report_root`` 的产出）。"""

    def update(self, *, output: Mapping[str, object]) -> object: ...


def update_eval_report_score(root: _ReportRoot | None, result: EvaluatorResult) -> None:
    """把分数写回已打开的 Langfuse 根 span。根不存在时跳过。"""
    if root is None:
        return
    try:
        root.update(output={"status": "completed", "score": result.score})
    except Exception:
        logger.debug("langfuse_report.root_update_failed", exc_info=True)


#: TriageOutcome.status values that fail the whole evaluate() request.
#: ``skipped`` / ``created`` remain success (good/watch skip or full success).
_TRIAGE_FAILURE_STATUSES: frozenset[str] = frozenset(
    {"attribution_failed", "dispatch_failed"},
)
_METRIC_COLLECTION_TIMEOUT_S = 1.0


# ════════════════════════════════════════════════════════════════════════════
# Service — select (runtime.agent_manager) → before_eval → eval → after_eval
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EvalRunOptions:
    """评测执行选项（``EvaluatorService.evaluate`` / ``after_eval`` 共用参数对象）。

    ``evaluate`` 消费全部字段；``after_eval`` 消费
    ``langfuse_client`` / ``skip_triage`` / ``writer`` / ``score_writeback`` /
    ``triage_classifier`` / ``attribution_agent_key`` / ``attribution_execution``，
    与 ``evaluate`` 内部透传保持同一对象。
    """

    resolve_trace: bool = True
    trace_client: Langfuse | None = None
    skip_triage: bool = False
    writer: "TraceRecordWriter | None" = None
    enable_trace_report: bool = True
    score_writeback: ScoreWritebackPolicy | None = None
    sidecar_observations: list[Any] | None = None
    #: streaming 任务级 triage 分类器覆盖（``tasks[].triage_class``）。
    triage_classifier: Any | None = None
    #: streaming 任务级 attribution agent 覆盖（``tasks[].attribution_class``）。
    attribution_agent_key: str | None = None
    #: streaming 归因执行方式：``sync`` / ``async`` / ``none``。
    attribution_execution: Literal["sync", "async", "none"] = "sync"


@dataclass(frozen=True)
class ScoreWritebackCall:
    """一次源项目分数回写所需的上下文。"""

    evaluator_name: str
    request: EvaluatorRequest
    result: EvaluatorResult
    policy: ScoreWritebackPolicy | None


class EvaluatorService:
    """Evaluator scheduling + before/after hooks over the domain types."""

    def __init__(
        self,
        *,
        runtime: AppRuntime,
        trace_record_service: TraceRecordService,
        triage_service: "TriageAttributionService | None" = None,
        llm_max_concurrency: int | None = None,
        source_score_client_provider: SourceScoreClientProvider | None = None,
    ) -> None:
        self._runtime = runtime
        self._trace_record_service = trace_record_service
        self._triage_service = triage_service
        runtime_settings = getattr(runtime, "settings", None) or settings
        self._source_score_client_provider = (
            source_score_client_provider
            or ProjectScoreClientProvider(
                trace_instances=runtime_settings.trace_instances,
                timeout_seconds=runtime_settings.trace_http_timeout_s,
            )
        )
        # LLM concurrency semaphore — shared across Path A (HTTP) + Path B (Kafka consumer)
        # within the same process; covers judge `_eval` + sync `after_eval` (triage).
        # IMPORTANT: this is a per-process semaphore, NOT a distributed rate limiter.
        # In multi-node deployments, each node has its own independent semaphore.
        # Set ``judge_kafka_concurrency`` per-node so that
        # node_count × judge_kafka_concurrency ≤ LLM provider rate limit.

        max_conc = (
            llm_max_concurrency if llm_max_concurrency is not None else settings.llm_max_concurrency
        )
        self._llm_semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_conc) if max_conc > 0 else None
        )
        self._default_writer: TraceRecordWriter | None = None

    async def evaluate(
        self,
        evaluator_name: str,
        request: EvaluatorRequest,
        *,
        options: EvalRunOptions | None = None,
    ) -> EvaluatorResult:
        """Select the evaluator and run before_eval → eval → after_eval.

        The request is already parsed (the API layer did the OpenAI → domain
        conversion); this layer only schedules. Raises :class:`EvaluatorError`
        (code + message) on any failure; the API layer maps the code to an HTTP
        status.

        ``options.resolve_trace`` controls whether ``before_eval`` runs（业务
        trace_id → Langfuse 主键解析）。缺省 ``True`` 保持既有行为；调用方（如
        MQ handler 透传消息声明）对自带数据、``trace_id`` 已是主键的来源传
        ``False`` 即可跳过。

        ``options.enable_langfuse_report`` controls ``eval_report_root``
        observability spans. Benchmark batch passes ``False`` (PG-only results,
        no Langfuse writeback); dataset-eval / HTTP / 洛书 MQ keep the default
        ``True``.

        Sidecar ingest 不调用本方法。它由
        ``BenchmarkStreamingEvaluator.judge_ingested`` 打开同一个
        :func:`judge_report`，再调用 :meth:`judge_and_record`。
        """
        opts = options or EvalRunOptions()

        sidecar_obs = opts.sidecar_observations
        if not isinstance(sidecar_obs, list):
            sidecar_obs = (request.metadata or {}).get(_SIDECAR_OBSERVATIONS_KEY)
        sidecar_cm = (
            bind_sidecar_observations(sidecar_obs)
            if isinstance(sidecar_obs, list)
            else nullcontext()
        )

        report_cm = judge_report(
            evaluator_name,
            request,
            enabled=opts.enable_trace_report,
        )
        with sidecar_cm, report_cm as root:
            result = await self.judge_and_record(evaluator_name, request, options=opts)
            update_eval_report_score(root, result)
            return result

    async def judge_and_record(
        self,
        evaluator_name: str,
        request: EvaluatorRequest,
        *,
        options: EvalRunOptions | None = None,
    ) -> EvaluatorResult:
        """算分并落库。不打开 Langfuse 根 span。

        调用方用 :func:`judge_report` 包住本方法，子 span 才能挂到同一根上。
        ``resolve_trace=False`` 时跳过主键解析。
        """
        opts = options or EvalRunOptions()
        started = time.perf_counter()
        missing = validate_evaluator_request(request)
        if missing:
            message = f"required metadata missing: {', '.join(missing)}"
            logger.error(
                "evaluator.metadata_missing slot=%s missing=%s",
                evaluator_slot_from_request(evaluator_name, request.metadata),
                ",".join(missing),
            )
            raise EvaluatorError(
                code=evaluator_status(METADATA_MISSING_CODE),
                error_type=METADATA_MISSING_CODE,
                message=message,
            )
        # Dataset-eval：Langfuse expected_output → request.expected_output；
        # 既有 judge 仍读 request.output，此处在 output 为空时桥接。
        if request.output is None and request.expected_output is not None:
            request = request.model_copy(update={"output": request.expected_output})
        agent = self._select(evaluator_name)
        # before_eval（业务 trace_id → Langfuse 主键解析）仅对
        # 「依赖平台 trace 数据」的评测器有意义；由调用方通过
        # resolve_trace 决定是否执行（handler 透传消息声明；
        # 缺省 True 保持既有行为）。
        if opts.resolve_trace:
            request = await self.before_eval(request)
        # Enrich request input/output from Langfuse trace when empty.
        # All paths (HTTP / Kafka / benchmark / dataset) benefit: Kafka
        # already has IO from the trace record so this is a no-op; others
        # fetch trace IO via TraceResolver so trace_meta_json is populated.
        request = await self._enrich_request_io(request)
        request = await self._enrich_vui_observation_context(evaluator_name, request)
        # LLM 信号量限流：Path A (HTTP) + sidecar benchmark judge 共享。
        # judge + 同步 triage/attribution 必须同槽位（勿把 after_eval 挪到外面）。
        # nullcontext：无信号量时也走同一路径，避免 if/else 双份 after_eval。
        llm_slot = self._llm_semaphore if self._llm_semaphore is not None else nullcontext()
        # 多实例分桶（§12.5 F1）：先过实例桶再过全局桶，被实例桶挡住的
        # 请求不白占全局槽位；无内部名（存量链路）直通，行为不变。
        internal_name = (
            request.metadata.get("internal_agent_name")
            if isinstance(request.metadata, dict)
            else None
        )
        async with get_instance_limiter().acquire(internal_name), llm_slot:
            result = await self._eval(agent, request)
            await self.after_eval(
                evaluator_name,
                request,
                result,
                options=opts,
            )
        logger.info(
            "evaluator.done slot=%s trace_id=%s project_id=%s score=%s elapsed_ms=%.2f",
            evaluator_slot_from_request(evaluator_name, request.metadata),
            request.trace_id,
            request.project_id,
            result.score,
            (time.perf_counter() - started) * 1000,
        )
        return result

    def _select(self, evaluator_name: str) -> BaseAgent:
        agent_key = stage_agent_key(STAGE_JUDGE, evaluator_name)
        try:
            return self._runtime.agent_manager.get(agent_key)
        except (KeyError, LookupError) as exc:
            logger.warning(
                "evaluator.select_miss evaluator_name=%r agent_key=%r",
                evaluator_name,
                agent_key,
            )
            raise EvaluatorError(
                code=evaluator_status(MODEL_NOT_FOUND_CODE),
                error_type=MODEL_NOT_FOUND_CODE,
                message=f"Unknown evaluator model {evaluator_name!r}.",
            ) from exc

    async def before_eval(self, request: EvaluatorRequest) -> EvaluatorRequest:
        """Rewrite ``trace_id`` from business metadata id → Langfuse primary id.

        Uses ``TraceResolver.resolve_primary_id_for`` (metadata + lookback
        datetime filter) once. Downstream judge / attribution / DB / trace
        tools all see the primary id.
        On miss raises :class:`EvaluatorError` ``trace_not_found`` (HTTP 400).
        On Langfuse transport / non-200 raises ``langfuse_unavailable`` (502)
        — never collapse upstream failures into not-found.
        """
        if not request.trace_id or not request.project_id:
            return request

        # Kafka consumer path (Path B) and dataset eval path already have
        # the Langfuse primary trace ID; skip metadata-based resolution
        # that would fail with trace_not_found.
        if request.skip_trace_resolve:
            return request

        provider = self._resolve_trace_provider()
        if provider is None:
            logger.warning(
                "evaluator.trace_resolve_skipped reason=no_provider trace_id=%s project_id=%s",
                request.trace_id,
                request.project_id,
            )
            return request

        business_id = request.trace_id
        resolve_span = start_report_child_span(
            name="trace.resolve_primary_id",
            input_value={
                "business_trace_id": business_id,
                "project_id": request.project_id,
            },
        )
        try:
            primary_id = await provider.resolver.resolve_primary_id_for(
                request.project_id,
                business_id,
            )
        except asyncio.CancelledError as exc:
            finish_report_child_span(resolve_span, exc=exc)
            raise
        except TraceHttpError as exc:
            finish_report_child_span(resolve_span, exc=exc)
            logger.warning(
                "evaluator.trace_resolve_langfuse_failed business_trace_id=%s "
                "project_id=%s status=%s error=%s",
                business_id,
                request.project_id,
                exc.status_code,
                exc,
            )
            raise EvaluatorError(
                code=evaluator_status(TRACE_UNAVAILABLE_CODE),
                error_type=TRACE_UNAVAILABLE_CODE,
                message=(
                    f"langfuse unavailable while resolving trace: "
                    f"trace_id={business_id!r} project_id={request.project_id!r} "
                    f"status={exc.status_code} error={exc}"
                ),
            ) from exc
        except Exception as exc:
            finish_report_child_span(resolve_span, exc=exc)
            logger.warning(
                "evaluator.trace_resolve_raised business_trace_id=%s project_id=%s error_type=%s",
                business_id,
                request.project_id,
                type(exc).__name__,
            )
            raise EvaluatorError(
                code=evaluator_status(TRACE_UNAVAILABLE_CODE),
                error_type=TRACE_UNAVAILABLE_CODE,
                message=(
                    f"langfuse resolve raised {type(exc).__name__}: "
                    f"trace_id={business_id!r} project_id={request.project_id!r}"
                ),
            ) from exc

        if primary_id is None:
            finish_report_child_span(
                resolve_span,
                output={
                    "business_trace_id": business_id,
                    "project_id": request.project_id,
                    "primary_id": None,
                    "status": "not_found",
                },
            )
            logger.warning(
                "evaluator.trace_resolve_miss business_trace_id=%s project_id=%s",
                business_id,
                request.project_id,
            )
            raise EvaluatorError(
                code=evaluator_status(TRACE_NOT_FOUND_CODE),
                error_type=TRACE_NOT_FOUND_CODE,
                message=(
                    f"trace not found in Langfuse: "
                    f"trace_id={business_id!r} project_id={request.project_id!r}"
                ),
            )

        finish_report_child_span(
            resolve_span,
            output={
                "business_trace_id": business_id,
                "project_id": request.project_id,
                "primary_id": primary_id,
                "status": "resolved",
            },
        )
        # Preserve the business ID for attribution; trace_id now holds the
        # resolved primary ID used by optional IO enrichment.
        meta = dict(request.metadata or {})
        meta.setdefault("business_trace_id", business_id)

        if primary_id == business_id:
            # trace_id is already the primary id; only update metadata.
            return request.model_copy(update={"metadata": meta})

        meta["trace_id"] = primary_id
        logger.info(
            "evaluator.trace_resolved business_trace_id=%s primary_id=%s project_id=%s",
            business_id,
            primary_id,
            request.project_id,
        )
        return request.model_copy(update={"trace_id": primary_id, "metadata": meta})

    def _resolve_trace_provider(self) -> TraceToolProvider | None:
        """Return ``TraceToolProvider`` from runtime tool_manager, or ``None``."""
        tool_manager = getattr(self._runtime, "tool_manager", None)
        if tool_manager is None:
            return None
        try:
            provider = tool_manager.get_provider("trace")
        except (KeyError, LookupError):
            return None

        if isinstance(provider, TraceToolProvider):
            return provider
        return None

    async def _enrich_request_io(self, request: EvaluatorRequest) -> EvaluatorRequest:
        """Backfill missing trace IO from the Langfuse primary trace.

        Kafka may publish lifecycle snapshots before one or both IO fields are
        persisted. Only ``None`` denotes a missing field: empty strings and
        empty JSON objects are valid payloads and must not be replaced. Kafka
        requests already carry the Langfuse primary trace ID and enforce complete IO.
        Other callers retain best-effort enrichment, including resolve_trace=False.
        """
        missing_input = request.input is None
        missing_output = request.output is None
        if not missing_input and not missing_output:
            return request

        trace_id = request.trace_id
        project_id = request.project_id
        missing_fields = ",".join(
            field
            for field, missing in (("input", missing_input), ("output", missing_output))
            if missing
        )
        source = (request.metadata or {}).get(PIPELINE_SOURCE_METADATA_KEY)
        strict_io = source == PipelineSource.KAFKA.value
        meta = request.metadata or {}
        # search_eval 族：CSV 已带全量 IO，禁止再打 Langfuse 回填。
        # 其他 benchmark（example / amap_chat 等）仍可依赖 trace IO 回填，不在此跳过。
        benchmark_name = meta.get("benchmark_name") or meta.get("benchmark_services")
        skip_search_eval_enrich = benchmark_name in SEARCH_EVAL_BENCHMARKS
        # Dataset and sidecar items have no Langfuse root trace to fetch.
        if (
            not trace_id
            or trace_id.startswith("dataset:")
            or not project_id
            or source == PipelineSource.SIDECAR.value
            or skip_search_eval_enrich
            or isinstance(meta.get(_SIDECAR_OBSERVATIONS_KEY), list)
        ):
            return request

        provider = self._resolve_trace_provider()
        if provider is None:
            if not strict_io:
                return request
            raise EvaluatorError(
                code=evaluator_status(TRACE_UNAVAILABLE_CODE),
                error_type=TRACE_UNAVAILABLE_CODE,
                message=(
                    f"trace IO incomplete: missing={missing_fields}; "
                    f"trace provider unavailable for trace_id={trace_id!r} "
                    f"project_id={project_id!r}"
                ),
            )

        try:
            trace = await provider.resolver.fetch_full_trace(
                project_id,
                trace_id,
                fields="core,io",
            )
        except Exception as exc:
            logger.warning(
                "evaluator.trace_io_enrich_failed trace_id=%s project_id=%s",
                trace_id,
                project_id,
                exc_info=True,
            )
            if not strict_io:
                return request
            error_code = (
                TRACE_UNAVAILABLE_CODE
                if isinstance(exc, (TraceHttpError, TimeoutError, OSError))
                else INCOMPLETE_TRACE_IO_CODE
            )
            raise EvaluatorError(
                code=evaluator_status(error_code),
                error_type=error_code,
                message=(
                    f"trace IO incomplete: full trace fetch failed for "
                    f"trace_id={trace_id!r} project_id={project_id!r}: {type(exc).__name__}"
                ),
            ) from exc

        updates: dict[str, Any] = {}
        if trace is not None:
            if missing_input and trace.input is not None:
                updates["input"] = trace.input
            if missing_output and trace.output is not None:
                updates["output"] = trace.output
        enriched = request.model_copy(update=updates) if updates else request
        if strict_io and (enriched.input is None or enriched.output is None):
            logger.info(
                "evaluator.trace_io_incomplete trace_id=%s project_id=%s missing=%s",
                trace_id,
                project_id,
                ",".join(
                    field
                    for field, value in (("input", enriched.input), ("output", enriched.output))
                    if value is None
                ),
            )
            raise EvaluatorError(
                code=evaluator_status(INCOMPLETE_TRACE_IO_CODE),
                error_type=INCOMPLETE_TRACE_IO_CODE,
                message=(
                    f"trace IO incomplete after full trace fetch: "
                    f"trace_id={trace_id!r} project_id={project_id!r}"
                ),
            )

        logger.info(
            "evaluator.trace_io_enriched trace_id=%s fields=%s",
            trace_id,
            ",".join(sorted(updates)),
        )
        return enriched

    async def _enrich_vui_observation_context(
        self,
        evaluator_name: str,
        request: EvaluatorRequest,
    ) -> EvaluatorRequest:
        """为旧 VUI 请求附加一页原始观测，证据统一由 evaluator 装配。

        仅作用于启用 trace-provider 开关的 ``vui_agent``；调用方显式提供的
        observations（包括空列表）不再重复读取。不请求观测详情、不在业务层
        重试；失败非致命，原始 input/output 保持不变。
        """

        if (
            evaluator_name != "vui_agent"
            or request.observations is not None
            or request.input is None
            or not request.trace_id
            or not request.project_id
            or not needs_observation_context_fallback(request.input)
        ):
            return request
        provider = self._resolve_trace_provider()
        if provider is None or not provider.enable_vui_observation_context_fallback:
            return request
        try:
            observations = await provider.resolver.fetch_observations_once(
                request.project_id,
                primary_id=request.trace_id,
            )
            compacted = compact_observations(observations)
            logger.info(
                "evaluator.vui_observation_context_fallback trace_id=%s "
                "observations=%s attached=%s",
                request.trace_id,
                len(observations),
                len(compacted),
            )
            return request.model_copy(update={"observations": compacted})
        except Exception:  # noqa: BLE001 - optional evidence fallback is non-fatal
            logger.warning(
                "evaluator.vui_observation_context_fallback_failed trace_id=%s project_id=%s",
                request.trace_id,
                request.project_id,
                exc_info=True,
            )
            return request

    async def _eval(self, agent: BaseAgent, request: EvaluatorRequest) -> EvaluatorResult:
        """Serialize the request as message-content JSON, invoke the agent, and read
        the verdict back from the response content (no metadata).

        traceid/session_id share one correlation id (falling back to a single
        generated one when no trace_id is supplied). A failed / absent response is
        turned into an :class:`EvaluatorError`.
        """

        correlation_id = request.trace_id or uuid.uuid4().hex
        agent_request = AgentRequest(
            traceid=correlation_id,
            session_id=correlation_id,
            messages=[ChatMessage(role="user", content=encode_eval_json(request))],
        )
        response = await agent.call(agent_request)

        if response is None or response.status == ResponseStatus.FAILED:
            raise self._failure_error(response)
        return EvaluatorResult.model_validate_json(decode_eval_text(response.content_parts))

    async def after_eval(
        self,
        evaluator_name: str,
        request: EvaluatorRequest,
        result: EvaluatorResult,
        *,
        options: EvalRunOptions | None = None,
    ) -> None:
        """评判结果落库；upsert TraceRecord(status=SCORED, judge_json=…)。

        ``judge_json`` 必含 ``score`` / ``reason``；若结果带 ``sub_scores`` 或非空
        ``extra`` 则一并写入（缺省不写键，保持老形状）。落库后**同步 await**
        ``TriageAttributionService.run``（如已配置），使归因在 ``eval_report_root``
        关闭前完成；未捕获异常上抛为 :class:`EvaluatorError`。

        缺 trace_id 的请求视为无法定位到业务 trace，跳过落库（evaluator 跑通但
        没法关联到任何 trace_pipeline_records 行）。**跳过路径必须 warn**—
        静默 return 会让"HTTP 200 + DB 空"的诡异组合无法从日志定位。

        合成 trace_id（``dataset:<itemId>``）同样执行落库 + 归因，使 dataset
        评测也能触发完整的 judge → DB → triage → attribution 流程。

        ``pipeline_source=benchmark`` 且未传入 ``writer`` 时提前返回：既不
        回落到实时流表，也不跑后面的 triage / attribution。批量明细由
        ``BenchmarkTraceRecordWriter`` 或 fused 路径负责；需要归因必须显式
        传入 writer。

        注意：upsert ``SCORED`` 成功后若归因失败，HTTP/MQ 会失败但 DB 已有
        judge 分。``TriageOutcome.status`` 为 ``attribution_failed`` /
        ``dispatch_failed`` 时上抛；``skipped`` / ``created`` 视为成功。
        """
        if request.trace_id is None or request.project_id is None or request.agent_name is None:
            logger.warning(
                "evaluator.after_eval.skipped trace_id=%r project_id=%r slot=%s score=%s reason=%r",
                request.trace_id,
                request.project_id,
                evaluator_slot_from_request(evaluator_name, request.metadata),
                result.score,
                result.reason,
            )
            return

        opts = options or EvalRunOptions()
        # 批量 benchmark 未传 writer：跳过实时流落库，并跳过其后的 triage。
        # 明细由 BenchmarkTraceRecordWriter / fused 负责；需要归因必须显式传 writer。
        if opts.writer is None and _is_benchmark_batch(request):
            logger.info(
                "evaluator.after_eval.skip_benchmark_side_effects trace_id=%s "
                "slot=%s reason=benchmark_batch_no_writer "
                "skipped=persist,triage",
                request.trace_id,
                evaluator_slot_from_request(evaluator_name, request.metadata),
            )
            return

        result = await self._enrich_vehicle_metrics(
            evaluator_name=evaluator_name, request=request, result=result
        )

        # 写入路径抽象：writer 优先，None 时走 DailyTraceRecordWriter
        w = opts.writer or self._default_writer
        if w is None:
            w = DailyTraceRecordWriter(self._trace_record_service)
            self._default_writer = w

        upsert_extra: dict[str, Any] = {
            "project_id": request.project_id,
            "agent_name": request.agent_name,
            "scene": request.metadata.get("scene") or "non_voice",
            "dataset_item_id": request.metadata.get("dataset_item_id"),
        }
        # Stamp pipeline source (+ optional dataset/kafka) and full IO into
        # trace_meta_json so rows are filterable / rehydratable without scalar
        # columns. Upsert shallow-merges this bucket. Kafka / 统一落库路径
        # 额外直拼 langfuse_url（trace_id 即主键），供 trace daily report
        # Trace 链接列读取。
        trace_meta = _build_trace_meta(
            request.metadata,
            input_value=request.input,
            output_value=request.output,
            project_id=request.project_id,
            trace_id=request.trace_id,
        )
        if trace_meta is not None:
            upsert_extra["trace_meta_json"] = trace_meta

        # 落库键：有 registrationId 写 stem@reg（同茎多版本隔离），否则写茎。
        persist_name = persist_evaluator_name(evaluator_name, request.metadata)
        record = await w.upsert_score(
            ScoreWrite(
                trace_id=request.trace_id,
                evaluator_ref=persist_name,
                score=result.score,
                judge_json=_judge_json_from_result(result),
            ),
            **upsert_extra,
        )

        # 将 record UUID 写入 metadata，供下游全链路 UUID 查询使用
        if record is not None:
            request.metadata["trace_record_id"] = str(record.id)

        await self._write_configured_score(
            ScoreWritebackCall(evaluator_name, request, result, opts.score_writeback)
        )

        slot = evaluator_slot_from_request(evaluator_name, request.metadata)
        if opts.skip_triage or self._triage_service is None:
            reason = "skip_triage" if opts.skip_triage else "no_service"
            logger.info(
                "evaluator.triage_skipped slot=%s trace_id=%s reason=%s",
                slot,
                request.trace_id,
                reason,
            )
            return

        logger.info(
            "evaluator.triage_started slot=%s trace_id=%s score=%s",
            slot,
            request.trace_id,
            result.score,
        )
        # Container span under still-open eval_report_root so nested
        # attribution LangGraph / LLM / tool spans inherit triage as parent.
        # TriageAttributionService.run mostly returns TriageOutcome (soft fail);
        # map failure statuses to EvaluatorError so HTTP/MQ see attribution fail.
        try:
            with report_child_scope(
                name="triage_attribution",
                input_value={
                    "evaluator_name": evaluator_name,
                    "trace_id": request.trace_id,
                    "score": result.score,
                },
            ) as span:
                outcome = await self._triage_service.run(
                    TriageRunInput(
                        evaluator_name=evaluator_name,
                        evaluator_request=request,
                        evaluator_result=result,
                        trace_client=opts.trace_client,
                        triage_classifier=opts.triage_classifier,
                        attribution_agent_key=opts.attribution_agent_key,
                        attribution_execution=opts.attribution_execution,
                    ),
                )
                if span is not None:
                    try:
                        # 展平归因结论与 notification_dispatch_json，dry_run 派单可直接在 trace 上核对。
                        span.update(output=_triage_span_output(evaluator_name, request, outcome))
                    except Exception:
                        logger.debug(
                            "langfuse_report.triage_update_failed",
                            exc_info=True,
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "evaluator.triage_failed slot=%s trace_id=%s",
                slot,
                request.trace_id,
            )
            raise EvaluatorError(
                code=evaluator_status(EVALUATION_FAILED_CODE),
                error_type=EVALUATION_FAILED_CODE,
                message=(f"triage/attribution failed: {type(exc).__name__}: {exc}"),
            ) from exc
        finally:
            await self._flush_notification_groups()

        # Raise outside report_child_scope so a disabled Langfuse CM cannot
        # swallow / rewrite the EvaluatorError on __exit__.
        if outcome.status in _TRIAGE_FAILURE_STATUSES:
            detail = outcome.error or outcome.reason or outcome.status
            logger.error(
                "evaluator.triage_failed slot=%s trace_id=%s status=%s detail=%s",
                slot,
                request.trace_id,
                outcome.status,
                detail,
            )
            raise EvaluatorError(
                code=evaluator_status(EVALUATION_FAILED_CODE),
                error_type=EVALUATION_FAILED_CODE,
                message=(f"triage/attribution failed: status={outcome.status} detail={detail}"),
            )
        logger.info(
            "evaluator.triage_done slot=%s trace_id=%s score=%s status=%s",
            slot,
            request.trace_id,
            result.score,
            outcome.status,
        )

    async def _flush_notification_groups(self) -> None:
        """单次归因结束后，按根因把缓冲通知发给对应人群。"""

        if self._triage_service is None:
            return
        try:
            await self._triage_service.flush_notification_groups()
        except Exception:
            logger.warning("evaluator.flush_notification_groups_failed", exc_info=True)

    async def _write_configured_score(self, call: ScoreWritebackCall) -> None:
        """Write a Kafka-configured score to its source Langfuse project only."""
        evaluator_name, request, result, policy = (
            call.evaluator_name,
            call.request,
            call.result,
            call.policy,
        )

        if policy is None or not policy.enabled:
            return
        try:
            client = self._source_score_client_provider.get_client(request.project_id or "")
        except Exception:
            logger.warning(
                "langfuse_score.source_client_failed trace_id=%s project_id=%s evaluator=%s",
                request.trace_id,
                request.project_id,
                evaluator_name,
                exc_info=True,
            )
            return
        if client is None:
            logger.warning(
                "langfuse_score.judge_write_skipped trace_id=%s project_id=%s "
                "evaluator=%s score_name=%s reason=source_project_unconfigured",
                request.trace_id,
                request.project_id,
                evaluator_name,
                policy.score_name,
            )
            return
        await write_judge_score(
            JudgeScoreWrite(
                trace_id=request.trace_id or "",
                evaluator_name=evaluator_name,
                result=result,
                client=client,
                score_name=policy.score_name,
            )
        )

    async def _enrich_vehicle_metrics(
        self,
        *,
        evaluator_name: str,
        request: EvaluatorRequest,
        result: EvaluatorResult,
    ) -> EvaluatorResult:
        """在归因前补齐 Vehicle 性能观测；外部读取失败只降级为 unknown。"""
        if evaluator_name != "vehicle_agent" or not request.trace_id or not request.project_id:
            return result
        extra = dict(result.extra)
        if extra.get("schema_version") != "agent_eval_extra.v1":
            return result
        profile = load_vehicle_profile()
        provider = self._resolve_trace_provider()
        if provider is None:
            observations = unknown_observations(profile.metrics)
        else:
            try:
                observations = await asyncio.wait_for(
                    TraceMetricCollector().collect(
                        resolver=provider.resolver,
                        project_id=request.project_id,
                        trace_id=request.trace_id,
                        bindings=profile.metrics,
                    ),
                    timeout=_METRIC_COLLECTION_TIMEOUT_S,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 性能采集不阻断评测与归因
                logger.warning(
                    "evaluator.metric_collection_failed trace_id=%s error_type=%s",
                    request.trace_id,
                    type(exc).__name__,
                )
                observations = unknown_observations(profile.metrics)
        extra["metric_observations"] = [
            observation.model_dump(mode="json") for observation in observations
        ]
        return result.model_copy(update={"extra": extra})

    @staticmethod
    def _failure_error(response: AgentResponse | None) -> EvaluatorError:
        """Turn a failed AgentResponse into an :class:`EvaluatorError` with a clean,
        client-safe message (internal detail stays in logs).

        The error is read from the response content via
        :meth:`EvaluatorError.from_payload`, the same way a success verdict is read.
        """

        if response is None:
            return EvaluatorError(
                code=evaluator_status(EVALUATION_FAILED_CODE),
                error_type=EVALUATION_FAILED_CODE,
                message="evaluation failed",
            )
        err = EvaluatorError.from_payload(decode_eval_text(response.content_parts))
        if err.error_type == PROMPT_NOT_FOUND_CODE:
            message = "No prompt is configured for the requested evaluator."
        elif err.error_type in (LLM_FAILED_CODE, LLM_UNPARSEABLE_CODE, LLM_SCHEMA_INVALID_CODE):
            logger.warning(
                "evaluator.verdict_failed error_type=%s status=%s",
                err.error_type,
                err.code,
            )
            message = "Evaluator failed to produce a verdict from the LLM."
        else:
            message = err.message or "evaluation failed"
        # 保留 details（如 prompt_customization），供 worker 在失败路径落库。
        # FUNC-PARAMS-001：details 构造后赋值（与 EvaluatorError 契约一致）。
        out = EvaluatorError(
            code=err.code,
            error_type=err.error_type,
            message=message,
        )
        if err.details:
            out.details = err.details
        return out

    # ── proxy fallback ──────────────────────────────────────────────────────
    #
    # Requests whose message content is NOT the evaluator JSON envelope (Langfuse
    # save-time connectivity probe, function-calling test, ad-hoc chat) bypass the
    # LangGraph evaluator and go straight to the evaluator's LLM. The reply is
    # projected back onto an OpenAI chat.completion verbatim (text in ``content``,
    # or ``tool_calls`` when the LLM used function calling), so whatever the client
    # negotiated (tools / response_format) is honoured by the model itself — we add
    # no verdict wrapping. This keeps the probe sub-second (no judge prompt) and
    # shape-correct (model speaks the client's structured-output protocol natively).

    async def proxy_chat_completion(self, payload: ChatCompletionRequest) -> ChatCompletion:
        """Forward a non-eval chat request to the evaluator's LLM, return OpenAI envelope.

        The LLM is resolved by the judge-stage model key (``judge_<model>``), which
        mirrors :meth:`_select` — the same model the evaluator would have used. The
        raw OpenAI ``tools`` are bound verbatim so function-calling clients get
        native ``tool_calls`` back rather than a text-embedded JSON the judge
        verdict path would have produced.
        """

        model_key = stage_agent_key(STAGE_JUDGE, payload.model)
        try:
            llm = self._runtime.get_llm(model_key)
        except (KeyError, LookupError) as exc:
            logger.warning(
                "evaluator.proxy_select_miss evaluator_name=%r model_key=%r",
                payload.model,
                model_key,
            )
            raise EvaluatorError(
                code=evaluator_status(MODEL_NOT_FOUND_CODE),
                error_type=MODEL_NOT_FOUND_CODE,
                message=f"Unknown evaluator model {payload.model!r}.",
            ) from exc
        if payload.tools:
            llm = llm.bind_tools(payload.tools)  # type: ignore[attr-defined]
        lc_messages = _to_langchain_messages(payload.messages)
        ai_message = await llm.ainvoke(lc_messages)
        return _ai_message_to_completion(payload.model, ai_message)


# ════════════════════════════════════════════════════════════════════════════
# Dependency injection — FastAPI seam for EvaluatorService (no app.state in routes)
# ════════════════════════════════════════════════════════════════════════════


def _is_benchmark_source(metadata: dict[str, Any] | None) -> bool:
    """``pipeline_source`` 去空白、忽略大小写后是否为 benchmark。"""
    raw = (metadata or {}).get(PIPELINE_SOURCE_METADATA_KEY)
    return isinstance(raw, str) and raw.strip().lower() == PipelineSource.BENCHMARK.value


def _is_benchmark_batch(request: EvaluatorRequest) -> bool:
    """批量 benchmark 请求：``pipeline_source=benchmark``。"""
    return _is_benchmark_source(request.metadata)


def _resolve_trace_source(
    metadata: dict[str, Any],
    dataset_meta: dict[str, Any] | None,
) -> str | None:
    """归一持久化来源，兼容历史 dataset 写法并保留未知来源。"""
    raw = metadata.get(PIPELINE_SOURCE_METADATA_KEY)
    if dataset_meta is not None:
        return format_dataset_pipeline_source(
            metadata.get("dataset_name"),
            metadata.get("dataset_run_name"),
        )
    if not isinstance(raw, str) or not raw.strip():
        return None

    source = raw.strip()
    lowered = source.lower()
    fixed = {
        PipelineSource.KAFKA.value,
        PipelineSource.METAQ.value,
        PipelineSource.HTTP.value,
        PipelineSource.BENCHMARK.value,
        PipelineSource.SIDECAR.value,
    }
    if lowered in fixed:
        return lowered
    if is_dataset_pipeline_source(source):
        return source
    if lowered == PipelineSource.DATASET.value or lowered.startswith(
        f"{PipelineSource.DATASET.value}+"
    ):
        return format_dataset_pipeline_source(
            metadata.get("dataset_name"),
            metadata.get("dataset_run_name"),
        )
    logger.warning(
        "evaluator.unknown_pipeline_source value=%r; persisting as-is",
        raw,
    )
    return source


def _build_trace_meta(
    metadata: dict[str, Any] | None,
    *,
    input_value: JsonValue = None,
    output_value: JsonValue = None,
    project_id: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any] | None:
    """Build ``trace_meta_json`` with ``source`` (+ optional ``dataset`` / ``kafka`` / IO).

        ``source`` ∈ {kafka, metaq, http, benchmark, sidecar} or composite
    ``dataset-{dataset_name}-{run_name}``. Bare ``dataset`` / legacy ``dataset+…``
    are rewritten to the composite form (never persisted as-is).
    Optional ``metadata['kafka']`` (Judge Path B) is copied into
    ``trace_meta_json.kafka``.
    Full evaluator ``input`` / ``output`` are appended when not ``None`` so
    downstream can rehydrate the scored case without re-fetching Langfuse.
    When ``project_id`` + ``trace_id`` 均非空，追加 ``langfuse_url``（直拼，
    不解析 primary_id），供 trace daily report Trace 链接列使用。
    Returns ``None`` only when the payload would be empty (no source / dataset /
    kafka / IO / langfuse_url); prefer always setting ``pipeline_source`` at
    entry points.
    """

    meta = metadata or {}
    dataset_meta = _dataset_trace_meta(meta)
    source = _resolve_trace_source(meta, dataset_meta)

    payload: dict[str, Any] = {}
    if source is not None:
        payload["source"] = source
    if dataset_meta is not None:
        payload["dataset"] = dataset_meta
    kafka_meta = meta.get("kafka")
    if isinstance(kafka_meta, dict) and kafka_meta:
        payload["kafka"] = kafka_meta
    # 注册 id 入 meta：example 多版本旧茎名行可按此回挂，不依赖 content_version
    reg_id = meta.get("agent_registration_id")
    if isinstance(reg_id, str) and reg_id.strip():
        payload["agent_registration_id"] = reg_id.strip()
    # Scheme A: persist full evaluator IO (not a preview / digest).
    if input_value is not None:
        payload["input"] = input_value
    if output_value is not None:
        payload["output"] = output_value
    pid = str(project_id).strip() if project_id else ""
    tid = str(trace_id).strip() if trace_id else ""
    if pid and tid:
        payload["trace_url"] = build_direct_trace_url(pid, tid)
    return payload or None


def _dataset_trace_meta(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build ``trace_meta_json.dataset`` payload when request looks like dataset-eval.

    Gate: ``dataset_item_id`` or ``dataset_name``.
    Returns ``None`` for Kafka / HTTP / Benchmark paths without those keys.

    Benchmark stamps ``dataset_item_id`` on the request metadata (for
    ``benchmark_run_items.item_id`` upsert), but must NOT be treated as
    dataset-eval — otherwise ``_build_trace_meta`` overwrites the ``benchmark``
    source with ``format_dataset_pipeline_source(None, None)`` = ``dataset--``.
    """
    if not metadata:
        return None
    # Benchmark path: pipeline_source=benchmark takes priority even when
    # dataset_item_id is present (stamped by _stamp_benchmark_metadata).
    if _is_benchmark_source(metadata):
        return None
    item_id = metadata.get("dataset_item_id")
    name = metadata.get("dataset_name")
    if not item_id and not name:
        return None
    payload: dict[str, Any] = {}
    if name is not None:
        payload["name"] = name
    if item_id is not None:
        payload["item_id"] = item_id
    run_name = metadata.get("dataset_run_name")
    if run_name is not None:
        payload["run_name"] = run_name
    return payload


def _judge_json_from_result(result: EvaluatorResult) -> dict[str, Any]:
    """Project an :class:`EvaluatorResult` into the ``judge_json`` bucket shape.

    Always writes ``score`` / ``reason``. Optional ``sub_scores`` / non-empty
    ``extra`` are included only when present so legacy rows stay ``{score, reason}``.
    """

    payload: dict[str, Any] = {
        "score": str(result.score),
        "reason": result.reason,
    }
    if result.sub_scores is not None:
        payload["sub_scores"] = result.sub_scores
    if result.extra:
        payload["extra"] = result.extra
    return payload


def get_evaluator_service(request: Request) -> EvaluatorService:
    """Resolve the lifespan-scoped :class:`EvaluatorService` from ``app.state``.

    The lifespan owns the singleton (``app.state.evaluator_service``); this is the
    only place that reads it back, so routes inject the service via
    :data:`EvaluatorServiceDep` (``Depends``) instead of touching
    ``request.app.state`` themselves. Tests override the seam with
    ``app.dependency_overrides[get_evaluator_service]``.
    """

    return request.app.state.evaluator_service


EvaluatorServiceDep = Annotated[EvaluatorService, Depends(get_evaluator_service)]


# ════════════════════════════════════════════════════════════════════════════
# Runtime — load conf/app.<env>.yaml via the standard agentgraph AppConfigManager
# ════════════════════════════════════════════════════════════════════════════


class _CustomSectionSchema(BaseModel):
    """Passthrough schema for the ``custom`` section in app.<env>.yaml.

    AppConfigManager rejects unregistered top-level keys; registering this
    permissive schema lets ``custom.triage_attribution`` coexist with the
    standard agentgraph sections without modifying the framework.
    """

    model_config = ConfigDict(extra="allow")


def build_app_runtime(config_path: Path, service_root: Path) -> AppRuntime:
    """Build an (un-bootstrapped) ``AppRuntime`` from ``conf/app.<env>.yaml``.

    The yaml carries the standard agentgraph sections (``discovery`` /
    ``observability`` / ``models`` / ``prompts`` / ``agents``) plus a
    ``custom`` section consumed by :func:`benchmark.settings.load_triage_configs`.
    ``AppRuntime`` assembles the evaluators (``agents`` section), models
    (``runtime.get_llm``) and prompts (``runtime.prompt_manager``) from it.
    Caller bootstraps the runtime (``await runtime.bootstrap()``), typically
    inside the app lifespan.
    """

    app_config = AppConfigManager()
    app_config.register_section("custom", _CustomSectionSchema, multi=False)
    register_section_extension("tools", "trace", TraceToolExt)
    app_config.load_yaml(str(config_path))
    return AgentEvalRuntime.from_app_config(app_config, service_root=service_root)


# ════════════════════════════════════════════════════════════════════════════
# Proxy fallback helpers — OpenAI ↔ LangChain message + envelope projection
# ════════════════════════════════════════════════════════════════════════════


def _to_langchain_messages(messages: list[ChatMessage]) -> list[Any]:
    """Project OpenAI chat messages onto LangChain message objects.

    ``content`` is taken verbatim when it is a string; a list content (multimodal
    parts) is flattened to the concatenated text of its ``text`` parts, matching
    the proxy's text-only scope. Unknown roles fall back to ``HumanMessage``.
    """

    projected: list[Any] = []
    for message in messages:
        text = _flatten_content_text(message.content)
        role = (message.role or "").lower()
        if role == "system":
            projected.append(SystemMessage(content=text))
        elif role == "assistant":
            projected.append(AIMessage(content=text))
        else:
            projected.append(HumanMessage(content=text))
    return projected


def _flatten_content_text(content: object) -> str:
    """Return the text carried by an OpenAI message ``content``."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _ai_message_to_completion(
    model: str,
    ai_message: Any,  # noqa: ANN401
) -> ChatCompletion:
    """Project a LangChain ``AIMessage`` onto an OpenAI ``chat.completion``.

    ``tool_calls`` (function calling) win over ``content`` — OpenAI's schema sets
    ``content=null`` and ``finish_reason="tool_calls"`` in that case, which is
    what function-calling clients (Langfuse probe via LangChain.js) parse.
    """

    raw_tool_calls = getattr(ai_message, "tool_calls", None)
    if raw_tool_calls:
        tool_calls = [_to_tool_call(tc) for tc in raw_tool_calls]
        return build_proxy_chat_completion(model=model, content=None, tool_calls=tool_calls or None)
    content = getattr(ai_message, "content", ai_message)
    text = content if isinstance(content, str) else str(content)
    return build_proxy_chat_completion(model=model, content=text)


def _to_tool_call(tool_call: Any) -> ToolCall:  # noqa: ANN401
    """Normalize one LangChain tool call (dict or dataclass) to :class:`ToolCall`."""

    if isinstance(tool_call, dict):
        name = tool_call.get("name", "")
        args = tool_call.get("args", {}) or {}
        call_id = tool_call.get("id")
    else:
        name = getattr(tool_call, "name", "")
        args = getattr(tool_call, "args", {}) or {}
        call_id = getattr(tool_call, "id", None)
    return ToolCall(
        id=call_id or f"call_{uuid.uuid4().hex[:24]}",
        function=ToolCallFunction(
            name=name,
            arguments=json.dumps(args, ensure_ascii=False),
        ),
    )


def _triage_span_output(  # noqa: PLR0912 - observability payload branches are explicit
    evaluator_name: str,
    request: EvaluatorRequest,
    outcome: Any,  # noqa: ANN401 - TriageOutcome, 避免为观测代码引入循环导入
) -> dict[str, Any]:
    """把 TriageOutcome 展平为 triage_attribution span 的 output。

    归因结论与派单 payload 本来只在服务日志里（``aone.dispatch_skipped``），Langfuse 侧
    直接展示 attribution_status / dispatch_status / dispatch_policy，避免把
    TriageOutcome.status 的内部控制流语义误读为归因或派单状态；同时把归因结论和
    ``notification_dispatch_json`` 一并写回，便于直接在 trace 上核对。
    """

    output: dict[str, Any] = {
        "evaluator_name": evaluator_name,
        "trace_id": request.trace_id,
    }
    if outcome is None:
        output["attribution_status"] = "not_started"
        return output

    outcome_status = str(getattr(outcome, "status", ""))
    attribution = getattr(outcome, "attribution", None)
    if attribution is not None:
        output["attribution_status"] = "completed"
    elif outcome_status == "attribution_failed":
        output["attribution_status"] = "failed"
    else:
        output["attribution_status"] = "not_started"

    reason = getattr(outcome, "reason", None)
    if reason:
        output["reason"] = reason
        if reason == "non_safety_aggregate_only":
            output["dispatch_policy"] = "aggregate_via_cluster_report"
        elif reason == "agent_should_dispatch_false":
            output["dispatch_policy"] = "suppressed_by_attribution"
    error = getattr(outcome, "error", None)
    if error:
        output["error"] = error

    if attribution is not None:
        for key in (
            "conclusion_type",
            "category",
            "root_cause_code",
            "confidence",
            "needs_human_review",
        ):
            value = getattr(attribution, key, None)
            if value is not None:
                output[key] = value

    notification_task = getattr(outcome, "notification_task", None)
    if notification_task is not None:
        dispatch_status = _plain(getattr(notification_task, "status", ""))
        output["dispatch_status"] = dispatch_status
        dispatch: dict[str, Any] = {
            "status": dispatch_status,
            "message": getattr(notification_task, "message", ""),
        }
        workitem_id = getattr(notification_task, "workitem_id", None)
        if workitem_id:
            dispatch["workitem_id"] = workitem_id
        payload = getattr(notification_task, "payload", None)
        if payload is not None:
            dispatch["payload"] = _trim_notification_body(_model_to_plain(payload))
        output["notification_dispatch_json"] = dispatch
    return output


_NOTIFICATION_BODY_PREVIEW_CHARS = 2000


def _trim_notification_body(payload: Any) -> Any:  # noqa: ANN401 - dict 或原对象
    """工单正文（markdown）可达几 KB，span 里只留预览；不影响 title/assignee 等关键字段。"""
    if not isinstance(payload, dict):
        return payload
    body = payload.get("body")
    if isinstance(body, str) and len(body) > _NOTIFICATION_BODY_PREVIEW_CHARS:
        payload = {**payload, "body": body[:_NOTIFICATION_BODY_PREVIEW_CHARS] + "...<已裁剪>"}
    return payload


def _model_to_plain(model: Any) -> Any:  # noqa: ANN401 - pydantic model 或任意对象
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return _plain(model)


def _plain(value: Any) -> Any:  # noqa: ANN401 - 枚举或标量
    return getattr(value, "value", value)
