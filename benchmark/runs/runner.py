"""BenchmarkRunner — benchmark evaluation executor.

Orchestrates multi-criterion evaluation: dataset fetch → per-criterion eval
→ fused triage → aggregation → report generation.  Modeled after
:class:`DatasetEvalRunner`.  Per-item scores / triage persist to
``benchmark_run_items`` via :class:`BenchmarkTraceRecordWriter`; run
summary lives only in ``benchmark_runs.metrics_json``.

Execution phases (single-node :meth:`BenchmarkRunner.run`):
1. Create Langfuse client + fetch dataset + sample (read-only; no score
   writeback / flush / ``eval_report_root``).  Inline datasets
   (``inline_items``) skip this phase entirely.
2. Per-criterion evaluation (``skip_triage=True``, pipeline writer,
   ``enable_langfuse_report=False``; failures write ``eval_failed`` stubs)
3. Fused triage (if ``triage_mode="fused"``)
4. Aggregate case scores from ``benchmark_run_items`` → ``AggregationCalculator`` →
   ``benchmark_runs.metrics_json`` (never overwrites case rows)
5. Report generation (via WebdevReportClient)

Multi-node mode is driven by BullMQ queues — workers consume case jobs
and the final report is elected by a conditional-UPDATE CAS on
``benchmark_runs.status`` so exactly one node aggregates + reports.

Results persist only to PG (``benchmark_run_items`` + ``benchmark_runs``).
Dataset-eval keeps Langfuse score writeback + flush unchanged.
"""

import asyncio
import dataclasses
import random
import statistics
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from benchmark.libs.agenthost.observability.logger import get_logger
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError

from benchmark.settings.setting import settings as default_settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import (
    BenchmarkRun,
    BenchmarkRunItem,
)
from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY, PipelineSource
from benchmark.evaluators.base import EvaluatorRequest, EvaluatorResult
from benchmark.libs.async_runtime import map_bounded
from benchmark.attribution.badcase.attribution_trigger_policy import (
    resolve_trigger_policy,
    should_trigger_attribution,
)
from benchmark.runs.common.attribution_stats import (
    resolve_attribution_stats_provider,
)
from benchmark.runs.common.config_helpers import (
    load_aggregation_from_config,
    load_rubric_from_config,
)
DispatchRequest = None
class schedule_notification_dispatch_for_triage:
    pass
from business.evaluators.example.grading import grade_fused_score
from benchmark.runs.schema import (
    FUSED_CRITERION_ID,
    BenchmarkRequest,
    BenchmarkRunResult,
    PipelineBenchmarkRequest,
    TriageMode,
)
from benchmark.reporting.framework.aggregation_calculator import (
    AggregationCalculator,
    EvaluatorResultEntry,
)
from benchmark.reporting.framework.base import (
    BaseBenchmarkServices,
    BenchmarkComponentMetadata,
)
from benchmark.reporting.framework.fusion import EvalFailure, ScoreFuser, is_failure
from benchmark.reporting.framework.report_spec import (
    ReportJobInput,
    generate_benchmark_report,
)
from benchmark.reporting.framework.schema import (
    AggregatedMetrics,
    AggregationConfig,
    AttributionStats,
    Criterion,
    CriterionStats,
    DatasetRef,
)
from benchmark.reporting.framework.webdev_client import WebdevReportClient
from benchmark.lifecycle import (
    RUN_HEARTBEAT_INTERVAL_S,
    HeartbeatSpec,
    cancel_and_wait,
    fail_eval_run,
    heartbeat_loop,
)
from benchmark.services.evaluation import EvalRunOptions
from benchmark.services.trace_writer import BenchmarkTraceRecordWriter, RunMeta, ScoreWrite
from benchmark.attribution.service.service import TriageRunInput
from benchmark.utils.trace_dataset import (
    TraceDatasetError,
    TraceEndpointNotFoundError,
    create_trace_client_for_project,
    fetch_dataset_items,
)

if TYPE_CHECKING:
    from benchmark.runs.run_service import BenchmarkService
    from benchmark.services.evaluation import EvaluatorService
    from benchmark.reporting.publish import ReportSiteReportPublisher
    from benchmark.attribution.service.service import TriageAttributionService

from benchmark.runs.run_helpers import percentile_cont
from benchmark.runs.run_service import (
    CompleteRunWrite,
    ExpandDimensionMetricsSpec,
    KafkaSampleQuery,
)

logger = get_logger("agent_eval.benchmark.runner")

EVALUATE_TIMEOUT_S = 300
MAX_TRIAGE_CONCURRENCY = 1

_JUDGE_PREFIX = "judge_"
_ATTRIBUTION_PREFIX = "attribution_"


def _eval_failure_judge_json(error: BaseException) -> dict[str, Any]:
    """Build judge_json for an evaluation failure (score=0, eval_failed=True)."""
    return {
        "score": "0.0",
        "reason": str(error)[:500],
        "eval_failed": True,
    }


@dataclasses.dataclass(frozen=True)
class BenchmarkRunExec:
    """``BenchmarkRunner.run`` 执行入参（FUNC-PARAMS-001）。"""

    benchmark_services: BaseBenchmarkServices
    run_id: str
    request: BenchmarkRequest
    generate_report: bool = True
    inline_items: list[Any] | None = None


@dataclasses.dataclass(frozen=True)
class PipelineRunExec:
    """``BenchmarkRunner.run_pipeline_benchmark`` 执行入参（FUNC-PARAMS-001）。"""

    benchmark_services: BaseBenchmarkServices
    run_id: str
    request: PipelineBenchmarkRequest
    generate_report: bool = True


@dataclasses.dataclass(frozen=True)
class FetchItemsSpec:
    """``BenchmarkRunner.fetch_items`` 入参（FUNC-PARAMS-001）。"""

    dataset_name: str
    project_id: str
    sample_size: int | None = None
    sample_seed: int = 42


@dataclasses.dataclass(frozen=True)
class _DistributedReportSpec:
    """``_try_generate_distributed_report`` 入参（FUNC-PARAMS-001）。"""

    aggregation: AggregationConfig
    generate_report: bool
    started_at: float


@dataclasses.dataclass(frozen=True)
class _EntryBuild:
    """``_build_entry`` 入参（FUNC-PARAMS-001）。"""

    criterion: Criterion
    results: list[Any]
    benchmark_services: BaseBenchmarkServices
    items: list[Any] | None = None


@dataclasses.dataclass(frozen=True)
class BenchmarkRunContext:
    """Shared execution context for a single benchmark run.

    Packs the common parameters threaded through ``_run_evaluators`` /
    ``_run_single_evaluator`` / ``_run_fused_triage`` so each function
    signature stays under the 7-argument limit.
    """

    benchmark_services: BaseBenchmarkServices
    run_id: str
    service_name: str
    dataset_ref: DatasetRef
    rubric: list[Criterion]
    items: list[Any]
    max_concurrency: int
    parallel_evaluators: bool
    triage_mode: str
    request: BenchmarkRequest
    # ── finalize params (consumed by _finalize / distributed report) ─
    aggregation: AggregationConfig | None = None
    generate_report: bool = True
    started_at: float = 0.0


@dataclasses.dataclass(frozen=True)
class BenchmarkRunnerDeps:
    """BenchmarkRunner 装配依赖包（``__init__`` 参数对象）。

    四个必传服务依赖 + 可选 triage / 评测超时，在 lifespan 装配时
    一次性收拢；运行期不再逐参传递。
    """

    evaluator_service: "EvaluatorService"
    benchmark_service: "BenchmarkService"
    aggregation_calculator: AggregationCalculator
    webdev_client: WebdevReportClient
    triage_service: "TriageAttributionService | None" = None
    evaluate_timeout_s: int = EVALUATE_TIMEOUT_S
    #: Aone Pages 二次发布器（可选；None 时报告仅 OSS 单链路，行为同接入前）
    pages_publisher: "AonePagesReportPublisher | None" = None


@dataclasses.dataclass(frozen=True)
class ReportGenerationSpec:
    """报告生成规格（``_generate_report`` 参数对象）。

    ``trace_ids`` + ``rubric`` 同时提供时走 pipeline（Kafka 明细）
    路径，否则回退 ``benchmark_run_items`` 明细。
    """

    benchmark_services: BaseBenchmarkServices
    run_id: str
    metrics: AggregatedMetrics
    final_score: float
    trace_ids: list[str] | None = None
    rubric: list[Criterion] | None = None


@dataclasses.dataclass(frozen=True)
class ReportCompletionSpec:
    """run 收尾规格（``_complete_with_report`` 参数对象）。

    Phase 5（报告生成）与 ``complete_run`` 所需的全部上下文。
    """

    benchmark_services: BaseBenchmarkServices
    run_id: str
    aggregation: AggregationConfig
    metrics: AggregatedMetrics
    final_score: float
    generate_report: bool
    started_at: float


def _logic_name(raw: str) -> str:
    """Strip stage prefixes so ``stage_agent_key`` does not double-prefix.

    Accepts both ``amap_agent`` and ``judge_amap_agent`` /
    ``attribution_amap_agent`` (spec historically used the full key).
    """
    name = (raw or "").strip()
    if name.startswith(_ATTRIBUTION_PREFIX):
        return name[len(_ATTRIBUTION_PREFIX) :]
    if name.startswith(_JUDGE_PREFIX):
        return name[len(_JUDGE_PREFIX) :]
    return name


def _resolve_attribution_logic_name(
    meta: BenchmarkComponentMetadata,
    *,
    evaluator_ref: str | None = None,
) -> str:
    """Pick attribution logic name for TriageAttributionService.run.

    Priority: explicit ``attribution_services[0]`` → criterion ``evaluator_ref``
    → ``service_name``.
    """
    services = getattr(meta, "attribution_services", None) or []
    if services:
        return _logic_name(str(services[0]))
    if evaluator_ref:
        return _logic_name(evaluator_ref)
    return _logic_name(getattr(meta, "service_name", "") or "example")


def _registration_id_from_run_config(config: object) -> str | None:
    """从 run ``config_json`` 取 registrationId（扁平键或 agent_registration 快照）。"""
    if not isinstance(config, dict):
        return None
    reg = str(config.get("agent_registration_id") or "").strip()
    if reg:
        return reg
    snap = config.get("agent_registration")
    if isinstance(snap, dict):
        reg = str(snap.get("id") or "").strip()
        if reg:
            return reg
    return None


def _stamp_benchmark_metadata(
    eval_request: EvaluatorRequest,
    *,
    item_id: str | None,
    registration_id: str | None = None,
) -> EvaluatorRequest:
    """Ensure ``pipeline_source=benchmark`` (+ dataset_item_id / registration) on the request.

    Subclass ``_build_evaluator_request`` should stamp source itself; this is a
    runner-level safety net so custom services cannot silently fall through as
    ``dataset`` via ``after_eval`` inference from dataset_* keys.
    ``agent_registration_id`` 与实时流对齐，供日志 slot / 落库键使用。
    """
    meta = dict(eval_request.metadata or {})
    meta[PIPELINE_SOURCE_METADATA_KEY] = PipelineSource.BENCHMARK.value
    if item_id and "dataset_item_id" not in meta:
        meta["dataset_item_id"] = item_id
    reg = (registration_id or "").strip()
    if reg and not str(meta.get("agent_registration_id") or "").strip():
        meta["agent_registration_id"] = reg
    return eval_request.model_copy(update={"metadata": meta})


class BenchmarkRunner:
    """Benchmark evaluation executor — modeled after DatasetEvalRunner.

    Data is fetched via Langfuse SDK ``get_dataset()``.  Each criterion's
    evaluation can run in parallel or serial.  Per-item results are written to
    ``benchmark_run_items`` via :class:`BenchmarkTraceRecordWriter`; criterion
    aggregates are held in memory until folded into ``metrics_json``.

    The ``benchmark_service`` parameter is duck-typed; it must implement:
        - ``get_run(run_id)``
        - ``update_run_status(run_id, status, **fields)``
        - ``heartbeat(run_id)``
        - ``merge_criterion_metrics(run_id, criterion_id=..., criterion_payload=...)``
        - ``update_evaluator_progress(run_id)``
        - ``complete_run(run_id, final_score, metrics_json, report_meta_json, duration_ms)``
        - ``fail_run(run_id, reason)``
    """

    _LOG_PREFIX = "benchmark"

    def __init__(self, deps: BenchmarkRunnerDeps) -> None:
        self._evaluator = deps.evaluator_service
        self._benchmark_service = deps.benchmark_service
        self._run_service = deps.benchmark_service
        self._aggregation = deps.aggregation_calculator
        self._webdev = deps.webdev_client
        self._triage = deps.triage_service
        self._evaluate_timeout = deps.evaluate_timeout_s
        self._pages_publisher = deps.pages_publisher

    # ═══════════════════════════════════════════════════════════════════════
    # Main entry point
    # ═══════════════════════════════════════════════════════════════════════

    # PLR0912: inline_items 分支（身份回读 / 跳过 Langfuse / 跳过 sampling）使
    # 分支数略超阈值；与既有 PLR0915 一样属流程编排函数的可接受复杂度。
    async def _resolve_run_rubric_aggregation(
        self,
        meta: BenchmarkComponentMetadata,
        run_id: str,
    ) -> tuple[list[Criterion], AggregationConfig]:
        """rubric/aggregation 来源（§6.2）：run config_json 快照优先，缺省 meta。

        Resolver 在 API 层执行、快照随 run 持久化；runner 从 run 记录重建，
        未走 profile 的存量 run（config 无快照键）行为与现网逐位一致。
        """
        run_record = await self._benchmark_service.get_run(run_id)
        run_config = (
            run_record.config_json
            if run_record is not None and isinstance(run_record.config_json, dict)
            else {}
        )
        rubric = load_rubric_from_config(run_config) if run_config.get("rubric") else meta.rubric
        aggregation = (
            load_aggregation_from_config(run_config)
            if run_config.get("aggregation")
            else meta.aggregation
        )
        return rubric, aggregation

    async def _run_eval_and_aggregate(
        self,
        ctx: BenchmarkRunContext,
        *,
        aggregation: AggregationConfig,
    ) -> tuple[float, AggregatedMetrics]:
        """Phase 2-4：per-criterion 评估 → fused triage → SQL 聚合。"""
        run_id = ctx.run_id
        rubric = ctx.rubric
        # Phase 2: per-criterion evaluation (entries for AggregationCalculator)
        eval_results, criterion_entries = await self._run_evaluators(ctx)

        # Phase 3: fused triage
        if ctx.triage_mode == TriageMode.FUSED.value and self._triage is not None:
            await self._run_fused_triage(ctx, eval_results, aggregation)

        # Phase 4: SQL-aggregate case scores from pipeline → job metrics
        await self._benchmark_service.update_run_status(
            run_id,
            BenchmarkStatus.AGGREGATING,
        )

        sql_entries = await self._benchmark_service.load_criterion_entries_from_pipeline(
            run_id=run_id,
            rubric=rubric,
            pass_threshold=aggregation.pass_threshold,
        )
        run_record = await self._benchmark_service.get_run(run_id)
        # react_rounds 优先从 metrics_json（DB）恢复——多机部署下本节点
        # 内存 entry 只覆盖本机处理的 criterion；缺失时回退内存进度 entry
        # （挂到首条，避免多 criterion 重复累加）。
        react = self._extract_persisted_react_rounds(
            run_record.metrics_json if run_record else None,
        ) or next(
            (e.react_rounds_json for e in criterion_entries if e.react_rounds_json),
            None,
        )
        if react and sql_entries:
            first = sql_entries[0]
            sql_entries[0] = EvaluatorResultEntry(
                criterion_id=first.criterion_id,
                evaluator_ref=first.evaluator_ref,
                sample_count=first.sample_count,
                success_count=first.success_count,
                failed_count=first.failed_count,
                criterion_stats_json=first.criterion_stats_json,
                attribution_stats_json=first.attribution_stats_json,
                react_rounds_json=react,
            )

        final_score, _, metrics = await self._aggregation.aggregate(
            aggregation,
            rubric,
            sql_entries,
        )
        run_config = (
            run_record.config_json
            if run_record is not None and isinstance(run_record.config_json, dict)
            else None
        )
        await self._benchmark_service.expand_dimension_metrics(
            ExpandDimensionMetricsSpec(
                run_id=run_id,
                benchmark_name=getattr(run_record, "benchmark_name", None) if run_record else None,
                metrics=metrics,
                config=run_config,
            ),
        )
        await self._attach_attribution_to_metrics(
            metrics,
            run_id,
            benchmark_name=getattr(run_record, "benchmark_name", None) if run_record else None,
        )
        return final_score, metrics

    async def _complete_with_report(
        self,
        req: ReportCompletionSpec,
    ) -> BenchmarkRunResult:
        """Phase 5 + Complete：报告生成 → ``complete_run``（心跳由调用方管理）。"""
        benchmark_services = req.benchmark_services
        run_id = req.run_id
        metrics = req.metrics
        final_score = req.final_score
        # Phase 5: report generation
        report_url: str | None = None
        if req.generate_report:
            # Stamp elapsed so ReportSpec includes duration; re-stamp after report.
            metrics.total_duration_ms = int(
                (time.perf_counter() - req.started_at) * 1000,
            )
            await self._benchmark_service.update_run_status(
                run_id,
                BenchmarkStatus.GENERATING_REPORT,
                final_score=final_score,
                metrics_json=metrics.model_dump(),
            )
            report_url = await self._generate_report(
                ReportGenerationSpec(
                    benchmark_services=benchmark_services,
                    run_id=run_id,
                    metrics=metrics,
                    final_score=final_score,
                ),
            )

        # Complete
        duration_ms = int((time.perf_counter() - req.started_at) * 1000)
        metrics.total_duration_ms = duration_ms
        await self._benchmark_service.complete_run(
            CompleteRunWrite(
                run_id=run_id,
                final_score=final_score,
                metrics_json=metrics.model_dump(),
                report_meta_json=(
                    {
                        "report_url": report_url,
                        "generated_at": datetime.now(UTC).isoformat(),
                    }
                    if report_url
                    else None
                ),
                duration_ms=duration_ms,
            ),
        )

        complete_rec = await self._benchmark_service.get_run(run_id)
        complete_reg = _registration_id_from_run_config(
            complete_rec.config_json if complete_rec is not None else None
        )
        logger.info(
            "benchmark.run_complete run_id=%s registration_id=%s score=%.4f duration=%dms",
            run_id,
            complete_reg or "-",
            final_score,
            duration_ms,
        )
        return BenchmarkRunResult(
            run_id=run_id,
            final_score=final_score,
            aggregation_method=req.aggregation.method,
            metrics=metrics,
            report_url=report_url,
            status=BenchmarkStatus.COMPLETED,
        )

    async def run(
        self,
        exec: BenchmarkRunExec,
    ) -> BenchmarkRunResult:
        """Execute benchmark evaluation (blocking ``await``).

        Orchestrates all phases and returns the final result.  The caller
        (API handler / BenchmarkService) controls timeout via
        ``asyncio.wait_for()``.

        ``inline_items`` 非 None 时走内联数据集路径：跳过 Langfuse
        fetch / sampling，身份参数（service/dataset/project）从 run 记录
        回读（API 层已随 ``create_run`` 持久化调用方 payload）。
        """
        benchmark_services, run_id, request, generate_report, inline_items = (
            exec.benchmark_services,
            exec.run_id,
            exec.request,
            exec.generate_report,
            exec.inline_items,
        )
        meta = benchmark_services.get_metadata()
        rubric, aggregation = await self._resolve_run_rubric_aggregation(meta, run_id)
        # Request-level overrides take priority over DatasetRef defaults
        effective_dataset_name = request.dataset_name or meta.dataset.dataset_name
        effective_project_id = request.project_id or meta.dataset.project_id
        dataset_ref = dataclasses.replace(
            meta.dataset,
            dataset_name=effective_dataset_name,
            project_id=effective_project_id,
        )
        service_name = meta.service_name

        sample_size = request.sample_size or dataset_ref.sample_size
        sample_seed = (
            request.sample_seed if request.sample_seed is not None else dataset_ref.sample_seed
        )
        max_concurrency = request.max_concurrency
        parallel_evaluators = request.parallel_evaluators
        triage_mode = request.triage_mode.value

        started_at = time.perf_counter()

        if inline_items is not None:
            # 内联数据集：身份参数（service/dataset/project）以持久化的 run
            # 记录为准，覆盖 @benchmark_component 默认值。
            identity = await self._resolve_run_identity(run_id, dataset_ref)
            if identity is None:
                return BenchmarkRunResult(
                    run_id=run_id,
                    status=BenchmarkStatus.FAILED,
                    error=f"Run record not found: {run_id}",
                )
            service_name, dataset_ref = identity

        # Phase 0: Langfuse client for dataset READ only (no score writeback)
        # 内联模式不拉取数据集，无需 Langfuse client。
        lf: Langfuse | None = None
        if inline_items is None:
            lf = self._create_trace_client(dataset_ref.project_id)
            if lf is None:
                await fail_eval_run(
                    self._run_service,
                    run_id,
                    f"Langfuse not found: {dataset_ref.project_id}",
                    log_prefix=self._LOG_PREFIX,
                )
                return BenchmarkRunResult(
                    run_id=run_id,
                    status=BenchmarkStatus.FAILED,
                    error=f"Langfuse instance not found for project: {dataset_ref.project_id}",
                )

        try:
            # Phase 1: fetch dataset + sample（内联模式直接用传入 items，不 sampling）
            if inline_items is not None:
                items = list(inline_items)
            else:
                assert lf is not None  # guarded above
                items = await self._fetch_dataset(lf, dataset_ref.dataset_name)
                if sample_size is not None and sample_size < len(items):
                    items = self._sample_items(items, sample_size, sample_seed)

            n_items = len(items)
            logger.info(
                "benchmark.run_start run_id=%s dataset=%s items=%d criteria=%d",
                run_id,
                dataset_ref.dataset_name,
                n_items,
                len(rubric),
            )

            await self._benchmark_service.update_run_status(
                run_id,
                BenchmarkStatus.RUNNING,
                total_samples=n_items,
                total_evaluators=len(rubric),
                started_at=datetime.now(UTC),
            )

            heartbeat_task = asyncio.create_task(
                heartbeat_loop(
                    HeartbeatSpec(
                        heartbeat_fn=lambda: self._benchmark_service.heartbeat(run_id),
                        logger=logger,
                        fail_log_message="benchmark.heartbeat_failed run_id=%s",
                        fail_log_args=(run_id,),
                    ),
                    interval_s=RUN_HEARTBEAT_INTERVAL_S,
                ),
            )
            try:
                ctx = BenchmarkRunContext(
                    benchmark_services=benchmark_services,
                    run_id=run_id,
                    service_name=service_name,
                    dataset_ref=dataset_ref,
                    rubric=rubric,
                    items=items,
                    max_concurrency=max_concurrency,
                    parallel_evaluators=parallel_evaluators,
                    triage_mode=triage_mode,
                    request=request,
                )
                # Phase 2-4: evaluation + fused triage + SQL aggregation
                final_score, metrics = await self._run_eval_and_aggregate(
                    ctx,
                    aggregation=aggregation,
                )

                # Phase 5 + Complete: report + complete_run
                return await self._complete_with_report(
                    ReportCompletionSpec(
                        benchmark_services=benchmark_services,
                        run_id=run_id,
                        aggregation=aggregation,
                        metrics=metrics,
                        final_score=final_score,
                        generate_report=generate_report,
                        started_at=started_at,
                    ),
                )
            finally:
                await cancel_and_wait(heartbeat_task)

        except Exception as exc:
            logger.exception("benchmark.run_failed run_id=%s", run_id)
            await fail_eval_run(self._run_service, run_id, str(exc), log_prefix=self._LOG_PREFIX)
            return BenchmarkRunResult(
                run_id=run_id,
                status=BenchmarkStatus.FAILED,
                error=str(exc),
            )
        finally:
            await self._flush_notification_groups()
            # No Langfuse flush: benchmark never writes scores/spans back.
            BenchmarkTraceRecordWriter.clear_run(run_id)

    async def _flush_notification_groups(self) -> None:
        """本轮评测结束后，按根因把缓冲通知发给对应人群。"""

        if self._triage is None:
            return
        try:
            await self._triage.flush_notification_groups()
        except Exception:
            logger.warning("benchmark.flush_notification_groups_failed", exc_info=True)

    async def fetch_items(
        self,
        spec: FetchItemsSpec,
    ) -> list[Any]:
        """Fetch dataset items from Langfuse (in-memory, no PG staging).

        Creates a per-project Langfuse client, fetches dataset items (SDK
        auto-paginates), optionally samples them. The Langfuse client is
        not flushed (benchmark never writes scores/spans back).

        Returns:
            List of Langfuse DatasetItem objects.
        """
        dataset_name = spec.dataset_name
        project_id = spec.project_id
        sample_size = spec.sample_size
        sample_seed = spec.sample_seed
        lf = self._create_trace_client(project_id)
        if lf is None:
            raise RuntimeError(f"Langfuse instance not found for project: {project_id}")
        items = await self._fetch_dataset(lf, dataset_name)
        if sample_size is not None and sample_size < len(items):
            items = self._sample_items(items, sample_size, sample_seed)
        logger.info(
            "benchmark.items_fetched dataset=%s count=%d",
            dataset_name,
            len(items),
        )
        return items

    # ═══════════════════════════════════════════════════════════════════════
    # Pipeline benchmark: Kafka retrospective aggregation (no LLM)
    # ═══════════════════════════════════════════════════════════════════════

    async def run_pipeline_benchmark(
        self,
        exec: PipelineRunExec,
    ) -> BenchmarkRunResult:
        """Aggregate existing Kafka judge scores — no LLM, no Langfuse.

        Unlike :meth:`run`, this method does **not** fetch a Langfuse
        dataset or call evaluators.  It scans ``trace_pipeline_records``
        for Kafka-sourced rows in a date range, samples them, and
        aggregates existing ``judge_json.score`` values via SQL +
        :class:`AggregationCalculator`.

        No detail rows are written to ``benchmark_run_items`` — the
        run summary lives only in ``benchmark_runs.metrics_json``。
        """
        benchmark_services = exec.benchmark_services
        run_id = exec.run_id
        request = exec.request
        generate_report = exec.generate_report
        meta = benchmark_services.get_metadata()
        rubric, aggregation = await self._resolve_run_rubric_aggregation(meta, run_id)
        from benchmark.domain.eval_config import sample_evaluator_names_for_report  # noqa: PLC0415

        run_record = await self._benchmark_service.get_run(run_id)
        run_cfg = (
            run_record.config_json
            if run_record is not None and isinstance(run_record.config_json, dict)
            else {}
        )
        registration_id = _registration_id_from_run_config(run_cfg)
        stream_instance_id = str(run_cfg.get("stream_instance_id") or "").strip() or None
        evaluator_names = sample_evaluator_names_for_report(
            rubric_refs=[c.evaluator_ref for c in rubric],
            registration_id=registration_id,
            stream_instance_id=stream_instance_id,
        )

        started_at = time.perf_counter()

        try:
            # Phase 1: scan + sample Kafka trace IDs
            await self._benchmark_service.update_run_status(
                run_id,
                BenchmarkStatus.RUNNING,
                started_at=datetime.now(UTC),
            )

            trace_ids = await self._benchmark_service.sample_kafka_pipeline_records(
                KafkaSampleQuery(
                    date_start=request.date_start,
                    date_end=request.date_end,
                    evaluator_names=evaluator_names,
                    sample_size=request.sample_size,
                    project_id=request.project_id,
                ),
            )
            n_items = len(trace_ids)
            logger.info(
                "pipeline_benchmark.run_start run_id=%s registration_id=%s records=%d criteria=%d",
                run_id,
                registration_id or "-",
                n_items,
                len(rubric),
            )

            if n_items == 0:
                await fail_eval_run(
                    self._run_service,
                    run_id,
                    f"No Kafka records found for date range "
                    f"{request.date_start}..{request.date_end}",
                    log_prefix=self._LOG_PREFIX,
                )
                return BenchmarkRunResult(
                    run_id=run_id,
                    status=BenchmarkStatus.FAILED,
                    error="No Kafka records found in the specified date range",
                )

            # Phase 2: SQL-aggregate existing scores per evaluator_name
            await self._benchmark_service.update_run_status(
                run_id,
                BenchmarkStatus.AGGREGATING,
                total_samples=n_items,
            )

            entries = await self._benchmark_service.load_kafka_criterion_entries(
                trace_ids=trace_ids,
                rubric=rubric,
                pass_threshold=aggregation.pass_threshold,
                registration_id=registration_id,
                stream_instance_id=stream_instance_id,
            )

            # Phase 3: aggregate → final score + metrics
            final_score, _per_criterion, metrics = await self._aggregation.aggregate(
                aggregation,
                rubric,
                entries,
            )

            # Phase 3.5: aggregate Kafka attribution_detail_json into metrics
            # Kafka records may carry attribution (written by daily triage);
            # fold field distributions (default primary_issue_category) into
            # metrics so the report shows attribution panels without
            # re-running triage.
            attr_by_cid = await self._benchmark_service.aggregate_kafka_attribution(
                trace_ids=trace_ids,
                rubric=rubric,
                registration_id=registration_id,
                stream_instance_id=stream_instance_id,
            )
            for cid, attr in attr_by_cid.items():
                metrics.attribution_metrics[cid] = AttributionStats(
                    field_distributions=attr.get("field_distributions", {}),
                    value_labels=attr.get("value_labels", {}),
                    total_attributed=attr.get("total_attributed", 0),
                )

            # Phase 4: report generation
            report_url: str | None = None
            if generate_report:
                metrics.total_duration_ms = int(
                    (time.perf_counter() - started_at) * 1000,
                )
                await self._benchmark_service.update_run_status(
                    run_id,
                    BenchmarkStatus.GENERATING_REPORT,
                    final_score=final_score,
                    metrics_json=metrics.model_dump(),
                )
                report_url = await self._generate_report(
                    ReportGenerationSpec(
                        benchmark_services=benchmark_services,
                        run_id=run_id,
                        metrics=metrics,
                        final_score=final_score,
                        trace_ids=trace_ids,
                        rubric=rubric,
                    ),
                )

            # Complete
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            metrics.total_duration_ms = duration_ms
            await self._benchmark_service.complete_run(
                CompleteRunWrite(
                    run_id=run_id,
                    final_score=final_score,
                    metrics_json=metrics.model_dump(),
                    report_meta_json=(
                        {
                            "report_url": report_url,
                            "generated_at": datetime.now(UTC).isoformat(),
                            # 冻结出报采样，避免 case 列表接口按「最新 trace」重采样漂移
                            "sampled_trace_ids": list(trace_ids),
                        }
                        if report_url
                        else None
                    ),
                    duration_ms=duration_ms,
                ),
            )

            logger.info(
                "pipeline_benchmark.run_complete run_id=%s score=%.4f duration=%dms records=%d",
                run_id,
                final_score,
                duration_ms,
                n_items,
            )
            return BenchmarkRunResult(
                run_id=run_id,
                final_score=final_score,
                aggregation_method=aggregation.method,
                metrics=metrics,
                report_url=report_url,
                status=BenchmarkStatus.COMPLETED,
            )

        except Exception as exc:
            logger.exception("pipeline_benchmark.run_failed run_id=%s", run_id)
            await fail_eval_run(self._run_service, run_id, str(exc), log_prefix=self._LOG_PREFIX)
            return BenchmarkRunResult(
                run_id=run_id,
                status=BenchmarkStatus.FAILED,
                error=str(exc),
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: per-criterion evaluation
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_evaluators(
        self,
        ctx: BenchmarkRunContext,
    ) -> tuple[
        dict[str, list[EvaluatorResult | EvalFailure]],
        list[EvaluatorResultEntry],
    ]:
        """Orchestrate all criteria. Returns (per-item results, criterion entries).

        ``parallel_evaluators`` 时各 criterion 并发（criterion 数量小，item 级并发
        上限由 :meth:`_run_single_evaluator` 内 ``map_bounded`` 控制）；否则顺序执行。
        某 criterion 整体抛错 → 该 criterion 全部 item 记 ``EvalFailure`` 并落失败桩。
        """
        if ctx.parallel_evaluators:
            outcomes = await asyncio.gather(
                *(self._run_single_evaluator(ctx, criterion) for criterion in ctx.rubric),
                return_exceptions=True,
            )
        else:
            outcomes = []
            for criterion in ctx.rubric:
                try:
                    outcomes.append(await self._run_single_evaluator(ctx, criterion))
                except Exception as exc:
                    outcomes.append(exc)

        results: dict[str, list[EvaluatorResult | EvalFailure]] = {}
        entries: list[EvaluatorResultEntry] = []
        for criterion, outcome in zip(ctx.rubric, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                item_results, entry = await self._fail_whole_criterion(ctx, criterion, outcome)
            else:
                item_results, entry = outcome
            results[criterion.id] = item_results
            entries.append(entry)
        return results, entries

    async def _fail_whole_criterion(
        self,
        ctx: BenchmarkRunContext,
        criterion: Criterion,
        error: BaseException,
    ) -> tuple[list[EvaluatorResult | EvalFailure], EvaluatorResultEntry]:
        """criterion 级失败：全部 item 记 EvalFailure、写失败桩、落进度。"""
        item_results: list[EvaluatorResult | EvalFailure] = [EvalFailure(error)] * len(ctx.items)
        await self._write_eval_failure_stubs(ctx, criterion, error)
        entry = self._build_entry(
            _EntryBuild(
                criterion=criterion,
                results=item_results,
                benchmark_services=ctx.benchmark_services,
                items=ctx.items,
            ),
        )
        await self._persist_criterion_progress(ctx.run_id, entry)
        return item_results, entry

    async def _write_eval_failure_stubs(
        self,
        ctx: BenchmarkRunContext,
        criterion: Criterion,
        error: BaseException,
    ) -> None:
        """Persist score=0 / eval_failed stubs so SQL aggregation sees failed cases."""
        writer = BenchmarkTraceRecordWriter(
            run_id=ctx.run_id,
            criterion_id=criterion.id,
            meta=RunMeta(
                benchmark_name=ctx.benchmark_services.get_metadata().name,
                service_name=ctx.service_name,
                dataset_name=ctx.dataset_ref.dataset_name,
                project_id=ctx.dataset_ref.project_id,
            ),
        )
        for item in ctx.items:
            try:
                item_id = str(getattr(item, "id", "") or "")
                item_trace_id = getattr(item, "source_trace_id", None) or f"dataset:{item_id}"
                await writer.upsert_score(
                    ScoreWrite(
                        trace_id=item_trace_id,
                        evaluator_ref=criterion.evaluator_ref,
                        score=0.0,
                        judge_json=_eval_failure_judge_json(error),
                        dataset_item_id=item_id,
                    ),
                    project_id=ctx.dataset_ref.project_id,
                    agent_name=ctx.service_name,
                    trace_meta_json={
                        "source": PipelineSource.BENCHMARK.value,
                        "input": getattr(item, "input", None),
                        "output": getattr(item, "output", None),
                    },
                )
            except Exception:
                logger.exception(
                    "benchmark.fail_stub_write_failed run_id=%s criterion=%s item=%s",
                    ctx.run_id,
                    criterion.id,
                    getattr(item, "id", None),
                )

    async def _run_single_evaluator(
        self,
        ctx: BenchmarkRunContext,
        criterion: Criterion,
        *,
        persist_progress: bool = True,
    ) -> tuple[list[EvaluatorResult | EvalFailure], EvaluatorResultEntry]:
        """Evaluate one criterion across all items with bounded concurrency.

        When ``persist_progress`` is ``False`` (shard mode), skip
        ``_persist_criterion_progress`` — the caller aggregates at shard
        level to avoid ``completed_evaluators`` overflow.
        """
        writer = BenchmarkTraceRecordWriter(
            run_id=ctx.run_id,
            criterion_id=criterion.id,
            meta=RunMeta(
                benchmark_name=ctx.benchmark_services.get_metadata().name,
                service_name=ctx.service_name,
                dataset_name=ctx.dataset_ref.dataset_name,
                project_id=ctx.dataset_ref.project_id,
            ),
        )
        run_rec = await self._benchmark_service.get_run(ctx.run_id)
        batch_reg = _registration_id_from_run_config(
            run_rec.config_json if run_rec is not None else None
        )

        async def eval_one(index: int, item: object) -> EvaluatorResult | EvalFailure:
            eval_request = _stamp_benchmark_metadata(
                ctx.benchmark_services._build_evaluator_request(
                    item,
                    ctx.run_id,
                    ctx.request,
                ),
                item_id=getattr(item, "id", None),
                registration_id=batch_reg,
            )

            try:
                result = await asyncio.wait_for(
                    self._evaluator.evaluate(
                        criterion.evaluator_ref,
                        eval_request,
                        options=EvalRunOptions(
                            resolve_trace=False,
                            skip_triage=True,
                            writer=writer,
                            enable_trace_report=False,
                        ),
                    ),
                    timeout=self._evaluate_timeout,
                )

                if ctx.triage_mode == TriageMode.PER_EVAL.value and self._triage is not None:
                    await self._triage.run(
                        TriageRunInput(
                            evaluator_name=_resolve_attribution_logic_name(
                                ctx.benchmark_services.get_metadata(),
                                evaluator_ref=criterion.evaluator_ref,
                            ),
                            evaluator_request=eval_request,
                            evaluator_result=result,
                        ),
                    )
                return result
            except Exception as e:
                logger.warning(
                    "benchmark.eval_failed run_id=%s registration_id=%s "
                    "criterion=%s item=%s err=%s",
                    ctx.run_id,
                    batch_reg or "-",
                    criterion.id,
                    getattr(item, "id", None),
                    e,
                )
                try:
                    await writer.upsert_score(
                        ScoreWrite(
                            trace_id=eval_request.trace_id,
                            evaluator_ref=criterion.evaluator_ref,
                            score=0.0,
                            judge_json=_eval_failure_judge_json(e),
                            dataset_item_id=str(getattr(item, "id", "") or ""),
                        ),
                        project_id=ctx.dataset_ref.project_id,
                        agent_name=ctx.service_name,
                        trace_meta_json={
                            "source": PipelineSource.BENCHMARK.value,
                            "input": eval_request.input,
                            "output": eval_request.output,
                        },
                    )
                except Exception:
                    logger.exception(
                        "benchmark.fail_stub_write_failed run_id=%s criterion=%s item=%s",
                        ctx.run_id,
                        criterion.id,
                        getattr(item, "id", None),
                    )
                return EvalFailure(e)

        outcomes = await map_bounded(
            ctx.items,
            eval_one,
            limit=max(1, ctx.max_concurrency),
        )
        final_results: list[EvaluatorResult | EvalFailure] = []
        for outcome in outcomes:
            if isinstance(outcome, (EvaluatorResult, EvalFailure)):
                final_results.append(outcome)
            elif isinstance(outcome, BaseException):
                final_results.append(EvalFailure(outcome))
            else:
                final_results.append(EvalFailure(RuntimeError("unknown None")))
        entry = self._build_entry(
            _EntryBuild(
                criterion=criterion,
                results=final_results,
                benchmark_services=ctx.benchmark_services,
                items=ctx.items,
            ),
        )
        if persist_progress:
            await self._persist_criterion_progress(ctx.run_id, entry)
        return final_results, entry

    # ═══════════════════════════════════════════════════════════════════════
    # Multi-node worker: CAS report election（context 由 run() 在单节点
    # 路径构造；BullMQ worker 直接消费 case job，不经 _resolve_context）
    # ═══════════════════════════════════════════════════════════════════════

    async def _finalize(self, ctx: BenchmarkRunContext) -> None:
        """Elect unique node to aggregate + generate report (two-phase CAS)."""
        if ctx.aggregation is None:
            return
        await self._try_generate_distributed_report(
            ctx,
            _DistributedReportSpec(
                aggregation=ctx.aggregation,
                generate_report=ctx.generate_report,
                started_at=ctx.started_at,
            ),
        )

    async def _cleanup(self, ctx: BenchmarkRunContext) -> None:
        BenchmarkTraceRecordWriter.clear_run(ctx.run_id)

    async def _try_generate_distributed_report(
        self,
        ctx: BenchmarkRunContext,
        spec: _DistributedReportSpec,
    ) -> None:
        """全部 task 完成后选举唯一节点聚合 + 生成 report（两段式条件 UPDATE）。

        并发安全：
        1. ``RUNNING → AGGREGATING`` 条件推进 —— 其他节点已推进/抢占时影响
           行数=0，不会把 ``GENERATING_REPORT`` 覆盖回去；
        2. ``AGGREGATING → GENERATING_REPORT`` CAS —— 多节点仅一个影响行数=1。
        """
        aggregation = spec.aggregation
        generate_report = spec.generate_report
        started_at = spec.started_at
        await BenchmarkRun.filter(
            run_id=ctx.run_id,
            status=BenchmarkStatus.RUNNING.value,
        ).update(status=BenchmarkStatus.AGGREGATING.value)

        won = await BenchmarkRun.filter(
            run_id=ctx.run_id,
            status=BenchmarkStatus.AGGREGATING.value,
        ).update(status=BenchmarkStatus.GENERATING_REPORT.value)
        if won == 0:
            logger.info("benchmark.report_election_lost run_id=%s", ctx.run_id)
            return

        logger.info("benchmark.report_election_won run_id=%s", ctx.run_id)
        try:
            # FUSED triage deferred from multi-node execution to the
            # elected report node — runs after all criteria complete.
            if ctx.triage_mode == TriageMode.FUSED.value and self._triage is not None:
                eval_results = await self._reconstruct_eval_results(ctx)
                if eval_results:
                    await self._run_fused_triage(ctx, eval_results, aggregation)

            sql_entries = await self._benchmark_service.load_criterion_entries_from_pipeline(
                run_id=ctx.run_id,
                rubric=ctx.rubric,
                pass_threshold=aggregation.pass_threshold,
            )
            # react_rounds 由执行 task 的节点随 criterion progress 持久化到
            # metrics_json；report 节点本地无内存 entry，从持久化数据恢复。
            run_record = await self._benchmark_service.get_run(ctx.run_id)
            react = self._extract_persisted_react_rounds(
                run_record.metrics_json if run_record else None,
            )
            if react and sql_entries:
                sql_entries[0] = dataclasses.replace(
                    sql_entries[0],
                    react_rounds_json=react,
                )

            final_score, _per_criterion, metrics = await self._aggregation.aggregate(
                aggregation,
                ctx.rubric,
                sql_entries,
            )
            run_cfg = (
                run_record.config_json
                if run_record is not None and isinstance(run_record.config_json, dict)
                else None
            )
            await self._benchmark_service.expand_dimension_metrics(
                ExpandDimensionMetricsSpec(
                    run_id=ctx.run_id,
                    benchmark_name=(
                        getattr(run_record, "benchmark_name", None) if run_record else None
                    ),
                    metrics=metrics,
                    config=run_cfg,
                ),
            )
            await self._attach_attribution_to_metrics(
                metrics,
                ctx.run_id,
                benchmark_name=(
                    getattr(run_record, "benchmark_name", None) if run_record else None
                ),
            )

            # 多节点下本进程计时不是 run 的真实起点，优先用 run.started_at
            duration_ms = self._run_duration_ms(run_record, started_at)
            metrics.total_duration_ms = duration_ms

            report_url: str | None = None
            if generate_report:
                await self._benchmark_service.update_run_status(
                    ctx.run_id,
                    BenchmarkStatus.GENERATING_REPORT,
                    final_score=final_score,
                    metrics_json=metrics.model_dump(),
                )
                report_url = await self._generate_report(
                    ReportGenerationSpec(
                        benchmark_services=ctx.benchmark_services,
                        run_id=ctx.run_id,
                        metrics=metrics,
                        final_score=final_score,
                    ),
                )

            await self._benchmark_service.complete_run(
                CompleteRunWrite(
                    run_id=ctx.run_id,
                    final_score=final_score,
                    metrics_json=metrics.model_dump(),
                    report_meta_json=(
                        {
                            "report_url": report_url,
                            "generated_at": datetime.now(UTC).isoformat(),
                        }
                        if report_url
                        else None
                    ),
                    duration_ms=duration_ms,
                ),
            )
            logger.info(
                "benchmark.distributed_run_complete run_id=%s score=%.4f duration=%dms",
                ctx.run_id,
                final_score,
                duration_ms,
            )
        except Exception as exc:
            logger.exception(
                "benchmark.distributed_finalize_failed run_id=%s",
                ctx.run_id,
            )
            await fail_eval_run(
                self._run_service,
                ctx.run_id,
                f"distributed finalize failed: {exc}",
                log_prefix=self._LOG_PREFIX,
            )

    @staticmethod
    def _extract_persisted_react_rounds(
        metrics_json: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """从 ``metrics_json.criterion_metrics`` 恢复首个 react_rounds 分布。"""
        criterion_metrics = (metrics_json or {}).get("criterion_metrics") or {}
        for payload in criterion_metrics.values():
            if isinstance(payload, dict) and payload.get("react_rounds"):
                return payload["react_rounds"]
        return None

    @staticmethod
    def _run_duration_ms(run_record: object, started_at: float) -> int:
        """run 级耗时：优先 ``started_at`` 列（跨节点一致），回退进程计时。"""
        run_started = getattr(run_record, "started_at", None)
        if run_started is not None:
            if run_started.tzinfo is None:
                run_started = run_started.replace(tzinfo=UTC)
            return int((datetime.now(UTC) - run_started).total_seconds() * 1000)
        return int((time.perf_counter() - started_at) * 1000)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: fused triage
    # ═══════════════════════════════════════════════════════════════════════

    async def _run_fused_triage(
        self,
        ctx: BenchmarkRunContext,
        eval_results: dict[str, list[EvaluatorResult | EvalFailure]],
        aggregation: AggregationConfig,
    ) -> None:
        """Fuse per-item scores and trigger attribution for low-score items.

        处理流程对齐 external judge 回调（external_runner §5）：fuse → 写 fused 行
        （judge_json 含 criterion_scores）→ grade 计算 + 归因门控 → 低分才 triage。
        归因：judge attribution_threshold / 大分制 pass_threshold / 业务策略。
        归因落地后触发 Aone 推送（对齐 external §5.5），判定 / 配置复用
        默认版本（分数门槛 + YAML/RemoteConfig 默认目标）。
        """
        if self._triage is None:
            return

        # 推送配置与 external 同源：per-run config_json（缺失回退
        # app-level YAML custom.benchmark.aone_dispatch 默认）。
        run = await self._benchmark_service.get_run(ctx.run_id)
        run_config = getattr(run, "config_json", None)

        writer = BenchmarkTraceRecordWriter(
            run_id=ctx.run_id,
            criterion_id=FUSED_CRITERION_ID,
            meta=RunMeta(
                benchmark_name=ctx.benchmark_services.get_metadata().name,
                service_name=ctx.service_name,
                dataset_name=ctx.dataset_ref.dataset_name,
                project_id=ctx.dataset_ref.project_id,
            ),
        )
        per_item_scores = self._transpose_results(eval_results, len(ctx.items))
        paired = list(zip(ctx.items, per_item_scores, strict=False))

        async def triage_one(_index: int, pair: tuple[object, dict[str, Any]]) -> None:
            item, scores = pair
            item_id = getattr(item, "id", None)
            eval_request = _stamp_benchmark_metadata(
                ctx.benchmark_services._build_evaluator_request(
                    item,
                    ctx.run_id,
                    ctx.request,
                ),
                item_id=item_id,
            )

            fused_score = await ScoreFuser.afuse(scores, ctx.rubric, aggregation)
            from benchmark.runs.common.dynamic_rubric import (  # noqa: PLC0415
                apply_dynamic_fused_override,
                first_attribution_threshold_from_scores,
                project_criterion_scores,
                resolve_gate_attribution_threshold,
                should_trigger_with_optional_threshold,
                stamp_fused_grade,
            )

            # 动态权重覆盖 fused；无 weights 时保持 ScoreFuser（自定义 Rubric 不变）。
            fused_score = apply_dynamic_fused_override(scores, fused_score)

            # 归因门控 — judge threshold → 大分制 pass_threshold → 业务策略。
            score_val = float(fused_score) if fused_score is not None else 0.0
            grade = (
                grade_fused_score(
                    score_val,
                    pass_threshold=aggregation.pass_threshold,
                )
                if fused_score is not None
                else "FAIL"
            )
            dynamic_threshold = first_attribution_threshold_from_scores(scores)
            if dynamic_threshold is None:
                dynamic_threshold = resolve_gate_attribution_threshold(
                    None,
                    pass_threshold=aggregation.pass_threshold,
                )
            dynamic_gate = should_trigger_with_optional_threshold(
                fused_score=fused_score,
                attribution_threshold=dynamic_threshold,
            )
            if dynamic_gate is not None:
                should_attr = dynamic_gate
            else:
                trigger_policy = resolve_trigger_policy(
                    ctx.benchmark_services.get_metadata().name,
                )
                should_attr = await should_trigger_attribution(
                    trigger_policy,
                    fused_score=fused_score,
                    criterion_scores={cid: getattr(r, "score", None) for cid, r in scores.items()},
                )

            criterion_scores = project_criterion_scores(scores)

            fused_result = EvaluatorResult(
                score=fused_score,
                reason=f"benchmark fused: {aggregation.method}",
                sub_scores={cid: r.score for cid, r in scores.items() if not is_failure(r)},
                extra={
                    "per_evaluator": {
                        cid: r.model_dump() for cid, r in scores.items() if not is_failure(r)
                    },
                    "failed_criteria": [cid for cid, r in scores.items() if is_failure(r)],
                },
            )

            attr_logic = _resolve_attribution_logic_name(
                ctx.benchmark_services.get_metadata(),
            )
            # Write fused score as judge_json before triage so the fused
            # row exists with judge data — upsert_triage will find it and
            # append attribution_detail_json, avoiding judge-less rows.
            fused_judge: dict[str, Any] = {
                "score": str(fused_result.score),
                "reason": fused_result.reason,
                # 对齐 external / benchmark_worker fused 行格式：
                # judge_json.criterion_scores 供 _reconstruct_eval_results
                # 优先解析（重放 / 多节点报告重建路径）。
                "criterion_scores": criterion_scores,
            }
            if fused_result.sub_scores is not None:
                fused_judge["sub_scores"] = fused_result.sub_scores
            if fused_result.extra:
                fused_judge["extra"] = fused_result.extra
            stamp_fused_grade(
                fused_judge,
                fused_score=fused_score,
                pass_threshold=aggregation.pass_threshold,
            )
            # 日志 grade 与 judge_json 同源（框架定级，非 LLM）
            grade = str(fused_judge.get("grade") or grade)
            try:
                await writer.upsert_score(
                    ScoreWrite(
                        trace_id=None,
                        evaluator_ref=attr_logic,
                        score=fused_result.score,
                        judge_json=fused_judge,
                        dataset_item_id=item_id,
                    ),
                )
                logger.info(
                    "benchmark.fused_attribution_gate run_id=%s item_id=%s "
                    "fused_score=%s grade=%s should_attribute=%s "
                    "pass_threshold=%s",
                    ctx.run_id,
                    item_id,
                    fused_score,
                    grade,
                    should_attr,
                    aggregation.pass_threshold,
                )
                if not should_attr:
                    return
                outcome = await self._triage.run(
                    TriageRunInput(
                        evaluator_name=attr_logic,
                        evaluator_request=eval_request,
                        evaluator_result=fused_result,
                    ),
                )
                # 对齐 external §5.5：归因落地后触发 Aone 推送
                # （判定/配置复用默认版本）。fused 行已 upsert_score，
                # 门槛分可得；归因推翻 judge 时 outcome.attribution
                # .should_dispatch=False → 核心拒派 attribution_overridden。
                if outcome.attribution is not None:
                    schedule_notification_dispatch_for_triage(
                        DispatchRequest(
                            store=self._benchmark_service,
                            run_id=ctx.run_id,
                            item_id=item_id or "",
                            config=run_config,
                        ),
                        attribution=outcome.attribution,
                    )
            except Exception:
                logger.warning(
                    "benchmark.fused_triage_failed run_id=%s item=%s attribution=%s",
                    ctx.run_id,
                    item_id,
                    attr_logic,
                    exc_info=True,
                )

        await map_bounded(
            paired,
            triage_one,
            limit=MAX_TRIAGE_CONCURRENCY,
            return_exceptions=True,
        )

    async def _reconstruct_eval_results(
        self,
        ctx: BenchmarkRunContext,
    ) -> dict[str, list[EvaluatorResult | EvalFailure]]:
        """Rebuild per-criterion eval results from ``benchmark_run_items``.

        Used by multi-node ``_try_generate_distributed_report`` to feed
        ``_run_fused_triage`` after all criterion tasks have completed.
        Rows with ``status="failed"`` become :class:`EvalFailure`; missing
        items are filled with ``EvalFailure`` so fusion weights stay constant.

        Tries fused rows first (external trigger mode — per-criterion scores
        are embedded in ``judge_json.criterion_scores``); falls back to
        per-criterion rows (inline/multi-node mode).
        """
        item_index: dict[str, int] = {
            getattr(item, "id", str(i)): i for i, item in enumerate(ctx.items)
        }

        by_criterion = await self._reconstruct_from_fused_rows(ctx, item_index)
        if not by_criterion:
            by_criterion = await self._reconstruct_from_criterion_rows(
                ctx,
                item_index,
            )
        return by_criterion

    @staticmethod
    def _missing_bucket(
        by_criterion: dict[str, list[EvaluatorResult | EvalFailure]],
        criterion_id: str,
        n_items: int,
    ) -> list[EvaluatorResult | EvalFailure]:
        """Get / create a per-criterion bucket pre-filled with missing-item stubs."""
        bucket = by_criterion.get(criterion_id)
        if bucket is None:
            bucket = [EvalFailure(RuntimeError("missing")) for _ in range(n_items)]
            by_criterion[criterion_id] = bucket
        return bucket

    async def _reconstruct_from_fused_rows(
        self,
        ctx: BenchmarkRunContext,
        item_index: dict[str, int],
    ) -> dict[str, list[EvaluatorResult | EvalFailure]]:
        """Parse per-criterion scores from fused rows' ``judge_json.criterion_scores``.

        External trigger mode embeds all criterion scores in the fused row;
        rows still ``running`` are excluded. Returns ``{}`` when no fused row
        carries scores (caller falls back to per-criterion rows).
        """
        n_items = len(ctx.items)
        by_criterion: dict[str, list[EvaluatorResult | EvalFailure]] = {}

        # 优先从 fused 行 judge_json.criterion_scores 解析（外部触发模式）
        async for row in BenchmarkRunItem.filter(
            run_id=ctx.run_id,
            criterion_id=FUSED_CRITERION_ID,
        ).exclude(status="running"):
            idx = item_index.get(row.item_id)
            if idx is None:
                continue
            cj = row.judge_json or {}
            cs = cj.get("criterion_scores")
            if not isinstance(cs, dict) or not cs:
                continue
            has_error = bool(cj.get("error"))
            for cid, cdata in cs.items():
                bucket = self._missing_bucket(by_criterion, cid, n_items)
                if isinstance(cdata, dict):
                    raw_score = cdata.get("score")
                    score = float(raw_score) if raw_score is not None else 0.0
                    reason = cdata.get("reason", "")
                else:
                    score = 0.0
                    reason = ""
                if has_error and score == 0.0:
                    bucket[idx] = EvalFailure(
                        RuntimeError(cj.get("error") or "eval failed"),
                    )
                else:
                    bucket[idx] = EvaluatorResult(score=score, reason=reason)
        return by_criterion

    async def _reconstruct_from_criterion_rows(
        self,
        ctx: BenchmarkRunContext,
        item_index: dict[str, int],
    ) -> dict[str, list[EvaluatorResult | EvalFailure]]:
        """Parse per-criterion rows directly (inline / multi-node fallback).

        Rows with ``status="failed"`` become :class:`EvalFailure`.
        """
        n_items = len(ctx.items)
        by_criterion: dict[str, list[EvaluatorResult | EvalFailure]] = {}

        rows = await BenchmarkRunItem.filter(
            run_id=ctx.run_id,
        ).exclude(criterion_id=FUSED_CRITERION_ID)

        for row in rows:
            idx = item_index.get(row.item_id)
            if idx is None:
                continue
            bucket = self._missing_bucket(by_criterion, row.criterion_id, n_items)
            if row.status == "failed":
                bucket[idx] = EvalFailure(
                    RuntimeError(row.error_message or "eval failed"),
                )
            else:
                bucket[idx] = EvaluatorResult(
                    score=row.score if row.score is not None else 0.0,
                    reason=row.reason,
                )
        return by_criterion

    # ═══════════════════════════════════════════════════════════════════════
    # Aggregation helpers (per-criterion entries → metrics_json)
    # ═══════════════════════════════════════════════════════════════════════

    async def _persist_criterion_progress(
        self,
        run_id: str,
        entry: EvaluatorResultEntry,
    ) -> None:
        """Write mid-run criterion stats + bump completed_evaluators."""
        payload = {
            "criterion_id": entry.criterion_id,
            "evaluator_ref": entry.evaluator_ref,
            "sample_count": entry.sample_count,
            "success_count": entry.success_count,
            "failed_count": entry.failed_count,
            **(entry.criterion_stats_json or {}),
        }
        if entry.react_rounds_json is not None:
            payload["react_rounds"] = entry.react_rounds_json
        await self._benchmark_service.merge_criterion_metrics(
            run_id,
            criterion_id=entry.criterion_id,
            criterion_payload=payload,
        )
        await self._benchmark_service.update_evaluator_progress(run_id)

    @staticmethod
    async def _attach_attribution_to_metrics(
        metrics: AggregatedMetrics,
        run_id: str,
        *,
        benchmark_name: str | None = None,
    ) -> None:
        """Fold triage / judge attribution into metrics (no broadcast).

        - 已注册 AttributionStatsProvider（如 search_eval）→ provider.collect
        - 否则回退 ``BenchmarkTraceRecordWriter.aggregate_attribution``：
          fused rows → ``attribution_metrics["fused"]``；
          per_eval rows → ``attribution_metrics[criterion_id]``
        """
        if benchmark_name is None:
            run = await BenchmarkRun.filter(run_id=run_id).first()
            benchmark_name = getattr(run, "benchmark_name", None) if run else None
        attr_provider = resolve_attribution_stats_provider(benchmark_name)
        if attr_provider is not None:
            attr_by_cid = await attr_provider.collect(run_id)
        else:
            attr_by_cid = await BenchmarkTraceRecordWriter.aggregate_attribution(run_id)
        if not attr_by_cid:
            return
        for cid, attr in attr_by_cid.items():
            metrics.attribution_metrics[cid] = AttributionStats(
                field_distributions=attr.get("field_distributions", {}),
                value_labels=attr.get("value_labels", {}),
                total_attributed=attr.get("total_attributed", 0),
            )

    def _build_entry(
        self,
        build: _EntryBuild,
    ) -> EvaluatorResultEntry:
        """Compute CriterionStats (+ optional react_rounds) as an in-memory entry."""
        criterion = build.criterion
        results = build.results
        benchmark_services = build.benchmark_services
        items = build.items
        meta = benchmark_services.get_metadata()
        stats = self._compute_criterion_stats(
            criterion=criterion,
            results=results,
            pass_threshold=meta.aggregation.pass_threshold,
        )
        react_rounds_json: dict[str, Any] | None = None
        if items:
            distribution: dict[str, int] = {}
            for item in items:
                rounds = benchmark_services._extract_react_rounds(item)
                if rounds is None:
                    continue
                key = str(rounds)
                distribution[key] = distribution.get(key, 0) + 1
            if distribution:
                react_rounds_json = {"distribution": distribution}

        return EvaluatorResultEntry(
            criterion_id=criterion.id,
            evaluator_ref=criterion.evaluator_ref,
            sample_count=stats.sample_count,
            success_count=stats.success_count,
            failed_count=stats.failed_count,
            criterion_stats_json=stats.model_dump(
                include={"mean", "median", "p95", "min", "max", "std", "pass_rate"},
            ),
            react_rounds_json=react_rounds_json,
        )

    @staticmethod
    def _compute_criterion_stats(
        *,
        criterion: Criterion,
        results: list[EvaluatorResult | EvalFailure],
        pass_threshold: float,
    ) -> CriterionStats:
        """Compute CriterionStats from a list of per-item results."""
        scores: list[float] = []
        success_count = 0
        failed_count = 0

        for r in results:
            if is_failure(r):
                scores.append(0.0)
                failed_count += 1
            else:
                scores.append(r.score)
                success_count += 1

        sample_count = len(results)
        if not scores:
            return CriterionStats(
                criterion_id=criterion.id,
                evaluator_ref=criterion.evaluator_ref,
                sample_count=0,
                success_count=0,
                failed_count=0,
                mean=0.0,
                median=0.0,
                p95=0.0,
                min=0.0,
                max=0.0,
                std=0.0,
                pass_rate=0.0,
            )

        sorted_scores = sorted(scores)
        pass_count = sum(1 for s in scores if s >= pass_threshold)

        return CriterionStats(
            criterion_id=criterion.id,
            evaluator_ref=criterion.evaluator_ref,
            sample_count=sample_count,
            success_count=success_count,
            failed_count=failed_count,
            mean=statistics.mean(scores),
            median=statistics.median(scores),
            p95=percentile_cont(sorted_scores, 0.95),
            min=sorted_scores[0],
            max=sorted_scores[-1],
            std=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            pass_rate=pass_count / sample_count if sample_count > 0 else 0.0,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 5: report generation
    # ═══════════════════════════════════════════════════════════════════════

    async def _generate_report(
        self,
        req: ReportGenerationSpec,
    ) -> str | None:
        """Generate HTML report via unified ``report_spec`` kernel.

        For regular benchmarks, case results are loaded from
        ``benchmark_run_items`` via :meth:`load_case_results`.

        For pipeline benchmarks (``trace_ids`` + ``rubric`` provided),
        case results are loaded from real Kafka ``trace_pipeline_records``
        via :meth:`load_kafka_case_results`, since pipeline benchmark
        does not write synthetic detail rows.
        """
        case_results_loader = None
        if req.trace_ids is not None and req.rubric is not None:
            trace_ids = req.trace_ids
            pipeline_rubric = req.rubric
            run_rec = await self._benchmark_service.get_run(req.run_id)
            run_cfg = (
                run_rec.config_json
                if run_rec is not None and isinstance(run_rec.config_json, dict)
                else {}
            )
            reg = str(run_cfg.get("agent_registration_id") or "").strip() or None
            if reg is None:
                snap = run_cfg.get("agent_registration")
                if isinstance(snap, dict):
                    reg = str(snap.get("id") or "").strip() or None
            stream = str(run_cfg.get("stream_instance_id") or "").strip() or None

            async def _load_kafka_cases() -> list[dict[str, Any]]:
                return await self._benchmark_service.load_kafka_case_results(
                    trace_ids=trace_ids,
                    rubric=pipeline_rubric,
                    registration_id=reg,
                    stream_instance_id=stream,
                )

            case_results_loader = _load_kafka_cases

        return await generate_benchmark_report(
            ReportJobInput(
                benchmark_service=self._benchmark_service,
                run_id=req.run_id,
                metrics=req.metrics,
                final_score=req.final_score,
                case_results_loader=case_results_loader,
                component_services=req.benchmark_services,
                log_prefix="benchmark",
            ),
            webdev=self._webdev,
            pages_publisher=self._pages_publisher,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _create_trace_client(self, project_id: str) -> Langfuse | None:
        """Create a per-project Langfuse SDK client; ``None`` when project is unconfigured."""
        try:
            client, _instance = create_trace_client_for_project(project_id)
        except TraceEndpointNotFoundError:
            logger.error("benchmark.langfuse_not_found project_id=%s", project_id)
            return None
        return client

    async def _fetch_dataset(self, lf: Langfuse, dataset_name: str) -> list[Any]:
        """Fetch dataset items from Langfuse (SDK auto-paginates)."""
        try:
            return await fetch_dataset_items(
                lf,
                dataset_name,
                attempts=default_settings.trace_fetch_attempts,
                page_size=default_settings.trace_fetch_page_size,
            )
        except ApiError as exc:
            status = exc.status_code or 502
            detail = exc.body if isinstance(exc.body, str) else str(exc.body)
            raise RuntimeError(f"Langfuse dataset fetch failed ({status}): {detail}") from exc
        except TraceDatasetError as exc:
            # 重试穷尽（超时/传输错误）；消息含原始 httpx 异常类型名
            raise RuntimeError(str(exc)) from exc

    @staticmethod
    def _sample_items(
        items: list[Any],
        sample_size: int,
        seed: int,
    ) -> list[Any]:
        """Sample ``sample_size`` items with a fixed seed for reproducibility."""
        rng = random.Random(seed)
        if sample_size >= len(items):
            return items
        return rng.sample(items, sample_size)

    @staticmethod
    def _transpose_results(
        eval_results: dict[str, list[EvaluatorResult | EvalFailure]],
        n_items: int,
    ) -> list[dict[str, EvaluatorResult | EvalFailure]]:
        """Transpose ``{criterion_id: [per_item]}`` → ``[{criterion_id: result}]``."""
        return [{cid: results[i] for cid, results in eval_results.items()} for i in range(n_items)]

    async def _resolve_run_identity(
        self,
        run_id: str,
        fallback: DatasetRef,
    ) -> tuple[str, DatasetRef] | None:
        """从持久化 run 记录回读 ``(service_name, dataset_ref)``。

        内联 / 流式数据集的身份参数（``agent_name`` / ``dataset_name`` /
        ``project_id``）由 API 层随 ``create_run`` 持久化；执行期以 run
        记录为准，覆盖 ``@benchmark_component`` 的默认值。run 记录缺失
        时返回 ``None``。
        """
        run = await self._benchmark_service.get_run(run_id)
        if run is None:
            logger.error("benchmark.run_record_missing run_id=%s", run_id)
            return None
        return run.service_name, dataclasses.replace(
            fallback,
            dataset_name=run.dataset_name or fallback.dataset_name,
            project_id=run.project_id or fallback.project_id,
        )


__all__ = ["BenchmarkRunner", "BenchmarkRunExec", "FetchItemsSpec", "PipelineRunExec"]
