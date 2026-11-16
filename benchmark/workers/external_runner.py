"""ExternalBenchmarkRunner — orchestrator for externally triggered benchmarks (§5.5/§11).

Responsibilities:
1. Phase 1: OSS fetch + parse + fill in run config + create items + submit judge
2. Callback handling: receive judge/attribution results → write PG (§11)
3. Completion detection: CAS-atomic counter check → trigger aggregation + report (§11)
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.setting import settings as app_settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun, BenchmarkRunItem
from benchmark.domain.customization_config import fetch_customization_config
from benchmark.domain.eval_config import logical_benchmark_run_name
from benchmark.libs.jobqueue.job_listing import get_queue_job
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
from benchmark.runs.common.dimension_stats import (
    deserialize_stat_dimensions,
    resolve_run_stat_dimensions,
    serialize_stat_dimensions,
)
from benchmark.runs.common.execution_classes import (
    ExecutionClassResolutionError,
    LocalAttributionImpl,
    LocalAttributionRequest,
    load_execution_instance,
    normalize_attribution_callback,
    normalize_judge_callback,
    resolve_attribution_class,
    resolve_judge_class,
    snapshot_class_ref,
)
from benchmark.runs.common.profile_resolver import (
    ProfileResolutionError,
    ProfileResolver,
    ResolvedProfile,
)
from benchmark.runs.common.rubric_classes import (
    RubricClassResolutionError,
    resolve_rubric_class,
)
from benchmark.runs.common.submit_pipeline import (
    SubmitPipelineSpec,
    submit_prepared_items,
)
from benchmark.runs.common.trace_url_builder import build_trace_url
from benchmark.runs.common.utils import extract_ds
parse_external_attribution_detail = None
ExternalAttributionProvider = None
BatchAttributionSubmit = None
ExternalEvalService = None
DispatchRequest = None
JudgeDispatchInput = None
class schedule_notification_dispatch:
    pass
schedule_notification_dispatch_for_judge = None
normalize_actual = None
DailyDataLoader = None
DailyLoadSpec = None
SubmitBatchSpec = None
LocalAttributionRunner = None
LocalAttributionRunnerDeps = None
LocalAttributionTask = None
LocalEvalService = None
from business.evaluators.example.grading import rubric_v2_grade
from benchmark.runs.run_service import (
    BenchmarkRunSpec,
    CompleteRunWrite,
    ExpandDimensionMetricsSpec,
)
from benchmark.runs.run_stop import (
    is_stop_requested,
    stop_items_ready_for_report,
)
from benchmark.runs.schema import (
    FUSED_CRITERION_ID,
    AttributionBatchCallback,
    AttributionCallback,
    AttributionProfileConfig,
    AttributionResult,
    BenchmarkCallback,
    BenchmarkRuntimeProfile,
    CriterionScore,
    DailyParseResult,
    ExternalBenchmarkRequest,
    ExternalJudgeItemResult,
    JudgeCallback,
    JudgeProfileConfig,
    TriageMode,
)
from benchmark.attribution.code.document import merge_evidence
from benchmark.reporting.framework.fusion import EvalFailure, ScoreFuser
from benchmark.reporting.framework.report_spec import (
    ReportJobInput,
    generate_benchmark_report,
    resolve_rule_link_url,
)
from benchmark.reporting.framework.schema import (
    AggregationConfig,
    AttributionStats,
    Criterion,
)
from benchmark.queue.callback_next_submit import (
    SlotReleaseRequest,
    judge_submit_gate,
    promote_delayed_for_run,
    queue_names_for_pair,
    release_slot_and_promote_next,
)
from benchmark.queue.inflight_slot import (
    SlotKeySpec,
    build_slot_key,
    release_cluster_slot,
    release_slot,
)
from benchmark.queue.queue_factory import (
    ATTRIBUTION_RUNNING,
    ATTRIBUTION_WAIT,
    BENCHMARK_CASES,
    CODE_ATTRIBUTION,
    case_queue_name,
)
from benchmark.queue.retry_requeue import (
    RequeueDeps,
    RequeueOpts,
    force_fail_from_running,
    requeue_or_fail_from_running,
    retry_policy_from_settings,
)
from benchmark.workers.code_attribution_gate import should_run_code_attribution
from benchmark.workers.finalize_trigger import (
    tick_and_maybe_finalize,
    try_finalize_external,
)
from benchmark.workers.fused_row import (
    FusedRunningPrewrite,
    prewrite_running_fused_item,
)
from benchmark.services.trace_writer import (
    BenchmarkTraceRecordWriter,
    RunMeta,
    TriageUpsert,
    TriageWrite,
)

if TYPE_CHECKING:
    from benchmark.libs.storage import StorageClient

    from benchmark.runs.common.external_client import (
        ExternalAttributionClient,
        ExternalJudgeClient,
    )
    from benchmark.runs.common.external_client_factory import ExternalClientFactory
    from benchmark.runs.run_service import BenchmarkService
    from benchmark.reporting.framework.aggregation_calculator import (
        AggregationCalculator,
    )
    from benchmark.reporting.framework.base import BaseBenchmarkServices
    from benchmark.reporting.framework.schema import AggregatedMetrics
    from benchmark.reporting.framework.webdev_client import WebdevReportClient
    from benchmark.queue.queue_factory import EvalQueueRegistry
    from benchmark.services.evaluation import EvaluatorService
    from benchmark.reporting.publish import ReportSiteReportPublisher
    from benchmark.attribution.service.service import TriageAttributionService

logger = get_logger("agent_eval.benchmark.external_runner")

# ════════════════════════════════════════════════════════════════════════════
# Concurrency constants (§13.2) — no fixed sleep; rate is the callback latch / BullMQ slots / semaphore
# ════════════════════════════════════════════════════════════════════════════

MAX_CONCURRENT_CALLBACKS = 20
LOCAL_ATTR_CONCURRENCY = 1

# ── Report generation retry (exponential backoff + jitter, aligned with the model_retry.py pattern) ──
_REPORT_MAX_ATTEMPTS = 4  # compatible with old imports; actual retries are report_spec.REPORT_MAX_ATTEMPTS


@dataclass(frozen=True)
class AttributionCaseSpec:
    """Attribution-case enqueue spec (parameter object for ATTRIBUTION_WAIT job data, §4.2).

    Assembled on the judge-callback side; ``_enqueue_attribution_case`` expands it into BullMQ
    job data (fields correspond one to one; see ``async-attribution-design.md`` §4.2
    ``params``). The upstream_* trio are reconciliation / upstream-link fields and are echoed as-is.
    """

    run_id: str
    item_id: str
    source_trace_id: str | None
    adiu: str
    ds_start: str
    issue: dict[str, Any] | None = None
    attribution_provider: str = "external"
    ds_end: str | None = None
    trace_id: str | None = None
    do_cross_verify: bool = True
    upstream_job_id: str | None = None
    upstream_celery_task_id: str | None = None
    upstream_pipeline: str | None = None
    query: str = ""
    response: str = ""
    full_tool_calls_json: str | None = None
    target_url: str | None = None
    """Per-run attribution target (full trigger URL, §6.4); ``None`` = process default."""
    class_ref: str | None = None
    """Attribution execution-class ref (run snapshot ``attribution.class_ref``); ``None`` = route by provider name."""


@dataclass(frozen=True)
class ExternalRunnerDeps:
    """Optional dependencies for ExternalBenchmarkRunner (constructor parameter object).

    Required benchmark_service / storage_client and callback_url_base are still
    explicit arguments; when semaphore / downstream client / service are omitted, the runner builds them from
    module constants (§13.2).
    """

    judge_submit_semaphore: asyncio.Semaphore | None = None
    callback_semaphore: asyncio.Semaphore | None = None
    attribution_rate_limiter: asyncio.Semaphore | None = None
    local_attr_semaphore: asyncio.Semaphore | None = None
    aggregation_calculator: "AggregationCalculator | None" = None
    webdev_client: "WebdevReportClient | None" = None
    external_judge_client: "AoneJudgeClient | None" = None
    external_attribution_client: "AoneAttributionClient | None" = None
    external_client_factory: "AoneClientFactory | None" = None
    evaluator_service: "EvaluatorService | None" = None
    triage_service: "TriageAttributionService | None" = None
    queue_manager: "EvalQueueRegistry | None" = None
    #: Aone Pages republisher (optional; None means the report is OSS-only, same as before it was wired)
    pages_publisher: "AonePagesReportPublisher | None" = None
    #: Callback URL prefix (used to build the item callback address); empty means unconfigured (callback path unavailable)
    callback_url_base: str = ""


@dataclass(frozen=True)
class JudgeCallbackCase:
    """Per-item context for a judge callback (parsed once by _handle_judge_callback;
    shared parameter object for failure branching / the attribution gate / async attribution enqueue).

    item_id is the item resolved from the callback (job_id reverse lookup §15.3); source_trace_id
    is the judge job_id; item_input / item_output are the fused prewrite-row payload
    (passed through as-is; the consumer normalizes dicts); ri_row is the prewrite row
    (used to pull expected when enqueueing async attribution).
    """

    payload: JudgeCallback
    run_id: str
    item_id: str | None
    source_trace_id: str | None
    item_input: object
    item_output: object
    ri_row: BenchmarkRunItem | None
    queue_pair: str = "external"
    """Queue-pair marker for the run (§8.3): ``benchmark`` = unified queues; default = legacy external."""


# ---------------------------------------------------------------------------
# Method-argument collapse (FUNC-PARAMS-001): a long parameter list → a parameter object, unpacked on the method's first line
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _UnifiedCasesSubmit:
    """Arguments for ``_enqueue_unified_cases``."""

    run_id: str
    items: list[dict[str, Any]]
    resolved: ResolvedProfile
    payload: ExternalBenchmarkRequest
    service_name: str


@dataclass(frozen=True)
class _JudgeSubmitRequest:
    """Arguments for ``_submit_judge_via_service``."""

    run_id: str
    resolved: ResolvedProfile
    items: list[dict[str, Any]]
    aggregation: AggregationConfig
    service_name: str = ""
    project_id: str | None = None


@dataclass(frozen=True)
class _AttributionOnlySubmit:
    """Arguments for ``_submit_attribution_only``."""

    run_id: str
    parse_result: DailyParseResult
    rubric_data: list[dict[str, Any]]
    resolved: ResolvedProfile


@dataclass(frozen=True)
class _CallbackRouteContext:
    """Shared arguments for ``_route_callback_attribution`` / ``_run_attribution_gate``."""

    case: JudgeCallbackCase
    run: BenchmarkRun
    aggregation: AggregationConfig
    config: dict[str, Any] | None = None
    fused_score: float | None = None


@dataclass(frozen=True)
class _JudgeSlotRelease:
    """Arguments for ``_release_and_promote_next_judge``."""

    run_id: str
    item_id: str | None = None
    notification_job_id: str | None = None
    slot_ctx: "_JudgeSlotContext | None" = None


@dataclass(frozen=True)
class _AttributionSlotRelease:
    """Arguments for ``_release_and_promote_next_attribution``."""

    run_id: str
    item_id: str | None = None
    attribution_job_id: str | None = None
    slot_key: str | None = None
    cluster_kind: str = "attr"


@dataclass(frozen=True)
class _CodeAttributionEnqueue:
    """Arguments for enqueueing code attribution."""

    run_id: str
    item_id: str
    source_trace_id: str | None
    attribution: dict[str, Any]
    fused_score: object
    advance_received: bool


@dataclass(frozen=True)
class _AttributionResultWrite:
    """Arguments for ``_write_attribution_result``."""

    run_id: str
    item_id: str
    source_trace_id: str | None
    payload: AttributionCallback


@dataclass(frozen=True)
class _AttributionFailureHandle:
    """Arguments for ``_handle_attribution_callback_failure``."""

    payload: AttributionCallback
    run_id: str
    item_id: str
    source_trace_id: str | None
    is_terminal: bool


@dataclass(frozen=True)
class _LocalAttributionRun:
    """Arguments for ``_run_single_local_attribution``."""

    run_id: str
    item_id: str
    fused_score: float
    provider: str
    #: Sync attribution execution-class ref (snapshot ``attribution.target``); None = LocalAttributionRunner
    target: str | None = None
    #: project_id from the same source as judge (run/payload); attribution depends on this when the item has no metadata
    project_id: str | None = None


@dataclass(frozen=True)
class _AsyncAttributionRoute:
    """Attribution routing for ``_enqueue_async_attribution_case`` (from the run snapshot attribution)."""

    provider: str = "external"
    target_url: str | None = None
    class_ref: str | None = None


@dataclass(frozen=True)
class ExternalReportPublish:
    """Arguments for ``publish_external_report`` (public: reused by main / scripts)."""

    run_id: str
    config: dict[str, Any]
    rubric: list[Criterion]
    aggregation: AggregationConfig


@dataclass(frozen=True)
class _ExternalReportGenerate:
    """Arguments for ``_generate_external_report``."""

    run_id: str
    config: dict[str, Any]
    metrics: "AggregatedMetrics"
    final_score: float


def _customization_report_summary(config: dict[str, Any]) -> dict[str, Any] | None:
    """Run snapshot → a short custom summary the report can trace (None when there is no registration snapshot)."""
    snapshot = config.get("agent_registration")
    if not isinstance(snapshot, dict):
        return None
    customization = snapshot.get("customization") or {}
    judge = customization.get("judge") or {}
    attribution = customization.get("attribution") or {}
    return {
        "registration_id": snapshot.get("id"),
        # Multi-instance ownership key (design §12.2): reports for several instances of the same code are distinguished by this
        "internal_agent_name": snapshot.get("internal_agent_name"),
        "business": snapshot.get("business"),
        "variant_key": snapshot.get("variant_key"),
        "version": snapshot.get("version"),
        "attribution_enabled": snapshot.get("attribution_enabled", True),
        "judge_content_version": judge.get("content_version"),
        "judge_commit": judge.get("git_commit"),
        "attribution_content_version": attribution.get("content_version"),
        "attribution_commit": attribution.get("git_commit"),
    }


def _report_meta(report_url: str | None, config: dict[str, Any]) -> dict[str, Any] | None:
    """report_meta_json for complete_run: do not write it when there is no report_url (existing semantics unchanged)."""
    if not report_url:
        return None
    meta: dict[str, Any] = {
        "report_url": report_url,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    summary = _customization_report_summary(config)
    if summary is not None:
        meta["customization"] = summary
    return meta


def _build_criterion_scores(
    rubric: list[Criterion],
    payload: JudgeCallback,
) -> dict[str, Any]:
    """Assemble per-criterion raw scores: a missing criterion is recorded as EvalFailure."""
    scores: dict[str, Any] = {}
    for criterion in rubric:
        cid = criterion.id
        if cid in (payload.criterion_scores or {}):
            cs: CriterionScore = payload.criterion_scores[cid]
            scores[cid] = type(
                "EvaluatorResult",
                (),
                {
                    "score": cs.score,
                    "reason": cs.reason,
                    "sub_scores": cs.sub_scores,
                },
            )()
        else:
            scores[cid] = EvalFailure(RuntimeError(f"missing criterion {cid}"))
    return scores


def _normalize_judge_payload(config: dict[str, Any], payload: JudgeCallback) -> JudgeCallback:
    """Normalize the callback by the run snapshot's judge execution class; a hook failure becomes a non-retryable failure callback."""
    class_ref = snapshot_class_ref(config.get("judge"))
    try:
        return normalize_judge_callback(class_ref, payload)
    except Exception as exc:
        logger.exception(
            "callback.judge_normalize_failed run_id=%s job_id=%s class_ref=%s",
            payload.run_id,
            payload.job_id,
            class_ref,
        )
        return payload.model_copy(update={"error": f"callback_normalize_failed: {exc}"})


def _normalize_attribution_payload(
    config: dict[str, Any],
    payload: AttributionCallback,
) -> AttributionCallback:
    """Normalize the callback by the run snapshot's attribution execution class; a hook failure becomes a non-retryable failure callback."""
    class_ref = snapshot_class_ref(config.get("attribution"))
    try:
        return normalize_attribution_callback(class_ref, payload)
    except Exception as exc:
        logger.exception(
            "callback.attribution_normalize_failed run_id=%s job_id=%s class_ref=%s",
            payload.run_id,
            payload.job_id,
            class_ref,
        )
        return payload.model_copy(update={"error": f"callback_normalize_failed: {exc}"})


def _resolve_rule_link_url(config: dict[str, Any]) -> str | None:
    """Compatible with old imports; implementation is ``report_spec.resolve_rule_link_url``."""
    return resolve_rule_link_url(config)


def _resolve_judge_cfg(payload: ExternalBenchmarkRequest) -> JudgeProfileConfig:
    """Judge config resolution: alias contract first → v1 structured config fallback → process default."""
    if payload.judge_class:
        try:
            spec = resolve_judge_class(payload.judge_class)
        except ExecutionClassResolutionError as exc:
            raise ProfileResolutionError("invalid_execution_class", str(exc)) from exc
        return JudgeProfileConfig(
            execution=spec.execution,
            target=spec.target,
            provider=spec.provider,
            class_ref=spec.class_ref,
        )
    if payload.judge is not None:
        return payload.judge
    return JudgeProfileConfig()


def _resolve_attribution_cfg(payload: ExternalBenchmarkRequest) -> AttributionProfileConfig:
    """Attribution config resolution: alias contract first → v1 structured config fallback → provider mapping."""
    if payload.attribution_class:
        try:
            spec = resolve_attribution_class(payload.attribution_class)
        except ExecutionClassResolutionError as exc:
            raise ProfileResolutionError("invalid_execution_class", str(exc)) from exc
        return AttributionProfileConfig(
            execution=spec.execution,
            target=spec.target,
            provider=spec.provider,
            class_ref=spec.class_ref,
        )
    if payload.attribution is not None:
        return payload.attribution
    return AttributionProfileConfig(provider=payload.attribution_provider)


def _resolve_rubric_ref(payload: ExternalBenchmarkRequest) -> tuple[str | None, bool]:
    """rubric_ref resolution: alias contract first → v1 rubric_ref passthrough → default mapping.

    Returns ``(rubric_ref, component_rubric)``; default = ``(None, False)`` (same as v1
    semantics: decided by the benchmark's default Rubric mapping / meta.rubric).
    """
    if payload.rubric_class:
        try:
            spec = resolve_rubric_class(payload.rubric_class)
        except RubricClassResolutionError as exc:
            raise ProfileResolutionError("invalid_rubric_class", str(exc)) from exc
        return spec.class_ref, spec.class_ref is None
    if payload.rubric_ref is not None:
        return payload.rubric_ref, False
    return None, False


def _legacy_contract_fields(payload: ExternalBenchmarkRequest) -> list[str]:
    """List of v1 compatibility fields in use (for the deprecation warning)."""
    used = [
        name
        for name in ("judge", "attribution", "rubric_ref")
        if getattr(payload, name) is not None
    ]
    if (
        payload.attribution_class is None
        and payload.attribution is None
        and payload.attribution_provider != "example"
    ):
        used.append("attribution_provider")
    return used


def _write_parallel_report_config(
    config_json: dict[str, Any],
    payload: ExternalBenchmarkRequest,
) -> None:
    """Write the Judge / Attribution lists and both reports into the run config."""
    if payload.report_parser_class:
        config_json["report_parser_class"] = payload.report_parser_class
    if payload.report_html:
        config_json["report_html"] = payload.report_html
    if payload.judges:
        config_json["judges"] = [item.model_dump(exclude_none=True) for item in payload.judges]
    if payload.attributions:
        config_json["attributions"] = [
            item.model_dump(exclude_none=True) for item in payload.attributions
        ]
    if payload.attribution_report_parser_class:
        config_json["attribution_report_parser_class"] = payload.attribution_report_parser_class
    if payload.attribution_report_html:
        config_json["attribution_report_html"] = payload.attribution_report_html


def profile_from_external_request(
    payload: ExternalBenchmarkRequest,
) -> BenchmarkRuntimeProfile:
    """External-trigger contract → internal profile mapping.

    - ``date`` → ``data_date`` name normalization;
    - execution-class name contract first (``judge_class`` / ``attribution_class`` /
      ``rubric_class``); otherwise fall back to v1 compatibility fields (``judge`` / ``attribution`` /
      ``rubric_ref`` / ``attribution_provider``, deprecated migration window, §9);
    - ``rubric`` (JSON) does not go through the profile; the caller writes the config_json snapshot directly
      (today's semantics of attribution_only injecting attribution dimensions are kept).
    """
    legacy_fields = _legacy_contract_fields(payload)
    if legacy_fields:
        logger.warning(
            "run-external.legacy_contract_fields benchmark=%s fields=%s — "
            "v1 兼容字段已弃用，请迁移 judge_class/attribution_class/rubric_class",
            payload.benchmark_name,
            legacy_fields,
        )
    judge_cfg = _resolve_judge_cfg(payload)
    attribution_cfg = _resolve_attribution_cfg(payload)
    rubric_ref, component_rubric = _resolve_rubric_ref(payload)
    return BenchmarkRuntimeProfile(
        benchmark_name=payload.benchmark_name,
        rubric_ref=rubric_ref,
        component_rubric=component_rubric,
        parse_interceptor_ref=payload.parse_interceptor_ref,
        judge=judge_cfg,
        attribution=attribution_cfg,
        aggregation=payload.aggregation,
        oss_prefix=payload.effective_oss_prefix or payload.oss_prefix,
        data_date=payload.effective_date or payload.date,
        limit=payload.limit,
        run_id_suffix=payload.run_id_suffix,
        mode=payload.mode,
        generate_report=payload.generate_report,
        dataset_name=payload.dataset_name,
        project_id=payload.project_id,
        agent_registration_id=payload.agent_registration_id,
    )


@dataclass(frozen=True)
class UnifiedCaseSpec:
    """Run-level arguments for a unified-queue (``benchmark-cases``) case payload (§8.3).

    Shared by the external_runner submit side and the ops script (rerun_failed), so the payload
    shape has a single definition.
    """

    run_id: str
    criteria: list[dict[str, Any]]
    benchmark_name: str
    service_name: str = ""
    dataset_name: str = ""
    project_id: str | None = None
    judge_execution: str = "sync"
    ds: str = ""
    generate_report: bool = True


def build_unified_case_payload(
    item: dict[str, Any],
    spec: UnifiedCaseSpec,
) -> dict[str, Any] | None:
    """Build a unified-queue case payload (§8.3). Return None when the item is invalid (not a dict / no id).

    Spread the business item first (keep passthrough columns such as ``metadata.csv_row``), then overlay the queue
    contract fields; ``first_enqueued_at`` is not written when enqueueing to wait (stamped only on the first entry
    into running, so wait time does not consume the per-case 1h retry budget).
    """
    if not isinstance(item, dict):
        return None
    item_id = str(item.get("item_id") or item.get("id") or "")
    if not item_id:
        return None
    payload: dict[str, Any] = {
        **item,
        "eval_type": "external" if spec.judge_execution == "async" else "benchmark",
        "run_id": spec.run_id,
        "item_id": item_id,
        "id": str(item.get("id") or item_id),
        "case_id": item.get("case_id") or item_id,
        "retry_count": 0,
        "ds": spec.ds,
        "criteria": spec.criteria,
        "benchmark_services": spec.benchmark_name,
        "service_name": spec.service_name,
        "dataset_name": spec.dataset_name,
        "project_id": spec.project_id,
        "triage_mode": TriageMode.FUSED.value,
        "generate_report": spec.generate_report,
    }
    payload.pop("first_enqueued_at", None)
    return payload


async def run_custom_local_attribution(
    benchmark_service: "BenchmarkService",
    impl: LocalAttributionImpl,
    req: LocalAttributionRequest,
) -> None:
    """Call the sync attribution class's ``attribute_item``; advance received per item idempotently whether it succeeds or fails.

    Aligned with LocalAttributionRunner's finally counting: the caller already did
    expected +1, so a failed attribution must still be counted or the run cannot converge on finalize.
    """
    try:
        await impl.attribute_item(req)
    except Exception:
        logger.exception(
            "attribution.custom_local_failed run_id=%s item_id=%s impl=%s",
            req.run_id,
            req.item_id,
            type(impl).__name__,
        )
    finally:
        await benchmark_service.increment_attribution_received_once(req.run_id, req.item_id)


async def enqueue_attribution_case(
    queue_manager: "EvalQueueRegistry | None",
    case: AttributionCaseSpec,
) -> None:
    """Enqueue ATTRIBUTION_WAIT (BullMQ-driven attribution submit, shared at module level).

    Arguments align with the ``async-attribution-design.md`` §4.2 ``params`` business fields
    (see :class:`AttributionCaseSpec`). Shared by the external_runner callback side and the unified
    case worker (sync judge + async attribution, §8.2).
    ``first_enqueued_at`` is not written when enqueueing to wait: it is stamped when the attribution Worker first
    puts the case on attribution-running (same as judge, 1h each).
    """
    if queue_manager is None:
        raise RuntimeError(
            f"attribution enqueue requires queue_manager run_id={case.run_id}",
        )

    wait_q = queue_manager.get_queue(ATTRIBUTION_WAIT)
    case_job_id = f"{case.run_id}:{case.item_id}"

    data = {
        # Attribution inputs (async-attribution-design §4.2)
        "adiu": case.adiu,
        "ds_start": case.ds_start,
        "ds_end": case.ds_end,
        "trace_id": case.trace_id,
        "do_cross_verify": case.do_cross_verify,
        "issue": case.issue or {},
        # Required pipeline business fields (§4.2)
        "query": case.query,
        "response": case.response,
        "full_tool_calls_json": case.full_tool_calls_json,
        # Callback matching fields
        "run_id": case.run_id,
        "item_id": case.item_id,
        "source_trace_id": case.source_trace_id or "",
        # Reconciliation / upstream-link fields (echoed as-is)
        "upstream_job_id": case.upstream_job_id,
        "upstream_celery_task_id": case.upstream_celery_task_id,
        "upstream_pipeline": case.upstream_pipeline,
        # BullMQ lifetime: retry is independent; first_enqueued_at is stamped only on the first entry into running
        "retry_count": 0,
        "attribution_provider": case.attribution_provider,
        # per-run attribution target (§6.4; None = process default URL)
        "attribution_target_url": case.target_url,
        # Attribution execution class: the worker / healer loads the provider implementation from it
        "attribution_class_ref": case.class_ref,
    }

    await wait_q.add(
        "attr_case",
        data,
        {
            "jobId": case_job_id,
            "attempts": 3,
            "removeOnComplete": True,
            "removeOnFail": False,
        },
    )
    logger.info(
        "attribution.enqueued run_id=%s item_id=%s source_trace_id=%s adiu=%s",
        case.run_id,
        case.item_id,
        case.source_trace_id,
        case.adiu,
    )


@dataclass(frozen=True)
class _JudgeSlotContext:
    """Slot / queue-pair context for judge-callback cleanup (defaults match historical behavior)."""

    slot_key: str | None = None
    cluster_kind: str = "judge"
    queue_pair: str = "external"


class ExternalBenchmarkRunner:
    """Orchestrator for externally triggered benchmarks.

    The scheduler (§5.1) and the manual endpoint (§3) normalize trigger arguments to effective values
    before the call (effective_date / effective_oss_prefix: request override or the schedule config default).

    Responsibilities:
    1. Phase 1: OSS fetch + parse + fill in run config + create items/tasks + submit judge
    2. Callback handling: receive judge/attribution results → write PG (§11)
    3. Completion detection: once every callback has arrived, trigger aggregation + report (§11 _finalize_run)
    """

    def __init__(
        self,
        *,
        benchmark_service: "BenchmarkService",
        storage_client: "StorageClient | None",
        deps: ExternalRunnerDeps | None = None,
    ) -> None:
        d = deps or ExternalRunnerDeps()
        self._benchmark_service = benchmark_service
        self._storage_client = storage_client
        self._callback_url_base = d.callback_url_base.rstrip("/")
        self._judge_submit_limiter = d.judge_submit_semaphore or asyncio.Semaphore(1)
        self._callback_semaphore = d.callback_semaphore or asyncio.Semaphore(
            MAX_CONCURRENT_CALLBACKS,
        )
        self._attribution_rate_limiter = d.attribution_rate_limiter or asyncio.Semaphore(1)
        self._local_attr_semaphore = d.local_attr_semaphore or asyncio.Semaphore(
            LOCAL_ATTR_CONCURRENCY,
        )
        self._aggregation_calculator = d.aggregation_calculator
        self._webdev_client = d.webdev_client
        self._external_judge_client = d.external_judge_client
        self._external_attribution_client = d.external_attribution_client
        # per-run judge/attribution target resolution (§6.3/§6.4); None → only local sync is available
        self._external_client_factory = d.external_client_factory
        self._profile_resolver = ProfileResolver(client_factory=d.external_client_factory)
        self._evaluator_service = d.evaluator_service
        self._triage_service = d.triage_service
        self._queue_manager = d.queue_manager
        self._pages_publisher = d.pages_publisher

    def set_queue_manager(self, queue_manager: "EvalQueueRegistry") -> None:
        """Inject after BullMQ is ready (called by the main lifespan after EvalQueueRegistry is created)."""
        self._queue_manager = queue_manager
        logger.info(
            "external.queue_manager_attached prefix=%s",
            getattr(queue_manager, "prefix", None),
        )

    def _build_callback_url(self, run_id: str) -> str:
        """Build the callback URL."""
        return f"{self._callback_url_base}/v1/benchmarks/callback"

    @property
    def profile_resolver(self) -> ProfileResolver:
        """ProfileResolver (the API precheck reuses the same instance, 400 fail-fast)."""
        return self._profile_resolver

    @property
    def external_client_factory(self) -> "AoneClientFactory | None":
        """Process-level Aone client factory (§6.3); ``None`` = not configured.

        Read by lifespan / worker assembly, so a cross-module ``getattr`` of a private field
        does not silently degrade on a rename.
        """
        return self._external_client_factory

    def _resolve_judge_service(
        self,
        resolved: ResolvedProfile,
    ) -> "IEvalBenchmarkService":
        """Pick the judge service from the run profile (§6.3 per-run target).

        - ``async`` → ``AoneEvalService`` (the client is taken from the factory by target URL;
          default = the process default URL);
        - ``sync`` → local eval (``LocalEvalService`` wraps ``EvaluatorService``);
        - compatibility fallback: a legacy call (no profile snapshot) matches today's F3 —
          async when Aone is configured, otherwise local fallback.
        """
        judge = resolved.judge
        if judge.execution == "async":
            if self._external_client_factory is not None:
                return ExternalEvalService(
                    external_client=self._external_client_factory.get_judge_client(judge.target),
                )
            if self._external_judge_client is not None:
                return ExternalEvalService(external_client=self._external_judge_client)
            raise RuntimeError("judge execution=async but Aone client factory not configured")
        if self._evaluator_service is not None:
            return LocalEvalService(evaluator_service=self._evaluator_service)
        raise RuntimeError("no judge service configured (aone factory or evaluator_service)")

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 1: initialize_and_submit (§5.5)
    # ═════════════════════════════════════════════════════════════════════════

    async def initialize_and_submit(
        self,
        *,
        run_id: str,
        payload: ExternalBenchmarkRequest,
        benchmark_services: "BaseBenchmarkServices",
    ) -> int:
        """Phase 1: OSS fetch → Resolver resolution → unified submit pipeline (the intercept point is mandatory).

        Returns:
            Total number of cases resolved

        Raises:
            DailyNoDataError: today's OSS file does not exist
            ProfileResolutionError: the profile is illegal (rubric_ref / target and similar)
            Exception: download/parse/submit failed
        """
        meta = benchmark_services.get_metadata()

        # 0. profile resolution (§5.2): v1 flat contract → profile → ResolvedProfile
        #    (rubric class load / judge·attribution target check / interceptor load)
        profile = profile_from_external_request(payload)
        resolved = self._profile_resolver.resolve(profile, meta)

        logger.info(
            "external.initialize_start run_id=%s benchmark_name=%s "
            "registration_id=%s mode=%s attribution_provider=%s "
            "rubric_ref=%s interceptor=%s judge=%s/%s",
            run_id,
            meta.name,
            (payload.agent_registration_id or "").strip() or "-",
            payload.mode,
            resolved.attribution.provider,
            resolved.rubric_ref,
            resolved.parse_interceptor_ref,
            resolved.judge.execution,
            resolved.judge.provider,
        )

        # 1. Direct GET + parse of today's OSS file (path convention §5.2, interface §5.4).
        #    The loader only emits generic full-KV rows (§7.1, no business semantics); when limit is set, only the
        #    first N rows are parsed (not processed / not stored), and downstream counts align to N.
        loader = DailyDataLoader(self._storage_client)
        parse_result = await loader.load_daily(
            DailyLoadSpec(
                run_id=run_id,
                oss_prefix=payload.effective_oss_prefix or "",
                date=payload.effective_date or "",
                limit=payload.limit,
                require_case_id=payload.require_case_id,
                object_key=payload.oss_object_key,
            ),
        )
        total_count = parse_result.total_count

        logger.info(
            "external.oss_loaded run_id=%s total_count=%d object_key=%s title_columns=%d",
            run_id,
            total_count,
            parse_result.object_key,
            len(parse_result.title_columns),
        )

        # Heartbeat: avoid a false stale after a large-file download (§5.6.4)
        await self._benchmark_service.heartbeat(run_id)

        # 2. Decide the mode: attribution_only takes effect only when the attribution provider is 'aone' (today's semantics)
        is_attr_only = (
            payload.mode == "attribution_only" and resolved.attribution.provider == "external"
        )

        # 3. config_json assembly is closed over into the Resolver snapshot (§5.2); today's semantics of v1 ``rubric``
        #    (JSON override) writing the snapshot directly are kept (attribution_only injects attribution dimensions)
        snapshot = resolved.config_snapshot()
        rubric_data = payload.rubric or snapshot["rubric"]
        aggregation_data = snapshot["aggregation"]
        aggregation = resolved.aggregation

        # 3.5 Fill in benchmark_runs config (merge semantics, §5.5).
        # When dataset_name is empty, fill in {date}_benchmark_final_results.
        # project_id must not fall back to that placeholder: it is not a Langfuse instance id, and would make
        # example/local attribution report unknown project_id; leave the default for the item-level
        # metadata.project_id / csv_row.project (passthrough promotion) fallback.
        date_str = payload.effective_date or ""
        default_dataset = f"{date_str}_benchmark_final_results" if date_str else None

        config_json: dict[str, Any] = {
            "external": True,
            "benchmark_services": payload.benchmark_name,
            "judge_class": payload.judge_class,
            "attribution_class": payload.attribution_class,
            "rubric_class": payload.rubric_class,
            "attribution_provider": resolved.attribution.provider,
            "generate_report": payload.generate_report,
            "triage_mode": TriageMode.FUSED.value,
            "max_concurrency": 1,
            "oss_prefix": payload.effective_oss_prefix,
            "data_date": payload.effective_date,
            "oss_object_key": parse_result.object_key,
            "oss_total_rows": total_count,
            "oss_limit": payload.limit,
            "oss_title_columns": parse_result.title_columns,
            "rubric": rubric_data,
            "aggregation": aggregation_data,
            # Configured runtime snapshot (§5.2): the worker/callback side routes execution from the snapshot
            "rubric_ref": resolved.rubric_ref,
            "parse_interceptor_ref": resolved.parse_interceptor_ref,
            # Scoring-rule link (Rubric class rule_link_url): source for filling the report slot
            "rule_link_url": resolved.rule_link_url,
            # Stat-dimension snapshot (Rubric class stat_dimensions, including the Chinese descriptions):
            # aggregation/report rebuilds them from config_json (consistent across nodes, preferred over the registry)
            "stat_dimensions": snapshot["stat_dimensions"],
            "judge": snapshot["judge"],
            "attribution": snapshot["attribution"],
            # §8.3 queue-merge marker: new runs all use benchmark-cases/running
            "queue_pair": "benchmark",
        }
        _write_parallel_report_config(config_json, payload)
        if is_attr_only:
            config_json["mode"] = "attribution_only"

        await self._benchmark_service.create_external_run(
            run_id,
            BenchmarkRunSpec(
                benchmark_name=meta.name,
                service_name=meta.service_name,
                dataset_name=payload.dataset_name or default_dataset,
                project_id=payload.project_id,
                total_samples=total_count,
                total_evaluators=0 if is_attr_only else len(resolved.rubric),
                config_json=config_json,
            ),
        )

        logger.info(
            "external.run_created run_id=%s total_samples=%d total_evaluators=%d is_attr_only=%s",
            run_id,
            total_count,
            0 if is_attr_only else len(resolved.rubric),
            is_attr_only,
        )

        await self._benchmark_service.heartbeat(run_id)

        logger.info(
            "external.items_loaded run_id=%s items=%d",
            run_id,
            total_count,
        )

        # 4. attribution-only branch: skip task splitting + judge, and attribute the whole batch directly
        if is_attr_only:
            return await self._submit_attribution_only(
                _AttributionOnlySubmit(
                    run_id=run_id,
                    parse_result=parse_result,
                    rubric_data=rubric_data,
                    resolved=resolved,
                ),
            )

        # 5. BullMQ queues drive task dispatch; PG shard tasks are not needed
        await self._benchmark_service.heartbeat(run_id)

        # 6. Update status to running (judge is about to be submitted, so it does not stay initializing during rate limiting)
        await self._benchmark_service.update_run_status(
            run_id,
            BenchmarkStatus.RUNNING,
            total_samples=total_count,
            started_at=datetime.now(UTC),
        )

        logger.info(
            "external.status_running run_id=%s total_samples=%d",
            run_id,
            total_count,
        )

        # 7. Unified submit pipeline (§7.2): [intercept point] → persist parse-failed items → enqueue.
        #    judge_expected_count = the raw total; a parse-failed item advances judge_received_count on its behalf
        #    via a failure stub (cas_advance failed), so the counts converge.
        await self._benchmark_service.merge_config_json(
            run_id,
            {
                "judge_expected_count": total_count,
            },
        )

        async def _enqueue(items: list[dict[str, Any]]) -> int:
            """Enqueue parsed items: prefer the unified queue; with no queue_manager, fall back to in-process submit."""
            if self._queue_manager is not None:
                return await self._enqueue_unified_cases(
                    _UnifiedCasesSubmit(
                        run_id=run_id,
                        items=items,
                        resolved=resolved,
                        payload=payload,
                        service_name=meta.service_name,
                    ),
                )
            return await self._submit_judge_via_service(
                _JudgeSubmitRequest(
                    run_id=run_id,
                    resolved=resolved,
                    items=items,
                    aggregation=aggregation,
                    service_name=meta.service_name,
                    project_id=payload.project_id,
                ),
            )

        summary = await submit_prepared_items(
            SubmitPipelineSpec(
                run_id=run_id,
                resolved=resolved,
                raw_items=parse_result.items,
                enqueue_items=_enqueue,
                mark_item_failed=lambda item, reason: self._mark_parse_failed(
                    run_id,
                    item,
                    reason,
                ),
                snapshot=config_json,
            ),
        )
        logger.info(
            "external.submit_pipeline_done run_id=%s total=%d parsed=%d "
            "parse_failed=%d enqueued=%d",
            run_id,
            summary.total_raw,
            summary.parsed,
            summary.parse_failed,
            summary.enqueued,
        )
        # Every parse failed (enqueued=0): no case is enqueued and no callback fires finalize,
        # so compensate with one tick on the counter model, or the run stays RUNNING forever (§7.6/§10)
        if summary.enqueued == 0 and summary.total_raw > 0:
            logger.error(
                "external.submit_all_parse_failed run_id=%s total=%d — trigger finalize",
                run_id,
                summary.total_raw,
            )
            if resolved.judge.execution == "async" or resolved.attribution.execution == "async":
                await try_finalize_external(run_id, self._benchmark_service)
            else:
                await tick_and_maybe_finalize(
                    run_id,
                    "benchmark",
                    benchmark_run_service=self._benchmark_service,
                )
        return total_count

    async def _enqueue_unified_cases(
        self,
        req: _UnifiedCasesSubmit,
    ) -> int:
        """Enqueue parsed items onto the unified queue ``benchmark-cases`` (§8.3 queue merge).

        Pass through the business item body (including ``metadata.csv_row``) for the worker to consume;
        do not write ``first_enqueued_at`` here (stamped only on the first entry into running, so
        wait time does not consume the per-case 1h retry budget).
        """
        run_id = req.run_id
        items = req.items
        resolved = req.resolved
        payload = req.payload
        service_name = req.service_name
        if self._queue_manager is None:
            raise RuntimeError(f"unified enqueue requires queue_manager run_id={run_id}")
        criteria = [{"id": c.id, "evaluator_ref": c.evaluator_ref} for c in resolved.rubric]
        # L2 process isolation (design §12.7): a run whose snapshot business hits the list uses the isolation
        # queue; a snapshot-read failure falls back to the default queue (a routing error must not block eval accept).
        queue_name = await self._case_queue_name_for_run(run_id)
        wait_q = self._queue_manager.get_queue(queue_name)
        job_opts = {
            "attempts": 1,
            "removeOnComplete": True,
            "removeOnFail": True,
        }
        # ds shares its source with the data_date / run_id partition day (yyyymmdd parsed from the full storage path).
        spec = UnifiedCaseSpec(
            run_id=run_id,
            criteria=criteria,
            benchmark_name=resolved.benchmark_name,
            service_name=service_name,
            dataset_name=payload.dataset_name or "",
            project_id=payload.project_id,
            judge_execution=resolved.judge.execution,
            ds=str(payload.effective_date or "").strip() or extract_ds(run_id),
            generate_report=payload.generate_report,
        )
        count = 0
        for item in items:
            case_payload = build_unified_case_payload(item, spec)
            if case_payload is None:
                continue
            await wait_q.add(
                "case",
                case_payload,
                {"jobId": f"{run_id}:{case_payload['item_id']}", **job_opts},
            )
            count += 1
        logger.info(
            "external.enqueued_unified run_id=%s queue=%s count=%d judge_execution=%s",
            run_id,
            queue_name,
            count,
            resolved.judge.execution,
        )
        return count

    async def _case_queue_name_for_run(self, run_id: str) -> str:
        """Pick the case queue from the run snapshot business (L2 routing; fall back to default on error)."""
        try:
            run = await self._benchmark_service.get_run(run_id)
            config = (
                run.config_json if run is not None and isinstance(run.config_json, dict) else {}
            )
            snapshot = config.get("agent_registration")
            business = snapshot.get("business") if isinstance(snapshot, dict) else None
            return case_queue_name(business, fetch_customization_config().l2_businesses)
        except Exception:
            logger.warning(
                "external.l2_route_fallback run_id=%s queue=%s",
                run_id,
                BENCHMARK_CASES,
                exc_info=True,
            )
            return BENCHMARK_CASES

    async def _mark_parse_failed(
        self,
        run_id: str,
        item: dict[str, Any],
        reason: str,
    ) -> None:
        """Interceptor parse-failed item: persist an eval_failed stub + advance the count (§7.6/§10).

        ``cas_advance_external_judge_item`` writes the failed row and increments
        ``judge_received_count`` in one transaction — the same advance path as the callback side, so
        ``judge_expected_count`` (= the raw total) converges and the run does not stick.
        """
        await self._benchmark_service.cas_advance_external_judge_item(
            result=ExternalJudgeItemResult(
                run_id=run_id,
                item_id=str(item.get("item_id") or item.get("id") or ""),
                failed=True,
                score=0.0,
                judge_json={"error": reason, "reason": reason, "stage": "parse"},
                error_message=reason,
                item_input=item.get("input") if isinstance(item.get("input"), dict) else None,
                item_output=item.get("output") if isinstance(item.get("output"), dict) else None,
            ),
        )

    async def _submit_judge_via_service(
        self,
        req: _JudgeSubmitRequest,
    ) -> int:
        """In-process submit when there is no queue_manager: serial via the callback latch (replaces the fixed sleep).

        ``wait_after_item`` registers the latch and submits the next item only after this judge callback arrives
        (design: callback-driven serial). item_map (item_id → original item, used by the per-item
        callback to prewrite run_items) is derived from parsed items (output of the unified submit pipeline).
        """
        run_id = req.run_id
        resolved = req.resolved
        aggregation = req.aggregation
        judge_service = self._resolve_judge_service(resolved)
        # No-queue fallback: stamp run-level identity onto the item so LocalEvalService can build
        # EvaluatorRequest (otherwise trace_id/project_id/agent_name are missing).
        items: list[dict[str, Any]] = []
        for raw in req.items:
            item = dict(raw)
            if req.service_name and not item.get("service_name"):
                item["service_name"] = req.service_name
            if req.project_id and not item.get("project_id"):
                item["project_id"] = req.project_id
            items.append(item)
        total_count = len(items)
        # In-memory item_map (do not write a PG staging table)
        item_map: dict[str, dict[str, Any]] = {item["item_id"]: item for item in items}
        logger.info(
            "external.judge_submit_start run_id=%s items=%d "
            "callback_url=%s judge_service=%s mode=callback_gated",
            run_id,
            total_count,
            self._build_callback_url(run_id),
            type(judge_service).__name__,
        )

        async def _on_item_submitted(item_id: str, job_id: str) -> None:
            """Per-item callback: prewrite run_items (one row, including input/output)."""
            item_data = item_map.get(item_id)
            if item_data is None:
                logger.error(
                    "external.pre_write_item_missing run_id=%s item_id=%s",
                    run_id,
                    item_id,
                )
                return
            await prewrite_running_fused_item(
                FusedRunningPrewrite(
                    run_id=run_id,
                    item_id=item_id,
                    source_trace_id=job_id,
                    item_input=item_data.get("input"),
                    item_output=item_data.get("output"),
                    create_score=0.0,
                    create_judge_json={
                        "job_id": job_id,
                        "status": "running",
                        "expected_output": item_data.get("expected_output"),
                    },
                ),
            )
            logger.info(
                "external.pre_write_run_item run_id=%s item_id=%s job_id=%s",
                run_id,
                item_id,
                job_id,
            )

        async def _wait_after_item(item_id: str, job_id: str) -> None:
            """Submit the next item only after this judge callback arrives (design: callback-driven serial)."""
            await judge_submit_gate.register(run_id, item_id)
            logger.info(
                "external.wait_callback_before_next run_id=%s item_id=%s job_id=%s",
                run_id,
                item_id,
                job_id,
            )
            await judge_submit_gate.wait(run_id, item_id)

        try:
            async with self._judge_submit_limiter:
                # Only the async (Aone async callback) path needs the latch; local sync eval is self-contained and must not wait
                use_gate = resolved.judge.execution == "async"
                job_id_map = await judge_service.submit_batch(
                    SubmitBatchSpec(
                        run_id=run_id,
                        items=items,
                        rubric=resolved.rubric,
                        aggregation=aggregation,
                        callback_url=self._build_callback_url(run_id),
                        on_item_submitted=_on_item_submitted,
                        wait_after_item=_wait_after_item if use_gate else None,
                    ),
                )
        finally:
            await judge_submit_gate.cancel_run(run_id)

        logger.info(
            "external.judge_submit_complete run_id=%s submitted=%d unique_job_ids=%d",
            run_id,
            len(job_id_map),
            len(set(job_id_map.values())),
        )
        return total_count

    async def _submit_attribution_only(
        self,
        req: _AttributionOnlySubmit,
    ) -> int:
        """attribution-only mode: create a full-rubric task + submit the whole-batch attribution directly.

        Differences from judge_attribution mode:
        - do not split by criterion; submit the full rubric as one Aone attribution task
        - do not go through judge; submit all items + the full rubric as one Aone attribution task
        - after Aone finishes it sends one ``AttributionBatchCallback``
        """
        run_id = req.run_id
        parse_result = req.parse_result
        rubric_data = req.rubric_data
        resolved = req.resolved
        if self._external_client_factory is None and self._external_attribution_client is None:
            raise RuntimeError(
                "attribution_only mode requires aone attribution configured",
            )

        logger.info(
            "external.attribution_only_start run_id=%s total_items=%d rubric_dims=%d",
            run_id,
            parse_result.total_count,
            len(rubric_data),
        )

        # 1. BullMQ queues drive task dispatch; a PG benchmark_task is not needed

        # 2. Submit the whole-batch Aone attribution (all items + the full rubric → 1 POST → 1 task_id).
        #    per-run target (§6.4): a factory-aware provider; default = the process default URL
        provider = (
            ExternalAttributionProvider(client_factory=self._external_client_factory)
            if self._external_client_factory is not None
            else ExternalAttributionProvider(external_client=self._external_attribution_client)
        )
        async with self._attribution_rate_limiter:
            notification_task_id = await provider.submit_batch_all(
                BatchAttributionSubmit(
                    run_id=run_id,
                    items=parse_result.items,
                    callback_url=self._build_callback_url(run_id),
                    target_url=resolved.attribution.target,
                ),
            )

        # 3. BullMQ queues drive the task lifetime; PG claim/heartbeat is not needed

        # 4. Record the aone task_id on config_json (so the callback can correlate it)
        await self._benchmark_service.merge_config_json(
            run_id,
            {"notification_task_id": notification_task_id},
        )

        # 5. Update status to running (attribution is submitted, waiting for the callback)
        await self._benchmark_service.update_run_status(
            run_id,
            BenchmarkStatus.RUNNING,
            total_samples=parse_result.total_count,
            started_at=datetime.now(UTC),
        )

        logger.info(
            "external.attribution_only_submitted run_id=%s aone=%s total=%d",
            run_id,
            notification_task_id,
            parse_result.total_count,
        )
        return parse_result.total_count

    # ═════════════════════════════════════════════════════════════════════════
    # callback handling (§11)
    # ═════════════════════════════════════════════════════════════════════════

    async def handle_callback(self, payload: BenchmarkCallback) -> None:
        """Unified callback-handling entry (rate-limited by the callback semaphore, §13.3.2).

        Catch ``BaseException`` (including ``asyncio.CancelledError``) so a
        BackgroundTask exception is not silently swallowed by Starlette.
        """
        try:
            async with self._callback_semaphore:
                if isinstance(payload, JudgeCallback):
                    await self._handle_judge_callback(payload)
                elif isinstance(payload, AttributionCallback):
                    await self._handle_attribution_callback(payload)
                elif isinstance(payload, AttributionBatchCallback):
                    await self._handle_attribution_batch_callback(payload)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            logger.exception(
                "callback.handler_exception type=%s run_id=%s",
                type(payload).__name__,
                getattr(payload, "run_id", ""),
            )

    async def _handle_judge_callback(self, payload: JudgeCallback) -> None:
        """Handle a judge callback (§11)."""
        run_id = payload.run_id

        logger.info(
            "callback.judge_received_raw run_id=%s job_id=%s item_id=%s "
            "error=%s criterion_scores_keys=%s payload_dump=%s",
            run_id,
            payload.job_id,
            payload.item_id,
            payload.error,
            list(payload.criterion_scores.keys()) if payload.criterion_scores else [],
            payload.model_dump_json(),
        )

        # 1. Terminal guard — still write run_items data, but skip shard / finalize
        loaded = await self._load_run_with_status(run_id)
        if loaded is None:
            await self._release_and_promote_next_judge(
                _JudgeSlotRelease(
                    run_id=run_id,
                    item_id=payload.item_id,
                    notification_job_id=payload.job_id,
                ),
            )
            return
        run, run_status, run_config_json = loaded
        # stopped is like completed/failed: still write the callback data, and only skip finalize.
        is_terminal = run_status in (
            BenchmarkStatus.COMPLETED.value,
            BenchmarkStatus.FAILED.value,
            BenchmarkStatus.STOPPED.value,
        )
        if is_terminal:
            logger.warning(
                "callback.terminal_run_late_callback run_id=%s status=%s "
                "item_id=%s — writing data but skipping finalize",
                run_id,
                run_status,
                payload.item_id,
            )
        config = run_config_json if isinstance(run_config_json, dict) else {}
        payload = _normalize_judge_payload(config, payload)

        # 1.5 Resolve job_id → item_id (§15.3).
        # A judge callback's payload.job_id = judge job_id = source_trace_id
        item_id, resolved = await self._resolve_callback_item_id(
            payload,
            kind="judge",
        )
        if resolved and item_id is None:
            await self._release_and_promote_next_judge(
                _JudgeSlotRelease(
                    run_id=run_id,
                    item_id=None,
                    notification_job_id=payload.job_id,
                    slot_ctx=_JudgeSlotContext(
                        queue_pair=str(config.get("queue_pair") or "external"),
                    ),
                ),
            )
            return

        # source_trace_id comes from payload.job_id (already written when run_items was prewritten)
        source_trace_id = payload.job_id

        logger.info(
            "callback.judge_resolved run_id=%s item_id=%s source_trace_id=%s",
            run_id,
            item_id,
            source_trace_id,
        )

        rubric = load_rubric_from_config(config)
        aggregation = load_aggregation_from_config(config)

        # Read input/output from the benchmark_run_items prewrite row
        ri_row = await BenchmarkRunItem.filter(
            run_id=run_id,
            item_id=item_id,
            criterion_id=FUSED_CRITERION_ID,
        ).first()
        item_input = ri_row.input if ri_row else None
        item_output = ri_row.output if ri_row else None

        logger.info(
            "callback.judge_item_lookup run_id=%s item_id=%s "
            "ri_row_found=%s has_input=%s has_output=%s",
            run_id,
            item_id,
            ri_row is not None,
            item_input is not None,
            item_output is not None,
        )

        # Per-item judge-callback context: shared by failure branching / the attribution gate / async attribution enqueue
        case = JudgeCallbackCase(
            payload=payload,
            run_id=run_id,
            item_id=item_id,
            source_trace_id=source_trace_id,
            item_input=item_input,
            item_output=item_output,
            ri_row=ri_row,
            queue_pair=str(config.get("queue_pair") or "external"),
        )

        # Failure callback: retryable → running→wait; non-retryable → PG failed (design §8.3)
        if not payload.is_success:
            await self._handle_judge_callback_failure(
                case,
                is_terminal=is_terminal,
            )
            return

        # A "fake success" with empty criterion_scores: no error but no dimension scores (for example
        # ieval_external-20260909:58). Fusing it as success would make fused=0 and the
        # fallback would enter attribution by mistake, raising expected with no job_id. Treat it as a non-retryable failure
        # and do not attribute.
        if not payload.criterion_scores:
            logger.error(
                "callback.judge_empty_criterion_scores run_id=%s item_id=%s "
                "job_id=%s — treat as non-retryable failure, skip attribution",
                run_id,
                item_id,
                payload.job_id,
            )
            empty_payload = payload.model_copy(
                update={"error": "empty_criterion_scores"},
            )
            await self._handle_judge_callback_failure(
                JudgeCallbackCase(
                    payload=empty_payload,
                    run_id=case.run_id,
                    item_id=case.item_id,
                    source_trace_id=case.source_trace_id,
                    item_input=case.item_input,
                    item_output=case.item_output,
                    ri_row=case.ri_row,
                    queue_pair=case.queue_pair,
                ),
                is_terminal=is_terminal,
            )
            return

        # 2. Assemble per-criterion raw scores (do not write yet; write once after aggregation)
        scores = _build_criterion_scores(rubric, payload)

        # 3. ScoreFuser.afuse() — aggregate (async HTTP when a remote fusion plugin is used)
        fused_score = await ScoreFuser.afuse(scores, rubric, aggregation)
        triage_mode = config.get("triage_mode", TriageMode.FUSED.value)
        is_fused = triage_mode == TriageMode.FUSED.value

        logger.info(
            "callback.judge_aggregated run_id=%s item_id=%s "
            "fused_score=%s triage_mode=%s is_fused=%s "
            "raw_scores=%s",
            run_id,
            item_id,
            fused_score,
            triage_mode,
            is_fused,
            {cid: getattr(s, "score", None) for cid, s in scores.items()},
        )

        # 3.5 trace_url (best-effort, does not block the main flow): go through the trace_url factory —
        #     the business-registered extractor pulls (project_id, trace_id) from the callback payload
        #     (unregistered = the default ieval shape, reading payload fields directly). The shared main flow uses
        #     TraceResolver to resolve the primary id and splice the agentfuse dashboard link; any
        #     missing / miss / exception falls back to the value passed through from the judge side.
        #     Extractors are registered by logical component name; the benchmark_name column may already be a registrationId.

        trace_url = await build_trace_url(
            payload,
            benchmark_name=logical_benchmark_run_name(
                column_name=getattr(run, "benchmark_name", None),
                config=config,
            ),
            fallback=payload.trace_url,
        )

        # 4. Conditional advance in one transaction: item terminal + evaluated/failed + judge_received
        judge_json = payload.model_dump()
        # Append the raw callback body to judge_json.raw_json (unfiltered/unparsed, aligned with attribution)
        judge_json["raw_json"] = payload.model_dump()
        logger.info(
            "callback.judge_db_write run_id=%s item_id=%s "
            "fused_score=%s criterion_scores=%s "
            "source_trace_id=%s",
            run_id,
            item_id,
            fused_score,
            {cid: getattr(s, "score", None) for cid, s in scores.items()},
            source_trace_id,
        )
        advanced = await self._benchmark_service.cas_advance_external_judge_item(
            result=ExternalJudgeItemResult(
                run_id=run_id,
                item_id=str(item_id),
                failed=False,
                score=float(fused_score) if fused_score is not None else 0.0,
                judge_json=judge_json,
                source_trace_id=source_trace_id,
                trace_id=payload.trace_id,
                trace_url=trace_url,
                item_input=item_input if isinstance(item_input, dict) else None,
                item_output=item_output if isinstance(item_output, dict) else None,
            ),
        )
        logger.info(
            "callback.judge_db_written run_id=%s item_id=%s "
            "fused_score=%s criterion_count=%d advanced=%s",
            run_id,
            item_id,
            fused_score,
            len(rubric),
            advanced,
        )

        # 4.5 Release the next submit immediately (callback-driven serial / BullMQ slot release + promote).
        #     This must happen before attribution: attribution can be slow and must not block the next Aone submit.
        await self._release_and_promote_next_judge(
            _JudgeSlotRelease(
                run_id=run_id,
                item_id=item_id,
                notification_job_id=payload.job_id,
                slot_ctx=_JudgeSlotContext(queue_pair=case.queue_pair),
            ),
        )

        # Terminal run / replayed callback: data is already written; skip attribution/finalize side effects
        if is_terminal or not advanced:
            logger.info(
                "callback.terminal_or_replay run_id=%s item_id=%s "
                "fused_score=%s is_terminal=%s advanced=%s — "
                "skipping attribution/finalize side effects",
                run_id,
                item_id,
                fused_score,
                is_terminal,
                advanced,
            )
            return

        # 5. A low score triggers attribution immediately (per-item, without waiting for every judge to finish)
        await self._route_callback_attribution(
            _CallbackRouteContext(
                case=case,
                run=run,
                aggregation=aggregation,
                config=config,
                fused_score=fused_score,
            ),
        )

        # 6. BullMQ queues drive the shard lifetime; PG shard-task management is not needed
        await self._benchmark_service.heartbeat(run_id)

        # 7. CAS completion detection (finalize_trigger.try_finalize_external)
        await try_finalize_external(run_id, self._benchmark_service)

    async def _route_callback_attribution(
        self,
        route_ctx: _CallbackRouteContext,
    ) -> None:
        """Attribution routing after a judge callback (§6.4/§8.2, extracted from _handle_judge_callback for PLR0915).

        The snapshot ``attribution.execution`` wins; a legacy run (no snapshot key) is derived from today's
        provider semantics (aone→async / example·local→sync /
        none→push straight to Aone by the judge fused score).
        """
        case = route_ctx.case
        config = route_ctx.config or {}
        fused_score = route_ctx.fused_score
        attr_snapshot = config.get("attribution")
        attr_snapshot = attr_snapshot if isinstance(attr_snapshot, dict) else {}
        attribution_provider = str(
            attr_snapshot.get("provider") or config.get("attribution_provider") or "example"
        )
        attr_execution = str(attr_snapshot.get("execution") or "")
        if not attr_execution:
            attr_execution = {"external": "async", "none": "none"}.get(
                attribution_provider,
                "sync",
            )
        if attr_execution == "none":
            # Push straight only for explicit none (including provider derivation / a no_attribution snapshot);
            # a default empty snapshot lands on sync (see the derivation above) and does not enter this branch.
            self._schedule_judge_notification_from_callback(route_ctx)
            return
        should_attr = await self._run_attribution_gate(route_ctx)
        if not should_attr:
            return
        if attr_execution == "async":
            # ← async attribution provider enqueues ATTRIBUTION_WAIT (per-run target passed through)
            await self._enqueue_async_attribution_case(
                case,
                _AsyncAttributionRoute(
                    provider=attribution_provider,
                    target_url=(
                        str(attr_snapshot["target"]) if attr_snapshot.get("target") else None
                    ),
                    class_ref=snapshot_class_ref(attr_snapshot),
                ),
            )
        elif attr_execution == "sync":
            (
                _count,
                _cfg,
                claimed,
            ) = await self._benchmark_service.increment_attribution_expected_once(
                case.run_id,
                str(case.item_id),
            )
            if not claimed:
                return
            await self._run_single_local_attribution(
                _LocalAttributionRun(
                    run_id=case.run_id,
                    item_id=str(case.item_id),
                    fused_score=fused_score or 0.0,
                    provider=attribution_provider,
                    target=(str(attr_snapshot["target"]) if attr_snapshot.get("target") else None),
                    project_id=getattr(route_ctx.run, "project_id", None),
                ),
            )

    def _schedule_judge_notification_from_callback(self, route_ctx: _CallbackRouteContext) -> None:
        """``attribution.execution=none``: after the HTTP judge callback, push straight to Aone by the fused score."""
        case = route_ctx.case
        fused_score = route_ctx.fused_score
        if fused_score is None or case.item_id is None:
            logger.info(
                "callback.judge_aone_dispatch_skip run_id=%s item_id=%s fused_score=%s",
                case.run_id,
                case.item_id,
                fused_score,
            )
            return
        payload = case.payload
        criterion_scores: dict[str, Any] = {}
        for cid, cs in (payload.criterion_scores or {}).items():
            criterion_scores[str(cid)] = {
                "score": getattr(cs, "score", None),
                "reason": getattr(cs, "reason", "") or "",
            }
        item_input = case.item_input if isinstance(case.item_input, dict) else {}
        query = str(item_input.get("query") or "")
        schedule_notification_dispatch_for_judge(
            DispatchRequest(
                store=self._benchmark_service,
                run_id=case.run_id,
                item_id=str(case.item_id),
                source_trace_id=case.source_trace_id,
                config=route_ctx.config,
            ),
            JudgeDispatchInput(
                fused_score=float(fused_score),
                criterion_scores=criterion_scores,
                query=query or None,
            ),
        )
        logger.info(
            "callback.judge_aone_dispatch_scheduled run_id=%s item_id=%s fused_score=%s",
            case.run_id,
            case.item_id,
            fused_score,
        )

    async def _run_attribution_gate(
        self,
        route_ctx: _CallbackRouteContext,
    ) -> bool:
        """Gate for triggering attribution immediately on a low score (per-item, without waiting for every judge).

        Shared policy (RemoteConfig binding → registry → default): ieval_external registers
        DimensionFullScoreTriggerPolicy (fire when any of D0–D4 is not a full score; D0 full mark is 1,
        D1–D4 full mark is 2); an unregistered business falls back to the default fused < 1.4 (FAIL/POOR/ACCEPTABLE).
        """

        case = route_ctx.case
        run = route_ctx.run
        fused_score = route_ctx.fused_score
        aggregation = route_ctx.aggregation
        payload, run_id, item_id = case.payload, case.run_id, case.item_id
        run_cfg = getattr(run, "config_json", None)
        trigger_policy = resolve_trigger_policy(
            logical_benchmark_run_name(
                column_name=getattr(run, "benchmark_name", None),
                config=run_cfg,
            ),
        )
        grade = rubric_v2_grade(float(fused_score)) if fused_score is not None else "FAIL"
        should_attr = await should_trigger_attribution(
            trigger_policy,
            fused_score=fused_score,
            criterion_scores={
                cid: cs.score for cid, cs in (payload.criterion_scores or {}).items()
            },
        )
        logger.info(
            "callback.judge_attribution_gate run_id=%s item_id=%s "
            "fused_score=%s grade=%s should_attribute=%s pass_threshold=%s "
            "trigger_policy=%s",
            run_id,
            item_id,
            fused_score,
            grade,
            should_attr,
            aggregation.pass_threshold,
            type(trigger_policy).__name__,
        )
        return should_attr

    async def _load_run_with_status(
        self,
        run_id: str,
    ) -> tuple[BenchmarkRun, str, object] | None:
        """Load the run and safely read status/config_json; return None when the run does not exist."""
        logger.info("callback.judge_before_get_run run_id=%s", run_id)
        run = await self._benchmark_service.get_run(run_id)
        # Log the get_run return immediately, without touching any model attribute (an attribute-eval exception would swallow the log)
        logger.info(
            "callback.judge_get_run_done run_id=%s run_is_none=%s run_type=%s",
            run_id,
            run is None,
            type(run).__name__,
        )
        if run is None:
            logger.error(
                "callback.run_not_found run_id=%s",
                run_id,
            )
            return None
        # Access model attributes safely — CharEnumField.to_python_value may raise ValueError
        try:
            run_status = run.status
            run_config_json = run.config_json
        except Exception:
            logger.exception(
                "callback.judge_run_attr_access_failed run_id=%s run_type=%s",
                run_id,
                type(run).__name__,
            )
            raise
        logger.info(
            "callback.judge_run_loaded run_id=%s run_found=%s status=%s config_json_type=%s",
            run_id,
            True,
            run_status,
            type(run_config_json).__name__,
        )
        return run, run_status, run_config_json

    async def _resolve_callback_item_id(
        self,
        payload: JudgeCallback,
        *,
        kind: str,
    ) -> tuple[str | None, bool]:
        """When payload.item_id is missing, reverse-look-up item_id by job_id (§15.3).

        Returns ``(item_id, resolved)``; ``resolved=False`` means both item_id
        and job_id are empty and no reverse lookup ran (the caller keeps the original semantics and continues).
        """
        if payload.item_id is not None:
            return payload.item_id, True
        if not payload.job_id:
            return None, False
        item_id = await self._benchmark_service.resolve_item_id_by_job_id(
            payload.run_id,
            payload.job_id,
            kind=kind,
        )
        if item_id is None:
            logger.error(
                "callback.unknown_job_id run_id=%s job_id=%s",
                payload.run_id,
                payload.job_id,
            )
        return item_id, True

    async def _enqueue_async_attribution_case(
        self,
        case: JudgeCallbackCase,
        route: "_AsyncAttributionRoute",
    ) -> None:
        """Async attribution path: assemble the full issue, enqueue ATTRIBUTION_WAIT, and count expected.

        Issue assembly aligns with async-attribution-design §4.2.1; only after a successful enqueue
        do expected +1 per item, idempotently, so a missing queue / enqueue failure does not inflate expected,
        and a repeated enqueue does not raise the ceiling so received can never catch up.
        """
        payload, run_id, source_trace_id = (
            case.payload,
            case.run_id,
            case.source_trace_id,
        )
        item_id, item_input, item_output, ri_row = (
            str(case.item_id),
            case.item_input,
            case.item_output,
            case.ri_row,
        )
        issue = payload.model_dump()
        # Extract query / response / tool_calls_json (required by pipeline §4.2)
        attr_query = ""
        attr_response = ""
        attr_tool_calls_json = ""
        if isinstance(item_input, dict):
            attr_query = str(item_input.get("query") or "")
            issue.setdefault("query", attr_query)
        if isinstance(item_output, dict):
            # When the response_text column is missing, fall back to result — today's ieval CSV model output is in
            # the result column (the data pipeline has no response_text column). Missing it makes the worker-side
            # required-field check reject it (submit_missing_fields), and attribution is never submitted.
            attr_response = str(item_output.get("response_text") or item_output.get("result") or "")
            attr_tool_calls_json = str(
                item_output.get("tool_calls_json") or "",
            )
        # Inject expected (extracted from the prewrite ri_row.judge_json)
        raw_expected = ""
        if ri_row is not None and isinstance(ri_row.judge_json, dict):
            raw_eo = ri_row.judge_json.get("expected_output")
            raw_expected = str(raw_eo) if raw_eo else ""
        issue.setdefault("expected", raw_expected)
        # Normalize actual (dict → str)
        issue.setdefault("actual", normalize_actual(item_output))
        # Extract adiu / ds_start
        adiu = item_input.get("adiu", "") if isinstance(item_input, dict) else ""
        ds_start = extract_ds(run_id)

        await self._enqueue_attribution_case(
            AttributionCaseSpec(
                run_id=run_id,
                item_id=item_id,
                source_trace_id=source_trace_id,
                adiu=adiu,
                ds_start=ds_start,
                issue=issue,
                attribution_provider=route.provider,
                query=attr_query,
                response=attr_response,
                full_tool_calls_json=attr_tool_calls_json or None,
                target_url=route.target_url,
                class_ref=route.class_ref,
            ),
        )
        await self._benchmark_service.increment_attribution_expected_once(
            run_id,
            item_id,
        )

    async def _enqueue_attribution_case(
        self,
        case: "AttributionCaseSpec",
    ) -> None:
        """Enqueue ATTRIBUTION_WAIT (delegates to the module-level :func:`enqueue_attribution_case`)."""
        await enqueue_attribution_case(self._queue_manager, case)

    async def _handle_judge_callback_failure(  # noqa: PLR0915
        self,
        case: JudgeCallbackCase,
        *,
        is_terminal: bool,
    ) -> None:
        """Branch a judge failure callback (design §8.3).

        - **Retryable**: do not write an item terminal state; ``requeue_or_fail_from_running`` (count or window)
          → release the slot / promote; on terminal fail ``mark_failed`` writes PG failed
        - **Non-retryable**: ``cas_advance`` failed → release the slot / promote; do not attribute
        """
        payload, run_id, source_trace_id = (
            case.payload,
            case.run_id,
            case.source_trace_id,
        )
        item_id = str(case.item_id)
        # Normalize dict | None (aligned with the isinstance narrowing at the original call site)
        item_input = case.item_input if isinstance(case.item_input, dict) else None
        item_output = case.item_output if isinstance(case.item_output, dict) else None
        err_msg = str(payload.error or "")[:2000]
        retryable = payload.is_retryable
        logger.info(
            "callback.judge_failure run_id=%s item_id=%s job_id=%s retryable=%s error=%.200s",
            run_id,
            item_id,
            payload.job_id,
            retryable,
            err_msg,
        )

        judge_json = payload.model_dump()
        # Append the raw callback body to judge_json.raw_json (unfiltered/unparsed, aligned with attribution)
        judge_json["raw_json"] = payload.model_dump()

        async def _mark_failed(  # noqa: PLR0915
            d: dict[str, Any],
            *,
            retry_count: int = 0,
            **kw: object,
        ) -> None:
            reason = str(kw.get("reason") or err_msg)
            await self._benchmark_service.cas_advance_external_judge_item(
                result=ExternalJudgeItemResult(
                    run_id=str(d.get("run_id") or run_id),
                    item_id=str(d.get("item_id") or item_id),
                    failed=True,
                    score=0.0,
                    judge_json={
                        **judge_json,
                        "error": err_msg,
                        "reason": reason,
                        "retry_count": retry_count,
                    },
                    source_trace_id=source_trace_id,
                    trace_id=payload.trace_id,
                    trace_url=payload.trace_url,
                    item_input=item_input,
                    item_output=item_output,
                    error_message=err_msg,
                ),
            )

        # BullMQ path + retryable: peek the slot → requeue/terminal fail (delete running) → explicitly release the rate limit
        if retryable and self._queue_manager is not None and payload.job_id and not is_terminal:
            wait_name, running_name = queue_names_for_pair(case.queue_pair)
            running_q = self._queue_manager.get_queue(running_name)
            wait_q = self._queue_manager.get_queue(wait_name)
            peek = await get_queue_job(running_q, payload.job_id)
            slot_key: str | None = None
            cluster_kind = "judge"
            if peek is not None:
                pdata = dict(peek.data or {})
                slot_key = pdata.get("slot_key")
                if not slot_key and pdata.get("submitter_node_id") and pdata.get("run_id"):
                    slot_key = build_slot_key(
                        self._queue_manager.prefix,
                        SlotKeySpec(
                            kind="judge",
                            node=str(pdata["submitter_node_id"]),
                            run_id=str(pdata["run_id"]),
                        ),
                    )
                cluster_kind = str(pdata.get("cluster_kind") or "judge")

            outcome = await requeue_or_fail_from_running(
                RequeueDeps(
                    running_queue=running_q,
                    wait_queue=wait_q,
                    policy=retry_policy_from_settings(
                        app_settings,
                        retry_max_duration_ms=app_settings.external_retry_max_duration_ms,
                    ),
                    mark_failed=_mark_failed,
                ),
                payload.job_id,
            )
            await self._release_and_promote_next_judge(
                _JudgeSlotRelease(
                    run_id=run_id,
                    item_id=item_id,
                    notification_job_id=payload.job_id,
                    slot_ctx=_JudgeSlotContext(
                        slot_key=slot_key,
                        cluster_kind=cluster_kind,
                        queue_pair=case.queue_pair,
                    ),
                ),
            )
            logger.info(
                "callback.judge_failure_requeue run_id=%s item_id=%s outcome=%s slot_released=%s",
                run_id,
                item_id,
                outcome,
                bool(slot_key),
            )
            if outcome == "failed":
                await try_finalize_external(run_id, self._benchmark_service)
            return

        # Non-retryable / no queue / terminal run: peek the slot → clear running + write failed → release
        slot_key = None
        cluster_kind = "judge"
        if self._queue_manager is not None and payload.job_id and not is_terminal:
            _wait_name, running_name = queue_names_for_pair(case.queue_pair)
            running_q = self._queue_manager.get_queue(running_name)
            peek = await get_queue_job(running_q, payload.job_id)
            if peek is not None:
                pdata = dict(peek.data or {})
                slot_key = pdata.get("slot_key")
                if not slot_key and pdata.get("submitter_node_id") and pdata.get("run_id"):
                    slot_key = build_slot_key(
                        self._queue_manager.prefix,
                        SlotKeySpec(
                            kind="judge",
                            node=str(pdata["submitter_node_id"]),
                            run_id=str(pdata["run_id"]),
                        ),
                    )
                cluster_kind = str(pdata.get("cluster_kind") or "judge")
            try:
                await force_fail_from_running(
                    running_queue=running_q,
                    running_job_id=payload.job_id,
                    mark_failed=_mark_failed,
                )
            except Exception:
                logger.exception(
                    "callback.judge_force_fail_running_err run_id=%s item=%s",
                    run_id,
                    item_id,
                )
        # When running is already gone, force_fail may not write PG; cas fills it in idempotently
        await _mark_failed(
            {"run_id": run_id, "item_id": item_id},
            retry_count=0,
            reason="non_retryable",
        )

        await self._release_and_promote_next_judge(
            _JudgeSlotRelease(
                run_id=run_id,
                item_id=item_id,
                notification_job_id=payload.job_id,
                slot_ctx=_JudgeSlotContext(
                    slot_key=slot_key,
                    cluster_kind=cluster_kind,
                    queue_pair=case.queue_pair,
                ),
            ),
        )
        if not is_terminal:
            await try_finalize_external(run_id, self._benchmark_service)

    async def _release_and_promote_next_judge(
        self,
        release: _JudgeSlotRelease,
    ) -> None:
        """Judge-callback cleanup: release the latch + release the BullMQ slot + promote the next item.

        If the failure path already deleted running, pass ``slot_ctx.slot_key``, or the rate limit is not released.
        ``slot_ctx.queue_pair`` is the run's queue-pair marker (§8.3): a unified run operates on
        ``benchmark-cases/running``, a legacy run on ``external-wait/running``.
        """
        run_id = release.run_id
        item_id = release.item_id
        notification_job_id = release.notification_job_id
        slot_ctx = release.slot_ctx
        ctx = slot_ctx or _JudgeSlotContext()
        wait_name, running_name = queue_names_for_pair(ctx.queue_pair)
        try:
            await release_slot_and_promote_next(
                SlotReleaseRequest(
                    queue_manager=self._queue_manager,
                    notification_job_id=notification_job_id,
                    run_id=run_id,
                    item_id=item_id,
                    slot_key=ctx.slot_key,
                    cluster_kind=ctx.cluster_kind,
                    wait_queue_name=wait_name,
                    running_queue_name=running_name,
                ),
            )
        except Exception:
            logger.exception(
                "callback.release_promote_failed run_id=%s item_id=%s job_id=%s",
                run_id,
                item_id,
                notification_job_id,
            )

    async def _release_and_promote_next_attribution(
        self,
        release: _AttributionSlotRelease,
    ) -> None:
        """Attribution-callback cleanup: release the slot + promote the next item.

        Aligned with ``_release_and_promote_next_judge``, but it operates on the
        ATTRIBUTION_RUNNING/WAIT queues and has no judge_submit_gate (attribution does not use the latch).
        If the failure path already deleted running, pass ``slot_key``, or the rate limit is not released.
        """
        run_id = release.run_id
        item_id = release.item_id
        attribution_job_id = release.attribution_job_id
        slot_key = release.slot_key
        cluster_kind = release.cluster_kind
        if self._queue_manager is None:
            return
        try:
            running_q = self._queue_manager.get_queue(ATTRIBUTION_RUNNING)
            # requeue_or_fail / force_fail already deleted the running job; this is a fallback remove
            if attribution_job_id:
                leftover = await get_queue_job(running_q, attribution_job_id)
                if leftover is not None:
                    data = dict(leftover.data or {})
                    if not slot_key:
                        slot_key = data.get("slot_key")
                        if not slot_key and data.get("submitter_node_id") and data.get("run_id"):
                            slot_key = build_slot_key(
                                self._queue_manager.prefix,
                                SlotKeySpec(
                                    kind="attr",
                                    node=str(data["submitter_node_id"]),
                                    run_id=str(data["run_id"]),
                                ),
                            )
                    if data.get("cluster_kind"):
                        cluster_kind = str(data["cluster_kind"])
                    try:
                        await leftover.remove()
                    except Exception:
                        logger.warning(
                            "callback.attr_remove_leftover_failed job_id=%s",
                            attribution_job_id,
                            exc_info=True,
                        )
            if slot_key:
                await release_slot(str(slot_key))
            await release_cluster_slot(cluster_kind)
            wait_q = self._queue_manager.get_queue(ATTRIBUTION_WAIT)
            await promote_delayed_for_run(wait_q, run_id, limit=1)
        except Exception:
            logger.exception(
                "callback.attr_release_promote_failed run_id=%s item_id=%s job_id=%s",
                run_id,
                item_id,
                attribution_job_id,
            )

    async def _handle_attribution_callback(self, payload: AttributionCallback) -> None:
        """Handle an Aone attribution callback (§11)."""
        run_id = payload.run_id

        logger.info(
            "callback.attribution_received_raw run_id=%s job_id=%s "
            "item_id=%s attribution=%s payload_dump=%s",
            run_id,
            payload.job_id,
            payload.item_id,
            payload.attribution.model_dump() if payload.attribution else None,
            payload.model_dump_json(),
        )

        # Terminal guard: drop completed/failed; stopped still writes the attribution result and skips finalize.
        run = await self._benchmark_service.get_run(run_id)
        if run is None:
            logger.warning(
                "callback.attribution_terminal_run run_id=%s status=%s",
                run_id,
                "none",
            )
            return
        run_status = str(getattr(run.status, "value", run.status))
        if run_status in (
            BenchmarkStatus.COMPLETED.value,
            BenchmarkStatus.FAILED.value,
        ):
            logger.warning(
                "callback.attribution_terminal_run run_id=%s status=%s",
                run_id,
                run_status,
            )
            return
        is_terminal = run_status == BenchmarkStatus.STOPPED.value
        if is_terminal:
            logger.warning(
                "callback.attribution_stopped_late_write run_id=%s status=%s — "
                "writing data but skipping finalize",
                run_id,
                run_status,
            )
        run_config = getattr(run, "config_json", None)
        payload = _normalize_attribution_payload(
            run_config if isinstance(run_config, dict) else {},
            payload,
        )

        # 1. Resolve job_id → item_id (§15.3; the double fallback is _resolve_attribution_item_id)
        item_id = await self._resolve_attribution_item_id(payload)
        if not item_id:
            return

        # ← correction: take the real source_trace_id (judge job_id) from the DB
        ri_row = await BenchmarkRunItem.filter(
            run_id=run_id,
            item_id=item_id,
            criterion_id=FUSED_CRITERION_ID,
        ).first()
        source_trace_id = ri_row.source_trace_id if ri_row else None
        logger.info(
            "callback.attribution_resolved run_id=%s item_id=%s source_trace_id=%s ri_row_found=%s",
            run_id,
            item_id,
            source_trace_id,
            ri_row is not None,
        )

        # Failure callback: retryable → running→wait; non-retryable → increment received (aligned with judge)
        if not payload.is_success:
            await self._handle_attribution_callback_failure(
                _AttributionFailureHandle(
                    payload=payload,
                    run_id=run_id,
                    item_id=str(item_id),
                    source_trace_id=source_trace_id,
                    is_terminal=is_terminal,
                ),
            )
            return

        # 3. Clear the ATTRIBUTION_RUNNING BullMQ job + release the slot + promote
        await self._cleanup_attribution_running_job(
            run_id=run_id,
            item_id=item_id,
            payload=payload,
        )

        # 4. Write the attribution result
        await self._write_attribution_result(
            _AttributionResultWrite(
                run_id=run_id,
                item_id=item_id,
                source_trace_id=source_trace_id,
                payload=payload,
            ),
        )

        attribution_doc = (
            payload.attribution.model_dump() if payload.attribution is not None else {}
        )
        fused_score = getattr(ri_row, "score", None) if ri_row is not None else None
        deferred = await self._enqueue_code_attribution(
            _CodeAttributionEnqueue(
                run_id=run_id,
                item_id=str(item_id),
                source_trace_id=source_trace_id,
                attribution=attribution_doc,
                fused_score=fused_score,
                advance_received=True,
            ),
        )
        if deferred:
            await self._benchmark_service.heartbeat(run_id)
            return

        # 5. Increment the count per item, idempotently (deduped with the healer timeout)
        (
            new_count,
            config,
            claimed,
        ) = await self._benchmark_service.increment_attribution_received_once(
            run_id,
            str(item_id),
        )
        logger.info(
            "callback.attribution_received run_id=%s item_id=%s count=%s claimed=%s",
            run_id,
            item_id,
            new_count,
            claimed,
        )

        # 5.5 Aone ticket push (SDD benchmark-attribution-aone-dispatch):
        # fire only on the first callback with claimed=True (HTTP replay / healer replay stay idempotent),
        # create_task runs it in the background, exceptions are fully caught inside the coroutine, and it does not block heartbeat / CAS / finalize.
        if claimed and payload.attribution is not None:
            schedule_notification_dispatch(
                DispatchRequest(
                    store=self._benchmark_service,
                    run_id=run_id,
                    item_id=str(item_id),
                    source_trace_id=source_trace_id,
                    config=config,
                ),
                data=payload.attribution,
            )

        # Keep the heartbeat
        await self._benchmark_service.heartbeat(run_id)

        # 6. CAS completion detection (atomic check of the judge + attribution counters).
        # A late callback on stopped only persists; it does not trigger finalize / the report.
        if is_terminal:
            logger.info(
                "callback.attribution_skip_finalize_terminal run_id=%s item_id=%s",
                run_id,
                item_id,
            )
            return
        attr_expected = config.get("attribution_expected_count", 0)
        if new_count >= attr_expected:
            logger.info(
                "callback.attribution_cas_trigger run_id=%s item_id=%s received=%s expected=%s",
                run_id,
                item_id,
                new_count,
                attr_expected,
            )
            await try_finalize_external(run_id, self._benchmark_service)
        else:
            logger.info(
                "callback.attribution_cas_skip run_id=%s item_id=%s received=%s expected=%s",
                run_id,
                item_id,
                new_count,
                attr_expected,
            )

    async def _resolve_attribution_item_id(
        self,
        payload: AttributionCallback,
    ) -> str | None:
        """Resolve job_id → item_id (§15.3), double fallback: ieval_job_id first.

        An attribution callback's payload.job_id is params.job_id echoed as-is
        (= judge source_trace_id), not the attribution task_id; payload.ieval_job_id
        is the attribution task_id. Look up the attribution_job_id column by ieval_job_id first,
        so a wrongly echoed params.job_id does not hit the wrong item's row.
        """
        run_id = payload.run_id
        item_id = payload.item_id
        if item_id is None:
            # Prefer the attribution task_id (ieval_job_id) for resolution
            resolve_id = payload.ieval_job_id or payload.job_id
            if resolve_id:
                item_id = await self._benchmark_service.resolve_item_id_by_job_id(
                    run_id,
                    resolve_id,
                    kind="attribution",
                )
            # When ieval_job_id misses, fall back to payload.job_id (the echoed params.job_id)
            if item_id is None and payload.job_id and payload.job_id != resolve_id:
                item_id = await self._benchmark_service.resolve_item_id_by_job_id(
                    run_id,
                    payload.job_id,
                    kind="attribution",
                )
            if item_id is None:
                logger.error(
                    "callback.unknown_job_id run_id=%s job_id=%s ieval_job_id=%s",
                    run_id,
                    payload.job_id,
                    payload.ieval_job_id,
                )
                return None
        if not item_id:
            logger.error(
                "callback.attribution_missing_item_id run_id=%s job_id=%s",
                run_id,
                payload.job_id,
            )
            return None
        return item_id

    async def _cleanup_attribution_running_job(
        self,
        *,
        run_id: str,
        item_id: str,
        payload: AttributionCallback,
    ) -> None:
        """Clear the ATTRIBUTION_RUNNING BullMQ job + release the slot + promote the next item."""
        qm = self._queue_manager
        if qm is None:
            return
        running_q = qm.get_queue(ATTRIBUTION_RUNNING)
        callback_job = None
        # Look up the BullMQ job by case_job_id (= {run_id}:{item_id}) first,
        # because pending_callback's jobId is now case_job_id
        # (so a duplicate job_id returned by Aone cannot overwrite/mismatch).
        # Fall back to ieval_job_id / job_id for old pending_callback jobs.
        callback_job = await get_queue_job(running_q, f"{run_id}:{item_id}")
        if callback_job is None and payload.ieval_job_id:
            callback_job = await get_queue_job(
                running_q,
                payload.ieval_job_id,
            )
        if callback_job is None and payload.job_id:
            callback_job = await get_queue_job(running_q, payload.job_id)

        if callback_job is None:
            return
        if (callback_job.name or "") == "inflight":
            # The Worker is still submitting: do not steal inflight; wait for the Worker to upgrade, then take the early-callback path
            logger.info(
                "callback.attribution_inflight_left job_id=%s run_id=%s (worker still submitting)",
                callback_job.id,
                run_id,
            )
            return
        data = dict(callback_job.data or {})
        try:
            await callback_job.remove()
        except Exception:
            # remove is atomic (Lua): when concurrent callbacks clean the same job, only one delete wins
            # and the other raises. release_slot is a non-idempotent DECR, so the loser must return immediately,
            # or a double release decrements another run's slot count.
            logger.info(
                "callback.attribution_running_remove_race job_id=%s run_id=%s "
                "(already removed by concurrent callback)",
                callback_job.id,
                run_id,
            )
            return
        logger.info(
            "callback.attribution_running_removed job_id=%s run_id=%s",
            callback_job.id,
            run_id,
        )
        slot_key = data.get("slot_key")
        if not slot_key:
            node = data.get("submitter_node_id")
            rid = data.get("run_id") or run_id
            if node and rid:
                slot_key = build_slot_key(
                    qm.prefix,
                    SlotKeySpec(
                        kind="attr",
                        node=str(node),
                        run_id=str(rid),
                    ),
                )
        cluster_kind = str(data.get("cluster_kind") or "attr")
        if slot_key:
            await release_slot(str(slot_key))
        await release_cluster_slot(cluster_kind)
        wait_q = qm.get_queue(ATTRIBUTION_WAIT)
        await promote_delayed_for_run(wait_q, run_id, limit=1)

    async def _write_attribution_result(
        self,
        write: _AttributionResultWrite,
    ) -> None:
        """Write the attribution result to the triage table (when payload.attribution is present)."""
        run_id = write.run_id
        item_id = write.item_id
        source_trace_id = write.source_trace_id
        payload = write.payload
        if payload.attribution is None:
            return
        writer = BenchmarkTraceRecordWriter(
            run_id=run_id,
            criterion_id=FUSED_CRITERION_ID,
        )
        triage_level = payload.attribution.effective_triage_level
        root_cause = payload.attribution.effective_root_cause
        logger.info(
            "callback.attribution_db_write run_id=%s item_id=%s "
            "source_trace_id=%s primary_issue_layer=%s "
            "root_cause_chain=%.200s is_eval_misjudge=%s",
            run_id,
            item_id,
            source_trace_id,
            triage_level,
            root_cause,
            payload.attribution.is_eval_misjudge,
        )
        dumped = payload.attribution.model_dump()
        detail = merge_evidence(
            {
                **parse_external_attribution_detail(dumped),
                "raw_json": dict(dumped),
            },
            dumped,
            payload.evidence or {},
            payload.model_extra or {},
        )
        await writer.upsert_triage(
            TriageUpsert(
                trace_id=source_trace_id,
                evaluator_ref="fused",
                triage=TriageWrite(
                    triage_level=triage_level,
                    root_cause=root_cause,
                    # raw_json: the attribution's raw return value as-is, plus the three evidence bodies from the callback
                    attribution_detail_json=detail,
                ),
                dataset_item_id=item_id,
            ),
        )
        logger.info(
            "callback.attribution_db_written run_id=%s item_id=%s",
            run_id,
            item_id,
        )

    async def _enqueue_code_attribution(self, request: _CodeAttributionEnqueue) -> bool:
        """Enqueue when code attribution is needed. Return False if enqueue fails; the caller finishes as today."""
        run_id = request.run_id
        item_id = request.item_id
        source_trace_id = request.source_trace_id
        attribution = request.attribution
        advance_received = request.advance_received
        score_value = request.fused_score
        if score_value is None and app_settings.code_attribution_enabled:
            row = await BenchmarkRunItem.filter(
                run_id=run_id,
                item_id=item_id,
                criterion_id=FUSED_CRITERION_ID,
            ).first()
            score_value = row.score if row is not None else None
        score: float | None
        try:
            score = None if score_value is None else float(score_value)
        except (TypeError, ValueError):
            score = None
        if not should_run_code_attribution(attribution, score):
            return False
        queue_manager = self._queue_manager
        if queue_manager is None:
            return False
        try:
            queue = queue_manager.get_queue(CODE_ATTRIBUTION)
            await queue.add(
                "code_attribution",
                {
                    "run_id": run_id,
                    "item_id": item_id,
                    "source_trace_id": source_trace_id or "",
                    "advance_received": advance_received,
                },
                {
                    "jobId": f"{run_id}:{item_id}",
                    "attempts": 2,
                    "removeOnComplete": True,
                    "removeOnFail": False,
                },
            )
        except Exception:
            logger.exception(
                "code_attribution.enqueue_failed run_id=%s item_id=%s",
                run_id,
                item_id,
            )
            return False
        logger.info(
            "code_attribution.enqueued run_id=%s item_id=%s advance_received=%s",
            run_id,
            item_id,
            advance_received,
        )
        return True

    async def _handle_attribution_callback_failure(
        self,
        handle: _AttributionFailureHandle,
    ) -> None:
        """Branch an attribution failure callback (aligned with judge _handle_judge_callback_failure).

        - **Retryable**: do not write an item terminal state; ``requeue_or_fail_from_running``
          (count or window) → release the slot / promote; on terminal fail ``mark_failed``
          increments attribution_received_count
        - **Non-retryable**: ``force_fail_from_running`` → release the slot / promote;
          increment received
        """
        payload = handle.payload
        run_id = handle.run_id
        item_id = handle.item_id
        is_terminal = handle.is_terminal
        err_msg = str(payload.error or "")[:2000]
        retryable = payload.is_retryable
        logger.info(
            "callback.attribution_failure run_id=%s item_id=%s job_id=%s "
            "ieval_job_id=%s retryable=%s error=%.200s",
            run_id,
            item_id,
            payload.job_id,
            payload.ieval_job_id,
            retryable,
            err_msg,
        )

        async def _mark_failed(
            d: dict[str, Any],
            *,
            retry_count: int = 0,
            **kw: object,
        ) -> None:
            reason = str(kw.get("reason") or err_msg)
            logger.info(
                "callback.attribution_mark_failed run_id=%s item_id=%s reason=%s retry_count=%d",
                d.get("run_id") or run_id,
                d.get("item_id") or item_id,
                reason,
                retry_count,
            )
            await self._benchmark_service.increment_attribution_received_once(
                str(d.get("run_id") or run_id),
                str(d.get("item_id") or item_id),
            )

        # resolve_id: prefer case_job_id (= {run_id}:{item_id}),
        # because pending_callback's BullMQ jobId is now case_job_id.
        # Fall back to ieval_job_id / job_id for old pending_callback jobs.
        resolve_id = (
            f"{run_id}:{item_id}" if item_id else (payload.ieval_job_id or payload.job_id or "")
        )

        # BullMQ path + retryable: peek the slot → requeue/terminal fail (delete running) → release the rate limit
        if retryable and self._queue_manager is not None and resolve_id and not is_terminal:
            running_q = self._queue_manager.get_queue(ATTRIBUTION_RUNNING)
            wait_q = self._queue_manager.get_queue(ATTRIBUTION_WAIT)
            peek = await get_queue_job(running_q, resolve_id)
            slot_key: str | None = None
            cluster_kind = "attr"
            if peek is not None:
                pdata = dict(peek.data or {})
                slot_key = pdata.get("slot_key")
                if not slot_key and pdata.get("submitter_node_id") and pdata.get("run_id"):
                    slot_key = build_slot_key(
                        self._queue_manager.prefix,
                        SlotKeySpec(
                            kind="attr",
                            node=str(pdata["submitter_node_id"]),
                            run_id=str(pdata["run_id"]),
                        ),
                    )
                cluster_kind = str(pdata.get("cluster_kind") or "attr")

            outcome = await requeue_or_fail_from_running(
                RequeueDeps(
                    running_queue=running_q,
                    wait_queue=wait_q,
                    policy=retry_policy_from_settings(
                        app_settings,
                        retry_max_duration_ms=app_settings.external_retry_max_duration_ms,
                    ),
                    mark_failed=_mark_failed,
                ),
                resolve_id,
                RequeueOpts(wait_job_name="attr_case"),
            )
            await self._release_and_promote_next_attribution(
                _AttributionSlotRelease(
                    run_id=run_id,
                    item_id=item_id,
                    attribution_job_id=resolve_id,
                    slot_key=slot_key,
                    cluster_kind=cluster_kind,
                ),
            )
            logger.info(
                "callback.attribution_failure_requeue run_id=%s item_id=%s "
                "outcome=%s slot_released=%s",
                run_id,
                item_id,
                outcome,
                bool(slot_key),
            )
            if outcome == "failed":
                await try_finalize_external(run_id, self._benchmark_service)
            return

        # Non-retryable / no queue / terminal run: peek the slot → clear running + increment received
        slot_key = None
        cluster_kind = "attr"
        if self._queue_manager is not None and resolve_id and not is_terminal:
            running_q = self._queue_manager.get_queue(ATTRIBUTION_RUNNING)
            peek = await get_queue_job(running_q, resolve_id)
            if peek is not None:
                pdata = dict(peek.data or {})
                slot_key = pdata.get("slot_key")
                if not slot_key and pdata.get("submitter_node_id") and pdata.get("run_id"):
                    slot_key = build_slot_key(
                        self._queue_manager.prefix,
                        SlotKeySpec(
                            kind="attr",
                            node=str(pdata["submitter_node_id"]),
                            run_id=str(pdata["run_id"]),
                        ),
                    )
                cluster_kind = str(pdata.get("cluster_kind") or "attr")
            try:
                await force_fail_from_running(
                    running_queue=running_q,
                    running_job_id=resolve_id,
                    mark_failed=_mark_failed,
                )
            except Exception:
                logger.exception(
                    "callback.attribution_force_fail_running_err run_id=%s item=%s",
                    run_id,
                    item_id,
                )
        # When running is already gone, force_fail may not write; fill it in idempotently
        await _mark_failed(
            {"run_id": run_id, "item_id": item_id},
            retry_count=0,
            reason="non_retryable",
        )

        await self._release_and_promote_next_attribution(
            _AttributionSlotRelease(
                run_id=run_id,
                item_id=item_id,
                attribution_job_id=resolve_id,
                slot_key=slot_key,
                cluster_kind=cluster_kind,
            ),
        )
        if not is_terminal:
            await try_finalize_external(run_id, self._benchmark_service)

    async def _handle_attribution_batch_callback(
        self,
        payload: AttributionBatchCallback,
    ) -> None:
        """Handle the whole-batch attribution callback (attribution-only mode).

        After the Aone attribution service finishes every item it sends one batch callback,
        and the ``results`` list carries each item's attribution result.
        """
        run_id = payload.run_id

        logger.info(
            "callback.attribution_batch_received_raw run_id=%s results_count=%d",
            run_id,
            len(payload.results) if payload.results else 0,
        )

        # Terminal guard: drop completed/failed; stopped still writes the batch attribution and skips finalize.
        run = await self._benchmark_service.get_run(run_id)
        if run is None:
            logger.warning(
                "callback.attribution_batch_terminal_run run_id=%s status=%s",
                run_id,
                "none",
            )
            return
        run_status = str(getattr(run.status, "value", run.status))
        if run_status in (
            BenchmarkStatus.COMPLETED.value,
            BenchmarkStatus.FAILED.value,
        ):
            logger.warning(
                "callback.attribution_batch_terminal_run run_id=%s status=%s",
                run_id,
                run_status,
            )
            return
        is_terminal = run_status == BenchmarkStatus.STOPPED.value
        if is_terminal:
            logger.warning(
                "callback.attribution_batch_stopped_late_write run_id=%s status=%s — "
                "writing data but skipping finalize",
                run_id,
                run_status,
            )

        config = run.config_json if isinstance(run.config_json, dict) else {}

        # 1. Write per-item attribution results
        if payload.results:
            # Batch-prefetch the item_id → source_trace_id map
            job_id_map = await self._benchmark_service.get_external_job_ids(run_id)
            if not job_id_map:
                logger.error(
                    "callback.attribution_batch_empty_job_id_map run_id=%s results=%d "
                    "— source_trace_id 将缺失（judge 提交未回写 source_trace_id）",
                    run_id,
                    len(payload.results),
                )

            for result in payload.results:
                item_id = result.get("item_id", "")
                attribution_data = result.get("attribution")
                if attribution_data is not None:
                    writer = BenchmarkTraceRecordWriter(
                        run_id=run_id,
                        criterion_id=FUSED_CRITERION_ID,
                        meta=RunMeta(
                            benchmark_name=config.get("benchmark_services", ""),
                        ),
                    )
                    batch_triage = (
                        attribution_data.get("primary_issue_layer")
                        or attribution_data.get("triage_level")
                        or "minor_bad"
                    )
                    batch_root_cause = (
                        attribution_data.get("root_cause_chain")
                        or attribution_data.get("root_cause")
                        or ""
                    )
                    logger.info(
                        "callback.attribution_batch_db_write run_id=%s "
                        "item_id=%s source_trace_id=%s "
                        "primary_issue_layer=%s root_cause_chain=%.200s",
                        run_id,
                        item_id,
                        job_id_map.get(item_id),
                        batch_triage,
                        batch_root_cause,
                    )
                    raw_attribution = dict(attribution_data)
                    await writer.upsert_triage(
                        TriageUpsert(
                            trace_id=job_id_map.get(item_id),
                            evaluator_ref="fused",
                            triage=TriageWrite(
                                triage_level=batch_triage,
                                root_cause=batch_root_cause,
                                attribution_detail_json=merge_evidence(
                                    {
                                        **parse_external_attribution_detail(attribution_data),
                                        "raw_json": raw_attribution,
                                    },
                                    attribution_data,
                                    result,
                                ),
                            ),
                            dataset_item_id=item_id,
                        ),
                    )
                    logger.info(
                        "callback.attribution_batch_db_written run_id=%s item_id=%s",
                        run_id,
                        item_id,
                    )
                    await self._enqueue_code_attribution(
                        _CodeAttributionEnqueue(
                            run_id=run_id,
                            item_id=str(item_id),
                            source_trace_id=job_id_map.get(item_id),
                            attribution=attribution_data,
                            fused_score=None,
                            advance_received=False,
                        ),
                    )
                    # A batch attribution callback does not take the per-item claimed path; code_attribution
                    # also does not dispatch when advance_received=False. Push to Aone per item here.
                    try:
                        attr_payload = AttributionResult.model_validate(attribution_data)
                    except Exception:
                        logger.warning(
                            "callback.attribution_batch_aone_skip_invalid run_id=%s item_id=%s",
                            run_id,
                            item_id,
                            exc_info=True,
                        )
                    else:
                        schedule_notification_dispatch(
                            DispatchRequest(
                                store=self._benchmark_service,
                                run_id=run_id,
                                item_id=str(item_id),
                                source_trace_id=job_id_map.get(item_id),
                                config=config,
                            ),
                            data=attr_payload,
                        )

        # 2. Atomically increment the count (the whole batch is 1 callback → expected=1)
        new_count, config = await self._benchmark_service.increment_config_counter(
            run_id,
            "attribution_received_count",
        )

        # Keep the heartbeat
        await self._benchmark_service.heartbeat(run_id)

        # 3. BullMQ queues drive the task lifetime; PG complete_task is not needed

        # 4. CAS completion detection (atomic check of the judge + attribution counters).
        # A late batch callback on stopped only persists; it does not trigger finalize / the report.
        if is_terminal:
            logger.info(
                "callback.attribution_batch_skip_finalize_terminal run_id=%s",
                run_id,
            )
            return
        if new_count >= config.get("attribution_expected_count", 0):
            await try_finalize_external(run_id, self._benchmark_service)

    # ═════════════════════════════════════════════════════════════════════════
    # attribution submit
    # ═════════════════════════════════════════════════════════════════════════

    async def _run_single_local_attribution(
        self,
        run: _LocalAttributionRun,
    ) -> None:
        """Run one local attribution synchronously (fired immediately per item).

        Reuses ``LocalAttributionRunner._attribute_one``:
        - look up the item → call triage_service → write the attribution result
        - increment ``attribution_received_count`` (guaranteed in finally)
        - ``finalize_callback=None``: finalization is handled uniformly by the judge callback
        """
        run_id = run.run_id
        item_id = run.item_id
        fused_score = run.fused_score
        provider = run.provider
        logger.info(
            "external.local_attribution_start run_id=%s item_id=%s fused_score=%s provider=%s",
            run_id,
            item_id,
            fused_score,
            provider,
        )

        if run.target:
            await run_custom_local_attribution(
                self._benchmark_service,
                cast(
                    "LocalAttributionImpl",
                    load_execution_instance(run.target, what="attribution_class"),
                ),
                LocalAttributionRequest(
                    run_id=run_id,
                    item_id=item_id,
                    fused_score=fused_score,
                    provider=provider,
                ),
            )
            return

        runner = LocalAttributionRunner(
            LocalAttributionRunnerDeps(
                triage_service=self._triage_service,
                benchmark_service=self._benchmark_service,
                semaphore=self._local_attr_semaphore,
            ),
        )
        await runner._attribute_one(
            LocalAttributionTask(
                run_id=run_id,
                item_id=item_id,
                fused_score=fused_score,
                provider=provider,
                finalize_callback=None,
                project_id=run.project_id,
            ),
        )

        logger.info(
            "external.local_attribution_done run_id=%s item_id=%s",
            run_id,
            item_id,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # _finalize_run (§11)
    # ═════════════════════════════════════════════════════════════════════════

    async def _finalize_run(self, run_id: str) -> None:
        """Cleanup after every callback has arrived: CAS election → aggregate → report → complete (§11).

        Two-phase CAS (§12.2):
        1. RUNNING → AGGREGATING conditional advance (idempotent, safe under concurrent callbacks)
        2. AGGREGATING → GENERATING_REPORT CAS (only one executor succeeds)

        Note: when ``try_finalize_external`` already moved status to ``aggregating`` before calling
        this method, Phase 1 affects 0 rows, and this node can still win Phase 2.
        """
        # stop-run: refuse the report while any item is still running (same rule as try_finalize_*)
        run_gate = await self._benchmark_service.get_run(run_id)
        gate_cfg = (
            run_gate.config_json
            if run_gate is not None and isinstance(run_gate.config_json, dict)
            else {}
        )
        if is_stop_requested(gate_cfg) and not await stop_items_ready_for_report(run_id):
            logger.info(
                "external.finalize_deferred_running_items run_id=%s",
                run_id,
            )
            # If it already entered aggregating by mistake, pull it back to running and CAS again after items are terminal
            await BenchmarkRun.filter(
                run_id=run_id,
                status=BenchmarkStatus.AGGREGATING.value,
            ).update(status=BenchmarkStatus.RUNNING.value)
            return

        # Phase 1: RUNNING → AGGREGATING (conditional advance, idempotent).
        # try_finalize_external may already have advanced status → 0 rows affected (normal)
        phase1_updated = await BenchmarkRun.filter(
            run_id=run_id,
            status=BenchmarkStatus.RUNNING.value,
        ).update(status=BenchmarkStatus.AGGREGATING.value)
        if phase1_updated:
            logger.info(
                "external.phase1_advanced run_id=%s running→aggregating",
                run_id,
            )
        else:
            logger.info(
                "external.phase1_skipped run_id=%s already_aggregating",
                run_id,
            )

        # Phase 2: AGGREGATING → GENERATING_REPORT (CAS election)
        won = await BenchmarkRun.filter(
            run_id=run_id,
            status=BenchmarkStatus.AGGREGATING.value,
        ).update(status=BenchmarkStatus.GENERATING_REPORT.value)
        if won == 0:
            logger.info(
                "external.phase2_cas_lost run_id=%s "
                "aggregating→generating_report election_defeated",
                run_id,
            )
            return

        logger.info(
            "external.phase2_cas_won run_id=%s aggregating→generating_report",
            run_id,
        )
        try:
            # Reload the run (to get the latest config_json)
            run = await self._benchmark_service.get_run(run_id)
            if run is None:
                logger.error("external.finalize_run_missing run_id=%s", run_id)
                return
            # fused external: there is no per-criterion worker; align completed_evaluators at finalize
            if run.total_evaluators and run.completed_evaluators < run.total_evaluators:
                await BenchmarkRun.filter(run_id=run_id).update(
                    completed_evaluators=run.total_evaluators,
                )
            config = run.config_json if isinstance(run.config_json, dict) else {}
            # Report-dimension snapshot fallback: when config lacks stat_dimensions, resolve it from the registry /
            # rubric_ref and write it back, so online finalize and regenerate share one source.
            config = await self._ensure_stat_dimensions_snapshot(run_id, config)
            rubric = load_rubric_from_config(config)
            aggregation = load_aggregation_from_config(config)

            # Phase 4-6: aggregate + report + complete (shared with regenerate_report)
            await self.publish_external_report(
                ExternalReportPublish(
                    run_id=run_id,
                    config=config,
                    rubric=rubric,
                    aggregation=aggregation,
                ),
            )
        except Exception as exc:
            logger.exception("external.finalize_failed run_id=%s", run_id)
            await self._benchmark_service.fail_run(
                run_id=run_id,
                reason=f"external finalize failed: {exc}",
            )

    async def _ensure_stat_dimensions_snapshot(
        self,
        run_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Ensure config contains a ``stat_dimensions`` snapshot (backfill and persist when missing).

        For an early online run / a missing snapshot key, building the report from the process registry alone can
        disagree with regenerate (a cold start must import the component). This always goes through
        ``resolve_run_stat_dimensions``, and when resolution succeeds and the snapshot is empty it writes back
        ``config_json``, so later finalize/regenerate share one source.
        """
        cfg = dict(config) if isinstance(config, dict) else {}
        if deserialize_stat_dimensions(cfg.get("stat_dimensions")):
            return cfg
        run = await self._benchmark_service.get_run(run_id)

        name = (
            logical_benchmark_run_name(
                column_name=getattr(run, "benchmark_name", None) if run is not None else None,
                config=cfg,
            )
            or None
        )
        dims = resolve_run_stat_dimensions(cfg, name)
        if not dims:
            return cfg
        snapshot = serialize_stat_dimensions(dims)
        cfg["stat_dimensions"] = snapshot
        try:
            await self._benchmark_service.merge_config_json(
                run_id,
                {"stat_dimensions": snapshot},
            )
            logger.info(
                "external.stat_dimensions_backfilled run_id=%s dims=%d",
                run_id,
                len(dims),
            )
        except Exception:
            logger.warning(
                "external.stat_dimensions_backfill_failed run_id=%s",
                run_id,
                exc_info=True,
            )
        return cfg

    async def publish_external_report(
        self,
        publish: ExternalReportPublish,
    ) -> str | None:
        """Aggregate metrics → generate the report → ``complete_run`` (shared by finalize / regenerate).

        Aligned with ``scripts/regenerate_report.py``: the same
        ``_aggregate_external_metrics`` + ``_generate_external_report``,
        so automatic online cleanup and an ops-regenerated report do not diverge.
        """
        run_id = publish.run_id
        config = publish.config
        rubric = publish.rubric
        aggregation = publish.aggregation
        final_score, metrics = await self._aggregate_external_metrics(
            run_id,
            rubric,
            aggregation,
        )
        logger.info(
            "external.aggregated run_id=%s score=%.4f method=%s",
            run_id,
            final_score,
            getattr(metrics, "aggregation_method", "unknown"),
        )
        await self._benchmark_service.update_run_status(
            run_id,
            BenchmarkStatus.GENERATING_REPORT,
            final_score=final_score,
            metrics_json=metrics.model_dump() if hasattr(metrics, "model_dump") else {},
        )
        logger.info(
            "external.status_updated run_id=%s →generating_report",
            run_id,
        )

        report_url: str | None = None
        # webdev may be None: a report can still be emitted when the business has registered external_http.
        # A search_eval console task must emit a report (the page "view report" depends on report_site_url),
        # so generate one even if upstream wrongly passed generate_report=false.
        # The config-side field name is benchmark_services (written at accept time).
        benchmark_name = str(config.get("benchmark_name") or config.get("benchmark_services") or "")
        must_report = benchmark_name.startswith("search_eval")
        if config.get("generate_report", True) or must_report:
            logger.info(
                "external.report_start run_id=%s must_report=%s",
                run_id,
                must_report,
            )
            report_url = await self._generate_external_report(
                _ExternalReportGenerate(
                    run_id=run_id,
                    config=config,
                    metrics=metrics,
                    final_score=final_score,
                ),
            )
            logger.info(
                "external.report_done run_id=%s report_url=%s",
                run_id,
                report_url,
            )

        logger.info(
            "external.completing run_id=%s score=%.4f report_url=%s",
            run_id,
            final_score,
            report_url,
        )
        await self._benchmark_service.complete_run(
            CompleteRunWrite(
                run_id=run_id,
                final_score=final_score,
                metrics_json=metrics.model_dump() if hasattr(metrics, "model_dump") else {},
                report_meta_json=_report_meta(report_url, config),
            ),
        )
        logger.info(
            "external.run_complete run_id=%s score=%.4f",
            run_id,
            final_score,
        )
        return report_url

    async def _aggregate_external_metrics(
        self,
        run_id: str,
        rubric: list[Criterion],
        aggregation: AggregationConfig,
    ) -> tuple[float, Any]:
        """Aggregate benchmark eval metrics (§8).

        When ``fusion_fn_ref`` applies a gate at item level, the per-criterion
        mean cannot reflect the gate — override ``final_score`` with the average of the fused rows.
        Per-criterion stats are still used for the report's criterion_metrics panel.
        """
        if self._aggregation_calculator is None:
            logger.error("external.aggregation_calculator_not_configured run_id=%s", run_id)
            return 0.0, {}

        sql_entries = await self._benchmark_service.load_criterion_entries_from_pipeline(
            run_id=run_id,
            rubric=rubric,
            pass_threshold=aggregation.pass_threshold,
        )
        final_score, _per_criterion, metrics = await self._aggregation_calculator.aggregate(
            aggregation,
            rubric,
            sql_entries,
        )

        # Dimension-stat expansion (plugin registration extension point): expand fused-row sub_scores via
        # resolve_run_stat_dimensions (snapshot → registry → rubric_ref).

        run_for_dims = await self._benchmark_service.get_run(run_id)
        run_config = (
            getattr(run_for_dims, "config_json", None) if run_for_dims is not None else None
        )
        dims_cfg = run_config if isinstance(run_config, dict) else None
        logical_name = (
            logical_benchmark_run_name(
                column_name=getattr(run_for_dims, "benchmark_name", None),
                config=dims_cfg,
            )
            or None
        )
        await self._benchmark_service.expand_dimension_metrics(
            ExpandDimensionMetricsSpec(
                run_id=run_id,
                benchmark_name=logical_name,
                metrics=metrics,
                config=dims_cfg,
            ),
        )

        # Override final_score with average of per-item fused scores.
        # When fusion_fn_ref applies gate rules at item level, the
        # per-criterion weighted average doesn't reflect gates.
        fused_avg = await self._benchmark_service.load_fused_score_average(run_id)
        if fused_avg is not None:
            logger.info(
                "external.fused_avg_override run_id=%s criterion_weighted=%.4f fused_avg=%.4f",
                run_id,
                final_score,
                fused_avg,
            )
            final_score = fused_avg
            metrics.aggregation_method = "fused_avg"

        # Aggregate attribution stats (aligned with the inline runner's _attach_attribution_to_metrics).
        # Externally triggered attribution is written to benchmark_run_items.attribution_detail_json,
        # and here it is aggregated from the DB into metrics.attribution_metrics for the report.
        # Pluginized (§8 extension point): a business-registered attribution-stats provider (ieval = field distribution;
        # search_eval = distribution of judge's six dimensions that are not a full score) wins; unregistered falls back to today's default.
        attr_provider = resolve_attribution_stats_provider(logical_name)
        if attr_provider is not None:
            attr_by_cid = await attr_provider.collect(run_id)
        else:
            attr_by_cid = await BenchmarkTraceRecordWriter.aggregate_attribution(run_id)
        if attr_by_cid:
            for cid, attr in attr_by_cid.items():
                metrics.attribution_metrics[cid] = AttributionStats(
                    field_distributions=attr.get("field_distributions", {}),
                    value_labels=attr.get("value_labels", {}),
                    total_attributed=attr.get("total_attributed", 0),
                )
            logger.info(
                "external.attribution_aggregated run_id=%s keys=%s",
                run_id,
                list(attr_by_cid.keys()),
            )

        return final_score, metrics

    async def _generate_external_report(
        self,
        report: _ExternalReportGenerate,
    ) -> str | None:
        """Generate the benchmark report (§9) — shares the ``report_spec`` core with BenchmarkRunner."""
        return await generate_benchmark_report(
            ReportJobInput(
                benchmark_service=self._benchmark_service,
                run_id=report.run_id,
                metrics=report.metrics,
                final_score=report.final_score,
                config=report.config,
                rubric=load_rubric_from_config(report.config),
                log_prefix="external",
            ),
            webdev=self._webdev_client,
            pages_publisher=self._pages_publisher,
        )


__all__ = ["ExternalBenchmarkRunner", "ExternalReportPublish", "ExternalRunnerDeps"]
