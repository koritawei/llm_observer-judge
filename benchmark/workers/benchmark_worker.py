"""benchmark_worker — unified case worker (design doc §8.3 queue merge).

Each node's lifespan starts a resident ``bullmq.Worker`` that competes for the ``benchmark-cases`` queue.
The processor reads the full case from ``job.data`` and routes execution by the run ``config_json`` snapshot's
``judge.execution`` (§8.2):

- ``sync`` (default): local per-criterion eval (fused / per-criterion),
  results written to ``benchmark_run_items`` (three-phase write D12);
- ``async``: one-step submit to the Aone ``judge.target`` URL + hold the slot for the HTTP callback
  (implementation moved from ``external_worker``, :func:`build_async_judge_handlers`).

A sync judge has no HTTP callback; attribution is routed inside the worker by ``attribution.execution``
(sync → local attribution / async → the attribution-wait queue /
none → push straight to Aone by the judge fused score).
"""

import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Job

from benchmark.settings.setting import Settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRunItem
from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY, PipelineSource
from benchmark.evaluators.base import (
    EvaluatorRequest,
    EvaluatorResult,
    PromptBundleRef,
    SkillBundleRef,
)
from benchmark.evaluators.customization_resolution import (
    bundle_refs_from_config,
    prompt_provenance,
)
from benchmark.libs.jobqueue.worker import Worker
from benchmark.attribution.badcase.attribution_trigger_policy import (
    resolve_trigger_policy,
    should_trigger_attribution,
)
from benchmark.runs.common.config_helpers import (
    load_aggregation_from_config,
    load_rubric_from_config,
)
from benchmark.runs.common.criterion_eval import (
    can_fanout_sub_scores,
    fanout_criterion_scores,
    group_criteria_by_evaluator_ref,
)
from benchmark.runs.common.dynamic_rubric import (
    apply_dynamic_fused_override,
    resolve_gate_attribution_threshold,
    should_trigger_with_optional_threshold,
    stamp_fused_grade,
)
from benchmark.runs.common.execution_classes import (
    AsyncJudgeImpl,
    LocalAttributionImpl,
    LocalAttributionRequest,
    load_execution_instance,
    snapshot_class_ref,
)
from benchmark.runs.common.profile_resolver import load_class_ref
from benchmark.runs.common.raw_result_store import is_raw_result_store
from benchmark.runs.common.trace_url_builder import build_trace_url_for_item
DispatchRequest = None
JudgeDispatchInput = None
schedule_notification_dispatch_for_judge = None
LocalAttributionRunner = None
LocalAttributionRunnerDeps = None
LocalAttributionTask = None
from benchmark.runs.run_service import BenchmarkService
from benchmark.runs.run_stop import is_stop_requested
from benchmark.runs.schema import FUSED_CRITERION_ID
SEARCH_EVAL_BENCHMARKS = None
resolve_search_eval_source_trace_id = None
from benchmark.reporting.framework.fusion import EvalFailure, ScoreFuser
from benchmark.reporting.framework.schema import AggregationConfig, Criterion
from benchmark.queue.eval_case import EvalCase
from benchmark.queue.queue_factory import (
    BENCHMARK_CASES,
    BENCHMARK_RUNNING,
    EvalQueueRegistry,
)
from benchmark.queue.worker_skeleton import (
    JudgeWriteCall,
    SlotConfig,
    SyncEvalHandlers,
    SyncEvalRuntime,
    process_sync_eval_case,
)
from benchmark.workers.external_runner import (
    AttributionCaseSpec,
    enqueue_attribution_case,
    run_custom_local_attribution,
)
from benchmark.workers.external_worker import (
    AsyncJudgeHandlersBuild,
    AsyncJudgeOverrides,
    build_async_judge_handlers,
)
from benchmark.workers.finalize_trigger import EvalType, tick_and_maybe_finalize
from benchmark.workers.fused_row import (
    FusedRunningPrewrite,
    prewrite_running_fused_item,
)
from benchmark.services.evaluation import EvalRunOptions, EvaluatorService
from benchmark.services.trace_writer import BenchmarkTraceRecordWriter, RunMeta
from benchmark.attribution.service.service import TriageAttributionService
from benchmark.utils.exceptions import EvaluatorError

if TYPE_CHECKING:
    from benchmark.runs.common.external_client_factory import ExternalClientFactory


@dataclass(frozen=True)
class LocalJudgeRequest:
    """Arguments for ``LocalJudgeImpl.judge_item`` (FUNC-PARAMS-001).

    ``evaluator_service`` is an optional injection: a class pointed at by judge.target takes it itself when it
    must delegate to a registry evaluator (for example SearchEvalLocalJudge → ``judge_search_ai_card``);
    a class that already contains the full eval logic may ignore the field.
    """

    run_id: str
    item_id: str
    case: EvalCase
    criteria: list[dict[str, Any]]
    evaluator_service: "EvaluatorService | None" = None
    #: Business-custom judge bundle ref (run snapshot agent_registration.judge; absent = platform default)
    prompt_bundle_ref: "PromptBundleRef | None" = None
    skill_bundle_ref: "SkillBundleRef | None" = None


@dataclass(frozen=True)
class _FusedSuccessRowWrite:
    """Arguments for ``_update_fused_success_row`` (FUNC-PARAMS-001)."""

    run_id: str
    item_id: str
    case: EvalCase
    fused_score: float
    judge_json: dict[str, Any]
    trace_url: str | None = None
    #: Already resolved by ``_resolved_case_source_trace_id``; a dirty search_eval item_id is None
    source_trace_id: str | None = None


@dataclass(frozen=True)
class _JudgeFieldsWrite:
    """Write payload for ``_update_judge_fields`` (orthogonal to the ItemRowKey row key)."""

    score: float
    reason: str | None
    judge_json: dict[str, Any]
    sub_scores: dict[str, Any] | None = None


@dataclass(frozen=True)
class _FailedRowWrite:
    """Arguments for ``_update_to_failed`` (FUNC-PARAMS-001)."""

    run_id: str
    item_id: str
    criteria: list[dict[str, Any]]
    error: Exception
    is_fused: bool = False


@dataclass(frozen=True)
class _AsyncJudgeCaseCtx:
    """Arguments for ``_process_async_judge_case`` (FUNC-PARAMS-001)."""

    queue_manager: "EvalQueueRegistry"
    deps: "BenchmarkWorkerDeps"
    judge_cfg: dict[str, Any]


@dataclass(frozen=True)
class _SyncHandlersBuild:
    """Arguments for ``_build_sync_handlers`` (FUNC-PARAMS-001)."""

    run_id: str
    queue_manager: "EvalQueueRegistry"
    deps: "BenchmarkWorkerDeps"
    judge_cfg: dict[str, Any]
    attr_cfg: dict[str, Any]


class LocalJudgeImpl(Protocol):
    """Protocol for a judge.target local eval class (§6.3 sync shape, defined by the consumer)."""

    async def judge_item(
        self,
        req: "LocalJudgeRequest",
    ) -> dict[str, Any]:
        """Return ``{"criterion_scores": {cid: {"score", "reason", "sub_scores"}}}``."""
        ...


logger = get_logger("agent_eval.eval_workers.benchmark_worker")

#: Node identity (slot-key isolation)
WORKER_NAME = socket.gethostname()

_TERMINAL_STATUSES = frozenset(
    {
        BenchmarkStatus.COMPLETED.value,
        BenchmarkStatus.FAILED.value,
        BenchmarkStatus.STOPPED.value,
    }
)


async def _is_run_terminal(run_id: str, run_service: BenchmarkService) -> bool:
    """Whether the run is already terminal, or already has the stop latch (§6.8)."""
    run = await run_service.get_run(run_id)
    if run is None:
        return True
    if run.status in _TERMINAL_STATUSES:
        return True
    return is_stop_requested(getattr(run, "config_json", None))


#: Per-run (rubric, aggregation) in-process cache. config_json does not change during eval
#: (the rubric is not edited mid-run), so re-reading the run row per case is N+1 (significant DB
#: load on a large run); a FIFO cap stops a resident worker from growing without bound. Run count is far
#: below item count, so simple eviction does not hurt the hit rate.
_RUN_RUBRIC_CACHE_MAX = 1024
_run_rubric_cache: dict[str, tuple[list[Criterion], AggregationConfig]] = {}

#: Per-run execution-config (judge / attribution snapshot keys) in-process cache (§8.2,
#: same lifetime as ``_run_rubric_cache``: config_json does not change during eval).
_run_exec_cfg_cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

#: Per-run cache of local eval / attribution class instances (when judge.target / attribution.target
#: is ``module:Class``, loaded via the allowlist; no shared state per run → cache the instance by
#: run_id; class loading happens after Resolver validation).
_run_local_impl_cache: dict[str, tuple[LocalJudgeImpl | None, LocalAttributionImpl | None]] = {}


async def _get_run_exec_cfg(
    run_id: str,
    run_service: BenchmarkService,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the run's (judge config, attribution config) snapshot (in-process cache removes the N+1).

    Default (legacy run with no snapshot keys) = ``({}, {})`` → today's judge-sync behavior /
    attribution is not routed (today's push has no per-item attribution).
    """
    cached = _run_exec_cfg_cache.get(run_id)
    if cached is not None:
        return cached
    run = await run_service.get_run(run_id)
    if run is None:
        # Leftover job / deleted run: fail explicitly (aligned with the existing semantics of _get_run_rubric).
        # The old fallback to ({}, {}) disagreed with the raise on the rubric path in this file — after the
        # fallback the next rubric load still failed, so it never really "skipped gracefully"; a uniform fail-fast
        # lets a leftover job converge on the skeleton exception path at the earliest contact (visible, observable).
        # Run-level graceful skip is the _is_run_terminal guard (run None → True).
        raise RuntimeError(f"benchmark run not found: {run_id}")
    config = run.config_json if isinstance(run.config_json, dict) else {}
    judge_raw = config.get("judge")
    attr_raw = config.get("attribution")
    entry = (
        judge_raw if isinstance(judge_raw, dict) else {},
        attr_raw if isinstance(attr_raw, dict) else {},
    )
    if len(_run_exec_cfg_cache) >= _RUN_RUBRIC_CACHE_MAX:
        _run_exec_cfg_cache.pop(next(iter(_run_exec_cfg_cache)))
    _run_exec_cfg_cache[run_id] = entry
    return entry


#: Per-run registration-snapshot refs in-process cache (same lifetime semantics as _run_exec_cfg_cache).
#: The key is run_id and the value includes the snapshot's internal name — concurrent instances of the same
#: code key by run, and never cache by component name (design §12.4 cache-key discipline).
_run_registration_refs_cache: dict[
    str, tuple[PromptBundleRef | None, SkillBundleRef | None, str | None]
] = {}


async def _get_run_judge_refs(
    run_id: str,
    run_service: BenchmarkService,
    metadata: dict[str, Any] | None = None,
) -> tuple[PromptBundleRef | None, SkillBundleRef | None]:
    """Load judge refs from the run snapshot (no snapshot = (None, None), behavior unchanged).

    Judge stage only: attribution refs are read independently from the same snapshot by LocalAttributionRunner
    (attribution requests and judge requests are separate and do not cross stages). When ``metadata`` is passed,
    the snapshot's internal name is stamped onto the request (multi-instance ownership key; not used to look up
    an evaluator).
    """
    cached = _run_registration_refs_cache.get(run_id)
    if cached is not None:
        _stamp_internal_agent_name(metadata, cached[2])
        return cached[0], cached[1]
    run = await run_service.get_run(run_id)
    config = run.config_json if run is not None and isinstance(run.config_json, dict) else {}
    prompt_ref, skill_ref = bundle_refs_from_config(config, "judge")
    snapshot = config.get("agent_registration")
    internal_name = snapshot.get("internal_agent_name") if isinstance(snapshot, dict) else None
    entry = (
        prompt_ref,
        skill_ref,
        internal_name if isinstance(internal_name, str) and internal_name else None,
    )
    if len(_run_registration_refs_cache) >= _RUN_RUBRIC_CACHE_MAX:
        _run_registration_refs_cache.pop(next(iter(_run_registration_refs_cache)))
    _run_registration_refs_cache[run_id] = entry
    _stamp_internal_agent_name(metadata, entry[2])
    return entry[0], entry[1]


def _stamp_internal_agent_name(metadata: dict[str, Any] | None, internal_name: str | None) -> None:
    """Write the snapshot's internal name into request metadata (multi-instance ownership key; skip when there is no snapshot)."""
    if metadata is not None and internal_name:
        metadata["internal_agent_name"] = internal_name


@dataclass(frozen=True)
class JudgeRequestContext:
    """Eval context needed to build one judge request (the case itself + ownership key + routing)."""

    case: EvalCase
    item_id: str
    service_name: str
    project_id: str | None


async def _prepare_judge_request(
    run_id: str,
    run_service: BenchmarkService,
    context: JudgeRequestContext,
) -> EvaluatorRequest:
    """Build one judge eval request (metadata stamp + custom refs; behavior unchanged when there is no customization)."""
    case = context.case
    metadata = dict(case.metadata)
    metadata[PIPELINE_SOURCE_METADATA_KEY] = PipelineSource.BENCHMARK.value
    metadata["dataset_item_id"] = context.item_id
    prompt_ref, skill_ref = await _get_run_judge_refs(run_id, run_service, metadata)
    return EvaluatorRequest(
        input=case.input,
        output=case.output,
        expected_output=case.expected_output,
        metadata=metadata,
        trace_id=case.source_trace_id,
        agent_name=context.service_name,
        project_id=_resolve_eval_project_id(context.project_id, metadata),
        prompt_bundle_ref=prompt_ref,
        skill_bundle_ref=skill_ref,
    )


def _load_local_class(ref: object, *, what: str) -> type | None:
    """Load a ``module:Class`` local implementation (allowlist mechanism same as Resolver §6.1).

    A non-class ref (URL / empty) returns ``None``; a load failure raises and converges on the skeleton
    exception path (requeue / terminal fail).
    """
    if not isinstance(ref, str) or not ref or "://" in ref:
        return None
    return load_class_ref(ref, what=what)


async def _get_run_local_impls(
    run_id: str,
    judge_cfg: dict[str, Any],
    attr_cfg: dict[str, Any],
) -> tuple[LocalJudgeImpl | None, LocalAttributionImpl | None]:
    """Per-run local eval class / local attribution class instance (None = today's default implementation)."""
    cached = _run_local_impl_cache.get(run_id)
    if cached is not None:
        return cached
    judge_cls = _load_local_class(judge_cfg.get("target"), what="judge_target")
    attr_cls = _load_local_class(attr_cfg.get("target"), what="attribution_target")
    judge_impl = cast("LocalJudgeImpl | None", judge_cls() if judge_cls else None)
    attr_impl = cast("LocalAttributionImpl | None", attr_cls() if attr_cls else None)
    entry = (judge_impl, attr_impl)
    if len(_run_local_impl_cache) >= _RUN_RUBRIC_CACHE_MAX:
        _run_local_impl_cache.pop(next(iter(_run_local_impl_cache)))
    _run_local_impl_cache[run_id] = entry
    return entry


def _finalize_eval_type(
    judge_cfg: dict[str, Any],
    attr_cfg: dict[str, Any],
) -> "EvalType":
    """Finalize counter-model routing (§8.2):

    - either judge or attribution is ``async`` → ``external`` (config_json counter model:
      the callback / worker advances judge_received_count on its behalf);
    - all sync / none → ``benchmark`` (progress-column CAS model, today's sync semantics).
    """
    if judge_cfg.get("execution") == "async" or attr_cfg.get("execution") == "async":
        return "external"
    return "benchmark"


async def _get_run_rubric(
    run_id: str,
    run_service: BenchmarkService,
) -> tuple[list[Criterion], AggregationConfig]:
    """Load the run's (rubric, aggregation); the in-process cache removes the per-case N+1 read."""
    cached = _run_rubric_cache.get(run_id)
    if cached is not None:
        return cached
    run = await run_service.get_run(run_id)
    if run is None:
        raise RuntimeError(f"benchmark run not found: {run_id}")
    config = run.config_json if isinstance(run.config_json, dict) else {}
    rubric = load_rubric_from_config(config)
    aggregation = load_aggregation_from_config(config)
    entry = (rubric, aggregation)
    if len(_run_rubric_cache) >= _RUN_RUBRIC_CACHE_MAX:
        _run_rubric_cache.pop(next(iter(_run_rubric_cache)))
    _run_rubric_cache[run_id] = entry
    return entry


@dataclass(frozen=True)
class ItemRowKey:
    """Row-location key for benchmark_run_items (run/item/criterion + evaluator_ref).

    Orthogonal to the write payload (input/output/score and similar): this object only says which row to write.
    The D12 three-phase writes (``_insert_running_row`` / ``_update_judge_fields``)
    take it as the row-key argument.
    """

    run_id: str
    item_id: str
    criterion_id: str
    evaluator_ref: str


def _is_placeholder_project_id(project_id: str) -> bool:
    """Historical external run creation filled project_id with ``{date}_benchmark_final_results``.

    That string is not a Langfuse instance id; treat it as unset so it does not cover the real csv/metadata value.
    """
    return project_id.strip().endswith("_benchmark_final_results")


def _resolve_eval_project_id(
    project_id: str | None,
    metadata: dict[str, Any],
) -> str | None:
    """Normalize project_id for judge input: run-level first, fall back to item-level (taken straight from csv).

    **Batch benchmark-cases worker only**; the example realtime stream carries observation.project_id directly
    via ``to_evaluator_request`` in the Kafka consumer and does not pass through this function.

    Langfuse project identity for OSS businesses such as search_eval / example passthrough lives on the
    data row (parse or passthrough writes ``metadata.project_id``, or it is still only in
    ``metadata.csv_row.project``); when run-level ``payload.project_id`` was not passed, fall back to
    item-level so ``validate_evaluator_request`` identity checks pass. Legacy businesses whose
    metadata has no ``project_id`` → read ``csv_row`` next; if still missing, return the run-level value unchanged.
    """
    if project_id and project_id.strip() and not _is_placeholder_project_id(project_id):
        return project_id
    item_project = metadata.get("project_id")
    if isinstance(item_project, str) and item_project.strip():
        return item_project
    csv_row = metadata.get("csv_row")
    if isinstance(csv_row, dict):
        for key in ("project", "project_id"):
            value = csv_row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return project_id if project_id and project_id.strip() else None


def _attach_judge_extra(
    cs_payload: dict[str, Any],
    extra: Any,  # noqa: ANN401 — EvaluatorResult.extra is a heterogeneous dict
) -> None:
    """Projectable fields of judge extra are persisted with criterion_scores (cs_payload updated in place).

    - ``dimension`` / ``dimensions``: attribution labels such as search_ai_card;
    - ``prompt_customization``: custom-prompt provenance such as example;
    - ``weights`` / ``attribution_threshold`` / ``grade``: dynamic Rubric and
      framework grading (project when present, skip when absent; do not override a custom static rubric / threshold policy).

    An evaluator whose extra is not a dict / has no matching key writes nothing (legacy behavior unchanged).
    """
    if not isinstance(extra, dict):
        return
    if extra.get("dimension"):
        cs_payload["dimension"] = str(extra["dimension"])
    dimensions = extra.get("dimensions")
    if isinstance(dimensions, dict) and dimensions:
        cs_payload["dimensions"] = dimensions
    # Custom judges such as example: persist prompt-version provenance with criterion_scores / judge_json,
    # so the template actually used can be looked up by content_version / commit.
    prompt_customization = extra.get("prompt_customization")
    if isinstance(prompt_customization, dict) and prompt_customization:
        cs_payload["prompt_customization"] = prompt_customization
    weights = extra.get("weights")
    if isinstance(weights, dict) and weights:
        cs_payload["weights"] = weights
    attr_threshold = extra.get("attribution_threshold")
    if isinstance(attr_threshold, (int, float)) and not isinstance(attr_threshold, bool):
        cs_payload["attribution_threshold"] = float(attr_threshold)
    for key in (
        "dynamic_rubric_opt_in",
        "dynamic_rubric_opt_in_source",
        "declared_weights",
        "sub_score_scale",
        "computed_fused_score",
        "model_score",
        "score_mismatch",
        "dynamic_rubric_fallback_reason",
        "raw_judge_json",
    ):
        value = extra.get(key)
        if value is not None:
            cs_payload[key] = value
    grade = extra.get("grade")
    if isinstance(grade, str) and grade.strip():
        cs_payload["grade"] = grade.strip()


def _prompt_customization_on_failure(
    eval_exc: BaseException,
    eval_request: EvaluatorRequest,
) -> dict[str, Any] | None:
    """On judge failure, still try to obtain prompt provenance (content_version / git_commit / git_path).

    Prefer ``EvaluatorError.details.prompt_customization`` (example and similar already resolved
    the version); otherwise derive it from the bundle ref on the request (injected at accept time).
    """
    if isinstance(eval_exc, EvaluatorError) and isinstance(eval_exc.details, dict):
        prompt_meta = eval_exc.details.get("prompt_customization")
        if isinstance(prompt_meta, dict) and prompt_meta:
            return prompt_meta
    return prompt_provenance(eval_request.prompt_bundle_ref, eval_request.skill_bundle_ref)


async def _insert_running_row(
    key: ItemRowKey,
    *,
    input_val: object,
    output_val: object,
    source_trace_id: str | None = None,
) -> None:
    """D12 phase 1: INSERT the base row (status=running, input, output).

    Use ``get_or_create`` to avoid a TOCTOU race (two nodes creating the same row).
    Concurrency safety depends on the DB unique constraint ``BenchmarkRunItem.Meta.unique_together
    ("run_id", "item_id", "criterion_id")`` — on a conflict get_or_create falls back to the get branch;
    if the constraint is missing, concurrency produces duplicate rows.
    """
    await BenchmarkRunItem.get_or_create(
        run_id=key.run_id,
        item_id=key.item_id,
        criterion_id=key.criterion_id,
        defaults={
            "evaluator_ref": key.evaluator_ref,
            "status": "running",
            "input": input_val,
            "output": output_val,
            "source_trace_id": source_trace_id,
        },
    )


@dataclass(frozen=True)
class FusedCaseContext:
    """Single-case processing context for fused mode (parameter object for ``_process_fused_case``).

    ``run_id`` / ``item_id`` / ``case`` locate the case to eval; ``criteria`` /
    ``service_name`` / ``project_id`` carry the rubric and dataset identity from job data;
    ``evaluator_service`` / ``run_service`` are service dependencies.
    ``local_judge`` is the local eval-class instance named by profile ``judge.target``
    (``None`` = today's default EvaluatorService per-criterion eval, §8.2).
    ``count_judge_received``: when judge sync + attribution async uses the external
    counter model, the worker increments ``judge_received_count`` on its behalf (no HTTP callback).
    """

    run_id: str
    item_id: str
    case: EvalCase
    criteria: list[dict[str, Any]]
    service_name: str
    project_id: str | None
    evaluator_service: EvaluatorService
    run_service: BenchmarkService
    local_judge: "LocalJudgeImpl | None" = None
    count_judge_received: bool = False
    #: Benchmark component name (job data ``benchmark_services``) — the trace_url
    #: factory resolves the item-shape identity extractor from it; default ``""`` → the default extractor
    #: (no link when metadata has no ``project_id``; today's behavior unchanged).
    benchmark_name: str = ""


def _resolved_case_source_trace_id(case: EvalCase, *, benchmark_name: str) -> str | None:
    """Resolve the ``source_trace_id`` that should be persisted, by benchmark.

    search_eval family: prefer csv_row / ctx.trace_id, and reject a dirty value equal to item_id;
    other businesses: keep case.source_trace_id.
    """
    if benchmark_name not in SEARCH_EVAL_BENCHMARKS:
        return case.source_trace_id
    return resolve_search_eval_source_trace_id(
        {
            "item_id": case.item_id,
            "source_trace_id": case.source_trace_id,
            "input": case.input,
            "metadata": case.metadata,
        },
    )


@dataclass(frozen=True)
class FusedJudgeOutcome:
    """Fused-mode judge output (input to attribution routing after a sync judge, §8.2)."""

    fused_score: float
    criterion_scores: dict[str, dict[str, Any]]


async def _judge_via_local_class(
    local_judge: LocalJudgeImpl,
    req: LocalJudgeRequest,
) -> dict[str, dict[str, Any]]:
    """Call the ``judge.target`` local eval class (§6.3 sync shape, allowlist already checked).

    Protocol: ``async judge_item(LocalJudgeRequest)``
    returns ``{"criterion_scores": {cid: {"score":..,"reason":..,"sub_scores":..}}}``.
    A missing criterion is recorded as a 0-score failed row, matching today's semantics.
    """
    criteria = req.criteria
    result = await local_judge.judge_item(req)
    raw_scores = result.get("criterion_scores") if isinstance(result, dict) else None
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    criterion_scores: dict[str, dict[str, Any]] = {}
    for criterion in criteria:
        cid = criterion["id"]
        cs = raw_scores.get(cid)
        if isinstance(cs, dict) and "score" in cs:
            criterion_scores[cid] = {
                "score": float(cs["score"]),
                "reason": str(cs.get("reason") or ""),
                "sub_scores": cs.get("sub_scores"),
            }
        else:
            criterion_scores[cid] = {
                "score": 0.0,
                "reason": f"missing criterion {cid}",
            }
    return criterion_scores


def _build_judge_raw_json(
    scores: dict[str, "EvaluatorResult | EvalFailure"],
) -> dict[str, Any]:
    """Assemble judge raw_json (a small function, to keep _process_fused_case's complexity down).

    A full dump of each criterion's ``EvaluatorResult`` (including all of extra, with no processing);
    an eval failure records an error summary.
    """
    raw_json: dict[str, Any] = {}
    for cid, outcome in scores.items():
        if isinstance(outcome, EvalFailure):
            raw_json[cid] = {"error": str(outcome.error)[:500]}
        else:
            raw_json[cid] = outcome.model_dump(mode="json")
    return raw_json


async def _process_fused_case(  # noqa: PLR0912, PLR0915 — fused judge main-flow orchestration; branches are the stage chain (prewrite/eval/aggregate/persist)
    ctx: FusedCaseContext,
) -> FusedJudgeOutcome | None:
    """Fused mode: eval each criterion → ScoreFuser aggregate → write 1 fused row.

    Unlike per-criterion mode (one row per criterion_id=c.id), fused mode
    packs every criterion score into judge_json.criterion_scores and writes a single row
    with criterion_id=FUSED_CRITERION_ID, aligned with the external-mode data model.
    Arguments are in :class:`FusedCaseContext`.

    Returns:
        ``FusedJudgeOutcome`` when this call actually wrote a success row;
        ``None`` (resume already success / concurrent write failed — do not route attribution again).
    """
    run_id, item_id, case = ctx.run_id, ctx.item_id, ctx.case
    criteria = ctx.criteria
    service_name, project_id = ctx.service_name, ctx.project_id
    evaluator_service, run_service = ctx.evaluator_service, ctx.run_service
    # Rebuild rubric + aggregation from config_json (in-process cache; see
    # _get_run_rubric: config does not change during eval).
    rubric, aggregation = await _get_run_rubric(run_id, run_service)
    source_trace_id = _resolved_case_source_trace_id(
        case,
        benchmark_name=ctx.benchmark_name,
    )

    # Prewrite the running row (including input/output); skip eval if it is already terminal.
    still_open = await prewrite_running_fused_item(
        FusedRunningPrewrite(
            run_id=run_id,
            item_id=item_id,
            source_trace_id=source_trace_id,
            item_input=case.input,
            item_output=case.output,
            create_score=0.0,
        ),
    )
    if not still_open:
        logger.info(
            "benchmark.fused_already_terminal run_id=%s item=%s",
            run_id,
            item_id,
        )
        return None

    metadata = dict(case.metadata)
    metadata[PIPELINE_SOURCE_METADATA_KEY] = PipelineSource.BENCHMARK.value
    metadata["dataset_item_id"] = item_id
    if ctx.benchmark_name:
        metadata["benchmark_name"] = ctx.benchmark_name
    if source_trace_id:
        metadata["source_trace_id"] = source_trace_id

    # Business customization: judge refs come from the run registration snapshot (None when there is no snapshot; behavior unchanged)
    judge_prompt_ref, judge_skill_ref = await _get_run_judge_refs(run_id, run_service, metadata)

    eval_request = EvaluatorRequest(
        input=case.input,
        output=case.output,
        expected_output=case.expected_output,
        metadata=metadata,
        trace_id=source_trace_id,
        agent_name=service_name,
        project_id=_resolve_eval_project_id(project_id, metadata),
        # search_eval family: the CSV already carries IO, so skip the Langfuse primary-key lookup (a second guard,
        # same effect as EvalRunOptions.resolve_trace=False; covers a forgotten options argument).
        skip_trace_resolve=ctx.benchmark_name in SEARCH_EVAL_BENCHMARKS,
        prompt_bundle_ref=judge_prompt_ref,
        skill_bundle_ref=judge_skill_ref,
    )

    # Eval criterion by criterion and collect results
    # (when judge.target names a local eval class, use that custom implementation, §8.2 sync judge)
    scores: dict[str, EvaluatorResult | EvalFailure] = {}
    criterion_scores: dict[str, dict[str, Any]] = {}

    if ctx.local_judge is not None:
        criterion_scores = await _judge_via_local_class(
            ctx.local_judge,
            LocalJudgeRequest(
                run_id=run_id,
                item_id=item_id,
                case=case,
                criteria=criteria,
                evaluator_service=evaluator_service,
                prompt_bundle_ref=judge_prompt_ref,
                skill_bundle_ref=judge_skill_ref,
            ),
        )
        for cid, cs in criterion_scores.items():
            if cs.get("reason", "").startswith("missing criterion"):
                scores[cid] = EvalFailure(RuntimeError(cs["reason"]))
            else:
                # Keep weights / attribution_threshold / grade for dynamic fused and grading.
                extra: dict[str, Any] = {}
                weights = cs.get("weights")
                if isinstance(weights, dict) and weights:
                    extra["weights"] = weights
                elif isinstance(cs.get("extra"), dict):
                    nested_w = cs["extra"].get("weights")
                    if isinstance(nested_w, dict) and nested_w:
                        extra["weights"] = nested_w
                thr = cs.get("attribution_threshold")
                if thr is None and isinstance(cs.get("extra"), dict):
                    thr = cs["extra"].get("attribution_threshold")
                if isinstance(thr, (int, float)) and not isinstance(thr, bool):
                    extra["attribution_threshold"] = float(thr)
                elif isinstance(thr, str) and thr.strip():
                    try:
                        extra["attribution_threshold"] = float(thr.strip())
                    except ValueError:
                        pass
                grade = cs.get("grade")
                if not (isinstance(grade, str) and grade.strip()) and isinstance(
                    cs.get("extra"), dict
                ):
                    grade = cs["extra"].get("grade")
                if isinstance(grade, str) and grade.strip():
                    extra["grade"] = grade.strip()
                scores[cid] = EvaluatorResult(
                    score=float(cs["score"]),
                    reason=str(cs.get("reason") or ""),
                    sub_scores=cs.get("sub_scores"),
                    extra=extra,
                )
    else:
        # Multi-dimension under one evaluator_ref: if sub_scores covers every criterion.id, dispatch in one call;
        # otherwise fall back dimension by dimension inside the group (compatible with evaluators whose axes are not aligned). One failure does not affect other dimensions.

        async def _evaluate_ref(evaluator_ref: str) -> EvaluatorResult:
            logger.info(
                "benchmark.fused_evaluator_start run_id=%s item=%s evaluator=%s",
                run_id,
                item_id,
                evaluator_ref,
            )
            return await evaluator_service.evaluate(
                evaluator_ref,
                eval_request,
                options=EvalRunOptions(
                    resolve_trace=False,
                    skip_triage=True,
                    enable_trace_report=False,
                ),
            )

        def _record_success(criterion_id: str, evaluator_ref: str, result: EvaluatorResult) -> None:
            logger.info(
                "benchmark.fused_evaluator_raw run_id=%s item=%s criterion=%s "
                "evaluator=%s result=%s",
                run_id,
                item_id,
                criterion_id,
                evaluator_ref,
                result.model_dump_json()[:4000],
            )
            scores[criterion_id] = result
            if is_raw_result_store(ctx.benchmark_name):
                criterion_scores[criterion_id] = result.model_dump(mode="json")
                return
            cs_payload: dict[str, Any] = {
                "score": result.score,
                "reason": result.reason,
                "sub_scores": result.sub_scores,
            }
            _attach_judge_extra(cs_payload, getattr(result, "extra", None))
            criterion_scores[criterion_id] = cs_payload

        def _record_failure(criterion_id: str, eval_exc: Exception) -> None:
            scores[criterion_id] = EvalFailure(eval_exc)
            failed_payload: dict[str, Any] = {
                "score": 0.0,
                "reason": str(eval_exc)[:500],
            }
            prompt_meta = _prompt_customization_on_failure(eval_exc, eval_request)
            if prompt_meta is not None:
                failed_payload["prompt_customization"] = prompt_meta
            criterion_scores[criterion_id] = failed_payload
            logger.warning(
                "benchmark.fused_criterion_failed run_id=%s item=%s criterion=%s err=%s",
                run_id,
                item_id,
                criterion_id,
                eval_exc,
                exc_info=True,
            )

        for evaluator_ref, group in group_criteria_by_evaluator_ref(criteria).items():
            try:
                first = await _evaluate_ref(evaluator_ref)
            except Exception as eval_exc:
                for criterion in group:
                    _record_failure(criterion["id"], eval_exc)
                continue

            if len(group) == 1:
                _record_success(group[0]["id"], evaluator_ref, first)
                continue

            if can_fanout_sub_scores(first, group):
                for cid, result in fanout_criterion_scores(first, group).items():
                    _record_success(cid, evaluator_ref, result)
                continue

            _record_success(group[0]["id"], evaluator_ref, first)
            for criterion in group[1:]:
                criterion_id = criterion["id"]
                try:
                    result = await _evaluate_ref(evaluator_ref)
                except Exception as eval_exc:
                    _record_failure(criterion_id, eval_exc)
                    continue
                _record_success(criterion_id, evaluator_ref, result)

    # ScoreFuser aggregate (async HTTP when a remote fusion plugin is used)
    fused_score = await ScoreFuser.afuse(scores, rubric, aggregation)
    # Dynamic Rubric: override when weights+sub_scores are valid; otherwise keep the static / fusion_fn_ref result
    fused_score = apply_dynamic_fused_override(scores, fused_score)

    judge_json: dict[str, Any] = {
        "criterion_scores": criterion_scores,
        "fused_score": float(fused_score),
        # raw_json: the judge's raw return value as-is (no processing)
        "raw_json": _build_judge_raw_json(scores),
    }
    # Plan A: the framework writes grade from the fused score scale (the prompt declares the grade table; the LLM does not emit it)
    stamp_fused_grade(
        judge_json,
        fused_score=float(fused_score),
        pass_threshold=aggregation.pass_threshold,
    )
    # Persist project_id: BenchmarkRunItem has no metadata, so attribution/replay reads it back from the fused row
    # (same source as EvaluatorRequest.project_id; skip when empty so other business fields are not covered).
    if eval_request.project_id:
        judge_json["project_id"] = eval_request.project_id

    # trace_url (best-effort, does not block the main flow): the sync path takes (project_id, trace_id)
    # from the business item via the item-shape extractor (search_eval = the csv
    # ``project``/``trace_id`` columns written to metadata/source_trace_id by the parse interceptor);
    # a business with no extracted identity does not resolve and does not write the field (legacy behavior unchanged).
    # The search_eval family has register_direct_trace_url: splice the dashboard link directly, do not call Langfuse.
    trace_url = await build_trace_url_for_item(
        {
            "item_id": item_id,
            "source_trace_id": source_trace_id,
            "input": case.input,
            "metadata": case.metadata,
        },
        benchmark_name=ctx.benchmark_name,
    )

    # Write 1 fused row (upsert; shared write function, phase 3 of the three-phase write)
    updated = await _update_fused_success_row(
        _FusedSuccessRowWrite(
            run_id=run_id,
            item_id=item_id,
            case=case,
            fused_score=fused_score,
            judge_json=judge_json,
            trace_url=trace_url,
            source_trace_id=source_trace_id,
        ),
    )

    if updated:
        await run_service.increment_progress(run_id, evaluated_delta=1)
        if ctx.count_judge_received:
            # The judge-sync + attribution-async combination uses the external counter model (§8.2):
            # there is no HTTP callback, so the worker increments judge_received_count on its behalf
            await run_service.increment_config_counter(run_id, "judge_received_count")

    logger.info(
        "benchmark.fused_case_completed run_id=%s item=%s fused_score=%s criteria=%d",
        run_id,
        item_id,
        fused_score,
        len(criteria),
    )
    if not updated:
        return None
    return FusedJudgeOutcome(
        fused_score=float(fused_score),
        criterion_scores=criterion_scores,
    )


async def _update_fused_success_row(
    write: _FusedSuccessRowWrite,
) -> int:
    """Shared function for a successful fused-row write (phase 3 of the three-phase write; the single sync-path landing point).

    ``WHERE status != 'success'`` is idempotent (a resume that is already success returns 0);
    ``trace_url`` enters the update fields only when it has a value (None is not persisted; legacy behavior
    stays bit-for-bit with today). ``trace_id`` / ``source_trace_id`` trust only
    ``write.source_trace_id`` (already resolved by benchmark in ``_process_fused_case``;
    a dirty search_eval item_id → None, and falling back to ``case.source_trace_id`` is forbidden;
    None is not persisted). Returns the number of rows updated.
    """
    run_id = write.run_id
    item_id = write.item_id
    case = write.case
    fused_score = write.fused_score
    judge_json = write.judge_json
    trace_url = write.trace_url
    source_trace_id = write.source_trace_id
    update_fields: dict[str, Any] = {
        "status": "success",
        "evaluator_ref": "fused",
        "score": float(fused_score),
        "judge_json": judge_json,
        "input": case.input,
        "output": case.output,
    }
    if trace_url:
        update_fields["trace_url"] = trace_url
    if source_trace_id:
        update_fields["trace_id"] = source_trace_id
        update_fields["source_trace_id"] = source_trace_id
    return (
        await BenchmarkRunItem.filter(
            run_id=run_id,
            item_id=item_id,
            criterion_id=FUSED_CRITERION_ID,
        )
        .exclude(status="success")
        .update(**update_fields)
    )


async def _is_item_success(
    run_id: str,
    item_id: str,
    criterion_id: str,
) -> bool:
    """Whether the given (item, criterion) is already success."""
    row = await BenchmarkRunItem.filter(
        run_id=run_id,
        item_id=item_id,
        criterion_id=criterion_id,
    ).first()
    return row is not None and row.status == "success"


async def _update_judge_fields(
    key: ItemRowKey,
    write: _JudgeFieldsWrite,
) -> bool:
    """D12 phase 2: UPDATE score fields (status=success).

    ``WHERE status != 'success'`` is idempotent.
    """
    score = write.score
    reason = write.reason
    judge_json = write.judge_json
    sub_scores = write.sub_scores
    update_fields: dict[str, Any] = {
        "status": "success",
        "score": score,
        "reason": reason,
        "judge_json": judge_json,
        "evaluator_ref": key.evaluator_ref,
    }
    if sub_scores is not None:
        update_fields["sub_scores"] = sub_scores

    updated = (
        await BenchmarkRunItem.filter(
            run_id=key.run_id,
            item_id=key.item_id,
            criterion_id=key.criterion_id,
        )
        .exclude(status="success")
        .update(**update_fields)
    )
    return updated > 0


async def _update_to_failed(
    write: _FailedRowWrite,
) -> int:
    """Terminal fail: UPDATE run_items to failed.

    Returns:
        The number of rows actually updated.

    Idempotency guard: exclude success (do not overturn it) and failed (do not fail terminally twice) —
    counting failed→failed again (stalled redelivery / repeated terminal-fail callbacks) once over-collected
    judge_received_count (ieval_external-20260909: 275/200).
    """
    run_id = write.run_id
    item_id = write.item_id
    criteria = write.criteria
    error = write.error
    is_fused = write.is_fused
    # The error_message column stores a truncated copy (column-width limit); the full stack goes to the error log,
    # and debugging uses the log, not the DB column.
    err_msg = str(error)[:2000]
    logger.error(
        "benchmark.item_marked_failed run_id=%s item=%s err=%s",
        run_id,
        item_id,
        error,
        exc_info=error,
    )
    if is_fused:
        updated = (
            await BenchmarkRunItem.filter(
                run_id=run_id,
                item_id=item_id,
                criterion_id=FUSED_CRITERION_ID,
            )
            .exclude(status__in=["success", "failed"])
            .update(
                status="failed",
                score=0.0,
                error_message=err_msg,
            )
        )
    else:
        criterion_ids = [c["id"] for c in criteria]
        updated = (
            await BenchmarkRunItem.filter(
                run_id=run_id,
                item_id=item_id,
                criterion_id__in=criterion_ids,
            )
            .exclude(status__in=["success", "failed"])
            .update(
                status="failed",
                score=0.0,
                error_message=err_msg,
            )
        )
    return updated


async def _mark_benchmark_failed_from_wait(
    data: dict[str, Any],
    run_service: BenchmarkService,
    *,
    reason: str = "retry_deadline",
) -> int:
    """Terminal fail on the wait path: write input/output to PG, then mark failed.

    Aligned with external_worker._mark_external_failed_from_wait:
    the job payload carries case input/output, written to PG together on terminal fail,
    so the report case detail is not entirely empty (never entered running → no prewrite row → input/output NULL).

    Returns:
        Rows actually flipped to failed (0 = already terminal, idempotent re-entry). The caller uses this to decide
        whether to advance the external counter model's ``judge_received_count`` on its behalf (avoid double counting).
    """
    run_id: str = data["run_id"]
    item_id: str = data["item_id"]
    err = RuntimeError(reason)
    case = EvalCase.from_payload(data)
    triage_mode = data.get("triage_mode", "fused")
    is_fused = triage_mode == "fused"
    criteria: list[dict[str, Any]] = data.get("criteria") or []

    # Lay down the row first (including input/output), then mark failed
    if is_fused:
        still_open = await prewrite_running_fused_item(
            FusedRunningPrewrite(
                run_id=run_id,
                item_id=item_id,
                source_trace_id=case.source_trace_id,
                item_input=case.input,
                item_output=case.output,
                create_score=0.0,
            ),
        )
        if not still_open:
            return 0
    else:
        for criterion in criteria:
            await BenchmarkRunItem.get_or_create(
                run_id=run_id,
                item_id=item_id,
                criterion_id=criterion["id"],
                defaults={
                    "evaluator_ref": criterion.get("evaluator_ref", ""),
                    "status": "running",
                    "input": case.input,
                    "output": case.output,
                    "source_trace_id": case.source_trace_id,
                },
            )

    updated = await _update_to_failed(
        _FailedRowWrite(
            run_id=run_id,
            item_id=item_id,
            criteria=criteria,
            error=err,
            is_fused=is_fused,
        ),
    )
    if updated > 0:
        await run_service.increment_progress(run_id, failed_delta=updated)
    return updated


@dataclass(frozen=True)
class FusedJudgeAndRouteCtx:
    """Context linking fused judge and attribution routing (PLR0915/PLR0913 parameter collapse)."""

    case_ctx: FusedCaseContext
    data: dict[str, Any]
    attr_cfg: dict[str, Any]
    attr_impl: "LocalAttributionImpl | None"
    deps: "BenchmarkWorkerDeps"
    queue_manager: EvalQueueRegistry


async def _judge_fused_and_route(ctx: FusedJudgeAndRouteCtx) -> None:
    """Link fused judge → attribution routing (§8.2; extracted from _judge_and_write for PLR0915)."""
    outcome = await _process_fused_case(ctx.case_ctx)
    # A sync judge has no HTTP callback; attribution is routed inside the worker from the run config (§8.2)
    await _route_sync_attribution(
        SyncAttributionRouteCtx(
            run_id=ctx.case_ctx.run_id,
            item_id=ctx.case_ctx.item_id,
            case=ctx.case_ctx.case,
            data=ctx.data,
            outcome=outcome,
            attr_cfg=ctx.attr_cfg,
            attr_impl=ctx.attr_impl,
            deps=ctx.deps,
            queue_manager=ctx.queue_manager,
        ),
    )


async def _bump_judge_received_if_external(
    run_id: str,
    run_service: BenchmarkService,
) -> None:
    """Runs on the external counter model: a terminally failed item increments judge_received_count on its behalf.

    (The judge-sync + attribution-async combination has no judge callback; count convergence depends on this
    increment, or the run waits forever for ``judge_expected_count`` and sticks.)
    """
    judge_cfg, attr_cfg = await _get_run_exec_cfg(run_id, run_service)
    if _finalize_eval_type(judge_cfg, attr_cfg) == "external":
        await run_service.increment_config_counter(run_id, "judge_received_count")


@dataclass(frozen=True)
class SyncAttributionRouteCtx:
    """Parameter object for ``_route_sync_attribution`` (PLR0913 parameter collapse)."""

    run_id: str
    item_id: str
    case: EvalCase
    data: dict[str, Any]
    outcome: "FusedJudgeOutcome | None"
    attr_cfg: dict[str, Any]
    attr_impl: "LocalAttributionImpl | None"
    deps: "BenchmarkWorkerDeps"
    queue_manager: EvalQueueRegistry


async def _schedule_judge_notification_from_sync(ctx: SyncAttributionRouteCtx) -> None:
    """``attribution.execution=none``: after judge finishes, push straight to Aone by the fused score.

    Config comes from the run ``config_json`` (including ``aone_dispatch``); query prefers case
    input. Background ``create_task``; exceptions are fully caught inside the dispatch orchestration.
    """
    outcome = ctx.outcome
    if outcome is None:
        return
    run = await ctx.deps.run_service.get_run(ctx.run_id)
    config = getattr(run, "config_json", None) if run is not None else None
    input_data = ctx.case.input if isinstance(ctx.case.input, dict) else {}
    query = str(input_data.get("query") or "")
    schedule_notification_dispatch_for_judge(
        DispatchRequest(
            store=ctx.deps.run_service,
            run_id=ctx.run_id,
            item_id=str(ctx.item_id),
            source_trace_id=ctx.case.source_trace_id,
            config=config if isinstance(config, dict) else None,
        ),
        JudgeDispatchInput(
            fused_score=float(outcome.fused_score),
            criterion_scores=outcome.criterion_scores,
            query=query or None,
        ),
    )
    logger.info(
        "benchmark.judge_aone_dispatch_scheduled run_id=%s item=%s fused=%s",
        ctx.run_id,
        ctx.item_id,
        outcome.fused_score,
    )


async def _route_sync_attribution(ctx: SyncAttributionRouteCtx) -> None:
    """Attribution routing after a sync judge (§8.2; a sync judge has no HTTP callback, so the worker routes).

    - ``none`` / default → no separate attribution; push straight to Aone by the judge fused score (score
      threshold same as post-attribution dispatch: FAIL ∪ POOR by default);
    - ``outcome=None`` (resume already success) → do not route again;
    - ``sync`` → the local attribution class at ``attribution.target``; default = today's
      ``LocalAttributionRunner`` (wraps TriageAttributionService);
    - ``async`` → enqueue ``attribution-wait`` + expected count (§8.2 new combination:
      local judge → Aone attribution).
    """
    run_id, item_id, case, data = ctx.run_id, ctx.item_id, ctx.case, ctx.data
    outcome, attr_cfg = ctx.outcome, ctx.attr_cfg
    execution = str(attr_cfg.get("execution") or "none")
    if outcome is None:
        return
    if execution == "none":
        # Push straight from judge only for explicit no_attribution (snapshot provider=none).
        # An empty snapshot {} whose default execution=none keeps the old "do not route" semantics,
        # so a legacy run does not accidentally turn dispatch on (_get_run_exec_cfg's documented contract).
        if str(attr_cfg.get("provider") or "") == "none":
            await _schedule_judge_notification_from_sync(ctx)
        return
    provider = str(attr_cfg.get("provider") or "example")
    # Gate: judge attribution_threshold → large-scale pass_threshold fallback →
    # the business registration policy (so falling back to the default <1.4 does not misread example 2.9 as not firing).
    _rubric, aggregation = await _get_run_rubric(run_id, ctx.deps.run_service)
    dynamic_threshold = resolve_gate_attribution_threshold(
        outcome.criterion_scores,
        pass_threshold=aggregation.pass_threshold,
    )
    dynamic_gate = should_trigger_with_optional_threshold(
        fused_score=outcome.fused_score,
        attribution_threshold=dynamic_threshold,
    )
    if dynamic_gate is not None:
        should_attr = dynamic_gate
    else:
        benchmark_name = str(data.get("benchmark_services") or data.get("benchmark_name") or "")
        trigger_policy = resolve_trigger_policy(benchmark_name)
        criterion_score_map = {
            cid: float(cs.get("score", 0.0)) for cid, cs in outcome.criterion_scores.items()
        }
        should_attr = await should_trigger_attribution(
            trigger_policy,
            fused_score=outcome.fused_score,
            criterion_scores=criterion_score_map,
        )
    logger.info(
        "benchmark.attribution_gate run_id=%s item=%s fused=%s "
        "should_attr=%s execution=%s provider=%s",
        run_id,
        item_id,
        outcome.fused_score,
        should_attr,
        execution,
        provider,
    )
    if not should_attr:
        return

    if execution == "async":
        # sync judge + async attribution (new combination): build the attribution case and enqueue it
        input_data = case.input if isinstance(case.input, dict) else {}
        output_data = case.output if isinstance(case.output, dict) else {}
        await enqueue_attribution_case(
            queue_manager=ctx.queue_manager,
            case=AttributionCaseSpec(
                run_id=run_id,
                item_id=item_id,
                source_trace_id=case.source_trace_id,
                adiu=str(input_data.get("adiu") or ""),
                # Pass ds_start empty: the only date source of truth is aone_attribution_provider's
                # ``ds_start or extract_ds(run_id)`` fallback; a blind date guess on the worker side
                # ValueError-crashes attribution routing for a run_id with no 8-digit date segment.
                ds_start="",
                issue={
                    "fused_score": outcome.fused_score,
                    "criterion_scores": outcome.criterion_scores,
                },
                attribution_provider=provider,
                do_cross_verify=bool(attr_cfg.get("do_cross_verify", True)),
                query=str(input_data.get("query") or ""),
                # response takes only response_text: the unified submit pipeline (daily_data_loader)
                # already standardizes the CSV model-output column as response_text; the old result-column fallback
                # served historical ieval CSV format, and that path has moved to external_runner (async
                # judge). This sync attribution path has no remaining result consumer.
                response=str(output_data.get("response_text") or ""),
                full_tool_calls_json=(
                    str(output_data.get("tool_calls_json"))
                    if output_data.get("tool_calls_json")
                    else None
                ),
                target_url=(str(attr_cfg["target"]) if attr_cfg.get("target") else None),
                class_ref=snapshot_class_ref(attr_cfg),
            ),
        )
        await ctx.deps.run_service.increment_attribution_expected_once(
            run_id,
            item_id,
        )
        return

    # Sync local attribution: a profile-named attribution class → custom implementation; default = LocalAttributionRunner.
    # Counts pair with the async path (expected +1 per item, idempotent → received+1 after completion is advanced by
    # LocalAttributionRunner._attribute_one's finally), so the
    # config_json counter semantics stay consistent (for the healer / reconciliation).
    (
        _count,
        _cfg,
        claimed,
    ) = await ctx.deps.run_service.increment_attribution_expected_once(
        run_id,
        item_id,
    )
    if not claimed:
        return
    if ctx.attr_impl is not None:
        await run_custom_local_attribution(
            ctx.deps.run_service,
            ctx.attr_impl,
            LocalAttributionRequest(
                run_id=run_id,
                item_id=item_id,
                fused_score=outcome.fused_score,
                provider=provider,
            ),
        )
        return
    # Resolved from the same source as judge: case.metadata / job project_id. BenchmarkRunItem
    # has no metadata column, so it must be passed here, or attribution only sees run.project_id=null → skip.
    case_metadata = case.metadata if isinstance(case.metadata, dict) else {}
    job_project = data.get("project_id")
    judge_project_id = _resolve_eval_project_id(
        job_project if isinstance(job_project, str) else None,
        case_metadata,
    )
    runner = LocalAttributionRunner(
        LocalAttributionRunnerDeps(
            triage_service=ctx.deps.triage_service,
            benchmark_service=ctx.deps.run_service,
        ),
    )
    await runner._attribute_one(
        LocalAttributionTask(
            run_id=run_id,
            item_id=item_id,
            fused_score=outcome.fused_score,
            provider=provider,
            finalize_callback=None,
            project_id=judge_project_id,
        ),
    )


@dataclass(frozen=True)
class BenchmarkWorkerDeps:
    """Service dependency bundle for the benchmark worker (parameter object for ``process_benchmark_case``).

    ``queue_manager`` is a required skeleton argument and stays outside; the judge callback unpacks
    the three services and settings from ``deps``. ``aone_client_factory`` supports per-run target resolution
    for the unified execution layer's async judge (§6.3, ``None`` = Aone not configured).
    """

    run_service: BenchmarkService
    evaluator_service: EvaluatorService
    triage_service: TriageAttributionService
    settings: Settings
    external_client_factory: "AoneClientFactory | None" = None


async def _process_async_judge_case(
    job: Job,
    token: str,
    case_ctx: _AsyncJudgeCaseCtx,
) -> None:
    """judge.execution=async branch: Aone submit + hold the slot for the callback (§8.2).

    Handlers are built by :func:`build_async_judge_handlers` (the same implementation shared with the
    external-wait legacy drain), and the queue names are the unified ``benchmark-cases/running``.
    """
    queue_manager = case_ctx.queue_manager
    deps = case_ctx.deps
    judge_cfg = case_ctx.judge_cfg
    run_id = str(job.data.get("run_id") or "")
    if deps.external_client_factory is None:
        raise RuntimeError(
            f"judge execution=async but AoneClientFactory not configured (run_id={run_id})",
        )
    # per-run target URL → client (factory LRU cache, §6.3)
    judge_client = deps.external_client_factory.get_judge_client(
        str(judge_cfg["target"]) if judge_cfg.get("target") else None,
    )
    class_ref = snapshot_class_ref(judge_cfg)
    judge_impl = (
        cast("AsyncJudgeImpl", load_execution_instance(class_ref, what="judge_class"))
        if class_ref
        else None
    )
    await process_sync_eval_case(
        job,
        token,
        SyncEvalRuntime(
            handlers=build_async_judge_handlers(
                AsyncJudgeHandlersBuild(
                    run_service=deps.run_service,
                    settings=deps.settings,
                    external_judge_client=judge_client,
                    queue_manager=queue_manager,
                    overrides=AsyncJudgeOverrides(
                        wait_queue_name=BENCHMARK_CASES,
                        running_queue_name=BENCHMARK_RUNNING,
                        judge_dims=(str(judge_cfg["dims"]) if judge_cfg.get("dims") else None),
                        judge_prompt_version=(
                            str(judge_cfg["prompt_version"])
                            if judge_cfg.get("prompt_version")
                            else None
                        ),
                        judge_impl=judge_impl,
                    ),
                ),
            ),
            queue_manager=queue_manager,
            deps=deps,
        ),
    )


async def process_benchmark_case(
    job: Job,
    token: str,
    *,
    queue_manager: EvalQueueRegistry,
    deps: BenchmarkWorkerDeps,
) -> None:
    """Unified case-worker entry (§8.3): route judge execution from the run snapshot.

    - run ``config_json.judge.execution == "async"`` → Aone submit + hold the slot for the callback
      (:func:`_process_async_judge_case`);
    - otherwise (default / ``sync``) → local per-criterion eval (today's logic; the skeleton
      ``process_sync_eval_case`` is unchanged).
    """
    run_id = str(job.data.get("run_id") or "")
    judge_cfg, attr_cfg = await _get_run_exec_cfg(run_id, deps.run_service)

    if judge_cfg.get("execution") == "async":
        await _process_async_judge_case(
            job,
            token,
            _AsyncJudgeCaseCtx(
                queue_manager=queue_manager,
                deps=deps,
                judge_cfg=judge_cfg,
            ),
        )
        return

    await process_sync_eval_case(
        job,
        token,
        SyncEvalRuntime(
            handlers=await _build_sync_handlers(
                _SyncHandlersBuild(
                    run_id=run_id,
                    queue_manager=queue_manager,
                    deps=deps,
                    judge_cfg=judge_cfg,
                    attr_cfg=attr_cfg,
                ),
            ),
            queue_manager=queue_manager,
            deps=deps,
        ),
    )


async def _finalize_sync_case(run_id: str, run_service: BenchmarkService) -> None:
    """Sync-path finalize: route the counter model from the run snapshot (§8.2, module-level PLR0915 collapse).

    Either judge or attribution async → external counter model; all sync → benchmark progress-column model.
    """
    jcfg, acfg = await _get_run_exec_cfg(run_id, run_service)
    await tick_and_maybe_finalize(
        run_id,
        _finalize_eval_type(jcfg, acfg),
        benchmark_run_service=run_service,
    )


async def _build_sync_handlers(
    build: _SyncHandlersBuild,
) -> SyncEvalHandlers:
    """Build sync judge handlers (today's per-criterion eval path; the skeleton is unchanged).

    PLR0915 collapse: the nested-closure bulk was extracted here from ``process_benchmark_case``;
    the logic matches today bit for bit, plus two §8.2 extension points, ``local_judge`` / attribution
    routing (inactive under the default config).
    """
    run_id, queue_manager, deps, judge_cfg, attr_cfg = (
        build.run_id,
        build.queue_manager,
        build.deps,
        build.judge_cfg,
        build.attr_cfg,
    )
    _judge_impl, attr_impl = await _get_run_local_impls(run_id, judge_cfg, attr_cfg)
    count_model_external = _finalize_eval_type(judge_cfg, attr_cfg) == "external"

    async def _judge_and_write(
        call: JudgeWriteCall,
    ) -> None:
        run_id, item_id, case, data, deps = (
            call.run_id,
            call.item_id,
            call.case,
            call.data,
            cast(BenchmarkWorkerDeps, call.deps),
        )
        evaluator_service, run_service = deps.evaluator_service, deps.run_service
        triage_mode = data.get("triage_mode", "fused")
        is_fused = triage_mode == "fused"
        criteria: list[dict[str, Any]] = data.get("criteria") or []
        service_name: str = data.get("service_name") or ""
        dataset_name: str = data.get("dataset_name") or ""
        project_id: str | None = data.get("project_id")

        if is_fused:
            await _judge_fused_and_route(
                FusedJudgeAndRouteCtx(
                    case_ctx=FusedCaseContext(
                        run_id=run_id,
                        item_id=item_id,
                        case=case,
                        criteria=criteria,
                        service_name=service_name,
                        project_id=project_id,
                        evaluator_service=evaluator_service,
                        run_service=run_service,
                        local_judge=_judge_impl,
                        count_judge_received=count_model_external,
                        benchmark_name=str(data.get("benchmark_services") or ""),
                    ),
                    data=data,
                    attr_cfg=attr_cfg,
                    attr_impl=attr_impl,
                    deps=deps,
                    queue_manager=queue_manager,
                ),
            )
        else:
            for criterion in criteria:
                criterion_id: str = criterion["id"]
                evaluator_ref: str = criterion["evaluator_ref"]
                row_key = ItemRowKey(
                    run_id=run_id,
                    item_id=item_id,
                    criterion_id=criterion_id,
                    evaluator_ref=evaluator_ref,
                )

                await _insert_running_row(
                    row_key,
                    input_val=case.input,
                    output_val=case.output,
                    source_trace_id=case.source_trace_id,
                )

                if await _is_item_success(run_id, item_id, criterion_id):
                    continue

                eval_request = await _prepare_judge_request(
                    run_id,
                    run_service,
                    JudgeRequestContext(
                        case=case,
                        item_id=item_id,
                        service_name=service_name,
                        project_id=project_id,
                    ),
                )

                writer = BenchmarkTraceRecordWriter(
                    run_id=run_id,
                    criterion_id=criterion_id,
                    meta=RunMeta(
                        benchmark_name=dataset_name,
                        service_name=service_name,
                        dataset_name=dataset_name,
                        project_id=project_id,
                    ),
                )

                logger.info(
                    "benchmark.criterion_start run_id=%s item=%s criterion=%s evaluator=%s",
                    run_id,
                    item_id,
                    criterion_id,
                    evaluator_ref,
                )
                result = await evaluator_service.evaluate(
                    evaluator_ref,
                    eval_request,
                    options=EvalRunOptions(
                        resolve_trace=False,
                        skip_triage=True,
                        writer=writer,
                        enable_trace_report=False,
                    ),
                )

                judge_json: dict[str, Any] = {
                    "score": result.score,
                    "reason": result.reason,
                }
                if result.sub_scores:
                    judge_json["sub_scores"] = result.sub_scores

                updated = await _update_judge_fields(
                    row_key,
                    _JudgeFieldsWrite(
                        score=result.score,
                        reason=result.reason,
                        judge_json=judge_json,
                        sub_scores=result.sub_scores,
                    ),
                )
                if updated:
                    await run_service.increment_progress(
                        run_id,
                        evaluated_delta=1,
                    )
                logger.info(
                    "benchmark.criterion_success run_id=%s item=%s criterion=%s "
                    "score=%s updated=%s",
                    run_id,
                    item_id,
                    criterion_id,
                    result.score,
                    updated,
                )

    async def _mark_failed_from_wait(
        *,
        data: dict[str, Any],
        reason: str,
    ) -> None:
        updated = await _mark_benchmark_failed_from_wait(
            data,
            deps.run_service,
            reason=reason,
        )
        # Runs on the external counter model (judge sync + attribution async):
        # a terminally failed item advances judge_received_count on its behalf (only the first transition to terminal, to avoid double counting)
        if updated > 0:
            await _bump_judge_received_if_external(str(data["run_id"]), deps.run_service)

    async def _mark_failed_from_running(
        *,
        data: dict[str, Any],
        exc: Exception,
    ) -> None:
        run_id = data["run_id"]
        item_id = data["item_id"]
        triage_mode = data.get("triage_mode", "fused")
        is_fused = triage_mode == "fused"
        crit = data.get("criteria") or []
        updated = await _update_to_failed(
            _FailedRowWrite(
                run_id=run_id,
                item_id=item_id,
                criteria=crit,
                error=exc,
                is_fused=is_fused,
            ),
        )
        if updated > 0:
            await deps.run_service.increment_progress(
                run_id,
                failed_delta=updated,
            )
            await _bump_judge_received_if_external(str(run_id), deps.run_service)

    return SyncEvalHandlers(
        eval_type="benchmark",
        wait_queue_name=BENCHMARK_CASES,
        running_queue_name=BENCHMARK_RUNNING,
        slot_config=SlotConfig(
            retry_max_duration_ms=deps.settings.sync_retry_max_duration_ms,
            cluster_soft_cap=deps.settings.sync_cluster_soft_cap,
            slots_per_run_per_node=deps.settings.sync_slots_per_run_per_node,
            slot_busy_delay_ms=deps.settings.sync_slot_busy_delay_ms,
            slot_ttl_s=deps.settings.sync_slot_ttl_s,
        ),
        is_run_terminal=lambda rid: _is_run_terminal(rid, deps.run_service),
        heartbeat=deps.run_service.heartbeat,
        judge_and_write=_judge_and_write,
        mark_failed_from_wait=_mark_failed_from_wait,
        mark_failed_from_running=_mark_failed_from_running,
        # The lambda points straight at the module-level finalize (two fewer lines than a closure def; PLR0915 collapse)
        finalize_fn=lambda rid: _finalize_sync_case(rid, deps.run_service),
    )


def create_benchmark_worker(
    queue_manager: EvalQueueRegistry,
    deps: BenchmarkWorkerDeps,
    *,
    queue_name: str = BENCHMARK_CASES,
) -> Worker:
    """Create the BullMQ Worker for the benchmark case queue (unified case worker, §8.3).

    ``queue_name``: the default queue or the L2 isolation queue (design §12.7). Job payloads
    are the same shape and the processor is the same implementation; separate queues exist only for process isolation and an independent concurrency pool.
    ``aone_client_factory``: per-run target resolution for async judge (§6.3);
    ``None`` = only sync eval is available (legacy behavior).
    """
    queue = queue_manager.get_queue(queue_name)
    opts = queue_manager.build_worker_opts(
        queue_name=queue_name,
        lock_duration_ms=deps.settings.eval_queue_lock_duration_ms,
        stalled_interval_ms=deps.settings.eval_queue_stalled_interval_ms,
    )
    opts["concurrency"] = deps.settings.eval_queue_worker_concurrency

    async def processor(job: Job, token: str) -> None:
        await process_benchmark_case(
            job,
            token,
            queue_manager=queue_manager,
            deps=deps,
        )

    worker = Worker(
        queue.name,
        processor,
        opts,
    )
    logger.info(
        "benchmark_worker.started queue=%s concurrency=%d lockDuration=%dms",
        queue.name,
        deps.settings.eval_queue_worker_concurrency,
        deps.settings.eval_queue_lock_duration_ms,
    )
    return worker


__all__ = [
    "BenchmarkWorkerDeps",
    "create_benchmark_worker",
    "process_benchmark_case",
]
