"""Unified ReportSpec assembly and the report-generation retry kernel.

Shared by ``BenchmarkRunner._generate_report`` and
``ExternalBenchmarkRunner._generate_external_report``, so sync / async finalize do not
drift on fields (rule_link_url / total_failed / strip_per_criterion_failed_count /
build_report_criteria).

Completion-detection CAS (progress column vs counter) still forks. This module only
unifies "after aggregation → generate the report".
"""

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.models import BenchmarkRun, BenchmarkRunItem
from benchmark.domain.eval_config import logical_benchmark_run_name
from benchmark.domain.trace_report_config import normalize_report_html_name
from benchmark.runs.common.config_helpers import (
    load_rubric_from_config,
    strip_per_criterion_failed_count,
)
from benchmark.runs.common.dimension_stats import (
    build_report_criteria,
    resolve_run_stat_dimensions,
)
from benchmark.runs.trace_daily_report_adapter import apply_weight_dimension_section
from benchmark.reporting.framework.performance_metrics import summarize_performance_metrics
from benchmark.reporting.framework.report_dispatch import dispatch_report_generation
from benchmark.reporting.framework.schema import (
    AggregatedMetrics,
    Criterion,
    aggregated_metrics_from_json,
)

logger = get_logger("agent_eval.benchmark.report_spec")

REPORT_MAX_ATTEMPTS = 4  # 1 initial + 3 retries
REPORT_BACKOFF_BASE_S = 5.0
REPORT_BACKOFF_MAX_S = 40.0
_REPORT_JITTER_RATIO = 0.25


class _WebdevLike(Protocol):
    async def generate_report(
        self,
        spec: dict[str, Any],
    ) -> tuple[str, str | None]: ...


class _PagesPublisherLike(Protocol):
    async def publish(self, run_id: str, html: str) -> str | None: ...


class _BenchmarkServiceLike(Protocol):
    async def get_run(self, run_id: str) -> BenchmarkRun | None: ...

    async def load_case_results(
        self,
        *,
        run_id: str,
        benchmark_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def attach_daily_report_context(self, spec: dict[str, Any]) -> None: ...

    async def merge_performance_metrics(
        self,
        run_id: str,
        *,
        performance_metrics: dict[str, Any],
    ) -> None: ...

    async def merge_report_meta(
        self,
        run_id: str,
        *,
        patch: dict[str, Any],
    ) -> None: ...


CaseResultsLoader = Callable[[], Awaitable[list[dict[str, Any]]]]


@dataclass(slots=True)
class ReportJobInput:
    """Report-generation inputs (FUNC-PARAMS-001: assembly and generation share one parameter object)."""

    benchmark_service: _BenchmarkServiceLike
    run_id: str
    metrics: AggregatedMetrics
    final_score: float
    config: dict[str, Any] | None = None
    rubric: Sequence[Criterion] | None = None
    case_results_loader: CaseResultsLoader | None = None
    component_services: object | None = None
    log_prefix: str = "report"


def resolve_rule_link_url(config: dict[str, Any]) -> str | None:
    """Resolve the report "scoring rules" link: the config snapshot wins, and an existing run falls back to the rubric class.

    A new run already writes ``Rubric.rule_link_url`` into ``config_json`` at creation.
    When a run created before this feature has no such key, fall back to loading the
    current Rubric class from ``config_json.rubric_ref`` (whitelist prefix
    ``benchmark.runs.``).
    """
    url = config.get("rule_link_url")
    if isinstance(url, str) and url:
        return url
    rubric_ref = config.get("rubric_ref")
    if not isinstance(rubric_ref, str) or not rubric_ref:
        return None
    module_name, _, class_name = rubric_ref.rpartition(":")
    if not module_name.startswith("benchmark.runs."):
        return None
    try:
        import importlib  # noqa: PLC0415

        rubric_cls = getattr(importlib.import_module(module_name), class_name)
        fallback = getattr(rubric_cls, "rule_link_url", None)
    except Exception:
        logger.warning("report_spec.rule_link_url_fallback_failed rubric_ref=%s", rubric_ref)
        return None
    return fallback if isinstance(fallback, str) and fallback else None


async def count_distinct_failed_items(run_id: str) -> int:
    """Item-level distinct failure count (avoids inflating per-criterion failed_count)."""
    try:
        failed_item_ids = (
            await BenchmarkRunItem.filter(
                run_id=run_id,
                status="failed",
            )
            .distinct()
            .values_list("item_id", flat=True)
        )
        return len(failed_item_ids)
    except Exception:
        logger.warning(
            "report_spec.count_failed_items_failed run_id=%s — default 0",
            run_id,
            exc_info=True,
        )
        return 0


def _as_metrics_map(raw: object) -> dict[str, Any]:
    """Normalize an aggregate-metrics sub-dict: a non-dict (including MagicMock) → an empty dict."""
    return raw if isinstance(raw, dict) else {}


def _dump_attr_stats(stats: object) -> dict[str, Any]:
    dump = getattr(stats, "model_dump", None)
    if callable(dump):
        dumped = dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(stats, dict):
        return dict(stats)
    return {}


def prefer_db_metrics(
    run: BenchmarkRun | None,
    metrics: AggregatedMetrics,
    final_score: float,
) -> tuple[AggregatedMetrics, float]:
    """DB metrics_json / final_score is the only source of truth (no in-memory fallback).

    Report generation must trust only the DB: under multi-node deployment the in-memory
    metrics passed by the caller may cover only items handled by this node (or a healer
    recovery node may have no in-memory state at all). Every finalize path persists the
    full aggregate via ``update_run_status(GENERATING_REPORT, metrics_json=...)`` before
    generating the report.

    Raises:
        RuntimeError: the run is missing / metrics_json has no criterion_metrics /
            deserialization fails / final_score is not numeric — the caller should retry
            or fail, and must not fall back to in-memory metrics.
    """
    if run is None:
        raise RuntimeError("prefer_db_metrics: run not found (no DB truth)")
    db_metrics = aggregated_metrics_from_json(run.metrics_json)
    if db_metrics is None:
        raise RuntimeError(
            "prefer_db_metrics: metrics_json missing/invalid (no criterion_metrics) — "
            "refusing to fall back to in-memory metrics",
        )
    db_final = getattr(run, "final_score", None)
    if not isinstance(db_final, (int, float)) or isinstance(db_final, bool):
        raise RuntimeError(
            "prefer_db_metrics: final_score missing/invalid in DB — "
            "refusing to fall back to in-memory final_score",
        )
    return db_metrics, float(db_final)


@dataclass(slots=True)
class ReportSpecFields:
    """Inputs for pure field assembly (FUNC-PARAMS-001 consolidation)."""

    run: BenchmarkRun | None
    run_id: str
    config: dict[str, Any]
    metrics: AggregatedMetrics
    final_score: float
    rubric: Sequence[Criterion]
    total_failed: int


def build_report_spec_dict(fields: ReportSpecFields) -> dict[str, Any]:
    """Pure ReportSpec field assembly (no IO)."""
    run, run_id, config = fields.run, fields.run_id, fields.config
    metrics, final_score = fields.metrics, fields.final_score
    rubric, total_failed = fields.rubric, fields.total_failed

    total_samples = int(getattr(metrics, "total_samples", 0) or 0)
    total_success = max(total_samples - total_failed, 0)
    # The report header shows the logical name; the column may already store a registrationId
    bm_name = logical_benchmark_run_name(
        column_name=getattr(run, "benchmark_name", None) if run is not None else None,
        config=config,
    )
    criterion_metrics = _as_metrics_map(getattr(metrics, "criterion_metrics", None))
    attribution_metrics = _as_metrics_map(getattr(metrics, "attribution_metrics", None))
    react_rounds = getattr(metrics, "react_rounds_summary", None)
    react_rounds = dict(react_rounds) if isinstance(react_rounds, dict) else {}

    # When stat_dimensions exist, criteria show only sub-dimensions by default. If this run
    # collected no sub_scores (a custom prompt omitted format B), sub-dimension metrics are empty → a blank report.
    # Fall back to the rubric's primary dimension so the overall score / reason stay visible.
    stat_dims = resolve_run_stat_dimensions(config, bm_name or None)
    has_dim_metrics = any(d.key in criterion_metrics for d in stat_dims)
    report_dims = stat_dims if has_dim_metrics else ()

    duration = getattr(metrics, "total_duration_ms", None)
    spec: dict[str, Any] = {
        "run_id": run_id,
        "benchmark_name": bm_name,
        "service_name": getattr(run, "service_name", None) or "",
        "dataset_name": getattr(run, "dataset_name", None) or "",
        "rule_link_url": resolve_rule_link_url(config),
        "final_score": final_score,
        "aggregation_method": getattr(metrics, "aggregation_method", "weighted_avg"),
        "total_samples": total_samples,
        "total_failed": total_failed,
        "total_success": total_success,
        "criteria": build_report_criteria(
            list(rubric),
            report_dims,
        ),
        "criterion_metrics": {
            cid: strip_per_criterion_failed_count(stats) for cid, stats in criterion_metrics.items()
        },
        "attribution_metrics": {
            cid: _dump_attr_stats(stats) for cid, stats in attribution_metrics.items()
        },
        "react_rounds_summary": react_rounds,
    }
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        spec["total_duration_ms"] = int(duration)
    # Trace Daily Report: pass through the full score and the daily-report flag (other businesses omit these keys; the template/waterfall default to 2.0)
    if config.get("trace_daily_report"):
        spec["trace_daily_report"] = True
        raw_mark = config.get("score_full_mark", 1.0)
        try:
            mark = float(raw_mark)
        except (TypeError, ValueError):
            mark = 1.0
        spec["score_full_mark"] = mark
        # Namespace used to group trends (matches the run_id prefix); the benchmark_name column may be an internal agent.
        report_ns = config.get("trace_report_namespace")
        if isinstance(report_ns, str) and report_ns.strip():
            spec["trace_report_namespace"] = report_ns.strip()
    _copy_report_binding(spec, config)
    return spec


def _copy_report_binding(spec: dict[str, Any], config: dict[str, Any]) -> None:
    """Carry the page-registered parser class, template filename, rubric class, and document URL onto the report spec."""
    parser_alias = config.get("report_parser_class")
    if isinstance(parser_alias, str) and parser_alias.strip():
        spec["report_parser_class"] = parser_alias.strip()
    report_html = normalize_report_html_name(config.get("report_html"))
    if report_html is not None:
        spec["report_html"] = report_html
    rubric_class = config.get("rubric_class")
    if isinstance(rubric_class, str) and rubric_class.strip():
        spec["rubric_class"] = rubric_class.strip()
    if "rubric_doc_url" in config:
        raw_url = config.get("rubric_doc_url")
        spec["rubric_doc_url"] = raw_url.strip() if isinstance(raw_url, str) else ""


def _merge_component_extra(
    spec: dict[str, Any],
    extra: dict[str, Any] | None,
) -> None:
    """Keep business ``_build_report_spec`` extension keys; standard fields follow the unified assembly."""
    if not extra:
        return
    for key, value in extra.items():
        if key not in spec:
            spec[key] = value


async def assemble_report_spec(job: ReportJobInput) -> dict[str, Any]:
    """Assemble a full ReportSpec (case details plus daily-report context)."""
    benchmark_service = job.benchmark_service
    run_id, log_prefix = job.run_id, job.log_prefix
    run = await benchmark_service.get_run(run_id)
    metrics, final_score = prefer_db_metrics(run, job.metrics, job.final_score)

    run_config = run.config_json if run is not None and isinstance(run.config_json, dict) else {}
    cfg = dict(job.config) if isinstance(job.config, dict) and job.config else dict(run_config)
    resolved_rubric = list(job.rubric) if job.rubric is not None else load_rubric_from_config(cfg)
    total_failed = await count_distinct_failed_items(run_id)

    spec = build_report_spec_dict(
        ReportSpecFields(
            run=run,
            run_id=run_id,
            config=cfg,
            metrics=metrics,
            final_score=final_score,
            rubric=resolved_rubric,
            total_failed=total_failed,
        ),
    )

    builder = getattr(job.component_services, "_build_report_spec", None)
    if callable(builder):
        try:
            extra = builder(
                run=run,
                aggregated=metrics,
                final_score=final_score,
            )
        except Exception:
            logger.warning(
                "%s.component_spec_failed run_id=%s",
                log_prefix,
                run_id,
                exc_info=True,
            )
            extra = None
        if isinstance(extra, dict):
            _merge_component_extra(spec, extra)

    if job.case_results_loader is not None:
        loaded = await job.case_results_loader()
    else:
        bm_name = (
            logical_benchmark_run_name(
                column_name=getattr(run, "benchmark_name", None) if run is not None else None,
                config=cfg,
            )
            or None
        )
        loaded = await benchmark_service.load_case_results(
            run_id=run_id,
            benchmark_name=bm_name,
            config=cfg,
        )

    case_count = len(loaded) if loaded else 0
    logger.info(
        "%s.report_spec run_id=%s case_results=%d criteria=%d "
        "criterion_metrics=%s attribution_metrics=%s final_score=%s "
        "total_failed=%d total_success=%d",
        log_prefix,
        run_id,
        case_count,
        len(spec.get("criteria") or []),
        list((spec.get("criterion_metrics") or {}).keys()),
        list((spec.get("attribution_metrics") or {}).keys()),
        final_score,
        spec.get("total_failed"),
        spec.get("total_success"),
    )
    if not loaded:
        logger.warning(
            "%s.report_no_case_results run_id=%s — Case-detail section omitted",
            log_prefix,
            run_id,
        )
    else:
        spec["case_results"] = loaded
        apply_weight_dimension_section(spec)

    # Performance and efficiency must come before daily-report context: today's trend point reads latency / Token from the spec's current values
    await attach_performance_metrics(
        spec,
        PerformanceMetricsContext(
            benchmark_service=benchmark_service,
            run_id=run_id,
            case_results=loaded,
            log_prefix=log_prefix,
        ),
    )
    await benchmark_service.attach_daily_report_context(spec)
    return spec


@dataclass(frozen=True)
class PerformanceMetricsContext:
    """Inputs for performance-metric aggregation (the parameter object of :func:`attach_performance_metrics`)."""

    benchmark_service: _BenchmarkServiceLike
    run_id: str
    case_results: list[dict[str, Any]] | None
    log_prefix: str = "report"


@dataclass(frozen=True)
class PagesPublishContext:
    """Inputs for Pages publishing (the parameter object of :func:`_publish_pages_if_needed`)."""

    run_id: str
    html: str
    report_url: str | None
    pages_publisher: _PagesPublisherLike | None
    benchmark_service: _BenchmarkServiceLike
    log_prefix: str


async def attach_performance_metrics(
    spec: dict[str, Any],
    ctx: PerformanceMetricsContext,
) -> None:
    """Aggregate performance and efficiency metrics onto the spec, and persist them for the daily trend.

    When there is no valid sample (latency / TTFT / Token / ReAct are all missing), no spec
    key is written, and the report skips the whole column. A persist failure only degrades
    to a warning: the trend is extra information and must not block report generation.
    """
    perf = summarize_performance_metrics(
        ctx.case_results,
        react_rounds_summary=spec.get("react_rounds_summary"),
    )
    if not perf:
        return
    spec["performance_metrics"] = perf
    merge = getattr(ctx.benchmark_service, "merge_performance_metrics", None)
    if not callable(merge):
        return
    try:
        await merge(ctx.run_id, performance_metrics=perf)
    except Exception:
        logger.warning(
            "%s.performance_metrics_persist_failed run_id=%s — 趋势将缺该日数据点",
            ctx.log_prefix,
            ctx.run_id,
            exc_info=True,
        )


async def _publish_pages_if_needed(ctx: PagesPublishContext) -> None:
    if not (ctx.html and ctx.report_url and ctx.pages_publisher is not None):
        return
    page_url = await ctx.pages_publisher.publish(ctx.run_id, ctx.html)
    if not page_url:
        # Business-level terminal conclusion: a missing Pages publish does not block run completion (the report still goes to OSS), but without this
        # log the only trail is an internal publisher warning after the fact (silent degradations such as a missing git binary
        # were missed that way). Search by run_id for "Pages publish did not succeed".
        logger.warning(
            "%s.report_site_publish_failed run_id=%s report_url=%s "
            "— Pages 推送未成功，报告仅 OSS 可达（具体原因见同 run 的 report_site_* warning）",
            ctx.log_prefix,
            ctx.run_id,
            ctx.report_url,
        )
        return
    await ctx.benchmark_service.merge_report_meta(
        ctx.run_id,
        patch={
            "report_site_url": page_url,
            "report_site_published_at": datetime.now(UTC).isoformat(),
        },
    )


async def generate_benchmark_report(
    job: ReportJobInput,
    *,
    webdev: _WebdevLike | None,
    pages_publisher: _PagesPublisherLike | None = None,
) -> str | None:
    """Unified report generation: assemble spec → dispatch → optional Pages; retry with exponential backoff on failure."""
    run_id, log_prefix = job.run_id, job.log_prefix
    for attempt in range(1, REPORT_MAX_ATTEMPTS + 1):
        try:
            spec = await assemble_report_spec(job)
            html, report_url = await dispatch_report_generation(
                spec=spec,
                webdev=webdev,
            )
            logger.info(
                "%s.report_generated run_id=%s html_len=%d report_url=%s",
                log_prefix,
                run_id,
                len(html) if html else 0,
                report_url,
            )
            await _publish_pages_if_needed(
                PagesPublishContext(
                    run_id=run_id,
                    html=html,
                    report_url=report_url,
                    pages_publisher=pages_publisher,
                    benchmark_service=job.benchmark_service,
                    log_prefix=log_prefix,
                ),
            )
            return report_url
        except Exception:
            if attempt >= REPORT_MAX_ATTEMPTS:
                logger.error(
                    "%s.report_failed run_id=%s attempts=%d",
                    log_prefix,
                    run_id,
                    attempt,
                    exc_info=True,
                )
                return None
            delay = min(
                REPORT_BACKOFF_BASE_S * (2 ** (attempt - 1)),
                REPORT_BACKOFF_MAX_S,
            )
            delay += random.uniform(0.0, delay * _REPORT_JITTER_RATIO)
            logger.warning(
                "%s.report_retry run_id=%s attempt=%d max_attempts=%d delay_s=%.2f",
                log_prefix,
                run_id,
                attempt,
                REPORT_MAX_ATTEMPTS,
                delay,
                exc_info=True,
            )
            await asyncio.sleep(delay)
    return None


__all__ = [
    "REPORT_BACKOFF_BASE_S",
    "REPORT_BACKOFF_MAX_S",
    "REPORT_MAX_ATTEMPTS",
    "PagesPublishContext",
    "PerformanceMetricsContext",
    "ReportJobInput",
    "ReportSpecFields",
    "assemble_report_spec",
    "attach_performance_metrics",
    "build_report_spec_dict",
    "count_distinct_failed_items",
    "generate_benchmark_report",
    "prefer_db_metrics",
    "resolve_rule_link_url",
]
