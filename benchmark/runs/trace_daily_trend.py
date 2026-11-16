"""Recent-N-day trend for the trace daily report (serves ``trace_daily_report`` only).

Other benchmarks still use ``load_daily_trend`` / ``benchmark_run_items``.
This module:

- The window ends on the data day inside run_id (T-1), not the wall clock of report day;
- Historical days prefer ``benchmark_runs.metrics_json.daily_trend_point``;
- With no cache, replay ``trace_pipeline_records`` for that day's window, then write the
  day point back onto the matching run so the next report does not scan row details.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.enums import BenchmarkStatus
from benchmark.domain.trace_report_config import amap_agent_legacy_daily_namespaces
from benchmark.runs import run_helpers as rh
from benchmark.runs.common.rubric_base import BaseRubric
from benchmark.runs.common.rubric_classes import (
    RubricClassResolutionError,
    resolve_rubric_class,
)
from benchmark.runs.registry import BENCHMARK_SERVICES_CLASSES
from benchmark.runs.trace_daily_report_adapter import build_weight_dimension_section
from benchmark.reporting.framework.performance_metrics import (
    PERF_TREND_KEYS,
    performance_trend_point,
    summarize_performance_metrics,
)
from benchmark.reporting.framework.schema import Criterion
from benchmark.plugins.class_loader import load_plugin_class

logger = get_logger("agent_eval.benchmark.trace_daily_trend")

_TREND_KEYS: tuple[str, ...] = (
    "fused_avg",
    "excellent_good_rate",
    "poor_fail_rate",
    "eval_success_rate",
    "task_satisfaction_rate",
    *PERF_TREND_KEYS,
)
_SHANGHAI = timezone(timedelta(hours=8))
_DEFAULT_PASS_THRESHOLD = 0.5
_YMD_LEN = 8


class TraceTrendService(Protocol):
    """BenchmarkService methods the daily-report trend needs to call."""

    def _build_today_trend_point(self, spec: dict[str, Any]) -> dict[str, Any] | None:
        """Build today's trend point from the current spec."""
        ...

    async def sample_trace_pipeline_records(
        self,
        query: rh.TraceRecordSampleQuery,
    ) -> list[str]:
        """Sample trace_id values inside the window."""
        ...

    async def load_trace_daily_case_results(
        self,
        *,
        trace_ids: list[str],
        rubric: list[Criterion],
        pass_threshold: float,
    ) -> list[dict[str, Any]]:
        """Pack trace rows into daily-report cases."""
        ...

    async def _patch_metrics(self, run_id: str, patch: dict[str, Any]) -> None:
        """Merge-write metrics_json."""
        ...


@dataclass(frozen=True, slots=True)
class TraceDailySampleRequest:
    """Inputs needed to freeze the sampling rule at report time."""

    source: str
    timezone_name: str
    pass_threshold: float
    rubric: list[Any]
    agent_name: str | None = None
    project_id: str | None = None
    sample_size: int | None = None
    registration_id: str | None = None


@dataclass(frozen=True, slots=True)
class TraceTrendRunLookup:
    """Fetch historical daily-report runs by day (FUNC-PARAMS-001: namespace + optional legacy namespace fallback)."""

    report_namespace: str
    ymd: str
    legacy_namespaces: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceHistoricalPointQuery:
    """Query spec for assembling one historical day's trend point (FUNC-PARAMS-001)."""

    day: date
    report_namespace: str
    sample: dict[str, Any] | None
    legacy_namespaces: tuple[str, ...] = ()


def trace_daily_sample_snapshot(request: TraceDailySampleRequest) -> dict[str, Any]:
    """Sampling condition frozen at report time, so later days replay the same trace window."""
    from benchmark.domain.eval_config import sample_evaluator_names_for_report  # noqa: PLC0415

    criteria: list[dict[str, Any]] = []
    rubric_refs: list[str] = []
    for item in request.rubric:
        evaluator_ref = str(getattr(item, "evaluator_ref", "") or "").strip()
        criterion_id = str(getattr(item, "id", "") or "").strip()
        if not evaluator_ref or not criterion_id:
            continue
        rubric_refs.append(evaluator_ref)
        weight = getattr(item, "weight", 0.0)
        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            weight_value = 0.0
        criteria.append(
            {
                "id": criterion_id,
                "name": str(getattr(item, "name", "") or criterion_id),
                "weight": weight_value,
                "evaluator_ref": evaluator_ref,
                "description": str(getattr(item, "description", "") or ""),
            }
        )
    evaluator_names = sample_evaluator_names_for_report(
        rubric_refs=rubric_refs,
        registration_id=request.registration_id,
    )
    snapshot: dict[str, Any] = {
        "source": request.source,
        "agent_name": request.agent_name,
        "project_id": request.project_id,
        "sample_size": request.sample_size,
        "timezone": request.timezone_name,
        "pass_threshold": request.pass_threshold,
        "evaluator_names": evaluator_names,
        "criteria": criteria,
    }
    reg = (request.registration_id or "").strip()
    if reg:
        snapshot["agent_registration_id"] = reg
    return snapshot


def parse_trace_daily_data_date(run_id: str, benchmark_name: str) -> date | None:
    """``{name}-{yyyymmdd}`` / ``{name}-{yyyymmdd}-{suffix}`` → data day."""
    prefix = f"{benchmark_name}-"
    if not run_id.startswith(prefix):
        return None
    rest = run_id[len(prefix) :]
    if len(rest) < _YMD_LEN or not rest[:_YMD_LEN].isdigit():
        return None
    if len(rest) > _YMD_LEN and rest[_YMD_LEN] != "-":
        return None
    try:
        return date(int(rest[:4]), int(rest[4:6]), int(rest[6:8]))
    except ValueError:
        return None


async def attach_trace_daily_trend(
    service: TraceTrendService,
    spec: dict[str, Any],
    *,
    benchmark_name: str,
    days: int,
) -> None:
    """Assemble the daily-report ``trend_7d`` / yesterday comparison. Does not read ``benchmark_run_items``.

    ``benchmark_name`` here is the report namespace (same as the run_id prefix).
    The logical component name comes from ``spec.benchmark_name``: only ``amap_agent_benchmark``
    falls back to the legacy namespace ``amap_agent_daily`` so historical points still connect.
    """
    run_id = str(spec.get("run_id") or "")
    data_date = parse_trace_daily_data_date(run_id, benchmark_name)
    if data_date is None:
        logger.warning(
            "trace_daily_trend.bad_run_id run_id=%s benchmark=%s",
            run_id,
            benchmark_name,
        )
        return
    logical_name = str(spec.get("benchmark_name") or "").strip()
    legacy_namespaces = amap_agent_legacy_daily_namespaces(logical_name)
    current = await _run_by_id(run_id)
    sample = resolve_trace_daily_sample(current) if current is not None else None
    today_blob = _blob_from_spec(service, spec)
    day_list = [data_date - timedelta(days=days - 1 - offset) for offset in range(days)]
    blobs: dict[date, dict[str, Any]] = {}
    for day in day_list:
        if day == data_date:
            blobs[day] = today_blob
            if current is not None and "fused_avg" in today_blob:
                await service._patch_metrics(run_id, {"daily_trend_point": today_blob})
            continue
        blobs[day] = await _historical_point(
            service,
            TraceHistoricalPointQuery(
                day=day,
                report_namespace=benchmark_name,
                sample=sample,
                legacy_namespaces=legacy_namespaces,
            ),
        )
    series = _series(day_list, blobs)
    if series is not None:
        spec["trend_7d"] = series
    _attach_yesterday(spec, blobs.get(data_date - timedelta(days=1)) or {})
    logger.info(
        "trace_daily_trend.attached benchmark=%s logical=%s legacy=%s data_date=%s points=%d",
        benchmark_name,
        logical_name or "-",
        ",".join(legacy_namespaces) or "-",
        data_date.isoformat(),
        len(series["dates"]) if series else 0,
    )


def resolve_trace_daily_sample(run: object) -> dict[str, Any] | None:
    """Sampling snapshot of the current run; an old run with no snapshot falls back to the rubric config."""
    config = run.config_json if isinstance(getattr(run, "config_json", None), dict) else {}
    snapshot = config.get("trace_daily_sample")
    if _sample_usable(snapshot):
        return snapshot
    return _fallback_sample(run, config)


async def _historical_point(
    service: TraceTrendService,
    query: TraceHistoricalPointQuery,
) -> dict[str, Any]:
    run = await _completed_run(
        TraceTrendRunLookup(
            report_namespace=query.report_namespace,
            ymd=query.day.strftime("%Y%m%d"),
            legacy_namespaces=query.legacy_namespaces,
        ),
    )
    cached = _cached_point(getattr(run, "metrics_json", None) if run is not None else None)
    if cached is not None:
        return cached
    if query.sample is None:
        blob = _blob_from_final_score(run)
    else:
        blob = await _point_from_traces(
            service,
            day=query.day,
            sample=query.sample,
            run=run,
        )
    if run is not None and blob:
        await service._patch_metrics(run.run_id, {"daily_trend_point": blob})
    return blob


async def _point_from_traces(
    service: TraceTrendService,
    *,
    day: date,
    sample: dict[str, Any],
    run: object | None,
) -> dict[str, Any]:
    criteria = _criteria_from_sample(sample)
    evaluator_names = [str(name) for name in sample.get("evaluator_names") or [] if name]
    registration_id = _optional_str(sample.get("agent_registration_id"))
    cases: list[dict[str, Any]] = []
    if criteria and evaluator_names:
        window_start, window_end = _day_window(day, _resolve_tz(sample.get("timezone")))
        trace_ids = await service.sample_trace_pipeline_records(
            rh.TraceRecordSampleQuery(
                date_start=window_start,
                date_end=window_end,
                evaluator_names=evaluator_names,
                source=str(sample.get("source") or "kafka"),
                agent_name=_optional_str(sample.get("agent_name")),
                project_id=_optional_str(sample.get("project_id")),
                sample_size=_optional_sample_size(sample.get("sample_size")),
            ),
        )
        if trace_ids:
            cases = await service.load_trace_daily_case_results(
                trace_ids=trace_ids,
                rubric=criteria,
                pass_threshold=_pass_threshold(sample.get("pass_threshold")),
                registration_id=registration_id,
            )
    return _blob_from_cases(service, cases, run=run, scanned=True)


def _blob_from_spec(service: TraceTrendService, spec: dict[str, Any]) -> dict[str, Any]:
    raw = service._build_today_trend_point(spec) or {}
    blob = _trend_fields(raw)
    cases = spec.get("case_results")
    if isinstance(cases, list):
        grades = _grade_distribution(cases)
        if grades:
            blob["grade_distribution"] = grades
    means = _means_from_criterion_metrics(spec.get("criterion_metrics"))
    if means:
        blob["criterion_means"] = means
    return blob


def _blob_from_cases(
    service: TraceTrendService,
    cases: list[dict[str, Any]],
    *,
    run: object | None,
    scanned: bool,
) -> dict[str, Any]:
    final_score = _final_score(run, cases)
    spec: dict[str, Any] = {"final_score": final_score, "case_results": cases}
    if cases:
        spec["total_samples"] = len(cases)
        spec["total_failed"] = _failed_count(cases)
    elif run is not None and isinstance(getattr(run, "total_samples", None), int):
        spec["total_samples"] = run.total_samples
    performance = _performance_metrics(run, cases)
    if performance:
        spec["performance_metrics"] = performance
    raw = service._build_today_trend_point(spec) or {}
    blob = _trend_fields(raw)
    if scanned:
        blob["trace_scanned"] = True
    grades = _grade_distribution(cases)
    if grades:
        blob["grade_distribution"] = grades
    means = _criterion_means(cases)
    if means:
        blob["criterion_means"] = means
    return blob


def _blob_from_final_score(run: object | None) -> dict[str, Any]:
    score = _numeric(getattr(run, "final_score", None) if run is not None else None)
    blob: dict[str, Any] = {}
    if score is not None:
        blob["fused_avg"] = round(score, 4)
    performance = _stored_performance(run)
    if performance:
        blob.update(performance_trend_point(performance))
    return blob


def _series(
    day_list: list[date],
    blobs: dict[date, dict[str, Any]],
) -> dict[str, Any] | None:
    series: dict[str, list[Any]] = {"dates": [], **{key: [] for key in _TREND_KEYS}}
    has_any = False
    for day in day_list:
        series["dates"].append(day.strftime("%m-%d"))
        blob = blobs.get(day) or {}
        for key in _TREND_KEYS:
            value = blob.get(key)
            if _numeric(value) is None:
                series[key].append(None)
                continue
            series[key].append(value)
            has_any = True
    if not has_any:
        return None
    return series


def _attach_yesterday(spec: dict[str, Any], blob: dict[str, Any]) -> None:
    means = blob.get("criterion_means")
    if isinstance(means, dict) and means:
        parsed: dict[str, float] = {}
        for key, value in means.items():
            number = _numeric(value)
            if number is not None and str(key):
                parsed[str(key)] = round(number, 4)
        if parsed:
            spec["yesterday_criterion_means"] = parsed
    grades = blob.get("grade_distribution")
    if isinstance(grades, dict) and grades:
        normalized = _normalize_grades(grades)
        if normalized:
            spec["yesterday_grade_distribution"] = normalized


def _normalize_grades(raw: dict[str, Any]) -> dict[str, int]:
    """Normalize grade counts, same rules as ``BenchmarkRunMetrics.normalize_grade_distribution``."""
    out: dict[str, int] = {}
    for key, val in raw.items():
        label = str(key or "").strip()
        if not label or isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            out[label] = int(round(float(val)))
            continue
        if isinstance(val, str):
            try:
                out[label] = int(round(float(val.strip())))
            except ValueError:
                continue
    return out


def _trend_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in _TREND_KEYS if key in raw and raw[key] is not None}


def _cached_point(metrics_json: object) -> dict[str, Any] | None:
    if not isinstance(metrics_json, dict):
        return None
    point = metrics_json.get("daily_trend_point")
    if not isinstance(point, dict):
        return None
    if "fused_avg" in point or point.get("trace_scanned") is True:
        return point
    return None


def _sample_usable(snapshot: object) -> bool:
    return (
        isinstance(snapshot, dict)
        and bool(snapshot.get("evaluator_names"))
        and bool(snapshot.get("criteria"))
    )


def _fallback_sample(run: object, config: dict[str, Any]) -> dict[str, Any] | None:
    resolved = _resolve_rubric(config)
    if resolved is None:
        return None
    rubric, pass_threshold = resolved
    return trace_daily_sample_snapshot(
        TraceDailySampleRequest(
            source=str(config.get("source") or "kafka"),
            timezone_name="Asia/Shanghai",
            pass_threshold=pass_threshold,
            rubric=rubric,
            project_id=_optional_str(getattr(run, "project_id", None)),
        )
    )


def _resolve_rubric(config: dict[str, Any]) -> tuple[list[Any], float] | None:
    """When an old run has no sampling snapshot, restore the sampling rule from the rubric ref in config."""
    rubric_ref = config.get("rubric_ref")
    if isinstance(rubric_ref, str) and rubric_ref.strip():
        return _rubric_from_class(rubric_ref.strip())
    component = str(config.get("benchmark_component") or "").strip()
    if not component:
        return None
    return _rubric_from_component(component)


def _rubric_from_class(alias: str) -> tuple[list[Any], float] | None:
    try:
        spec = resolve_rubric_class(alias)
    except RubricClassResolutionError:
        return None
    if spec.class_ref is None:
        return None
    rubric_cls = load_plugin_class(spec.class_ref, what="rubric_class")
    if not isinstance(rubric_cls, type) or not issubclass(rubric_cls, BaseRubric):
        return None
    threshold = float(rubric_cls.default_aggregation().pass_threshold)
    return rubric_cls.criteria(), threshold


def _rubric_from_component(component: str) -> tuple[list[Any], float] | None:
    for cls in BENCHMARK_SERVICES_CLASSES:
        meta = cls.get_metadata()
        if meta.name != component:
            continue
        threshold = float(getattr(meta.aggregation, "pass_threshold", _DEFAULT_PASS_THRESHOLD))
        return list(meta.rubric), threshold
    return None


def _criteria_from_sample(sample: dict[str, Any]) -> list[Criterion]:
    criteria: list[Criterion] = []
    for raw in sample.get("criteria") or []:
        if not isinstance(raw, dict):
            continue
        try:
            criteria.append(
                Criterion(
                    id=str(raw["id"]),
                    name=str(raw.get("name") or raw["id"]),
                    weight=float(raw.get("weight") or 0.0),
                    evaluator_ref=str(raw["evaluator_ref"]),
                    description=str(raw.get("description") or ""),
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return criteria


def _final_score(run: object | None, cases: list[dict[str, Any]]) -> float | None:
    stored = _numeric(getattr(run, "final_score", None) if run is not None else None)
    if stored is not None:
        return stored
    fused = _fused_scores(cases)
    if not fused:
        return None
    return round(sum(fused) / len(fused), 4)


def _fused_scores(cases: list[dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    for case in cases:
        bucket = case.get("scores")
        fused = bucket.get("fused") if isinstance(bucket, dict) else None
        number = _numeric(fused.get("score") if isinstance(fused, dict) else None)
        if number is not None:
            scores.append(number)
    return scores


def _failed_count(cases: list[dict[str, Any]]) -> int:
    failed = 0
    for case in cases:
        bucket = case.get("scores")
        fused = bucket.get("fused") if isinstance(bucket, dict) else None
        if not isinstance(fused, dict) or fused.get("eval_failed") is True:
            failed += 1
            continue
        if _numeric(fused.get("score")) is None:
            failed += 1
    return failed


def _grade_distribution(cases: list[dict[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        grade = case.get("grade")
        if isinstance(grade, str) and grade:
            distribution[grade] = distribution.get(grade, 0) + 1
    return distribution


def _criterion_means(cases: list[dict[str, Any]]) -> dict[str, float]:
    section = build_weight_dimension_section(cases)
    if section is None:
        return {}
    return _means_from_criterion_metrics(section.criterion_metrics)


def _means_from_criterion_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    means: dict[str, float] = {}
    for key, stats in metrics.items():
        raw = stats.get("mean") if isinstance(stats, dict) else getattr(stats, "mean", None)
        number = _numeric(raw)
        if number is not None and str(key):
            means[str(key)] = round(number, 4)
    return means


def _performance_metrics(
    run: object | None,
    cases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    stored = _stored_performance(run)
    if stored:
        return stored
    summarized = summarize_performance_metrics(cases)
    return summarized or None


def _stored_performance(run: object | None) -> dict[str, Any] | None:
    metrics = getattr(run, "metrics_json", None) if run is not None else None
    if not isinstance(metrics, dict):
        return None
    performance = metrics.get("performance_metrics")
    if isinstance(performance, dict) and performance:
        return performance
    return None


def _day_window(day: date, tz: tzinfo) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    return start, start + timedelta(days=1)


def _resolve_tz(name: object) -> tzinfo:
    text = str(name or "").strip() or "Asia/Shanghai"
    try:
        return ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError):
        return _SHANGHAI


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_sample_size(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _pass_threshold(value: object) -> float:
    number = _numeric(value)
    if number is None:
        return _DEFAULT_PASS_THRESHOLD
    return number


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


async def _run_by_id(run_id: str) -> object | None:
    if not run_id:
        return None
    return await rh.BenchmarkRun.filter(run_id=run_id).first()


async def _completed_run(lookup: TraceTrendRunLookup) -> object | None:
    """Load a historical completed daily report by run_id.

    Look up ``{report_namespace}-{ymd}`` first; on a miss, try ``legacy_namespaces`` in order
    (amap_agent_benchmark → ``amap_agent_daily``).
    ``report_namespace`` is the report item name (same as the run_id prefix), not the
    ``benchmark_name`` column (that column stores the internal agent registration name).
    """
    namespaces = (lookup.report_namespace, *lookup.legacy_namespaces)
    seen: set[str] = set()
    for namespace in namespaces:
        ns = (namespace or "").strip()
        if not ns or ns in seen:
            continue
        seen.add(ns)
        run = (
            await rh.BenchmarkRun.filter(
                status=BenchmarkStatus.COMPLETED,
                run_id=f"{ns}-{lookup.ymd}",
            )
            .order_by("-created_at")
            .first()
        )
        if run is not None:
            return run
    return None


__all__ = [
    "attach_trace_daily_trend",
    "parse_trace_daily_data_date",
    "resolve_trace_daily_sample",
    "trace_daily_sample_snapshot",
]
