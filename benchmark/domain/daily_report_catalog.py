"""Today's Pages report catalog: latest realtime / batch reports from ``benchmark_runs``.

For ops / downstream callers that pull Aone Pages reports updated today. Semantics:

- First keep completed runs whose ``updated_at`` falls in the Shanghai calendar day;
- **batch**: daily runs whose ``run_id`` ends in ``-YYYYMMDD`` (the date may be today, yesterday, or n days ago);
- **realtime**: unchanged (normal T+1; ``*-stopped-*`` is not listed);
- Each key keeps only the latest row, with that run's overall score and rubric stats.
- Every request scans the full set. There is no cache.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field

from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun
from benchmark.domain.agent_registration_store import (
    find_agent_scope_by_registration_id,
    list_active_trace_report_registration_ids,
)
from benchmark.domain.eval_config import benchmark_run_id_stem

logger = get_logger(__name__)

#: UTC+8 (same timezone as the daily schedule / T+1 report).
_SH_TZ = timezone(timedelta(hours=8))

#: DB-RDB-003: a single query must return fewer than 10000 rows.
_LIST_LIMIT = 9999

#: Legacy default eval target for live registrations that never bound a scope.
_LEGACY_UNBOUND_SCOPE = "amap-agent"

#: Public short name for batch: ``ieval_external`` → ``ieval``.
_EXTERNAL_SUFFIX = "_external"

#: ``run_id`` ending in ``-yyyymmdd`` (batch daily run; excludes ``-m*`` / ``-s*`` / ``-stopped-*``).
_DATE_ENDED_RUN_ID_RE = re.compile(r"-(\d{8})$")

#: Manually stopped live-stream window: ``{stem}-{yyyymmdd}-stopped-{HHmmss}`` (see ``streaming_stopped_run_id``).
_STOPPED_WINDOW_RUN_ID_RE = re.compile(r"-stopped-\d{6}(?:$|[-_])")

#: Same as Agent Park RubricBreakdown: pass_rate ≥ 0.8 Healthy, ≥ 0.5 Watch, otherwise Degraded.
_PASS_RATE_HEALTHY = 0.8
_PASS_RATE_WATCH = 0.5
STATUS_HEALTHY = "Healthy"
STATUS_WATCH = "Watch"
STATUS_DEGRADED = "Degraded"


class ReportStats(BaseModel):
    """One-dimension stats (field names follow the public contract, not the CriterionStats names).

    Domain and API share this model so TypedDict / BaseModel / hand-built dict do not drift.
    ``overall`` does not reuse it: the overall score exposes only ``current`` (see ``OverallScore``).
    """

    current: float | None = None
    mean: float | None = None
    median: float | None = None
    p95: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    pass_rate: float | None = None
    success: int | None = None
    failed: int | None = None
    samples: int | None = None
    evaluator: str | None = None


class OverallScore(BaseModel):
    """Overall-score snapshot for one catalog item.

    - ``current``: ``final_score``, falling back to fused
    - ``success``: ``evaluated_items`` (not above ``samples``)
    - ``failed``: ``failed_items``
    - ``samples``: ``success + failed`` (identity)
    - ``evaluator``: fused / service_name / the first rubric

    Does not return mean / median / p95 / min / max / std / pass_rate.
    """

    current: float | None = None
    success: int | None = None
    failed: int | None = None
    samples: int | None = None
    evaluator: str | None = None


class RubricStat(ReportStats):
    """One rubric dimension: the definition comes from ``config_json.rubric``, the stats from ``criterion_metrics``."""

    id: str
    name: str
    description: str = ""
    weight: float = 0.0
    status: str | None = None


@dataclass(frozen=True, slots=True)
class DayWindow:
    """上海日历日闭开区间 ``[start, end)``（供 ORM 过滤）。"""

    start: datetime
    end: datetime
    ymd: str


@dataclass(frozen=True, slots=True)
class ReportEntry:
    """单条已解析的 Pages 报告（中间态，组装前）。"""

    key: str
    url: str
    run_id: str
    finished_at: datetime | None
    batch_rank: int = 0
    #: 选出后用来拼 overall / rubrics；测试替身与 ORM 行均可
    source: object = None
    updated_at: datetime | None = None


class DailyReportLink(BaseModel):
    """某个 key 当天最新一条报告及其评分快照。"""

    url: str
    run_id: str
    overall: OverallScore
    rubrics: list[RubricStat] = Field(default_factory=list)


class DailyReportCatalog(BaseModel):
    """对外目录：realtime / batch 两组 key → 最新报告快照。"""

    realtime: dict[str, DailyReportLink] = Field(default_factory=dict)
    batch: dict[str, DailyReportLink] = Field(default_factory=dict)


class CatalogRunView(Protocol):
    """组装目录所需的 run 视图（ORM 行或测试替身）。

    ``metrics_json`` / ``final_score`` 用于评分快照；旧替身缺这些属性时按空统计处理。
    ``updated_at`` 用于「今日有更新」窗口筛选；缺失时回退 ``finished_at`` /
    ``created_at``（测试替身兼容）。
    """

    run_id: str
    benchmark_name: str
    finished_at: datetime | None
    created_at: datetime | None
    report_meta_json: object
    config_json: object
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class CatalogBuildOptions:
    """``build_daily_report_catalog`` 入参（FUNC-PARAMS-001 收敛）。"""

    now: datetime | None = None
    #: 测试注入：跳过 DB，直接喂入已完成 run 列表
    runs: Sequence[CatalogRunView] | None = None
    #: 测试注入：活跃评测对象 scope 集合；``None`` = 运行时探测
    active_scopes: set[str] | None = None
    #: 测试注入：registration_id → agent_scope
    scope_by_registration: dict[str, str] | None = None


def shanghai_day_window(now: datetime | None = None) -> DayWindow:
    """当前东八区日历日的 ``[00:00, +1day)`` 窗口与 ``yyyymmdd``。"""
    clock = now if now is not None else datetime.now(_SH_TZ)
    local = clock.astimezone(_SH_TZ)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return DayWindow(start=start, end=end, ymd=start.strftime("%Y%m%d"))


def pages_report_url(meta: object) -> str | None:
    """优先取 Pages 正式入口 ``report_site_url``；缺失则不算当日目录项。"""
    if not isinstance(meta, dict):
        return None
    url = meta.get("report_site_url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def is_realtime_daily_run(config: object) -> bool:
    """``config_json.trace_daily_report is True`` 标识 T+1 实时日报。"""
    return isinstance(config, dict) and config.get("trace_daily_report") is True


def is_streaming_stopped_run_id(run_id: object) -> bool:
    """是否人工停止窗口报告（与正常 T+1 ``{stem}-{ymd}`` 区分，不进本目录）。"""
    text = str(run_id or "")
    if not text:
        return False
    if _STOPPED_WINDOW_RUN_ID_RE.search(text):
        return True
    # 宽松兜底：历史/变体命名含 ``-stopped-`` 段也不进目录
    return "-stopped-" in text


def is_date_ended_run_id(run_id: object, *, ymd: str | None = None) -> bool:
    """``run_id`` 是否以 ``-yyyymmdd`` 结尾；可选要求等于目录日 ``ymd``。"""
    text = str(run_id or "")
    matched = _DATE_ENDED_RUN_ID_RE.search(text)
    if matched is None:
        return False
    if ymd is not None and matched.group(1) != ymd:
        return False
    return True


def batch_catalog_key(benchmark_name: object) -> str:
    """batch 对外 key：茎名；``*_external`` 去掉后缀（``ieval_external``→``ieval``）。"""
    stem = benchmark_run_id_stem(benchmark_name)
    if not stem:
        return ""
    if stem.endswith(_EXTERNAL_SUFFIX) and len(stem) > len(_EXTERNAL_SUFFIX):
        return stem[: -len(_EXTERNAL_SUFFIX)]
    return stem


def batch_run_rank(run_id: str, stem: str, ymd: str) -> int:
    """同 key 多条时优选：今日精确日跑(0) > 其它 ``-{yyyymmdd}`` 日跑(1) > 其他(2)。

    ``yyyymmdd`` 可为今天 / 昨天 / 前 n 天；只要 ``run_id`` 以 ``-YYYYMMDD``
    结尾即进 batch 候选（由 ``updated_at`` 当日窗口先行筛选）。
    """
    if not stem or not ymd:
        return 2
    exact = f"{stem}-{ymd}"
    if run_id == exact:
        return 0
    if is_date_ended_run_id(run_id):
        return 1
    return 2


def rubric_health_status(pass_rate: float | None) -> str | None:
    """pass_rate → Healthy / Watch / Degraded。缺失则不算状态。"""
    if pass_rate is None:
        return None
    if pass_rate >= _PASS_RATE_HEALTHY:
        return STATUS_HEALTHY
    if pass_rate >= _PASS_RATE_WATCH:
        return STATUS_WATCH
    return STATUS_DEGRADED


def _finite_number(value: object) -> float | None:
    """int/float → float；bool / NaN / 非数字丢弃。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _whole_number(value: object) -> int | None:
    number = _finite_number(value)
    if number is None:
        return None
    return int(number)


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    return None


def _rubric_definitions(config: object) -> list[dict[str, object]]:
    """Rubric 定义：``config_json.rubric``，其次 ``criteria``，再次 T+1 采样快照。"""
    raw: object = None
    if isinstance(config, dict):
        raw = config.get("rubric")
        if not isinstance(raw, list) or not raw:
            raw = config.get("criteria")
        if not isinstance(raw, list) or not raw:
            sample = config.get("trace_daily_sample")
            if isinstance(sample, dict):
                raw = sample.get("criteria")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _criterion_metrics(metrics: object) -> dict[str, dict[str, object]]:
    blob = _mapping(metrics)
    if blob is None:
        return {}
    raw = blob.get("criterion_metrics")
    if not isinstance(raw, dict):
        return {}
    return {str(key): row for key, row in raw.items() if isinstance(row, dict)}


def _stats_from_metric(metric: Mapping[str, object] | None) -> ReportStats:
    """CriterionStats 落库字段 → 对外统计名。``current`` 先空着，由调用方填。"""
    row: Mapping[str, object] = metric or {}
    evaluator = row.get("evaluator_ref")
    return ReportStats(
        current=None,
        mean=_finite_number(row.get("mean")),
        median=_finite_number(row.get("median")),
        p95=_finite_number(row.get("p95")),
        min=_finite_number(row.get("min")),
        max=_finite_number(row.get("max")),
        std=_finite_number(row.get("std")),
        pass_rate=_finite_number(row.get("pass_rate")),
        success=_whole_number(row.get("success_count")),
        failed=_whole_number(row.get("failed_count")),
        samples=_whole_number(row.get("sample_count")),
        evaluator=evaluator if isinstance(evaluator, str) and evaluator else None,
    )


def _fused_metric(metrics: object) -> dict[str, object] | None:
    return _criterion_metrics(metrics).get("fused")


def _overall_current(run: object, fused: Mapping[str, object] | None) -> float | None:
    """``final_score`` 优先；没有则用 fused 维均值或趋势点 ``fused_avg``。"""
    current = _finite_number(getattr(run, "final_score", None))
    if current is not None:
        return current
    if fused is not None:
        fused_mean = _finite_number(fused.get("mean"))
        if fused_mean is not None:
            return fused_mean
    metrics = _mapping(getattr(run, "metrics_json", None))
    if metrics is None:
        return None
    point = metrics.get("daily_trend_point")
    if isinstance(point, dict):
        return _finite_number(point.get("fused_avg"))
    return None


def _non_negative_int(value: object) -> int:
    number = _whole_number(value)
    if number is None or number < 0:
        return 0
    return number


def _overall_sample_cap(run: object, fused: Mapping[str, object] | None) -> int | None:
    """success 上限：``total_samples`` → metrics → fused ``sample_count``。"""
    cap = _whole_number(getattr(run, "total_samples", None))
    if cap is not None:
        return max(cap, 0)
    metrics = _mapping(getattr(run, "metrics_json", None))
    if metrics is not None:
        total = _whole_number(metrics.get("total_samples"))
        if total is not None:
            return max(total, 0)
    if fused is not None:
        fused_samples = _whole_number(fused.get("sample_count"))
        if fused_samples is not None:
            return max(fused_samples, 0)
    return None


def _overall_counts(
    run: object,
    fused: Mapping[str, object] | None,
) -> tuple[int, int, int]:
    """返回 ``(success, failed, samples)``，满足 ``success + failed == samples``。

    ``success`` = ``evaluated_items``（不超过 sample cap）；``failed`` = ``failed_items``；
    ``samples`` = ``success + failed``。
    """
    success = _non_negative_int(getattr(run, "evaluated_items", None))
    failed = _non_negative_int(getattr(run, "failed_items", None))
    cap = _overall_sample_cap(run, fused)
    if cap is not None:
        success = min(success, cap)
    samples = success + failed
    return success, failed, samples


def _overall_evaluator(
    run: object,
    fused: Mapping[str, object] | None,
    rubrics: Sequence[RubricStat],
) -> str | None:
    if fused is not None:
        evaluator = fused.get("evaluator_ref")
        if isinstance(evaluator, str) and evaluator.strip():
            return evaluator.strip()
    service_name = _text(getattr(run, "service_name", None))
    if service_name is not None:
        return service_name
    if rubrics:
        return rubrics[0].evaluator
    return None


def build_report_snapshot(run: object, *, url: str, run_id: str) -> DailyReportLink:
    """把选中 run 的 rubric 定义与 ``criterion_metrics`` 拼成一条目录项。

    关联方式与 Agent Park RubricBreakdown 相同：按 rubric ``id`` 取
    ``metrics_json.criterion_metrics[id]``。weight=0 的维度也保留。
    ``overall`` 含 ``current`` / ``success`` / ``failed`` / ``samples`` / ``evaluator``，
    不含分布统计。
    """
    config = getattr(run, "config_json", None)
    metrics = getattr(run, "metrics_json", None)
    by_id = _criterion_metrics(metrics)
    definitions = _rubric_definitions(config)
    rubrics: list[RubricStat] = []
    for item in definitions:
        criterion_id = _text(item.get("id"))
        if criterion_id is None:
            continue
        metric = by_id.get(criterion_id)
        stats = _stats_from_metric(metric)
        evaluator = stats.evaluator or _text(item.get("evaluator_ref"))
        weight = _finite_number(item.get("weight"))
        name = _text(item.get("name")) or criterion_id
        description = item.get("description")
        rubrics.append(
            RubricStat(
                id=criterion_id,
                name=name,
                description=description if isinstance(description, str) else "",
                weight=0.0 if weight is None else weight,
                status=rubric_health_status(stats.pass_rate),
                current=stats.mean,
                mean=stats.mean,
                median=stats.median,
                p95=stats.p95,
                min=stats.min,
                max=stats.max,
                std=stats.std,
                pass_rate=stats.pass_rate,
                success=stats.success,
                failed=stats.failed,
                samples=stats.samples,
                evaluator=evaluator,
            )
        )

    fused = _fused_metric(metrics)
    success, failed, samples = _overall_counts(run, fused)
    return DailyReportLink(
        url=url,
        run_id=run_id,
        overall=OverallScore(
            current=_overall_current(run, fused),
            success=success,
            failed=failed,
            samples=samples,
            evaluator=_overall_evaluator(run, fused, rubrics),
        ),
        rubrics=rubrics,
    )


def _entry_timestamp(entry: ReportEntry) -> datetime:
    return entry.finished_at or datetime.min.replace(tzinfo=_SH_TZ)


def _pick_newer(current: ReportEntry | None, candidate: ReportEntry) -> ReportEntry:
    """同 key 保留 ``finished_at`` 更新的一条；时间相同保留先看到的。"""
    if current is None:
        return candidate
    if _entry_timestamp(candidate) > _entry_timestamp(current):
        return candidate
    return current


def _registration_id_from_config(config: dict[str, object]) -> str | None:
    reg = str(config.get("agent_registration_id") or "").strip()
    if reg:
        return reg
    snap = config.get("agent_registration")
    if isinstance(snap, dict):
        rid = str(snap.get("id") or "").strip()
        if rid:
            return rid
    return None


def _scope_from_config_snapshot(config: dict[str, object]) -> str | None:
    snap = config.get("agent_registration")
    if not isinstance(snap, dict):
        return None
    scope = snap.get("agent_scope") or snap.get("agentScope")
    if isinstance(scope, str) and scope.strip():
        return scope.strip()
    return None


async def _resolve_realtime_scope(
    config: dict[str, object],
    *,
    scope_by_registration: dict[str, str] | None,
) -> str:
    """实时报告 key：优先快照 / 注入表 / 注册文档上的 agent_scope。"""
    from_snap = _scope_from_config_snapshot(config)
    if from_snap:
        return from_snap
    reg = _registration_id_from_config(config)
    if reg and scope_by_registration is not None:
        mapped = scope_by_registration.get(reg)
        if mapped:
            return mapped
    if reg:
        scope = await find_agent_scope_by_registration_id(reg)
        if scope:
            return scope
    return _LEGACY_UNBOUND_SCOPE


async def _detect_active_evaluator_scopes() -> set[str]:
    """自动探测当前仍存在的实时流评测对象（有 T+1 报表项的注册 → scope）。

    数据源：``kind=trace_report`` 活跃项 + 注册文档 ``agent_scope``。
    无登记项时返回空集——调用方对空集不做过滤（仍返回当日已有 realtime 报）。
    """
    registration_ids = await list_active_trace_report_registration_ids()
    scopes: set[str] = set()
    for reg in registration_ids:
        if not reg:
            scopes.add(_LEGACY_UNBOUND_SCOPE)
            continue
        scope = await find_agent_scope_by_registration_id(reg)
        scopes.add(scope or _LEGACY_UNBOUND_SCOPE)
    return scopes


def _run_updated_at(run: object) -> datetime | None:
    """读取 run 的 ``updated_at``；测试替身缺失时回退 finished/created。"""
    value = getattr(run, "updated_at", None)
    if isinstance(value, datetime):
        return value
    finished = getattr(run, "finished_at", None)
    if isinstance(finished, datetime):
        return finished
    created = getattr(run, "created_at", None)
    if isinstance(created, datetime):
        return created
    return None


async def _load_completed_runs_for_day(window: DayWindow) -> list[BenchmarkRun]:
    """拉取上海当日 ``updated_at`` 窗口内已完成的全部 run。"""
    runs = (
        await BenchmarkRun.filter(
            deleted_at__isnull=True,
            status=BenchmarkStatus.COMPLETED,
            updated_at__gte=window.start,
            updated_at__lt=window.end,
        )
        .order_by("-updated_at", "-finished_at", "-created_at")
        .limit(_LIST_LIMIT)
    )
    if len(runs) >= _LIST_LIMIT:
        logger.warning(
            "daily_report_catalog run list hit limit=%d date=%s; excess runs truncated",
            _LIST_LIMIT,
            window.ymd,
        )
    return runs


def _pick_better_batch(current: ReportEntry | None, candidate: ReportEntry) -> ReportEntry:
    """同 batch key 保留更优条目（rank 更小，其次 updated_at 更新）。"""
    if current is None:
        return candidate
    if candidate.batch_rank < current.batch_rank:
        return candidate
    if candidate.batch_rank > current.batch_rank:
        return current
    cur_ts = (
        current.updated_at
        or current.finished_at
        or datetime.min.replace(tzinfo=_SH_TZ)
    )
    cand_ts = (
        candidate.updated_at
        or candidate.finished_at
        or datetime.min.replace(tzinfo=_SH_TZ)
    )
    if cand_ts >= cur_ts:
        return candidate
    return current


async def build_daily_report_catalog(
    options: CatalogBuildOptions | None = None,
) -> DailyReportCatalog:
    """组装当天报告目录。

    - 先按 ``updated_at`` 落在上海当日窗口筛选 completed run；
    - realtime：规则不变（仅正常 T+1；``*-stopped-*`` 不收录；scope 探测）；
    - batch：``run_id`` 以 ``-YYYYMMDD`` 结尾的全部日跑（日期可为今天 /
      昨天 / 前 n 天；不含 ``-m*``/``-s*`` 等后缀）；
    - 同 key 多条时 realtime 取 ``finished_at`` 最新，batch 按日跑 rank +
      ``updated_at`` 优选；
    - 选中 run 附带 ``final_score`` / fused 统计与全部 rubric 维度。
    """
    opts = options or CatalogBuildOptions()
    window = shanghai_day_window(opts.now)
    runs = opts.runs if opts.runs is not None else await _load_completed_runs_for_day(window)
    active_scopes = (
        opts.active_scopes
        if opts.active_scopes is not None
        else await _detect_active_evaluator_scopes()
    )

    realtime_best: dict[str, ReportEntry] = {}
    batch_best: dict[str, ReportEntry] = {}

    for run in runs:
        url = pages_report_url(run.report_meta_json)
        if not url:
            continue
        # 人工停止窗口与 T+1 / 批量日跑命名区分，本接口一律不返回
        if is_streaming_stopped_run_id(run.run_id):
            continue
        config = run.config_json if isinstance(run.config_json, dict) else {}
        updated_at = _run_updated_at(run)
        if is_realtime_daily_run(config):
            scope = await _resolve_realtime_scope(
                config,
                scope_by_registration=opts.scope_by_registration,
            )
            if active_scopes and scope not in active_scopes:
                continue
            entry = ReportEntry(
                key=scope,
                url=url,
                run_id=run.run_id,
                finished_at=run.finished_at or run.created_at,
                source=run,
                updated_at=updated_at,
            )
            realtime_best[scope] = _pick_newer(realtime_best.get(scope), entry)
            continue

        # 批量：以 -YYYYMMDD 结尾即可（日期不限当天）
        if not is_date_ended_run_id(run.run_id):
            continue
        stem = benchmark_run_id_stem(run.benchmark_name)
        key = batch_catalog_key(run.benchmark_name)
        if not key:
            continue
        entry = ReportEntry(
            key=key,
            url=url,
            run_id=run.run_id,
            finished_at=run.finished_at or run.created_at,
            batch_rank=batch_run_rank(run.run_id, stem, window.ymd),
            source=run,
            updated_at=updated_at,
        )
        batch_best[key] = _pick_better_batch(batch_best.get(key), entry)

    catalog = DailyReportCatalog(
        realtime={
            key: build_report_snapshot(entry.source, url=entry.url, run_id=entry.run_id)
            for key, entry in realtime_best.items()
            if entry.source is not None
        },
        batch={
            key: build_report_snapshot(entry.source, url=entry.url, run_id=entry.run_id)
            for key, entry in sorted(batch_best.items())
            if entry.source is not None
        },
    )
    logger.info(
        "daily_report_catalog built date=%s realtime=%d batch=%d runs_scanned=%d",
        window.ymd,
        len(catalog.realtime),
        len(catalog.batch),
        len(runs),
    )
    return catalog


__all__ = [
    "CatalogBuildOptions",
    "CatalogRunView",
    "DailyReportCatalog",
    "DailyReportLink",
    "DayWindow",
    "OverallScore",
    "ReportEntry",
    "ReportStats",
    "RubricStat",
    "STATUS_DEGRADED",
    "STATUS_HEALTHY",
    "STATUS_WATCH",
    "batch_catalog_key",
    "batch_run_rank",
    "build_daily_report_catalog",
    "build_report_snapshot",
    "is_date_ended_run_id",
    "is_realtime_daily_run",
    "is_streaming_stopped_run_id",
    "pages_report_url",
    "rubric_health_status",
    "shanghai_day_window",
]
