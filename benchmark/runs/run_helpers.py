"""Shared helpers / dataclasses for BenchmarkService mixins.

Extracted from :mod:`run_service` so lifecycle / metrics / case-query /
external-item modules can share pure functions without circular imports.
Public re-exports remain on :mod:`benchmark.runs.run_service`.
"""

import json
import os
import re
import secrets
import socket
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from benchmark.libs.agenthost.observability.logger import get_logger
from tortoise.expressions import F
from tortoise.transactions import in_transaction

from benchmark.records.models import (
    BenchmarkRun,
    BenchmarkRunItem,
    TraceRecord,
)
from benchmark.domain.eval_config import benchmark_run_id_stem
from benchmark.runs.common.dimension_stats import (
    resolve_run_stat_dimensions,
)
from business.evaluators.example.grading import (
    grade_fused_score,
)
from benchmark.runs.run_stop import is_stop_requested
from benchmark.runs.schema import ExternalJudgeItemResult
from benchmark.reporting.framework.schema import AggregatedMetrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.reporting.framework.schema import Criterion

logger = get_logger("agent_eval.benchmark.run_helpers")

# UTC+8 — 与 TraceRecordService bucket stamp 对齐
_UTC_PLUS_8 = timezone(timedelta(hours=8))

#: ``run_id`` / job id 最大长度（与 ORM CharField(128) 对齐）。
_JOB_ID_MAX_LEN = 128
#: load_case_results 分页大小，避免全量明细进内存
_CASE_RESULTS_PAGE_SIZE = 20
#: sample_trace_pipeline_records 分页大小：T+1 日级窗口 trace 量级远大于
#: case 明细，async for 会把整窗结果集缓存进 Tortoise ``_result_cache``，
#: 分页 + sample_size 提前终止避免日级窗口全量载入内存。
_TRACE_SAMPLE_PAGE_SIZE = 500

#: 天级跟踪日报趋势窗口（天）。
_DAILY_TREND_DAYS = 7


def current_worker_id() -> str:
    """``hostname:pid`` —— run owner 标识（BullMQ 迁移后内联）。"""
    return f"{socket.gethostname()}:{os.getpid()}"


# ═══════════════════════════════════════════════════════════════════════════
# 聚合 helpers
# — benchmark_run_items 结构化列查询 + 纯 Python 统计（mean / median / p95 / std）
# — Kafka 路径仍查 trace_pipeline_records，judge_json JSONB 提取复用 helpers
# ═══════════════════════════════════════════════════════════════════════════


def _judge_has_score(judge: object) -> bool:
    """``judge_json ? 'score'`` — key 存在（值允许为 JSON null）。"""
    return isinstance(judge, dict) and "score" in judge


def _judge_score(judge: dict[str, Any]) -> float | None:
    """``(judge_json->>'score')::float8``；JSON null → None（聚合按 SQL NULL 忽略）。"""
    value = judge.get("score")
    return None if value is None else float(value)


def _judge_eval_failed(judge: dict[str, Any]) -> bool:
    """``judge_json->>'eval_failed' = 'true'``（``->>`` 对 JSON bool/串统一转 text）。"""
    value = judge.get("eval_failed")
    return value is not None and str(value).lower() == "true"


def percentile_cont(sorted_scores: list[float], p: float) -> float:
    """PG ``PERCENTILE_CONT(p)`` 等价：排序后 ``p*(n-1)`` 位置线性插值（``p`` ∈ [0, 1]）。

    空序列返回 ``0.0``（与聚合层「无样本 → 0 分」口径一致）。
    """
    n = len(sorted_scores)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_scores[0]
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return sorted_scores[lo] + (sorted_scores[hi] - sorted_scores[lo]) * (pos - lo)


def _criterion_stats(
    rows: list[tuple[float | None, bool]],
    *,
    pass_threshold: float,
) -> dict[str, Any]:
    """per-criterion ``(score, eval_failed)`` 行集合 → 聚合统计。

    对齐原 SQL 聚合语义：``sample_count`` 计入全部行（score 为 JSON null 的
    行仅计入计数，不参与均值/分位数/标准差）；空集各指标归 0。
    """
    sample_count = len(rows)
    failed_count = sum(1 for _, failed in rows if failed)
    valid = sorted(score for score, _ in rows if score is not None)
    if valid:
        mean = statistics.fmean(valid)
        median = percentile_cont(valid, 0.5)
        p95 = percentile_cont(valid, 0.95)
        min_score, max_score = valid[0], valid[-1]
        std = statistics.stdev(valid) if len(valid) > 1 else 0.0
    else:
        mean = median = p95 = min_score = max_score = std = 0.0
    pass_count = sum(1 for score, _ in rows if score is not None and score >= pass_threshold)
    return {
        "sample_count": sample_count,
        "success_count": sample_count - failed_count,
        "failed_count": failed_count,
        "criterion_stats_json": {
            "mean": mean,
            "median": median,
            "p95": p95,
            "min": min_score,
            "max": max_score,
            "std": std,
            "pass_rate": pass_count / sample_count if sample_count else 0.0,
        },
    }


def _merge_case_row(
    case: dict[str, Any],
    criterion_id: str,
    row: dict[str, Any],
) -> None:
    """将单条 ``(case_id, criterion_id)`` 行投影合并进 case 结构。

    benchmark 路径和 Kafka 路径共用：从 row dict 提取 score / reason /
    eval_failed / sub_scores / judge_json / attribution_detail_json，
    附到 ``case["scores"][criterion_id]``。case 级 ``input`` / ``output``
    / ``sub_scores`` 从首行取值（调用方保证行已按 case_id, criterion_id 排序）。
    """
    if not case.get("input"):
        case["input"] = row.get("input") or ""
    if not case.get("output"):
        case["output"] = row.get("output") or ""
    if "sub_scores" not in case and row.get("sub_scores") is not None:
        case["sub_scores"] = row.get("sub_scores")
    case["scores"][criterion_id] = {
        "score": row.get("score"),
        "reason": row.get("reason"),
        "eval_failed": row.get("eval_failed", False),
        "sub_scores": row.get("sub_scores"),
        "judge": row.get("judge_json"),
        "attribution": row.get("attribution_detail_json"),
    }


def _resolve_report_case_id(case: dict[str, Any]) -> str:
    """报告 Case 明细「用例ID」：``trace_id`` 优先，空则 ``case_id``（item_id）。"""
    trace_id = case.get("trace_id")
    if isinstance(trace_id, str):
        tid = trace_id.strip()
        if tid:
            return tid
    elif trace_id is not None:
        tid = str(trace_id).strip()
        if tid:
            return tid
    return str(case.get("case_id") or "")


def _extract_display_case_id(raw: object, fallback: str) -> str:
    """Extract ``task_id`` from input JSON（业务 task_id，非报告用例ID列口径）。

    报告「用例ID」列见 :func:`_resolve_report_case_id`（``trace_id`` →
    ``case_id``）。本函数保留供 input 内 ``task_id`` 提取等场景。
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return fallback
    if isinstance(raw, dict):
        task_id = raw.get("task_id")
        if task_id:
            return str(task_id)
    return fallback


def _extract_display_input(raw: object) -> str:
    """Extract user query from input for list display.

    Supports two input schemas:
    - **External OSS trigger**: ``{"query": "..."}`` — query is top-level
    - **Inline / Kafka trace**: ``{"request_body": {"query": "..."}}``

    The full ``input`` (with request body, user info, geography, etc.)
    is kept in the case_result for the detail modal; this method
    extracts just the user's query string for the compact table view.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    if isinstance(raw, dict):
        # External OSS trigger format: query is top-level
        q = raw.get("query")
        if q:
            return str(q)
        # Inline / Kafka trace format: nested under request_body
        rb = raw.get("request_body")
        if isinstance(rb, dict):
            q = rb.get("query")
            if q:
                return str(q)
    return ""


def _extract_display_output(raw: object) -> str:
    """Extract display text from output for list display.

    Supports two output schemas:
    - **Inline / Kafka trace**: ``{"component": {"displayText": ...}}``
    - **External OSS trigger**: ``{"response_text": "...", ...}`` —— ieval
      数据管道模型输出在 ``result`` 列（无 ``response_text`` 列时回退）。

    The full ``output`` (with component metadata, trace info, etc.)
    is kept in the case_result for the detail modal; this method
    extracts just the agent's display text for the compact table view
    （报告表格「output」列直接渲染本字段）。
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    if isinstance(raw, dict):
        # External OSS trigger format: response_text is the top-level key
        # （ieval 管道无 response_text 列，模型输出在 result 列——回退）
        for key in ("response_text", "result", "displayText", "text"):
            val = raw.get(key)
            if val:
                return str(val)
        # Inline / Kafka trace format: nested under component
        comp = raw.get("component")
        if isinstance(comp, dict):
            for key in ("displayText", "text"):
                val = comp.get(key)
                if val:
                    return str(val)
    return ""


def generate_benchmark_job_id(benchmark_name: str) -> str:
    """生成可读 benchmark job id（即 ``run_id``）。

    格式: ``bm_{sanitized_name}_{yyyyMMddHHmmss}_{6hex}``（UTC，总长 ≤ 128）。
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", benchmark_name or "benchmark")
    sanitized = re.sub(r"_+", "_", sanitized).strip("_") or "benchmark"
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    suffix = secrets.token_hex(3)
    # bm_ + _ + ts(14) + _ + suffix(6) = 3+1+14+1+6 = 25 fixed chars around name
    max_name_len = _JOB_ID_MAX_LEN - 25
    max_name_len = max(max_name_len, 1)
    name_part = sanitized[:max_name_len]
    return f"bm_{name_part}_{ts}_{suffix}"


#: ``metrics_json`` 内按子键 merge 的嵌套字典（非整块替换）。
_METRICS_NESTED_DICT_KEYS: frozenset[str] = frozenset(
    {
        "criterion_metrics",
        "attribution_metrics",
        "react_rounds_summary",
        # 出报时单独 patch（AggregatedMetrics 不含该字段），按子键 merge
        # 保证后续 complete_run 的整块写入不清掉性能指标。
        "performance_metrics",
    }
)

#: ``update_run_status`` / ``upsert`` 风格可合并的 bucket 字段。
_MERGE_BUCKET_FIELDS: frozenset[str] = frozenset(
    {
        "metrics_json",
        "report_meta_json",
        "config_json",
    }
)


def _stamp_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    """向 bucket 注入 ``updated_at``（UTC+8 ISO），对齐 TraceRecordService。"""
    bucket["updated_at"] = datetime.now(_UTC_PLUS_8).isoformat()
    return bucket


def merge_metrics_json(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """增量合并 ``metrics_json``。

    生命周期示例::

        merge_criterion → {criterion_metrics: {c0: …}}
        merge_criterion → {criterion_metrics: {c0: …, c1: …}}  # 保留 c0
        complete_run    → 覆盖/补齐顶层 + attribution_metrics，保留未提及 criterion

    ``criterion_metrics`` / ``attribution_metrics`` / ``react_rounds_summary``
    按 **子键** merge；其余顶层键同名覆盖。
    """
    merged: dict[str, Any] = dict(existing or {})
    for key, value in incoming.items():
        if key in _METRICS_NESTED_DICT_KEYS and isinstance(value, dict):
            nested = dict(merged.get(key) or {})
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return _stamp_bucket(merged)


def merge_report_meta_json(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """浅合并 ``report_meta_json``（url / size / generated_at 等逐步补齐）。"""
    merged = dict(existing or {})
    merged.update(incoming)
    return _stamp_bucket(merged)


def merge_config_json(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """浅合并 ``config_json``（配置快照补丁）。"""
    merged = dict(existing or {})
    merged.update(incoming)
    return _stamp_bucket(merged)


def _build_judge_bucket(result: ExternalJudgeItemResult) -> dict[str, Any]:
    """judge 回调 bucket：score 字符串化 + run/criterion 锚定。

    ``cas_advance_external_judge_item`` 的前置步骤，供 create / update
    两条路径共用（judge_json 快照落库格式）。
    """
    bucket = dict(result.judge_json)
    if "score" in bucket and not isinstance(bucket["score"], str):
        bucket["score"] = str(bucket["score"])
    bucket.setdefault("score", str(result.score))
    bucket["benchmark_run_id"] = result.run_id
    bucket["criterion_id"] = result.criterion_id
    return bucket


def _run_id_date_rest(run_id: str, benchmark_name: str) -> str | None:
    """``{stem}-{yyyymmdd}[...]`` → ``yyyymmdd...`` 余段；非该形态 → None。"""
    stem = benchmark_run_id_stem(benchmark_name) or benchmark_name
    prefix = f"{stem}-"
    if not run_id.startswith(prefix):
        return None
    return run_id[len(prefix) :]


def looks_like_daily_run_id(run_id: str, benchmark_name: str) -> bool:
    """是否呈日跑 ``{stem}-…`` 前缀形态（不论日期能否解析）。

    供趋势锚点回退日志分级等跨模块场景使用；实时流 / inline 等非该形态
    返回 False。内部与 ``parse_benchmark_run_data_date`` 共用前缀规则。
    """
    return _run_id_date_rest(run_id, benchmark_name) is not None


def parse_benchmark_run_data_date(run_id: str, benchmark_name: str) -> date | None:
    """从日跑 ``run_id`` 解析数据日（``{stem}-{yyyymmdd}[-suffix]``）。

    趋势 / 昨日对比必须以该日为锚，禁止用出报 ``now()``——延迟 finalize
    时否则会把分数贴到日历今天、数据日落空（如 11 月发布的
    ``ieval_external-20261031``）。
    """
    date_len = 8  # yyyymmdd
    rest = _run_id_date_rest(run_id, benchmark_name)
    if rest is None or len(rest) < date_len or not rest[:date_len].isdigit():
        return None
    if len(rest) > date_len and rest[date_len] != "-":
        return None
    try:
        return date(int(rest[:4]), int(rest[4:6]), int(rest[6:8]))
    except ValueError:
        return None


def _parse_run_id_suffix(run_id: str, benchmark_name: str) -> str | None:
    """从 run_id 解析 run_id_suffix（``{name}-{yyyymmdd}[-{suffix}]``）。

    ``{name}-{yyyymmdd}`` → None（无 suffix）；``{name}-{yyyymmdd}-q2`` →
    ``"q2"``；非该形态（inline ``bm_*`` 等）→ None。

    ``benchmark_name`` 可为裸茎或 ``stem@registrationId``；run_id 前缀
    一律按茎解析（与日调度 ``{stem}-{yyyymmdd}`` 一致）。
    """
    date_len = 8  # yyyymmdd
    rest = _run_id_date_rest(run_id, benchmark_name)
    if rest is None:
        return None
    # rest = yyyymmdd（8 位）或 yyyymmdd-suffix
    if len(rest) <= date_len:
        return None
    if not rest[:date_len].isdigit() or rest[date_len] != "-":
        return None
    suffix = rest[date_len + 1 :]
    return suffix or None


#: fused / per-criterion 行终态（与 ``fused_row.TERMINAL_ITEM_STATUSES`` 同值）。
#: 【仅 stop-run 阶段性报告】跳过 ``running`` 预写，避免未回调行被当成 0 分
#: failed；正常 completed/failed 出报不得启用该过滤（零语义扩散）。
#: 常量留在本模块：``run_helpers`` 不得反向依赖 ``eval_workers.fused_row``。
_TERMINAL_ITEM_STATUSES = frozenset({"success", "failed"})


def _is_terminal_item_status(status: object) -> bool:
    """run item 是否已终态（success/failed）；running 预写行返回 False。"""
    return str(status or "") in _TERMINAL_ITEM_STATUSES


#: stop-run 阶段性报告口径 ≡ 停止门闩（复用 run_stop 原语，禁止分叉语义）。
_is_partial_stop_config = is_stop_requested


def _collect_fused_criterion_scores(
    all_rows: list[Any],
    rubric: "list[Criterion]",
    *,
    partial_stop: bool = False,
) -> dict[str, list[tuple[float | None, bool]]]:
    """从 fused 行 ``judge_json.criterion_scores`` 提取 per-dimension 分数。

    外部触发模式可能所有回调都返回 error（criterion_scores 为空 dict），
    此时无法提取 per-criterion 分数；对 error 行按 score=0 / failed 计入
    每个 criterion，保证 criterion_stats 样本数与 fused 行数一致。
    ``partial_stop=True``（仅 stop-run）时跳过 ``running`` 预写行。
    返回空 dict 表示无 fused 行可解析（调用方回退 per-criterion 行）。
    """
    by_cid: dict[str, list[tuple[float | None, bool]]] = {}
    rows = (
        [r for r in all_rows if _is_terminal_item_status(getattr(r, "status", None))]
        if partial_stop
        else all_rows
    )
    is_external_only = not any(r.criterion_id != "fused" for r in rows)
    for rec in rows:
        if rec.criterion_id != "fused":
            continue
        cj = getattr(rec, "judge_json", None) or {}
        cs = cj.get("criterion_scores")
        if not isinstance(cs, dict) or not cs:
            # 外部触发 error 回调：criterion_scores 为空，按 failed 计入
            if is_external_only:
                for criterion in rubric:
                    by_cid.setdefault(criterion.id, []).append((0.0, True))
            continue
        has_error = bool(cj.get("error"))
        for cid, cdata in cs.items():
            if isinstance(cdata, dict):
                raw_score = cdata.get("score")
                score = float(raw_score) if raw_score is not None else 0.0
                is_failed = has_error and score == 0.0
            else:
                score = 0.0
                is_failed = True
            by_cid.setdefault(cid, []).append((score, is_failed))
    return by_cid


@dataclass(frozen=True)
class ExternalFusedRowProcess:
    """``_process_external_fused_row`` 入参（FUNC-PARAMS-001）。"""

    rec: BenchmarkRunItem
    stats: "_CaseRowStats"
    benchmark_name: str | None = None
    config: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExpandDimensionMetricsSpec:
    """``expand_dimension_metrics`` 入参（FUNC-PARAMS-001）。"""

    run_id: str
    benchmark_name: str | None
    metrics: "AggregatedMetrics"
    config: dict[str, Any] | None = None


@dataclass(frozen=True)
class DailyTrendQuery:
    """``load_daily_trend`` 入参（FUNC-PARAMS-001）。"""

    benchmark_name: str
    today_point: dict[str, Any] | None = None
    days: int = 7
    run_id_suffix: str | None = None
    #: 窗口右端（数据日）。None → 东八区今天（仅兼容旧调用）。
    anchor_date: date | None = None


@dataclass(frozen=True)
class KafkaSampleQuery:
    """``sample_kafka_pipeline_records`` 入参（FUNC-PARAMS-001）。"""

    date_start: datetime
    date_end: datetime
    evaluator_names: list[str]
    sample_size: int | None = None
    project_id: str | None = None


@dataclass(frozen=True)
class TraceRecordSampleQuery:
    """``sample_trace_pipeline_records`` 入参（FUNC-PARAMS-001）。

    与 :class:`KafkaSampleQuery` 的差异：``source`` 可配（kafka 版硬编码
    ``"kafka"``），新增 ``agent_name`` 标量列过滤；供 trace 日级报表
    （T+1）独立链路使用，不影响 pipeline benchmark 的 kafka 采样路径。
    ``judge_content_version`` 非空时只保留 judge lineage 对应该版本的行
    （example 双链按注册隔离出报，与调用次数同口径）。
    """

    date_start: datetime
    date_end: datetime
    evaluator_names: list[str]
    source: str = "kafka"
    agent_name: str | None = None
    sample_size: int | None = None
    project_id: str | None = None
    judge_content_version: str | None = None


@dataclass(frozen=True)
class ParseFailedStubWrite:
    """``write_parse_failed_stubs`` 入参（FUNC-PARAMS-001）。"""

    run_id: str
    item_id: str
    reason: str
    criterion_ids: list[str]
    item: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompleteRunWrite:
    """``complete_run`` 入参（FUNC-PARAMS-001）。"""

    run_id: str
    final_score: float
    metrics_json: dict[str, Any]
    report_meta_json: dict[str, Any] | None = None
    duration_ms: int | None = None


@dataclass(frozen=True)
class KafkaRubricQuery:
    """Kafka/trace × rubric 查询入参（FUNC-PARAMS-001）。

    供 ``load_kafka_case_results`` / ``load_trace_daily_case_results`` /
    ``load_kafka_criterion_entries`` / ``aggregate_kafka_attribution`` /
    ``aggregate_trace_daily_root_cause`` 共用；未用字段保留默认即可。
    """

    trace_ids: list[str]
    rubric: "list[Criterion]"
    registration_id: str | None = None
    stream_instance_id: str | None = None
    pass_threshold: float = 0.5
    fields: "Sequence[str] | None" = None


_KAFKA_RUBRIC_LEGACY_KEYS = frozenset(
    {
        "trace_ids",
        "rubric",
        "registration_id",
        "stream_instance_id",
        "pass_threshold",
        "fields",
    }
)


def resolve_kafka_rubric_query(
    query: KafkaRubricQuery | None,
    legacy: Mapping[str, object],
) -> KafkaRubricQuery:
    """解析 ``KafkaRubricQuery`` 或旧关键字入参（互斥）。"""
    if query is not None:
        if legacy:
            raise TypeError("cannot combine KafkaRubricQuery with legacy kwargs")
        return query
    unknown = sorted(set(legacy) - _KAFKA_RUBRIC_LEGACY_KEYS)
    if unknown:
        raise TypeError(f"unexpected kafka rubric query arguments: {unknown}")
    if "trace_ids" not in legacy or "rubric" not in legacy:
        raise TypeError("requires KafkaRubricQuery or trace_ids/rubric kwargs")
    pass_threshold = legacy.get("pass_threshold", 0.5)
    fields = legacy.get("fields")
    return KafkaRubricQuery(
        trace_ids=list(legacy["trace_ids"]),  # type: ignore[arg-type]
        rubric=list(legacy["rubric"]),  # type: ignore[arg-type]
        registration_id=(
            str(legacy["registration_id"]) if legacy.get("registration_id") is not None else None
        ),
        stream_instance_id=(
            str(legacy["stream_instance_id"])
            if legacy.get("stream_instance_id") is not None
            else None
        ),
        pass_threshold=float(pass_threshold) if pass_threshold is not None else 0.5,
        fields=fields if isinstance(fields, (list, tuple)) or fields is None else None,
    )


@dataclass(frozen=True)
class _AttributionCounterBump:
    """``_increment_attribution_counter_once`` 入参（FUNC-PARAMS-001）。"""

    run_id: str
    item_id: str
    count_key: str
    ids_key: str


@dataclass
class _CaseRowStats:
    """``load_case_results`` 行级计数器（日志输出 / null 列监控用）。"""

    total_rows: int = 0
    scored_rows: int = 0
    failed_rows: int = 0
    fused_count: int = 0
    null_score_rows: int = 0
    null_input_rows: int = 0
    null_output_rows: int = 0
    null_judge_rows: int = 0
    null_attr_rows: int = 0
    fused_with_attr: int = 0

    def count_scored(self, rec: BenchmarkRunItem) -> None:
        """记一行 scored 明细并累计 failed / null 列计数。"""
        self.scored_rows += 1
        if rec.status == "failed":
            self.failed_rows += 1
        if rec.score is None:
            self.null_score_rows += 1
        if rec.input is None:
            self.null_input_rows += 1
        if rec.output is None:
            self.null_output_rows += 1
        if rec.judge_json is None:
            self.null_judge_rows += 1
        if rec.attribution_detail_json is None:
            self.null_attr_rows += 1


def _attach_inline_fused_attribution(
    by_case: dict[str, dict[str, Any]],
    rec: BenchmarkRunItem,
    stats: _CaseRowStats,
) -> None:
    """inline 模式 fused 行：只贡献 case 级 attribution，分数走 per-criterion。"""
    fattr = rec.attribution_detail_json
    if isinstance(fattr, dict) and fattr:
        case = by_case.setdefault(
            rec.item_id,
            {"case_id": rec.item_id, "input": "", "output": "", "scores": {}},
        )
        case["attribution"] = fattr
        stats.fused_with_attr += 1


def _extract_criterion_scores(judge_json: dict[str, Any] | None) -> dict[str, float]:
    """external 模式 fused 行 ``judge_json.criterion_scores`` → {cid: 有效分}。

    score 为 None 或 error 行的 0 分（eval_failed，同
    ``_process_external_fused_row`` 判定）不计入；非 dict 结构返回空。
    """
    if not isinstance(judge_json, dict):
        return {}
    criterion_scores = judge_json.get("criterion_scores")
    if not isinstance(criterion_scores, dict):
        return {}
    has_error = bool(judge_json.get("error"))
    result: dict[str, float] = {}
    for cid, cdata in criterion_scores.items():
        if not isinstance(cdata, dict):
            continue
        raw_score = cdata.get("score")
        if raw_score is None:
            continue
        score = float(raw_score)
        if has_error and score == 0.0:
            continue  # eval_failed：error 行的 0 分不计入
        result[str(cid)] = score
    return result


def _process_external_fused_row(
    by_case: dict[str, dict[str, Any]],
    proc: ExternalFusedRowProcess,
) -> None:
    """external 模式 fused 行：计数 + case 初始化 + criterion_scores 展开落 case。"""
    rec = proc.rec
    stats = proc.stats
    benchmark_name = proc.benchmark_name
    config = proc.config
    stats.count_scored(rec)
    # 从 fused 行结构化列初始化 input/output，即使
    # criterion_scores 为空（error 回调 / 预写行）也能展示 case 明细
    case = by_case.setdefault(
        rec.item_id,
        {
            "case_id": rec.item_id,
            "input": rec.input if rec.input is not None else "",
            "output": rec.output if rec.output is not None else "",
            "scores": {},
        },
    )
    # fused 行 trace_id / trace_url（外部 judge 回调写入）→
    # case 顶层：trace_id 供报告「用例ID」列，trace_url 供
    # 「Trace链接」列与 Aone 工单正文复用
    if rec.trace_id:
        case["trace_id"] = rec.trace_id
    elif getattr(rec, "source_trace_id", None):
        case["trace_id"] = rec.source_trace_id
    trace_url = getattr(rec, "trace_url", None)
    if trace_url:
        case["trace_url"] = trace_url
    cj = rec.judge_json or {}
    cs = cj.get("criterion_scores", {})
    has_error = bool(cj.get("error"))
    for cid, cdata in cs.items():
        if isinstance(cdata, dict):
            raw_score = cdata.get("score")
            score = float(raw_score) if raw_score is not None else None
            reason = cdata.get("reason", "")
            sub_scores = cdata.get("sub_scores")
            eval_failed = has_error and (score is None or score == 0.0)
        else:
            score = None
            reason = ""
            sub_scores = None
            eval_failed = True
        _merge_case_row(
            case,
            cid,
            {
                "score": score,
                "reason": reason,
                "eval_failed": eval_failed,
                "sub_scores": sub_scores,
                "input": rec.input,
                "output": rec.output,
                "judge_json": rec.judge_json,
                "attribution_detail_json": None,
            },
        )
    # fused 行的 score 列（run_items.score）是 ScoreFuser
    # 计算的综合分，附加到 scores["fused"] 供报告"综合分"
    # 列直接读取（与 prompt 中 scores.fused.score 约定一致）。
    if cs:
        _merge_case_row(
            case,
            "fused",
            {
                "score": rec.score,
                "reason": None,
                "eval_failed": False,
                "sub_scores": None,
                "input": rec.input,
                "output": rec.output,
                "judge_json": rec.judge_json,
                "attribution_detail_json": None,
            },
        )
    # criterion_scores 为空（error/failed 回调）时，仍需附加
    # judge_json + 失败原因，否则报告 case 明细 scores 区域全空。
    if not cs:
        _merge_case_row(
            case,
            "fused",
            {
                "score": rec.score if rec.score is not None else 0.0,
                "reason": cj.get("reason") or str(cj.get("error", ""))[:2000],
                "eval_failed": True,
                "sub_scores": None,
                "input": rec.input,
                "output": rec.output,
                "judge_json": rec.judge_json,
                "attribution_detail_json": None,
            },
        )
    # fused 行归因内联附加（不再单独遍历 fused_rows）
    fattr = rec.attribution_detail_json
    if isinstance(fattr, dict) and fattr:
        case["attribution"] = fattr
        stats.fused_with_attr += 1

    # 注册统计维度的业务（如 search_eval）：case 明细的「各维度得分」
    # 直接使用 judge sub_scores 展开（每维一行，reason 取维度明细）。
    # 未注册业务不展开（ieval 等存量行为不变）。
    # config 与报告 criteria / expand_dimension_metrics 共用
    # resolve_run_stat_dimensions（快照 → 注册表 → rubric_ref）。
    _expand_case_dimension_scores(case, benchmark_name, config=config)


def _expand_case_dimension_scores(
    case: dict[str, Any],
    benchmark_name: str | None,
    *,
    config: dict[str, Any] | None = None,
) -> None:
    """case 明细按统计维度展开 sub_scores 各维度得分（search_eval 形态）。

    数据源：fused 行 ``judge_json.criterion_scores[*]``（直存形态顶层
    ``sub_scores``；``extra.dimensions[key].reason`` 作维度行理由）。
    维度解析与聚合/报告一致（``resolve_run_stat_dimensions``）；无维度则跳过。
    """
    dims = resolve_run_stat_dimensions(config, benchmark_name)
    if not dims:
        return
    scores = case.get("scores")
    if not isinstance(scores, dict):
        return
    pending: dict[str, dict[str, Any]] = {}
    # 遍历快照（展开写入会新增维度行，直接遍历会 RuntimeError）
    for cdata in list(scores.values()):
        if not isinstance(cdata, dict):
            continue
        judge = cdata.get("judge")
        judge = judge if isinstance(judge, dict) else {}
        cs = judge.get("criterion_scores")
        cs = cs if isinstance(cs, dict) else {}
        for payload in cs.values():
            if not isinstance(payload, dict):
                continue
            sub_scores = payload.get("sub_scores")
            if not isinstance(sub_scores, dict):
                continue
            extra = payload.get("extra")
            dim_details = extra.get("dimensions") if isinstance(extra, dict) else None
            dim_details = dim_details if isinstance(dim_details, dict) else {}
            parent_reason = payload.get("reason")
            parent_reason = parent_reason if isinstance(parent_reason, str) else ""
            for dim in dims:
                raw = sub_scores.get(dim.key)
                if raw is None:
                    continue
                try:
                    dim_score = float(raw)
                except (TypeError, ValueError):
                    continue
                detail = dim_details.get(dim.key)
                dim_reason = detail.get("reason") if isinstance(detail, dict) else None
                if not dim_reason and parent_reason:
                    dim_reason = _extract_dim_reason_snippet(parent_reason, dim.key)
                if not dim_reason and parent_reason:
                    # 无法按维切分时，仍在分数后展示 holistic 理由（报告可读）
                    dim_reason = parent_reason
                pending[dim.key] = {
                    "score": dim_score,
                    "reason": dim_reason,
                    "eval_failed": False,
                    "sub_scores": None,
                    "judge": None,
                    "attribution": None,
                }
    scores.update(pending)


def _process_criterion_row(
    by_case: dict[str, dict[str, Any]],
    rec: BenchmarkRunItem,
    stats: _CaseRowStats,
) -> None:
    """per-criterion 行（inline 模式）：计数 + 展开落 case。"""
    stats.count_scored(rec)
    case = by_case.setdefault(
        rec.item_id,
        {"case_id": rec.item_id, "input": "", "output": "", "scores": {}},
    )
    _merge_case_row(
        case,
        rec.criterion_id,
        {
            "score": rec.score,
            "reason": rec.reason,
            "eval_failed": rec.status == "failed",
            "sub_scores": rec.sub_scores,
            "input": rec.input,
            "output": rec.output,
            "judge_json": rec.judge_json,
            "attribution_detail_json": rec.attribution_detail_json,
        },
    )
    if rec.trace_id and not case.get("trace_id"):
        case["trace_id"] = rec.trace_id
    elif getattr(rec, "source_trace_id", None) and not case.get("trace_id"):
        case["trace_id"] = rec.source_trace_id


def _pass_threshold_from_config(config: dict[str, Any] | None) -> float | None:
    """从 run ``config_json.aggregation.pass_threshold`` 读取；缺失 → None。"""
    if not isinstance(config, dict):
        return None
    aggregation = config.get("aggregation")
    if not isinstance(aggregation, dict):
        return None
    raw = aggregation.get("pass_threshold")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _extract_dim_reason_snippet(reason: str, dim_key: str) -> str | None:
    """从 holistic reasoning 中截取「维名…：…」片段（example 逐维叙述）。

    匹配 ``准确度8：…`` / ``相关性：…`` 等，截到下一维名、``总评`` 或文末。
    """
    text = (reason or "").strip()
    if not text or not dim_key:
        return None
    escaped = re.escape(dim_key)
    pattern = (
        rf"(?:^|[；;\n。])\s*({escaped}\s*\d*(?:[./]\d+)?\s*[：:]\s*.*?)"
        rf"(?=(?:准确度|相关性|完整性|总评)\s*\d*\s*[：:]|$)"
    )
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip().rstrip("；;。")
    # 文首无前导分隔符时再试一次
    head = re.match(
        rf"^({escaped}\s*\d*(?:[./]\d+)?\s*[：:]\s*.*?)"
        rf"(?=(?:准确度|相关性|完整性|总评)\s*\d*\s*[：:]|$)",
        text,
        flags=re.DOTALL,
    )
    if head:
        return head.group(1).strip().rstrip("；;。")
    return None


def _grade_from_judge_payload(payload: object) -> str | None:
    """从 judge_json / criterion_scores 条目提取框架写入的 ``grade``。"""
    if not isinstance(payload, dict):
        return None
    direct = payload.get("grade")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    cs = payload.get("criterion_scores")
    if isinstance(cs, dict):
        for row in cs.values():
            if not isinstance(row, dict):
                continue
            g = row.get("grade")
            if isinstance(g, str) and g.strip():
                return g.strip()
            extra = row.get("extra")
            if isinstance(extra, dict):
                eg = extra.get("grade")
                if isinstance(eg, str) and eg.strip():
                    return eg.strip()
    return None


def _extract_case_grade(case: dict[str, Any]) -> str | None:
    """优先从 fused / criterion 的 judge_json 拉 grade（方案 A 落库契约）。"""
    scores = case.get("scores")
    if not isinstance(scores, dict):
        return None
    fused = scores.get("fused")
    if isinstance(fused, dict):
        found = _grade_from_judge_payload(fused.get("judge"))
        if found:
            return found
        top = fused.get("grade")
        if isinstance(top, str) and top.strip():
            return top.strip()
    for sc in scores.values():
        if not isinstance(sc, dict):
            continue
        found = _grade_from_judge_payload(sc.get("judge"))
        if found:
            return found
        top = sc.get("grade")
        if isinstance(top, str) and top.strip():
            return top.strip()
    return None


def _decorate_case_rows(
    result: list[dict[str, Any]],
    *,
    pass_threshold: float | None = None,
) -> None:
    """case 明细装饰：display_* 列 + pass 判定 + grade（优先 judge_json）。"""
    for case in result:
        case["display_case_id"] = _resolve_report_case_id(case)
        case["display_input"] = _extract_display_input(case.get("input"))
        case["display_output"] = _extract_display_output(case.get("output"))
        # 通过状态：有归因（badcase）或评测失败 → 不通过；
        # 否则默认通过。确保归因 case 不会被 LLM 误标为通过。
        has_attribution = bool(case.get("attribution"))
        any_eval_failed = any(
            s.get("eval_failed", False)
            for s in case.get("scores", {}).values()
            if isinstance(s, dict)
        )
        case["pass"] = not (has_attribution or any_eval_failed)
        # 方案 A：优先 judge_json.grade；旧 run 无字段时按分制回退计算。
        from_judge = _extract_case_grade(case)
        if from_judge:
            case["grade"] = from_judge
            continue
        fused_row = case.get("scores", {}).get("fused")
        if (
            isinstance(fused_row, dict)
            and not fused_row.get("eval_failed")
            and fused_row.get("score") is not None
        ):
            case["grade"] = grade_fused_score(
                float(fused_row["score"]),
                pass_threshold=pass_threshold,
            )


@dataclass(frozen=True)
class BenchmarkRunSpec:
    """benchmark run 创建规格（``create_run`` / ``create_placeholder_run`` /
    ``create_external_run`` 共用参数对象）。

    三个入口共享同一组 ``benchmark_runs`` 行字段；``owner`` 仅
    ``create_run`` 使用，占位/补全入口固定取 ``current_worker_id()``；
    ``run_id`` 由调用方生成（占位/补全）或服务内生成（``create_run``），
    不入 spec。
    """

    benchmark_name: str
    service_name: str
    dataset_name: str | None = None
    project_id: str | None = None
    total_samples: int = 0
    total_evaluators: int = 0
    config_json: dict[str, Any] | None = None
    owner: str | None = None


__all__ = [
    "BenchmarkRun",
    "BenchmarkRunItem",
    "TraceRecord",
    "F",
    "in_transaction",
    "_stamp_bucket",
    "_expand_case_dimension_scores",
    "_extract_case_grade",
    "_extract_dim_reason_snippet",
    "_extract_display_case_id",
    "_grade_from_judge_payload",
    "_pass_threshold_from_config",
    "_resolve_report_case_id",
    "_extract_display_input",
    "_extract_display_output",
    "BenchmarkRunSpec",
    "CompleteRunWrite",
    "DailyTrendQuery",
    "ExpandDimensionMetricsSpec",
    "ExternalFusedRowProcess",
    "KafkaSampleQuery",
    "ParseFailedStubWrite",
    "TraceRecordSampleQuery",
    "current_worker_id",
    "generate_benchmark_job_id",
    "merge_config_json",
    "merge_metrics_json",
    "merge_report_meta_json",
    "looks_like_daily_run_id",
    "parse_benchmark_run_data_date",
    "percentile_cont",
]
