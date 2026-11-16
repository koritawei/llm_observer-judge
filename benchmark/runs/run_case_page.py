"""分页列出某次 benchmark run 的 case，字段对齐报告 ``#cases-data``。

普通 run 走 ``load_case_results``（``benchmark_run_items``）。
T+1 日报与 pipeline-kafka run 不写明细行，按出报时同一套采样条件
回放 trace，再走 ``load_trace_daily_case_results`` / ``load_kafka_case_results``。

GitBook 详情：``build_case_detail`` 在 list 字段之上补 ``input`` /
``expected_output``（列表契约仍不带这两列，避免 TOC 膨胀）。
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.models import BenchmarkRunItem, TraceRecord
from benchmark.domain.benchmark_console import (
    find_agent_document_by_id,
    find_agent_document_by_name,
)
from benchmark.domain.eval_config import sample_evaluator_names_for_report
from benchmark.domain.trace_report_config import TRACE_REPORT_NAMESPACE_CONFIG_KEY
from benchmark.runs.case_display import CaseDisplayContext, normalize_case_display
from business.evaluators.example.grading import grade_fused_score
from benchmark.runs.run_helpers import (
    KafkaSampleQuery,
    TraceRecordSampleQuery,
    _extract_display_input,
)
from benchmark.runs.schema import FUSED_CRITERION_ID
from benchmark.runs.trace_daily_trend import parse_trace_daily_data_date
from benchmark.reporting.framework.schema import Criterion

logger = get_logger(__name__)

_DEFAULT_PASS_THRESHOLD = 0.5
_PIPELINE_SOURCE = "pipeline-kafka"
_NOTIFICATION_SOURCE_ITEMS = "items"
_NOTIFICATION_SOURCE_TRACE = "trace"
#: sample.timezone 缺失 / ZoneInfo 非法时的默认 IANA 时区。
_DEFAULT_TIMEZONE = "Asia/Shanghai"
_FIRST_PAGE = 1
_MIN_POSITIVE_INT = 1
#: case 列表默认状态筛选：与报告「不合格(FAIL+POOR)」对齐。
_DEFAULT_STATUS_FILTER: frozenset[str] = frozenset({"FAIL", "POOR"})
_STATUS_FILTER_ALL = "ALL"
#: search_eval / search_eval_v2 不展示归因分类列与筛选。
_SEARCH_EVAL_MARKER = "search_eval"


class RunCaseLoader(Protocol):
    """``BenchmarkService`` 上本页需要的明细加载方法。"""

    async def load_case_results(
        self,
        *,
        run_id: str,
        benchmark_name: str | None = None,
        config: dict[str, object] | None = None,
    ) -> list[dict[str, object]]: ...

    async def load_kafka_case_results(
        self,
        **kwargs: object,
    ) -> list[dict[str, object]]: ...

    async def load_trace_daily_case_results(
        self,
        **kwargs: object,
    ) -> list[dict[str, object]]: ...

    async def sample_kafka_pipeline_records(
        self,
        query: KafkaSampleQuery,
    ) -> list[str]: ...

    async def sample_trace_pipeline_records(
        self,
        query: TraceRecordSampleQuery,
    ) -> list[str]: ...


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _positive_int(value: object) -> int | None:
    """正整数；0 / 负数 / 非数字返回 None（避免和魔法数字比较）。"""
    number = _finite_number(value)
    if number is None:
        return None
    whole = int(number)
    if whole < _MIN_POSITIVE_INT:
        return None
    return whole


def _config_dict(run: object) -> dict[str, object]:
    config = getattr(run, "config_json", None)
    if isinstance(config, dict):
        return config
    return {}


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _case_id(case: dict[str, object]) -> str:
    raw = case.get("case_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if raw is not None and not isinstance(raw, bool):
        text = str(raw).strip()
        if text:
            return text
    display = case.get("display_case_id")
    if isinstance(display, str):
        return display.strip()
    return ""


def _query_text(case: dict[str, object]) -> str:
    display = case.get("display_input")
    if isinstance(display, str) and display:
        return display
    extracted = _extract_display_input(case.get("input"))
    return extracted if isinstance(extracted, str) else ""


def _score_from_score_row(row: dict[str, object]) -> float | None:
    """从 scores 一行取分：行级 score → judge.score。"""
    number = _finite_number(row.get("score"))
    if number is not None:
        return number
    judge = row.get("judge")
    if isinstance(judge, dict):
        return _finite_number(judge.get("score"))
    return None


def _case_score(case: dict[str, object]) -> float | None:
    """综合分：优先 fused.score，再任意 scores.*.score / judge.score，最后顶层。"""
    scores = case.get("scores")
    if isinstance(scores, dict):
        fused = scores.get(FUSED_CRITERION_ID)
        if isinstance(fused, dict):
            number = _score_from_score_row(fused)
            if number is not None:
                return number
        for row in scores.values():
            if not isinstance(row, dict):
                continue
            number = _score_from_score_row(row)
            if number is not None:
                return number
    return _finite_number(case.get("score"))


def _case_status(
    case: dict[str, object],
    *,
    pass_threshold: float | None = None,
) -> str:
    """与报告 ``case.grade`` / 档位徽章同源。"""
    grade = case.get("grade")
    if isinstance(grade, str) and grade.strip():
        return grade.strip().upper()
    score = _case_score(case)
    if score is None:
        return ""
    return grade_fused_score(score, pass_threshold=pass_threshold)


def parse_status_filter(raw: str | None) -> frozenset[str] | None:
    """解析 ``status`` query。

    - ``None`` / 空串未给出时由调用方决定默认；本函数对空串 / ``ALL`` 返回
      ``None``（不过滤）。
    - 逗号分隔多档（如 ``FAIL,POOR``）返回 frozenset。
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.upper() == _STATUS_FILTER_ALL:
        return None
    return frozenset(part.strip().upper() for part in text.split(",") if part.strip())


def supports_attribution_category(benchmark_name: str) -> bool:
    """search_eval / search_eval_v2 不展示归因分类；其余业务展示。"""
    return _SEARCH_EVAL_MARKER not in (benchmark_name or "").strip().lower()


def extract_attribution_category(
    case: dict[str, object],
    *,
    notification_source: str,
    benchmark_name: str,
) -> str | None:
    """归因分类列取值（读已归一化的 ``case["attribution"]``，非 ORM 列名）。

    - 实时流（``aone_source=trace`` / amap-agent）：
      ``case["attribution"]["category"]``（由 ``load_trace_daily_case_results``
      将 ``attribution_detail_json`` merge 进 ``attribution`` 后产出）
    - 批量流 ieval：``case["attribution"]["raw_json"]`` 下
      ``primary_issue_layer`` / ``primary_issue_category``，以 ``/`` 连接
    - search_eval / search_eval_v2：不支持，返回 None
    """
    if not supports_attribution_category(benchmark_name):
        return None
    blob = case.get("attribution")
    if not isinstance(blob, dict):
        return None
    if notification_source == _NOTIFICATION_SOURCE_TRACE:
        return _text(blob.get("category"))
    raw = blob.get("raw_json")
    if isinstance(raw, dict):
        layer = _text(raw.get("primary_issue_layer"))
        category = _text(raw.get("primary_issue_category"))
        if layer and category:
            return f"{layer}/{category}"
        return layer or category
    return _text(blob.get("category")) or _text(blob.get("primary_issue_category"))


@dataclass(frozen=True)
class CaseFilterSpec:
    """``filter_report_cases`` 筛选参数（FUNC-PARAMS-001 收敛）。"""

    query: str = ""
    statuses: frozenset[str] | None = None
    pass_threshold: float | None = None
    attribution_category: str = ""
    notification_source: str = _NOTIFICATION_SOURCE_ITEMS
    benchmark_name: str = ""


def filter_report_cases(
    cases: Sequence[dict[str, object]],
    *,
    spec: CaseFilterSpec | None = None,
) -> list[dict[str, object]]:
    """按 query / 状态 / 归因分类筛选；状态 None 表示不过滤。"""
    opts = spec or CaseFilterSpec()
    needle = opts.query.strip().lower()
    category_needle = opts.attribution_category.strip()
    selected: list[dict[str, object]] = []
    for case in cases:
        if needle and needle not in _query_text(case).lower():
            continue
        if opts.statuses is not None:
            status = _case_status(case, pass_threshold=opts.pass_threshold)
            if status not in opts.statuses:
                continue
        if category_needle:
            category = extract_attribution_category(
                case,
                notification_source=opts.notification_source,
                benchmark_name=opts.benchmark_name,
            )
            if category is None or category != category_needle:
                continue
        selected.append(case)
    return selected


def resolve_notification_source(config: dict[str, object]) -> str:
    """与 ``load_report_cases`` 同源的 aone 落表判定，不加载明细。"""
    if config.get("trace_daily_report") is True:
        return _NOTIFICATION_SOURCE_TRACE
    if config.get("source") == _PIPELINE_SOURCE:
        return _NOTIFICATION_SOURCE_TRACE
    return _NOTIFICATION_SOURCE_ITEMS


def report_url_of(run: object) -> str | None:
    """``report_meta_json.report_site_url``（Pages 正式入口）；缺失返回 None。

    case 页「查看报告」与 ``report-summary`` 拉取均用此地址，**不用** OSS
    ``report_url``（prod 常无公网 egress，直连 OSS 会 Network unreachable → 502）。
    """
    meta = getattr(run, "report_meta_json", None)
    if not isinstance(meta, dict):
        return None
    return _text(meta.get("report_site_url"))


@dataclass(frozen=True)
class CaseListItemSpec:
    """``case_list_item`` 展示/归属参数（FUNC-PARAMS-001 收敛）。"""

    notification_dispatch: dict[str, str] | None = None
    notification_source: str = _NOTIFICATION_SOURCE_ITEMS
    benchmark_name: str = ""
    pass_threshold: float | None = None
    include_attribution_category: bool = False


@dataclass(frozen=True)
class CasePageQuery:
    """``build_case_page`` 分页与筛选（FUNC-PARAMS-001 收敛）。"""

    page: int
    page_size: int
    query: str = ""
    status: str | None = None
    attribution_category: str = ""


@dataclass(frozen=True)
class CaseDetailQuery:
    """``build_case_detail`` 定位参数（FUNC-PARAMS-001 收敛）。"""

    case_id: str


def _json_text(value: object) -> str:
    """详情正文用：dict/list → 缩进 JSON；其余转可读字符串。"""
    if value is None:
        return ""
    payload: object = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return str(payload)


def _expected_from_scores(case: dict[str, object]) -> object:
    scores = case.get("scores")
    if not isinstance(scores, dict):
        return None
    ordered_keys = [FUSED_CRITERION_ID, *[key for key in scores if key != FUSED_CRITERION_ID]]
    for key in ordered_keys:
        row = scores.get(key)
        if not isinstance(row, dict):
            continue
        judge = row.get("judge")
        if not isinstance(judge, dict):
            continue
        if "expected_output" in judge:
            return judge.get("expected_output")
        if "expected" in judge:
            return judge.get("expected")
    return None


def detail_input_text(case: dict[str, object]) -> str:
    """GitBook Input 章节：完整 input（非 list 的 display query）。"""
    return _json_text(case.get("input"))


def detail_expected_text(case: dict[str, object]) -> str:
    """GitBook Expected 章节：顶层 expected_output，否则 dig scores.judge。"""
    if "expected_output" in case:
        return _json_text(case.get("expected_output"))
    dug = _expected_from_scores(case)
    if dug is not None:
        return _json_text(dug)
    return ""


def case_list_item(
    case: dict[str, object],
    *,
    spec: CaseListItemSpec | None = None,
) -> dict[str, object]:
    """一条对外 case：output / judge / attribution 经展示插件归一化。

    契约相对旧版：不再返回 ``input``；``output`` 为可渲染文本；
    ``judge`` / ``attribution`` 为归一化视图（非报告原对象透传）。
    """
    opts = spec or CaseListItemSpec()
    trace_url = _text(case.get("trace_url"))
    dispatch = opts.notification_dispatch or {}
    output, judge_view, attr_view = normalize_case_display(
        case,
        CaseDisplayContext(
            notification_source=opts.notification_source,
            benchmark_name=opts.benchmark_name,
        ),
    )
    score = _case_score(case)
    status = _case_status(case, pass_threshold=opts.pass_threshold)
    item: dict[str, object] = {
        "id": _case_id(case),
        "query": _query_text(case),
        "score": score,
        "status": status or None,
        "output": output,
        "judge": judge_view.model_dump() if judge_view is not None else None,
        "attribution": attr_view.model_dump() if attr_view is not None else None,
        "trace_url": trace_url,
        "notification_url": dispatch.get("notification_url"),
        "notification_ref": dispatch.get("notification_ref"),
        "notification_assigned_to": dispatch.get("notification_assigned_to"),
    }
    if opts.include_attribution_category:
        item["attribution_category"] = extract_attribution_category(
            case,
            notification_source=opts.notification_source,
            benchmark_name=opts.benchmark_name,
        )
    return item


def case_detail_item(
    case: dict[str, object],
    *,
    spec: CaseListItemSpec | None = None,
) -> dict[str, object]:
    """GitBook 详情：list 字段 + ``input`` / ``expected_output``。"""
    item = case_list_item(case, spec=spec)
    item["input"] = detail_input_text(case)
    item["expected_output"] = detail_expected_text(case)
    return item


def find_report_case(
    cases: Sequence[dict[str, object]],
    case_id: str,
) -> dict[str, object] | None:
    """按报告 case id 精确命中一条原始 case。"""
    needle = case_id.strip()
    if not needle:
        return None
    for case in cases:
        if _case_id(case) == needle:
            return case
    return None


def slice_cases(
    cases: Sequence[dict[str, object]],
    *,
    page: int,
    page_size: int,
) -> list[dict[str, object]]:
    """1-based 页切片；超出 total 时返回空页。"""
    if page < _FIRST_PAGE:
        return []
    start = (page - _FIRST_PAGE) * page_size
    return list(cases[start : start + page_size])


def _notification_dispatch_fields(bucket: object) -> dict[str, str]:
    if not isinstance(bucket, dict):
        return {}
    fields: dict[str, str] = {}
    for key in ("notification_url", "notification_ref", "notification_assigned_to"):
        text = _text(bucket.get(key))
        if text is not None:
            fields[key] = text
    return fields


def _rubric_rows(config: dict[str, object]) -> list[dict[str, object]]:
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


def _safe_rubric(config: dict[str, object]) -> list[Criterion]:
    criteria: list[Criterion] = []
    for item in _rubric_rows(config):
        criterion_id = _text(item.get("id"))
        evaluator_ref = _text(item.get("evaluator_ref"))
        if criterion_id is None or evaluator_ref is None:
            continue
        weight = _finite_number(item.get("weight"))
        name = _text(item.get("name")) or criterion_id
        description = item.get("description")
        criteria.append(
            Criterion(
                id=criterion_id,
                name=name,
                weight=0.0 if weight is None else weight,
                evaluator_ref=evaluator_ref,
                description=description if isinstance(description, str) else "",
            )
        )
    return criteria


def _registration_id(config: dict[str, object]) -> str | None:
    direct = _text(config.get("agent_registration_id"))
    if direct is not None:
        return direct
    snap = config.get("agent_registration")
    if isinstance(snap, dict):
        rid = _text(snap.get("id"))
        if rid is not None:
            return rid
    sample = config.get("trace_daily_sample")
    if isinstance(sample, dict):
        return _text(sample.get("agent_registration_id"))
    return None


def _pass_threshold(config: dict[str, object]) -> float:
    sample = config.get("trace_daily_sample")
    if isinstance(sample, dict):
        number = _finite_number(sample.get("pass_threshold"))
        if number is not None:
            return number
    aggregation = config.get("aggregation")
    if isinstance(aggregation, dict):
        number = _finite_number(aggregation.get("pass_threshold"))
        if number is not None:
            return number
    return _DEFAULT_PASS_THRESHOLD


def _sample_blob(config: dict[str, object]) -> dict[str, object] | None:
    sample = config.get("trace_daily_sample")
    if isinstance(sample, dict):
        return sample
    return None


def _trace_window(run: object, config: dict[str, object]) -> tuple[datetime, datetime] | None:
    namespace = _text(config.get(TRACE_REPORT_NAMESPACE_CONFIG_KEY))
    if namespace is None:
        return None
    data_date = parse_trace_daily_data_date(str(getattr(run, "run_id", "") or ""), namespace)
    if data_date is None:
        return None
    sample = _sample_blob(config)
    tz_name = _DEFAULT_TIMEZONE
    if sample is not None:
        named = _text(sample.get("timezone"))
        if named is not None:
            tz_name = named
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(_DEFAULT_TIMEZONE)
    start = datetime(data_date.year, data_date.month, data_date.day, tzinfo=tz)
    return start, start + timedelta(days=1)


def _copy_cases(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows]


def _frozen_sampled_trace_ids(run: object) -> list[str] | None:
    """出报时写入 ``report_meta_json.sampled_trace_ids``；缺失则返回 None。"""
    meta = getattr(run, "report_meta_json", None)
    if not isinstance(meta, dict):
        return None
    raw = meta.get("sampled_trace_ids")
    if not isinstance(raw, list):
        return None
    ids = [str(item).strip() for item in raw if str(item or "").strip()]
    return ids or None


async def _load_trace_daily_cases(
    service: RunCaseLoader,
    run: object,
    config: dict[str, object],
) -> list[dict[str, object]]:
    window = _trace_window(run, config)
    rubric = _safe_rubric(config)
    sample = _sample_blob(config)
    source = _text(sample.get("source") if sample else None) or _text(config.get("source"))
    if not rubric:
        return []
    registration_id = _registration_id(config)
    stream = _text(config.get("stream_instance_id"))
    frozen = _frozen_sampled_trace_ids(run)
    if frozen is None:
        if window is None or source is None:
            logger.info(
                "benchmark.case_page_trace_skipped run_id=%s window=%s rubric=%d source=%s",
                getattr(run, "run_id", ""),
                window is not None,
                len(rubric),
                source or "-",
            )
            return []
        evaluator_names = sample_evaluator_names_for_report(
            rubric_refs=[item.evaluator_ref for item in rubric],
            registration_id=registration_id,
            stream_instance_id=stream,
        )
        project_id = _text(sample.get("project_id") if sample else None) or _text(
            getattr(run, "project_id", None)
        )
        agent_name = _text(sample.get("agent_name") if sample else None)
        sample_size = _positive_int(sample.get("sample_size")) if sample is not None else None
        content_version = _text(config.get("judge_content_version"))
        trace_ids = await service.sample_trace_pipeline_records(
            TraceRecordSampleQuery(
                date_start=window[0],
                date_end=window[1],
                evaluator_names=evaluator_names,
                source=source,
                agent_name=agent_name,
                sample_size=sample_size,
                project_id=project_id,
                judge_content_version=content_version,
            )
        )
    else:
        trace_ids = frozen
    if not trace_ids:
        return []
    return _copy_cases(
        await service.load_trace_daily_case_results(
            trace_ids=trace_ids,
            rubric=rubric,
            pass_threshold=_pass_threshold(config),
            registration_id=registration_id,
            stream_instance_id=stream,
        )
    )


async def _load_pipeline_cases(
    service: RunCaseLoader,
    run: object,
    config: dict[str, object],
) -> list[dict[str, object]]:
    rubric = _safe_rubric(config)
    if not rubric:
        return []
    registration_id = _registration_id(config)
    stream = _text(config.get("stream_instance_id"))
    frozen = _frozen_sampled_trace_ids(run)
    if frozen is None:
        start_raw = _text(config.get("date_start"))
        end_raw = _text(config.get("date_end"))
        if start_raw is None or end_raw is None:
            return []
        try:
            date_start = datetime.fromisoformat(start_raw)
            date_end = datetime.fromisoformat(end_raw)
        except ValueError:
            logger.warning(
                "benchmark.case_page_pipeline_bad_window run_id=%s",
                getattr(run, "run_id", ""),
            )
            return []
        evaluator_names = sample_evaluator_names_for_report(
            rubric_refs=[item.evaluator_ref for item in rubric],
            registration_id=registration_id,
            stream_instance_id=stream,
        )
        sample_size = _positive_int(config.get("sample_size"))
        trace_ids = await service.sample_kafka_pipeline_records(
            KafkaSampleQuery(
                date_start=date_start,
                date_end=date_end,
                evaluator_names=evaluator_names,
                sample_size=sample_size,
                project_id=_text(config.get("project_id"))
                or _text(getattr(run, "project_id", None)),
            )
        )
    else:
        trace_ids = frozen
    if not trace_ids:
        return []
    return _copy_cases(
        await service.load_kafka_case_results(
            trace_ids=trace_ids,
            rubric=rubric,
            registration_id=registration_id,
            stream_instance_id=stream,
        )
    )


async def load_report_cases(
    service: RunCaseLoader,
    run: object,
) -> tuple[list[dict[str, object]], str]:
    """加载与报告 HTML 同源的 case 列表，并标明 aone 落在哪张表。"""
    config = _config_dict(run)
    if config.get("trace_daily_report") is True:
        cases = await _load_trace_daily_cases(service, run, config)
        return cases, _NOTIFICATION_SOURCE_TRACE
    if config.get("source") == _PIPELINE_SOURCE:
        cases = await _load_pipeline_cases(service, run, config)
        return cases, _NOTIFICATION_SOURCE_TRACE
    cases = await service.load_case_results(
        run_id=str(getattr(run, "run_id", "") or ""),
        benchmark_name=_text(getattr(run, "benchmark_name", None)),
        config=config,
    )
    return _copy_cases(cases), _NOTIFICATION_SOURCE_ITEMS


async def lookup_item_notification_dispatch(run_id: str, case_ids: list[str]) -> dict[str, dict[str, str]]:
    """``benchmark_run_items.notification_dispatch_json``，fused 行优先。"""
    if not case_ids:
        return {}
    rows = await BenchmarkRunItem.filter(run_id=run_id, item_id__in=case_ids)
    found: dict[str, dict[str, str]] = {}
    fused: dict[str, dict[str, str]] = {}
    for rec in rows:
        fields = _notification_dispatch_fields(rec.notification_dispatch_json)
        item_id = _text(getattr(rec, "item_id", None))
        if not fields or item_id is None:
            continue
        if getattr(rec, "criterion_id", None) == FUSED_CRITERION_ID:
            fused[item_id] = fields
        else:
            found.setdefault(item_id, fields)
    found.update(fused)
    return found


async def load_item_notification_row(run_id: str, case_id: str) -> BenchmarkRunItem | None:
    """选改派回写用的 ``benchmark_run_items`` 行，口径对齐 ``lookup_item_notification_dispatch``。

    fused 行有 ``notification_ref``+``notification_url`` 时用 fused；否则回退同 ``item_id`` 下
    带链接的其它 criterion 行；都没有链接时仍返回 fused（或任意一行）供上层 400。
    """
    rows = await BenchmarkRunItem.filter(run_id=run_id, item_id=case_id)
    if not rows:
        return None
    fused_row: BenchmarkRunItem | None = None
    fallback_row: BenchmarkRunItem | None = None
    for rec in rows:
        fields = _notification_dispatch_fields(rec.notification_dispatch_json)
        has_link = bool(fields.get("notification_ref") and fields.get("notification_url"))
        if getattr(rec, "criterion_id", None) == FUSED_CRITERION_ID:
            fused_row = rec
            if has_link:
                return rec
        elif has_link and fallback_row is None:
            fallback_row = rec
    if fallback_row is not None:
        return fallback_row
    return fused_row if fused_row is not None else rows[0]


async def lookup_trace_notification_dispatch(case_ids: list[str]) -> dict[str, dict[str, str]]:
    """``trace_pipeline_records.notification_dispatch_json``。"""
    if not case_ids:
        return {}
    rows = await TraceRecord.filter(trace_id__in=case_ids)
    found: dict[str, dict[str, str]] = {}
    for rec in rows:
        trace_id = _text(getattr(rec, "trace_id", None))
        if trace_id is None or trace_id in found:
            continue
        fields = _notification_dispatch_fields(rec.notification_dispatch_json)
        if fields:
            found[trace_id] = fields
    return found


async def owners_for_run(config: dict[str, object]) -> list[str]:
    """从评测注册文档取 owners（花名列表）；缺省空。

    查询与字段读取失败只打 warning 并返回空列表，不抬成 500；ImportError
    不再被吞掉（模块顶层已导入 find_*）。
    """
    reg_id = _registration_id(config)
    try:
        document = await find_agent_document_by_id(reg_id) if reg_id else None
        if document is None:
            snap = config.get("agent_registration")
            name = ""
            if isinstance(snap, dict):
                name = str(snap.get("name") or "").strip()
            if not name:
                name = str(config.get("agent_name") or config.get("agentName") or "").strip()
            if name:
                document = await find_agent_document_by_name(name)
        if document is None:
            return []
        # AgentDocument 必有 owners/owner；一并纳入兜底，避免异常结构抬成 500。
        owners = [str(item).strip() for item in document.owners if str(item).strip()]
        if owners:
            return owners
        owner = str(document.owner or "").strip()
        return [owner] if owner else []
    except Exception:
        logger.warning("run_case_page.owners_lookup_failed", exc_info=True)
        return []


async def trace_case_in_run(run: object, case_id: str) -> bool:
    """TRACE 源改派归属校验。

    优先用出报时写入的 ``sampled_trace_ids``（精确归属本 run）。
    缺失冻结列表时**拒绝**（避免仅凭 TraceRecord 存在造成跨 run 越权改派）。
    """
    frozen = _frozen_sampled_trace_ids(run)
    if frozen is None:
        logger.warning(
            "run_case_page.trace_case_no_sampled_ids run_id=%s case_id=%s",
            getattr(run, "run_id", ""),
            case_id,
        )
        return False
    return case_id in frozenset(frozen)


def _collect_attribution_categories(
    cases: Sequence[dict[str, object]],
    *,
    notification_source: str,
    benchmark_name: str,
) -> list[str]:
    """去重后的归因分类值（按出现顺序），供筛选项下拉。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for case in cases:
        category = extract_attribution_category(
            case,
            notification_source=notification_source,
            benchmark_name=benchmark_name,
        )
        if category is None or category in seen:
            continue
        seen.add(category)
        ordered.append(category)
    return ordered


async def build_case_page(
    service: RunCaseLoader,
    run: object,
    *,
    page_query: CasePageQuery,
) -> dict[str, object]:
    """组装 ``{data, meta}``。``total`` 是筛选后的 case 总数。

    ``status`` 默认 ``FAIL,POOR``（不合格）；传 ``ALL`` 或空串不过滤状态。
    """
    config = _config_dict(run)
    cases, notification_source = await load_report_cases(service, run)
    threshold = _pass_threshold(config)
    benchmark_name = _text(getattr(run, "benchmark_name", None)) or ""
    show_category = supports_attribution_category(benchmark_name)
    status_filter = (
        _DEFAULT_STATUS_FILTER
        if page_query.status is None
        else parse_status_filter(page_query.status)
    )
    categories = (
        _collect_attribution_categories(
            cases,
            notification_source=notification_source,
            benchmark_name=benchmark_name,
        )
        if show_category
        else []
    )
    filtered = filter_report_cases(
        cases,
        spec=CaseFilterSpec(
            query=page_query.query,
            statuses=status_filter,
            pass_threshold=threshold,
            attribution_category=page_query.attribution_category if show_category else "",
            notification_source=notification_source,
            benchmark_name=benchmark_name,
        ),
    )
    window = slice_cases(filtered, page=page_query.page, page_size=page_query.page_size)
    ids = [case_id for case in window if (case_id := _case_id(case))]
    if notification_source == _NOTIFICATION_SOURCE_TRACE:
        notification_map = await lookup_trace_notification_dispatch(ids)
    else:
        notification_map = await lookup_item_notification_dispatch(
            str(getattr(run, "run_id", "") or ""),
            ids,
        )
    owners = await owners_for_run(config)
    return {
        "data": [
            case_list_item(
                case,
                spec=CaseListItemSpec(
                    notification_dispatch=notification_map.get(_case_id(case)),
                    notification_source=notification_source,
                    benchmark_name=benchmark_name,
                    pass_threshold=threshold,
                    include_attribution_category=show_category,
                ),
            )
            for case in window
        ],
        "meta": {
            "page": page_query.page,
            "pageSize": page_query.page_size,
            "total": len(filtered),
            "owners": owners,
            "notificationSource": notification_source,
            "supportsAttributionCategory": show_category,
            "attributionCategories": categories,
            "reportUrl": report_url_of(run),
        },
    }


async def build_case_detail(
    service: RunCaseLoader,
    run: object,
    *,
    detail_query: CaseDetailQuery,
) -> dict[str, object] | None:
    """组装单 case GitBook 详情 ``{data, meta}``；未命中返回 None。

    现状与列表同源：``load_report_cases`` 全量再 ``find_report_case``。
    功能正确，但深链单条成本 ≈ 拉整页列表。

    TODO(perf): 按 ``case_id`` 直查 ``benchmark_run_items`` / 复用分页缓存；
    trace_daily、kafka 出报回放路径需单独口径，避免高频深链放大 DB。
    """
    case_id = detail_query.case_id.strip()
    if not case_id:
        return None
    config = _config_dict(run)
    # TODO(perf): 勿长期依赖全量 load；见 docstring。
    cases, notification_source = await load_report_cases(service, run)
    matched = find_report_case(cases, case_id)
    if matched is None:
        return None
    threshold = _pass_threshold(config)
    benchmark_name = _text(getattr(run, "benchmark_name", None)) or ""
    show_category = supports_attribution_category(benchmark_name)
    run_id = str(getattr(run, "run_id", "") or "")
    if notification_source == _NOTIFICATION_SOURCE_TRACE:
        notification_map = await lookup_trace_notification_dispatch([case_id])
    else:
        notification_map = await lookup_item_notification_dispatch(run_id, [case_id])
    owners = await owners_for_run(config)
    return {
        "data": case_detail_item(
            matched,
            spec=CaseListItemSpec(
                notification_dispatch=notification_map.get(case_id),
                notification_source=notification_source,
                benchmark_name=benchmark_name,
                pass_threshold=threshold,
                include_attribution_category=show_category,
            ),
        ),
        "meta": {
            "owners": owners,
            "notificationSource": notification_source,
            "supportsAttributionCategory": show_category,
            "reportUrl": report_url_of(run),
        },
    }


#: 供改指派等兄弟模块复用（避免字段口径 / 枚举漂移）。
NOTIFICATION_SOURCE_ITEMS = _NOTIFICATION_SOURCE_ITEMS
NOTIFICATION_SOURCE_TRACE = _NOTIFICATION_SOURCE_TRACE
DEFAULT_STATUS_FILTER = _DEFAULT_STATUS_FILTER
notification_dispatch_fields = _notification_dispatch_fields
report_case_id = _case_id
run_config_dict = _config_dict


__all__ = [
    "NOTIFICATION_SOURCE_ITEMS",
    "NOTIFICATION_SOURCE_TRACE",
    "CaseDetailQuery",
    "CaseFilterSpec",
    "CaseListItemSpec",
    "CasePageQuery",
    "DEFAULT_STATUS_FILTER",
    "RunCaseLoader",
    "notification_dispatch_fields",
    "build_case_detail",
    "build_case_page",
    "case_detail_item",
    "case_list_item",
    "detail_expected_text",
    "detail_input_text",
    "extract_attribution_category",
    "filter_report_cases",
    "find_report_case",
    "load_item_notification_row",
    "load_report_cases",
    "lookup_item_notification_dispatch",
    "lookup_trace_notification_dispatch",
    "owners_for_run",
    "parse_status_filter",
    "report_case_id",
    "report_url_of",
    "resolve_notification_source",
    "run_config_dict",
    "slice_cases",
    "supports_attribution_category",
    "trace_case_in_run",
]
