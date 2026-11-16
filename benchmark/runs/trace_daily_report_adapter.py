"""Trace Daily Report 字段适配（仅日报链路，不影响其他 benchmark report）。

设计结论（case 状态）：
- ``TraceStatus``（new/scored/triaged/dispatched）是管线生命周期，
  **不能**直接替代报告 Case「case状态」与 KPI「档位」所用的
  EXCELLENT/GOOD/ACCEPTABLE/POOR/FAIL。
- ``triage_level``（good/watch/mild_bad/severe_bad）是质量相关分级，
  经兼容映射打到既有 grade 词表，KPI / 筛选 / 档位分布无需改模板。
- ``attribution_overridden`` 为归因推翻标记（非分级本身）：日报取
  ``attribution_detail_json.corrected_triage_level``（与标量同一次
  UPDATE 写入，见 ``update_attribution``）二次映射为 grade；解析不出
  合法分级时按综合质量门槛回退。
- ``triage_level`` 缺失 / 弃权（``no_decision``）时回退：综合质量分
  ≥ ``pass_threshold`` → GOOD，否则 FAIL（不影响 ieval / search_eval
  等其他业务）。
"""

import math
import statistics
from dataclasses import dataclass
from typing import Any

from benchmark.records.enums import TriageLevel
from benchmark.runs.common.trace_url_builder import build_direct_trace_url

#: triage_level → 报告 grade（兼容既有 KPI / 筛选词表）。
#: ``attribution_overridden`` / ``no_decision`` 不在此表：前者走
#: ``corrected_triage_level`` 二次解析（见 :func:`map_triage_to_grade`），
#: 后者与缺失值同策略按综合质量门槛回退。
_TRIAGE_TO_GRADE: dict[str, str] = {
    TriageLevel.GOOD.value: "GOOD",
    TriageLevel.WATCH.value: "ACCEPTABLE",
    TriageLevel.MILD_BAD.value: "POOR",
    TriageLevel.SEVERE_BAD.value: "FAIL",
}

#: 综合质量维名称关键字（Rubric ``Criterion.name``）
_OVERALL_QUALITY_NAME = "综合质量"

#: ``judge_json.sub_scores`` 展开的维度名 → 报表维度展示名。
#: Kafka 链路 evaluator 落库的 ``sub_scores`` key 为业务自定义 axes
#: （如 search_ai_card 维度 id）；未知 key 原样透传作展示名。
_SUB_SCORE_DISPLAY_NAMES: dict[str, str] = {
    "core_completion": "核心完成度",
    "expression": "表达质量",
}

#: ``extra.expression.score`` 为 1-5 制，报告口径为 [0,1]；按
#: ``aggregate_scores`` 同公式归一化（weights 不参与单维展示）。
_EXPRESSION_MIN = 1.0
_EXPRESSION_MAX = 5.0

#: 未知权重维度按 0/1 制。expression 为 1-5，core_completion 为 0/1。
_DEFAULT_WEIGHT_FULL_SCORE = 1.0
#: 与报告页黄/绿分界一致：达到满分的该比例记为通过。
_WEIGHT_PASS_RATIO = 0.8
_P95_QUANTILE = 0.95
_WEIGHT_FULL_SCORES: dict[str, float] = {
    "expression": _EXPRESSION_MAX,
    "core_completion": _DEFAULT_WEIGHT_FULL_SCORE,
}


@dataclass(frozen=True)
class TraceDailyDecorFields:
    """decorate_trace_daily_case 的装饰元数据收敛（FUNC-PARAMS-001）。

    来源 ``trace_pipeline_records`` 行（多 evaluator 行取首次非空）：
    ``attribution_overridden`` 场景依赖 ``attribution_detail``（标量与
    ``corrected_triage_level`` 同一次 UPDATE 写入）；各维度评分依赖
    ``judge_json``（维度分展开见 :func:`_expand_judge_sub_dimensions`）。
    """

    triage_level: str | None = None
    root_cause: str | None = None
    pipeline_status: str | None = None
    project_id: str | None = None
    meta: dict[str, Any] | None = None
    attribution_detail: dict[str, Any] | None = None
    judge_json: dict[str, Any] | None = None


def resolve_overall_criterion_id(
    rubric: list[Any],
    scores: dict[str, Any] | None,
) -> str | None:
    """定位「综合质量」criterion id：名称命中 > 单维 rubric > 首个非 fused 分数键。"""
    for c in rubric:
        name = str(getattr(c, "name", "") or "")
        if _OVERALL_QUALITY_NAME in name:
            cid = getattr(c, "id", None)
            if cid:
                return str(cid)
    if len(rubric) == 1:
        cid = getattr(rubric[0], "id", None)
        return str(cid) if cid else None
    if isinstance(scores, dict):
        for key in scores:
            if key and key != "fused":
                return str(key)
    return None


def map_triage_to_grade(
    triage_level: str | None,
    *,
    overall_score: float | None,
    pass_threshold: float,
    attribution_detail: dict[str, Any] | None = None,
) -> str | None:
    """日报 case 状态：优先 triage_level 映射，缺失则按综合质量门槛回退。

    ``triage_level == attribution_overridden`` 时取
    ``attribution_detail['corrected_triage_level']``（归因推翻给出的
    纠正分级）二次映射；bucket 缺失 / 值非法（含 ``no_decision``）时
    与缺失值同策略回退综合质量门槛。归因推翻场景标量与 bucket 由
    ``update_attribution`` 同一次 UPDATE 写入，正常不会不一致。
    """
    level = triage_level.strip().lower() if isinstance(triage_level, str) else ""
    if level == TriageLevel.ATTRIBUTION_OVERRIDDEN.value:
        level = _corrected_level_from(attribution_detail)
    if level:
        mapped = _TRIAGE_TO_GRADE.get(level)
        if mapped:
            return mapped
    if overall_score is None:
        return None
    try:
        score = float(overall_score)
    except (TypeError, ValueError):
        return None
    return "GOOD" if score >= pass_threshold else "FAIL"


def _corrected_level_from(attribution_detail: dict[str, Any] | None) -> str:
    """归因推翻：从 ``attribution_detail_json`` 提取纠正分级（小写归一）。

    非法 / 缺失返回空串（调用方按缺失回退门槛，不误报 GOOD/FAIL）。
    """
    if not isinstance(attribution_detail, dict):
        return ""
    raw = attribution_detail.get("corrected_triage_level")
    if isinstance(raw, str):
        return raw.strip().lower()
    return ""


#: ``extra`` 下的非维度 key（权重配置/元数据源标记），展开时排除。
_EXTRA_NON_DIMENSION_KEYS: frozenset[str] = frozenset({"weights"})


def _expand_judge_sub_dimensions(
    judge_json: dict[str, Any] | None,
    scores: dict[str, Any],
) -> list[str]:
    """把 ``judge_json`` 维度分展开进 ``scores``，返回展开的维度展示名。

    两个取值源，优先级递减：

    1. ``sub_scores``（dict[str, float]）——evaluator 声明 axes 的维度分；
       key 映射展示名，值为原始分（不做量纲归一，量纲由 evaluator 自管）；
    2. ``extra`` 下**除 ``weights`` 外所有值为含 ``score`` 的 dict 的 key**
       （amap_agent 复合 judge 落库形态：``core_completion`` / ``expression``
       /未来新增维度自动识别；``query_category`` 无 ``score`` 字段自然
       排除，``weights`` 为权重配置显式排除）——key 映射展示名展开；
       其中 ``expression`` 为 1-5 制，按 ``aggregate_scores`` 同公式归一
       到 [0,1]，``raw_score`` 保留原始分供展示标注。

    展开条目写入形状与 ``_merge_case_row`` 的 criterion 行对齐
    （``{score, reason}``），且**不覆盖既有同名 key**（rubric criterion
    已提供的维度优先）。返回的展示名列表写入 ``case['dim_scores']``
    供报告模板渲染展开行（key 为展示名而非 criterion id）。
    """
    expanded: list[str] = []
    if not isinstance(judge_json, dict):
        return expanded
    sub_scores = judge_json.get("sub_scores")
    if isinstance(sub_scores, dict):
        for key, value in sub_scores.items():
            if value is None or key in scores or key == "fused":
                continue
            display = _SUB_SCORE_DISPLAY_NAMES.get(str(key), str(key))
            if display in scores:
                continue
            scores[display] = {"score": value, "reason": None}
            expanded.append(display)
    extra = judge_json.get("extra")
    if not isinstance(extra, dict):
        return expanded
    for key, entry in extra.items():
        # 维度条目判定：非排除 key + dict 且含非 None 的 score 字段
        if (
            key in _EXTRA_NON_DIMENSION_KEYS
            or not isinstance(entry, dict)
            or entry.get("score") is None
        ):
            continue
        display = _SUB_SCORE_DISPLAY_NAMES.get(str(key), str(key))
        if display in scores:
            continue
        score_value = entry["score"]
        row: dict[str, Any] = {
            "score": score_value,
            "reason": entry.get("reason"),
        }
        # expression 为 1-5 制：归一到报告 [0,1] 口径，raw_score 保留原始分
        if key == "expression" and isinstance(score_value, (int, float)):
            raw = float(score_value)
            norm = (raw - _EXPRESSION_MIN) / (_EXPRESSION_MAX - _EXPRESSION_MIN)
            row["score"] = round(norm, 4)
            row["raw_score"] = raw
        scores[display] = row
        expanded.append(display)
    return expanded


def resolve_trace_url(
    *,
    meta: dict[str, Any] | None,
    project_id: str | None,
    trace_id: str | None,
) -> str | None:
    """日报 Trace 链接：优先 ``trace_meta_json.langfuse_url``，否则直拼回填旧数据。"""
    if isinstance(meta, dict):
        raw = meta.get("trace_url")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    pid = str(project_id).strip() if project_id else ""
    tid = str(trace_id).strip() if trace_id else ""
    if pid and tid:
        return build_direct_trace_url(pid, tid)
    return None


def decorate_trace_daily_case(
    case: dict[str, Any],
    *,
    rubric: list[Any],
    pass_threshold: float,
    fields: TraceDailyDecorFields,
) -> None:
    """原地装饰日报 case：综合分 / grade / Trace 链接 / 根因（幂等字段覆盖）。"""
    scores = case.get("scores")
    if not isinstance(scores, dict):
        scores = {}
        case["scores"] = scores

    overall_cid = resolve_overall_criterion_id(rubric, scores)
    overall_row = scores.get(overall_cid) if overall_cid else None
    overall_score: float | None = None
    if isinstance(overall_row, dict) and overall_row.get("score") is not None:
        try:
            overall_score = float(overall_row["score"])
        except (TypeError, ValueError):
            overall_score = None
        # 综合得分列读 scores.fused.score —— 与 综合质量 同值
        fused = dict(overall_row)
        scores["fused"] = fused

    # 各维度评分：judge_json 维度分展开（sub_scores / extra 复合 judge 维度），
    # 与 rubric criterion 行并存且不覆盖（多维度名共存时展开仅补缺）。
    _attach_judge_dimensions(case, scores, fields.judge_json)

    case["grade"] = map_triage_to_grade(
        fields.triage_level,
        overall_score=overall_score,
        pass_threshold=pass_threshold,
        attribution_detail=fields.attribution_detail,
    )
    # 管线 status 仅作元数据保留，不覆盖 grade（语义不同）
    if fields.pipeline_status:
        case["pipeline_status"] = fields.pipeline_status

    tid = str(case.get("trace_id") or case.get("case_id") or "")
    url = resolve_trace_url(
        meta=fields.meta,
        project_id=fields.project_id,
        trace_id=tid or None,
    )
    if url:
        case["trace_url"] = url

    # 详细归因数据：``attribution_detail_json`` bucket 打底 + 标量
    # ``root_cause`` 覆盖（两者同源，``update_attribution`` 同一次 UPDATE
    # 写入）。报告「详细归因数据」折叠栏直接消费 bucket 全量字段
    # （category / owner_group / evidence_raw / 分级纠正等）。
    attr = case.get("attribution")
    if not isinstance(attr, dict):
        attr = {}
    detail = fields.attribution_detail
    if isinstance(detail, dict):
        for key, value in detail.items():
            if value is not None and key not in attr:
                attr[key] = value
    rc = fields.root_cause.strip() if isinstance(fields.root_cause, str) else ""
    if rc:
        case["root_cause_display"] = rc
        attr["root_cause"] = rc
    if attr:
        case["attribution"] = attr


def _attach_judge_dimensions(
    case: dict[str, Any],
    scores: dict[str, Any],
    judge_json: dict[str, Any] | None,
) -> None:
    """展开 case 明细维度，并记下 weights 维度供评测维度栏聚合。"""
    expanded_dims = _expand_judge_sub_dimensions(judge_json, scores)
    if expanded_dims:
        case["dim_scores"] = expanded_dims
    weight_rows = _weight_dimension_rows(judge_json)
    if not weight_rows:
        return
    case["weight_dimensions"] = weight_rows
    # 评测维度栏 criteria.id 是 weights 的原始 key，Case 表按该 id 取分。
    # 写入与 full_score 同量纲的原始分。展示名 key 仍是归一化分，留给
    # dim_scores；模板在 criteria 已覆盖该展示名时不再重复渲染。
    for row in weight_rows:
        cid = str(row["id"])
        if cid in scores:
            continue
        scores[cid] = {"score": row["score"], "reason": row.get("reason")}


def _weight_full_score(key: str) -> float:
    """权重维度的原始满分；未知 key 按 0/1 制。"""
    return _WEIGHT_FULL_SCORES.get(key, _DEFAULT_WEIGHT_FULL_SCORE)


def _weight_dimension_rows(judge_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``extra.weights`` 的 key → ``extra[key].score``（原始分，不归一）。

    只取 weights 里能在 extra 找到含 ``score`` 的条目。amap_agent 为
    ``core_completion`` / ``expression`` 两项。
    """
    if not isinstance(judge_json, dict):
        return []
    extra = judge_json.get("extra")
    if not isinstance(extra, dict):
        return []
    weights = extra.get("weights")
    if not isinstance(weights, dict) or not weights:
        return []
    rows: list[dict[str, Any]] = []
    for key, weight in weights.items():
        entry = extra.get(key)
        if not isinstance(entry, dict) or entry.get("score") is None:
            continue
        try:
            score = float(entry["score"])
            weight_value = float(weight)
        except (TypeError, ValueError):
            continue
        name = str(key)
        rows.append(
            {
                "id": name,
                "name": _SUB_SCORE_DISPLAY_NAMES.get(name, name),
                "weight": weight_value,
                "score": score,
                "full_score": _weight_full_score(name),
                "reason": entry.get("reason"),
            }
        )
    return rows


def _percentile(sorted_scores: list[float], pct: float) -> float:
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    rank = math.ceil(pct * len(sorted_scores)) - 1
    return sorted_scores[min(max(rank, 0), len(sorted_scores) - 1)]


def _metric_payload(criterion_id: str, scores: list[float], full_score: float) -> dict[str, Any]:
    ordered = sorted(scores)
    mean = statistics.fmean(ordered)
    std = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    passed = sum(
        1 for score in ordered if full_score > 0 and score / full_score >= _WEIGHT_PASS_RATIO
    )
    return {
        "criterion_id": criterion_id,
        "evaluator_ref": "judge_extra",
        "sample_count": len(ordered),
        "success_count": len(ordered),
        "failed_count": 0,
        "mean": mean,
        "median": float(statistics.median(ordered)),
        "p95": _percentile(ordered, _P95_QUANTILE),
        "min": ordered[0],
        "max": ordered[-1],
        "std": std,
        "pass_rate": passed / len(ordered),
    }


@dataclass(frozen=True)
class WeightDimensionSection:
    """评测维度栏：weights 两项及其聚合分。"""

    criteria: list[dict[str, Any]]
    criterion_metrics: dict[str, dict[str, Any]]


def build_weight_dimension_section(
    cases: list[dict[str, Any]],
) -> WeightDimensionSection | None:
    """用 case 上的 ``weight_dimensions`` 替换评测维度栏。

    无 weights 数据时返回 ``None``（调用方保持 rubric 维度）。
    """
    buckets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for case in cases:
        rows = case.get("weight_dimensions")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            cid = str(row["id"])
            try:
                score = float(row["score"])
            except (TypeError, ValueError, KeyError):
                continue
            bucket = buckets.get(cid)
            if bucket is None:
                bucket = {"meta": row, "scores": []}
                buckets[cid] = bucket
                order.append(cid)
            bucket["scores"].append(score)
    if not order:
        return None
    criteria: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    for cid in order:
        meta = buckets[cid]["meta"]
        scores = buckets[cid]["scores"]
        raw_full = meta.get("full_score")
        full_score = _DEFAULT_WEIGHT_FULL_SCORE
        if isinstance(raw_full, (int, float)) and not isinstance(raw_full, bool):
            full_score = float(raw_full)
        criteria.append(
            {
                "id": cid,
                "name": meta.get("name") or cid,
                "description": "",
                "weight": meta.get("weight"),
                "evaluator_ref": "judge_extra",
                "full_score": full_score,
            }
        )
        metrics[cid] = _metric_payload(cid, scores, full_score)
    return WeightDimensionSection(criteria=criteria, criterion_metrics=metrics)


def apply_weight_dimension_section(spec: dict[str, Any]) -> None:
    """日报 spec：评测维度栏改为 ``extra.weights`` 的维度与原始分。"""
    if not spec.get("trace_daily_report"):
        return
    cases = spec.get("case_results")
    if not isinstance(cases, list):
        return
    section = build_weight_dimension_section(cases)
    if section is None:
        return
    spec["criteria"] = section.criteria
    spec["criterion_metrics"] = section.criterion_metrics
    spec.pop("today_criterion_means", None)


__all__ = [
    "TraceDailyDecorFields",
    "WeightDimensionSection",
    "apply_weight_dimension_section",
    "build_weight_dimension_section",
    "decorate_trace_daily_case",
    "map_triage_to_grade",
    "resolve_trace_url",
    "resolve_overall_criterion_id",
]
