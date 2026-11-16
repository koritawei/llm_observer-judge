"""Unit tests for trace daily report field adapter (isolation helpers)."""

from benchmark.runs.trace_daily_report_adapter import (
    TraceDailyDecorFields,
    apply_weight_dimension_section,
    decorate_trace_daily_case,
    map_triage_to_grade,
    resolve_trace_url,
    resolve_overall_criterion_id,
)
from benchmark.reporting.framework.schema import Criterion


def test_map_triage_to_grade_uses_triage_not_pipeline_status() -> None:
    assert map_triage_to_grade("good", overall_score=0.1, pass_threshold=0.5) == "GOOD"
    assert map_triage_to_grade("watch", overall_score=0.9, pass_threshold=0.5) == "ACCEPTABLE"
    assert map_triage_to_grade("mild_bad", overall_score=0.9, pass_threshold=0.5) == "POOR"
    assert map_triage_to_grade("severe_bad", overall_score=0.9, pass_threshold=0.5) == "FAIL"


def test_map_triage_falls_back_to_pass_threshold() -> None:
    assert map_triage_to_grade(None, overall_score=0.6, pass_threshold=0.5) == "GOOD"
    assert map_triage_to_grade(None, overall_score=0.4, pass_threshold=0.5) == "FAIL"
    assert map_triage_to_grade("no_decision", overall_score=0.4, pass_threshold=0.5) == "FAIL"


def test_map_triage_overridden_uses_corrected_level() -> None:
    """attribution_overridden takes a second mapping from corrected_triage_level (the attribution-overrides-the-judge case)."""
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=0.9,
            pass_threshold=0.5,
            attribution_detail={"corrected_triage_level": "watch"},
        )
        == "ACCEPTABLE"
    )
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=0.1,
            pass_threshold=0.5,
            attribution_detail={"corrected_triage_level": "good"},
        )
        == "GOOD"
    )
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=0.9,
            pass_threshold=0.5,
            attribution_detail={"corrected_triage_level": "mild_bad"},
        )
        == "POOR"
    )
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=0.9,
            pass_threshold=0.5,
            attribution_detail={"corrected_triage_level": "severe_bad"},
        )
        == "FAIL"
    )


def test_map_triage_overridden_falls_back_when_corrected_invalid() -> None:
    """corrected missing / illegal / abstained → fall back to the composite-quality threshold, and do not report the wrong grade."""
    # bucket missing / None / not a dict → fall back
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=0.6,
            pass_threshold=0.5,
            attribution_detail=None,
        )
        == "GOOD"
    )
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=0.4,
            pass_threshold=0.5,
            attribution_detail={"root_cause": "x"},
        )
        == "FAIL"
    )
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=0.4,
            pass_threshold=0.5,
            attribution_detail={"corrected_triage_level": None},
        )
        == "FAIL"
    )
    # An illegal value / an abstain value is not in _TRIAGE_TO_GRADE → fall back
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=0.6,
            pass_threshold=0.5,
            attribution_detail={"corrected_triage_level": "not_a_level"},
        )
        == "GOOD"
    )
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=0.4,
            pass_threshold=0.5,
            attribution_detail={"corrected_triage_level": "no_decision"},
        )
        == "FAIL"
    )
    # bucket missing and no composite score → None (it does not enter the grade-band distribution)
    assert (
        map_triage_to_grade(
            "attribution_overridden",
            overall_score=None,
            pass_threshold=0.5,
            attribution_detail=None,
        )
        is None
    )


def test_map_triage_handles_blank_and_unknown_levels() -> None:
    """An empty string / whitespace / an unknown value falls back to the threshold on the same policy as null."""
    assert map_triage_to_grade("", overall_score=0.6, pass_threshold=0.5) == "GOOD"
    assert map_triage_to_grade("   ", overall_score=0.4, pass_threshold=0.5) == "FAIL"
    assert map_triage_to_grade("unknown_level", overall_score=0.6, pass_threshold=0.5) == "GOOD"
    assert map_triage_to_grade("", overall_score=None, pass_threshold=0.5) is None


def test_map_triage_overridden_ignores_redundant_attribution_detail() -> None:
    """When the grade is not overridden, attribution_detail does not participate (it is not pulled into a second parse by mistake)."""
    assert (
        map_triage_to_grade(
            "severe_bad",
            overall_score=0.9,
            pass_threshold=0.5,
            attribution_detail={"corrected_triage_level": "good"},
        )
        == "FAIL"
    )


def test_resolve_overall_criterion_by_name() -> None:
    rubric = [
        Criterion(id="other", name="表达", weight=0.5, evaluator_ref="x"),
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    assert resolve_overall_criterion_id(rubric, {}) == "amap_agent_overall"


def test_resolve_trace_url_prefers_meta_then_direct() -> None:
    assert (
        resolve_trace_url(
            meta={"trace_url": "https://example/traces/t1"},
            project_id="p",
            trace_id="t",
        )
        == "https://example/traces/t1"
    )
    url = resolve_trace_url(meta={}, project_id="proj-a", trace_id="tid-b")
    assert url is not None
    assert "/project/proj-a/traces/tid-b?traceId=tid-b" in url


def test_decorate_sets_fused_grade_url_root_cause() -> None:
    rubric = [
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    case: dict = {
        "case_id": "tid-1",
        "trace_id": "tid-1",
        "scores": {
            "amap_agent_overall": {
                "score": 0.72,
                "reason": "ok",
                "eval_failed": False,
            },
        },
    }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="mild_bad",
            root_cause="tool_error",
            pipeline_status="triaged",
            project_id="proj-1",
            meta={"trace_url": "https://agentfuse.amap-inc.com/project/proj-1/traces/tid-1"},
        ),
    )
    assert case["grade"] == "POOR"
    assert case["pipeline_status"] == "triaged"
    assert case["scores"]["fused"]["score"] == 0.72
    assert case["root_cause_display"] == "tool_error"
    assert case["attribution"]["root_cause"] == "tool_error"
    assert "tid-1" in case["trace_url"]


def test_decorate_overridden_case_uses_corrected_level() -> None:
    """The decorate chain passes attribution_detail through: an attribution-override case enters the band by the corrected grade."""
    rubric = [
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    case: dict = {
        "case_id": "tid-2",
        "trace_id": "tid-2",
        "scores": {
            "amap_agent_overall": {
                "score": 0.1,
                "reason": "judge 判严重",
                "eval_failed": False,
            },
        },
    }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="attribution_overridden",
            root_cause="断句/分词错误",
            pipeline_status="triaged",
            project_id="proj-1",
            meta={"trace_url": "https://agentfuse.amap-inc.com/project/proj-1/traces/tid-2"},
            attribution_detail={
                "root_cause": "断句/分词错误",
                "original_triage_level": "severe_bad",
                "corrected_triage_level": "watch",
            },
        ),
    )
    # The composite score 0.1 < 0.5, but attribution corrected it to watch → ACCEPTABLE (do not fall back to the threshold)
    assert case["grade"] == "ACCEPTABLE"


def test_decorate_injects_full_attribution_detail() -> None:
    """Detailed attribution data: the bucket is injected as the base of case.attribution, and the scalar root_cause overwrites it."""
    rubric = [
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    case: dict = {
        "case_id": "tid-3",
        "trace_id": "tid-3",
        "scores": {
            "amap_agent_overall": {"score": 0.3, "reason": "bad", "eval_failed": False},
        },
    }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="mild_bad",
            root_cause="参数填充错误",
            pipeline_status="triaged",
            project_id="proj-1",
            attribution_detail={
                "category": "模型输出",
                "confidence": 0.0,
                "root_cause": "参数填充错误",
                "owner_group": "agent-eval-default",
                "owner_domain": "模型输出",
                "evidence_raw": '["1. 证据条目"]',
                "reasoning_raw": "推理原文",
                "attribution_raw": '{"trace_id": "t"}',
                "attribution_status": "completed",
                "needs_human_review": False,
                "candidate_root_cause": None,
                "original_triage_level": "mild_bad",
            },
        ),
    )
    attr = case["attribution"]
    # The bucket's full field set is in place (a None candidate_root_cause is not injected)
    assert attr["category"] == "模型输出"
    assert attr["owner_group"] == "agent-eval-default"
    assert attr["evidence_raw"] == '["1. 证据条目"]'
    assert attr["reasoning_raw"] == "推理原文"
    assert attr["original_triage_level"] == "mild_bad"
    assert "candidate_root_cause" not in attr
    # The scalar root_cause overwrites in sync (same source value, idempotent)
    assert attr["root_cause"] == "参数填充错误"
    assert case["root_cause_display"] == "参数填充错误"


def test_decorate_attribution_detail_without_scalar_root_cause() -> None:
    """The scalar root_cause is missing: the bucket alone can still populate case.attribution (the old path had only the one root_cause key)."""
    rubric = [
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    case: dict = {
        "case_id": "tid-4",
        "scores": {"amap_agent_overall": {"score": 0.9, "reason": "ok", "eval_failed": False}},
    }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="good",
            attribution_detail={"category": "模型输出", "root_cause": "桶内根因"},
        ),
    )
    # root_cause takes the bucket value (when there is no scalar overwrite, the bucket's original value is kept)
    assert case["attribution"]["root_cause"] == "桶内根因"
    assert "root_cause_display" not in case


def test_decorate_no_attribution_keeps_legacy_behavior() -> None:
    """Both the bucket and the scalar are missing: the case has no attribution key (consistent with the old path)."""
    rubric = [
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    case: dict = {
        "case_id": "tid-5",
        "scores": {"amap_agent_overall": {"score": 0.9, "reason": "ok", "eval_failed": False}},
    }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(triage_level="good"),
    )
    assert "attribution" not in case
    assert "root_cause_display" not in case


def test_decorate_existing_case_attribution_not_clobbered() -> None:
    """The case already has attribution (for example injected upstream): the bucket only fills missing keys and does not overwrite existing ones."""
    rubric = [
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    case: dict = {
        "case_id": "tid-6",
        "scores": {"amap_agent_overall": {"score": 0.3, "reason": "bad", "eval_failed": False}},
        "attribution": {"root_cause": "上游已有", "summary": "既有摘要"},
    }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="mild_bad",
            root_cause="标量根因",
            attribution_detail={"root_cause": "桶内根因", "category": "模型输出"},
        ),
    )
    attr = case["attribution"]
    # Existing keys win, and the bucket only fills category; the scalar root_cause overwrites last
    assert attr["summary"] == "既有摘要"
    assert attr["category"] == "模型输出"
    assert attr["root_cause"] == "标量根因"


def test_expand_judge_extra_dimensions() -> None:
    """amap_agent composite judge: extra.core_completion/expression expand into dimension rows."""
    rubric = [
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    case: dict = {
        "case_id": "tid-7",
        "scores": {
            "amap_agent_overall": {"score": 0.75, "reason": "综合 0.75", "eval_failed": False}
        },
    }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="mild_bad",
            judge_json={
                "score": "0.75",
                "reason": "core_completion=1，完成；expression=4，流畅。综合判定为 0.75。",
                "extra": {
                    "core_completion": {"score": 1, "reason": "任务完成"},
                    "expression": {"score": 4, "reason": "表达流畅"},
                },
            },
        ),
    )
    # The composite-quality row is kept, and dimension rows are appended
    assert case["scores"]["amap_agent_overall"]["score"] == 0.75
    assert case["dim_scores"] == ["核心完成度", "表达质量"]
    assert case["scores"]["核心完成度"] == {"score": 1, "reason": "任务完成"}
    # expression scores 4 (on a 1-5 scale) → normalized (4-1)/4 = 0.75; raw_score keeps the raw score
    assert case["scores"]["表达质量"]["score"] == 0.75
    assert case["scores"]["表达质量"]["raw_score"] == 4
    assert case["scores"]["表达质量"]["reason"] == "表达流畅"


def test_expand_judge_sub_scores_dimensions() -> None:
    """sub_scores axes expand: a key is mapped to a display name, and an unknown key is passed through as-is."""
    case: dict = {
        "case_id": "tid-8",
        "scores": {},
    }
    decorate_trace_daily_case(
        case,
        rubric=[],
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="good",
            judge_json={
                "score": "0.8",
                "reason": "ok",
                "sub_scores": {"core_completion": 1.0, "custom_axis": 0.6, "empty": None},
            },
        ),
    )
    assert case["dim_scores"] == ["核心完成度", "custom_axis"]
    assert case["scores"]["核心完成度"] == {"score": 1.0, "reason": None}
    assert case["scores"]["custom_axis"] == {"score": 0.6, "reason": None}
    # A None value is not expanded
    assert "empty" not in case["scores"]


def test_expand_judge_dimensions_not_overwrite_criterion() -> None:
    """When a rubric criterion row already provides a dimension of the same name, expansion does not overwrite it (the criterion wins)."""
    rubric = [
        Criterion(
            id="d_core",
            name="核心完成度",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    case: dict = {
        "case_id": "tid-9",
        "scores": {"d_core": {"score": 0.9, "reason": "rubric 行", "eval_failed": False}},
    }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="good",
            judge_json={
                "score": "0.8",
                "reason": "ok",
                "sub_scores": {"core_completion": 0.3},
            },
        ),
    )
    # The criterion row is indexed by id (d_core) and the expanded row by the display name (core completion):
    # the two coexist and do not overwrite each other, so the template can render both dimension rows
    assert case["scores"]["d_core"]["score"] == 0.9
    assert case["dim_scores"] == ["核心完成度"]
    assert case["scores"]["核心完成度"]["score"] == 0.3


def test_expand_judge_dimensions_absent_judge_json() -> None:
    """No judge_json / no dimension data: do not write dim_scores, and scores are unaffected (the old behavior)."""
    case: dict = {
        "case_id": "tid-10",
        "scores": {"c1": {"score": 0.5, "reason": "r", "eval_failed": False}},
    }
    decorate_trace_daily_case(
        case,
        rubric=[],
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(triage_level="good"),
    )
    assert "dim_scores" not in case
    # With an empty rubric the composite dimension falls back to the first score key c1 and mirrors fused (existing behavior, not introduced by this change)
    assert case["scores"]["c1"]["score"] == 0.5
    assert case["scores"]["fused"]["score"] == 0.5


def test_expand_judge_extra_generic_dimensions() -> None:
    """Generic walk of extra dimensions: every dict that contains score is expanded, and weights / a key with no score are excluded.

    The data shape is taken from a real production judge_json (core_completion=0, the gate did not pass, + expression=3).
    """
    rubric = [
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]
    case: dict = {
        "case_id": "tid-11",
        "scores": {
            "amap_agent_overall": {"score": 0.0, "reason": "综合 0.0", "eval_failed": False}
        },
    }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="attribution_overridden",
            attribution_detail={"corrected_triage_level": "watch"},
            judge_json={
                "score": "0.0",
                "reason": "core_completion=0，未满足……综合判定为 0.0。",
                "updated_at": "2026-10-08T16:07:46.894828+08:00",
                "triage_level": "attribution_overridden",
                "extra": {
                    "weights": {"expression": 0.5, "core_completion": 0.5},
                    "expression": {"score": 3, "reason": "整体更像范文，书面化明显，定为 3 分。"},
                    "query_category": {
                        "category": "other",
                        "confidence": 0.92,
                        "difficulty": "medium",
                        "sub_scenario": "other",
                        "reason": "",
                    },
                    "core_completion": {
                        "score": 0,
                        "reason": "未满足：缺少“上一篇”内容，Agent 未说明也未请求提供。",
                        "checkpoints": [
                            {
                                "id": "c1",
                                "text": "应说明缺少待融合内容",
                                "type": "positive",
                                "priority": "core",
                            }
                        ],
                        "checkpoint_source": "llm_generated",
                    },
                },
            },
        ),
    )
    # weights / query_category (no score) are not expanded; expansion order follows extra insertion order (expression first)
    assert case["dim_scores"] == ["表达质量", "核心完成度"]
    assert case["scores"]["核心完成度"]["score"] == 0
    assert "缺少" in case["scores"]["核心完成度"]["reason"]
    # expression scores 3 (on a 1-5 scale) → normalized (3-1)/4 = 0.5; raw_score keeps 3
    assert case["scores"]["表达质量"]["score"] == 0.5
    assert case["scores"]["表达质量"]["raw_score"] == 3
    # The evaluation-dimension section takes only the two keys in weights, and the scores are the raw scores in extra
    assert [row["id"] for row in case["weight_dimensions"]] == [
        "expression",
        "core_completion",
    ]
    assert case["weight_dimensions"][0]["score"] == 3
    assert case["weight_dimensions"][0]["weight"] == 0.5
    assert case["weight_dimensions"][1]["score"] == 0
    # The case table scores by criteria.id, and that score must be a raw score, the same as full_score
    assert case["scores"]["expression"]["score"] == 3
    assert case["scores"]["core_completion"]["score"] == 0
    # Attribution overrides the judge → corrected=watch → ACCEPTABLE (grade-band mapping regression)
    assert case["grade"] == "ACCEPTABLE"


def test_expand_judge_extra_future_dimension_auto_detected() -> None:
    """A dimension added to extra in the future (an unknown key) is recognized and expanded automatically (forward compatible, no code change)."""
    case: dict = {"case_id": "tid-12", "scores": {}}
    decorate_trace_daily_case(
        case,
        rubric=[],
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="good",
            judge_json={
                "score": "0.8",
                "reason": "ok",
                "extra": {
                    "safety": {"score": 1, "reason": "无安全问题"},
                    "weights": {"a": 0.5, "b": 0.5},
                    "meta_note": "非维度元数据",
                },
            },
        ),
    )
    assert case["dim_scores"] == ["safety"]
    assert case["scores"]["safety"] == {"score": 1, "reason": "无安全问题"}
    # A weights key that has no score in extra does not enter the evaluation-dimension section
    assert "weight_dimensions" not in case


def test_apply_weight_dimension_section_replaces_criteria() -> None:
    """amap_agent_daily evaluation-dimension section: the two weights items + the mean of the raw scores in extra."""
    spec = {
        "trace_daily_report": True,
        "criteria": [{"id": "overall", "name": "综合质量", "weight": 1.0}],
        "criterion_metrics": {"overall": {"mean": 0.4}},
        "today_criterion_means": {"overall": 0.4},
        "case_results": [
            {
                "weight_dimensions": [
                    {
                        "id": "core_completion",
                        "name": "核心完成度",
                        "weight": 0.5,
                        "score": 1,
                        "full_score": 1.0,
                    },
                    {
                        "id": "expression",
                        "name": "表达质量",
                        "weight": 0.5,
                        "score": 4,
                        "full_score": 5.0,
                    },
                ]
            },
            {
                "weight_dimensions": [
                    {
                        "id": "core_completion",
                        "name": "核心完成度",
                        "weight": 0.5,
                        "score": 0,
                        "full_score": 1.0,
                    },
                    {
                        "id": "expression",
                        "name": "表达质量",
                        "weight": 0.5,
                        "score": 2,
                        "full_score": 5.0,
                    },
                ]
            },
        ],
    }
    apply_weight_dimension_section(spec)
    assert [c["id"] for c in spec["criteria"]] == ["core_completion", "expression"]
    assert spec["criteria"][0]["weight"] == 0.5
    assert spec["criteria"][0]["full_score"] == 1.0
    assert spec["criterion_metrics"]["core_completion"]["mean"] == 0.5
    assert spec["criterion_metrics"]["expression"]["mean"] == 3.0
    assert spec["criteria"][1]["full_score"] == 5.0
    assert "today_criterion_means" not in spec


def test_apply_weight_dimension_section_keeps_rubric_without_weights() -> None:
    spec = {
        "trace_daily_report": True,
        "criteria": [{"id": "overall", "name": "综合质量", "weight": 1.0}],
        "case_results": [{"case_id": "t1"}],
    }
    apply_weight_dimension_section(spec)
    assert spec["criteria"][0]["id"] == "overall"
