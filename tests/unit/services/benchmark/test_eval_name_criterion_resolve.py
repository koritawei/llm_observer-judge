"""Regression for composite evaluator_name → criterion / fused mapping.

Locks that the streaming persistence key ``amap_agent@reg@ds-rt-…`` maps to
``amap_agent_overall`` and decorate writes ``scores.fused`` (previously, recognizing only the bare
stem left the case-detail composite score as ``-``).
"""

from benchmark.runs.run_case_query import (
    _eval_to_criteria,
    _resolve_eval_criterion_ids,
)
from benchmark.runs.trace_daily_report_adapter import (
    TraceDailyDecorFields,
    decorate_trace_daily_case,
)
from benchmark.reporting.framework.schema import Criterion

_COMPOSITE = "amap_agent@ba-1793104298670@ds-rt-1793090866482"
_REG = "ba-1793104298670"
_STREAM = "ds-rt-1793090866482"


def _amap_rubric() -> list[Criterion]:
    return [
        Criterion(
            id="amap_agent_overall",
            name="综合质量",
            weight=1.0,
            evaluator_ref="amap_agent",
        ),
    ]


def test_resolve_falls_back_to_stem_without_stream_aliases() -> None:
    """When the report does not pass stream, the stem-name fallback can still hit amap_agent_overall."""
    mapping = _eval_to_criteria(_amap_rubric())
    assert mapping == {"amap_agent": ["amap_agent_overall"]}
    assert _resolve_eval_criterion_ids(mapping, _COMPOSITE) == ["amap_agent_overall"]
    assert _resolve_eval_criterion_ids(mapping, "amap_agent") == ["amap_agent_overall"]
    assert _resolve_eval_criterion_ids(mapping, "unknown_judge") == []
    assert _resolve_eval_criterion_ids(
        mapping,
        "unknown_judge",
        fallback_raw=True,
    ) == ["unknown_judge"]


def test_resolve_exact_composite_when_stream_wired() -> None:
    mapping = _eval_to_criteria(
        _amap_rubric(),
        registration_id=_REG,
        stream_instance_id=_STREAM,
    )
    assert mapping[_COMPOSITE] == ["amap_agent_overall"]
    assert _resolve_eval_criterion_ids(mapping, _COMPOSITE) == ["amap_agent_overall"]


def test_decorate_fused_from_composite_mapped_score() -> None:
    """Simulated daily-report case: after the composite key is mapped, decorate must write scores.fused."""
    rubric = _amap_rubric()
    mapping = _eval_to_criteria(rubric)
    cids = _resolve_eval_criterion_ids(mapping, _COMPOSITE, fallback_raw=True)
    assert cids == ["amap_agent_overall"]

    case: dict = {"case_id": "d4c352e3fb8ac9aab8af25962c38ed84", "scores": {}}
    for cid in cids:
        case["scores"][cid] = {
            "score": 0.75,
            "reason": "综合判定为 0.75。",
            "eval_failed": False,
        }
    decorate_trace_daily_case(
        case,
        rubric=rubric,
        pass_threshold=0.5,
        fields=TraceDailyDecorFields(
            triage_level="watch",
            judge_json={
                "score": "0.75",
                "reason": "综合判定为 0.75。",
                "extra": {
                    "weights": {"expression": 0.5, "core_completion": 0.5},
                    "core_completion": {"score": 1, "reason": "ok"},
                    "expression": {"score": 3, "reason": "ok"},
                },
            },
        ),
    )
    assert case["scores"]["amap_agent_overall"]["score"] == 0.75
    assert case["scores"]["fused"]["score"] == 0.75
    assert "核心完成度" in case["dim_scores"]
    assert "表达质量" in case["dim_scores"]
