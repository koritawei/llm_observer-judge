"""Plan A: the report case.grade prefers judge_json.grade."""

from benchmark.runs.run_helpers import (
    _decorate_case_rows,
    _extract_case_grade,
)


def test_extract_case_grade_prefers_fused_judge_json() -> None:
    case = {
        "scores": {
            "fused": {
                "score": 2.9,
                "judge": {
                    "grade": "POOR",
                    "fused_score": 2.9,
                    "criterion_scores": {
                        "example": {"score": 2.9, "grade": "POOR"},
                    },
                },
            },
        },
    }
    assert _extract_case_grade(case) == "POOR"


def test_decorate_uses_judge_grade_over_recompute() -> None:
    """Even when the fused score would compute to EXCELLENT under the old rubric_v2, judge_json wins."""
    cases = [
        {
            "case_id": "c1",
            "scores": {
                "fused": {
                    "score": 2.9,
                    "eval_failed": False,
                    "judge": {"grade": "POOR", "fused_score": 2.9},
                },
            },
        },
    ]
    _decorate_case_rows(cases, pass_threshold=None)
    assert cases[0]["grade"] == "POOR"


def test_decorate_fallback_computes_when_judge_missing_grade() -> None:
    cases = [
        {
            "case_id": "c1",
            "scores": {
                "fused": {
                    "score": 2.9,
                    "eval_failed": False,
                    "judge": {"fused_score": 2.9},
                },
            },
        },
    ]
    _decorate_case_rows(cases, pass_threshold=6.0)
    assert cases[0]["grade"] == "POOR"
