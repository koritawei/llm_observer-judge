"""Unit tests for the example evaluator's score/reason extraction."""
# ruff: noqa: PLR2004  (test-only: literal asserts / fixture params / local imports)

import json

from business.evaluators.example.scoring import parse_evaluation


def test_parses_plain_json_content() -> None:
    result = parse_evaluation(json.dumps({"score": 7, "reason": "ok"}))
    assert result is not None
    assert result.score == 7.0
    assert result.reason == "ok"


def test_parses_float_score() -> None:
    result = parse_evaluation('{"score": 5.4, "reason": "x"}')
    assert result is not None
    assert result.score == 5.4


def test_strips_json_code_fence() -> None:
    content = '```json\n{"score": 8, "reason": "good"}\n```'
    result = parse_evaluation(content)
    assert result is not None
    assert result.score == 8.0
    assert result.reason == "good"


def test_repairs_unescaped_quotes_in_llm_reason() -> None:
    result = parse_evaluation('{"score": 8, "reason": "解释中有"引用""}')

    assert result is not None
    assert result.score == 8.0
    assert result.reason == '解释中有"引用"'


def test_reason_explanation_alias() -> None:
    result = parse_evaluation('{"score": 3, "explanation": "why"}')
    assert result is not None
    assert result.reason == "why"


def test_reason_reasoning_alias() -> None:
    # Langfuse Numeric evaluator contract emits `reasoning`; maps to the `reason` field.
    result = parse_evaluation('{"score": 3, "reasoning": "why"}')
    assert result is not None
    assert result.reason == "why"


def test_rejects_bool_score() -> None:
    # JSON `true` subclasses int but must not be read as 1.0
    assert parse_evaluation('{"score": true, "reason": "x"}') is None


def test_missing_score_returns_none() -> None:
    assert parse_evaluation('{"reason": "no score"}') is None


def test_non_numeric_score_returns_none() -> None:
    assert parse_evaluation('{"score": "high"}') is None


def test_non_json_content_returns_none() -> None:
    assert parse_evaluation("not json at all") is None


def test_empty_content_returns_none() -> None:
    assert parse_evaluation("") is None
    assert parse_evaluation("   ") is None


def test_reason_optional_when_absent() -> None:
    result = parse_evaluation('{"score": 9}')
    assert result is not None
    assert result.score == 9.0
    assert result.reason is None


def test_parses_sub_scores_when_present() -> None:
    result = parse_evaluation(
        json.dumps({"score": 8, "reason": "ok", "sub_scores": {"准确度": 8, "相关性": 9}})
    )
    assert result is not None
    assert result.score == 8.0
    assert result.sub_scores == {"准确度": 8.0, "相关性": 9.0}


def test_parses_weights_and_attribution_threshold() -> None:
    result = parse_evaluation(
        json.dumps(
            {
                "score": 7.2,
                "reasoning": "ok",
                "sub_scores": {"准确度": 7, "相关性": 8, "完整性": 6},
                "weights": {"准确度": 0.4, "相关性": 0.3, "完整性": 0.3},
                "attribution_threshold": 6.0,
            }
        )
    )
    assert result is not None
    assert result.weights == {"准确度": 0.4, "相关性": 0.3, "完整性": 0.3}
    assert result.attribution_threshold == 6.0


def test_parses_string_numbers_and_ignores_grade() -> None:
    """The LLM often writes threshold/weights as strings and mistakenly includes grade — that must not be llm_unparseable."""
    result = parse_evaluation(
        json.dumps(
            {
                "score": "6.5",
                "reasoning": "ok",
                "sub_scores": {"准确度": "8", "相关性": "7", "完整性": "4"},
                "weights": {"准确度": "0.4", "相关性": "0.3", "完整性": "0.3"},
                "attribution_threshold": "6.0",
                "grade": "POOR",
            }
        )
    )
    assert result is not None
    assert result.score == 6.5
    assert result.sub_scores == {"准确度": 8.0, "相关性": 7.0, "完整性": 4.0}
    assert result.weights == {"准确度": 0.4, "相关性": 0.3, "完整性": 0.3}
    assert result.attribution_threshold == 6.0
    # grade is an extra field; it is ignored while parsing and does not become a model field
    assert "grade" not in result.model_fields_set


def test_parses_json_with_preamble_text() -> None:
    content = '评测结果如下：\n{"score": 7, "reasoning": "ok"}\n'
    result = parse_evaluation(content)
    assert result is not None
    assert result.score == 7.0
    assert result.reason == "ok"


def test_sub_scores_absent_is_none() -> None:
    """Old LLM output (no sub_scores field) must be tolerated, and sub_scores defaults to None."""
    result = parse_evaluation(json.dumps({"score": 7, "reason": "ok"}))
    assert result is not None
    assert result.sub_scores is None


def test_malformed_dynamic_fields_do_not_invalidate_base_verdict() -> None:
    content = json.dumps(
        {
            "score": 7,
            "reasoning": "ok",
            "sub_scores": "not-a-dict",
            "weights": "not-a-dict",
            "attribution_threshold": "not-a-number",
        }
    )
    result = parse_evaluation(content)
    assert result is not None
    assert result.score == 7.0
    assert result.sub_scores is None
    assert result.weights is None
    assert result.attribution_threshold is None
    assert result.raw_json == content


def test_sub_scores_wrong_type_returns_none() -> None:
    """sub_scores that is not a dict only disables the dynamic fields, and does not break the base verdict."""
    result = parse_evaluation(json.dumps({"score": 7, "reason": "ok", "sub_scores": [1, 2, 3]}))
    assert result is not None
    assert result.score == 7.0
    assert result.sub_scores is None
