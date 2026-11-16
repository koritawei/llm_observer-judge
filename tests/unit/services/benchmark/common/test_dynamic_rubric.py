"""Dynamic Rubric: weighted scoring and attribution-threshold extraction."""

from benchmark.evaluators.base import EvaluatorResult
from benchmark.runs.common.dynamic_rubric import (
    apply_dynamic_fused_override,
    extract_attribution_threshold,
    extract_attribution_threshold_from_criterion_scores,
    extract_dynamic_weights,
    first_attribution_threshold_from_scores,
    fuse_from_weights,
    maybe_override_fused_score,
    project_criterion_score_row,
    project_criterion_scores,
    resolve_dynamic_rubric,
    should_trigger_with_optional_threshold,
)
from benchmark.reporting.framework.fusion import EvalFailure


def test_fuse_from_weights_normalizes_and_averages() -> None:
    fused = fuse_from_weights(
        {"准确度": 8.0, "相关性": 6.0, "完整性": 4.0},
        {"准确度": 0.4, "相关性": 0.3, "完整性": 0.3},
    )
    assert fused is not None
    assert abs(fused - (8 * 0.4 + 6 * 0.3 + 4 * 0.3)) < 1e-9


def test_fuse_from_weights_returns_none_when_keys_missing() -> None:
    assert (
        fuse_from_weights(
            {"准确度": 8.0},
            {"准确度": 0.5, "相关性": 0.5},
        )
        is None
    )


def test_fuse_from_weights_returns_none_without_inputs() -> None:
    assert fuse_from_weights(None, {"a": 1.0}) is None
    assert fuse_from_weights({"a": 1.0}, None) is None


def test_maybe_override_fused_score_uses_first_valid_result() -> None:
    scores = {
        "example": EvaluatorResult(
            score=9.0,
            reason="ok",
            sub_scores={"a": 8.0, "b": 6.0},
            extra={
                "dynamic_rubric_opt_in": True,
                "declared_weights": {"a": 0.5, "b": 0.5},
                "weights": {"a": 0.5, "b": 0.5},
            },
        )
    }
    assert maybe_override_fused_score(scores) == 7.0


def test_maybe_override_fused_score_skips_without_weights() -> None:
    """No weights → do not override, so a custom static Rubric stays protected."""
    scores = {
        "custom": EvaluatorResult(
            score=0.42,
            reason="static",
            sub_scores={"D1": 1.0},
        )
    }
    assert maybe_override_fused_score(scores) is None


def test_maybe_override_skips_failures() -> None:
    scores = {
        "x": EvalFailure(RuntimeError("boom")),
        "y": EvaluatorResult(
            score=1.0,
            sub_scores={"a": 10.0},
            extra={
                "dynamic_rubric_opt_in": True,
                "declared_weights": {"a": 1.0},
                "weights": {"a": 1.0},
            },
        ),
    }
    assert maybe_override_fused_score(scores) == 10.0


def test_extract_attribution_threshold_from_result_and_payload() -> None:
    result = EvaluatorResult(
        score=7.0,
        extra={
            "dynamic_rubric_opt_in": True,
            "declared_weights": {"a": 1.0},
            "weights": {"a": 1.0},
            "attribution_threshold": 6.0,
        },
    )
    assert extract_attribution_threshold(result) == 6.0
    assert extract_dynamic_weights(result) == {"a": 1.0}
    assert (
        extract_attribution_threshold_from_criterion_scores(
            {"example": {"score": 7.0, "attribution_threshold": 8.5, "dynamic_rubric_opt_in": True}}
        )
        == 8.5
    )
    assert (
        extract_attribution_threshold_from_criterion_scores(
            {"example": {"score": 7.0, "attribution_threshold": "6", "dynamic_rubric_opt_in": True}}
        )
        == 6.0
    )


def test_resolve_gate_attribution_threshold_pass_threshold_fallback() -> None:
    from benchmark.runs.common.dynamic_rubric import (
        resolve_gate_attribution_threshold,
        should_trigger_with_optional_threshold,
    )

    # When the criterion did not project a threshold, example pass_threshold=6 should still fire at 2.9
    thr = resolve_gate_attribution_threshold(None, pass_threshold=6.0)
    assert thr == 6.0
    assert (
        should_trigger_with_optional_threshold(
            fused_score=2.9,
            attribution_threshold=thr,
        )
        is True
    )
    # A pass_threshold on the small score scale is not backfilled (so it does not disturb rubric_v2)
    assert resolve_gate_attribution_threshold(None, pass_threshold=1.0) is None
    assert (
        resolve_gate_attribution_threshold(
            {"example": {"attribution_threshold": 6.0, "dynamic_rubric_opt_in": True}},
            pass_threshold=1.0,
        )
        == 6.0
    )


def test_should_trigger_with_optional_threshold_fallback_none() -> None:
    assert (
        should_trigger_with_optional_threshold(
            fused_score=5.0,
            attribution_threshold=None,
        )
        is None
    )
    assert (
        should_trigger_with_optional_threshold(
            fused_score=5.0,
            attribution_threshold=6.0,
        )
        is True
    )
    assert (
        should_trigger_with_optional_threshold(
            fused_score=6.0,
            attribution_threshold=6.0,
        )
        is False
    )
    assert (
        should_trigger_with_optional_threshold(
            fused_score=None,
            attribution_threshold=6.0,
        )
        is True
    )


def test_apply_dynamic_fused_override_keeps_static_when_no_weights() -> None:
    scores = {
        "c1": EvaluatorResult(score=9.0, reason="ok", sub_scores={"a": 9.0}),
    }
    assert apply_dynamic_fused_override(scores, 9.0) == 9.0


def test_first_threshold_and_project_criterion_scores() -> None:
    scores = {
        "c1": EvaluatorResult(
            score=7.0,
            reason="ok",
            sub_scores={"准确度": 7.0, "相关性": 8.0},
            extra={
                "dynamic_rubric_opt_in": True,
                "declared_weights": {"准确度": 0.6, "相关性": 0.4},
                "weights": {"准确度": 0.6, "相关性": 0.4},
                "attribution_threshold": 6.0,
                "grade": "GOOD",
            },
        ),
    }
    assert first_attribution_threshold_from_scores(scores) == 6.0
    projected = project_criterion_scores(scores)
    assert projected["c1"]["weights"] == {"准确度": 0.6, "相关性": 0.4}
    assert projected["c1"]["attribution_threshold"] == 6.0
    assert projected["c1"]["sub_scores"] == {"准确度": 7.0, "相关性": 8.0}
    assert projected["c1"]["computed_fused_score"] == 7.4
    assert projected["c1"]["dynamic_rubric_opt_in"] is True
    assert projected["c1"]["declared_weights"] == {"准确度": 0.6, "相关性": 0.4}
    assert projected["c1"]["model_score"] == 7.0
    assert projected["c1"]["score_mismatch"] is True
    assert projected["c1"]["grade"] == "GOOD"
    # The threshold has only the single attribution_threshold key and no longer emits selected/default variants
    assert "selected_attribution_threshold" not in projected["c1"]
    assert "default_attribution_threshold" not in projected["c1"]


def test_project_prefers_extra_model_score_when_result_score_is_fused() -> None:
    """When result.score is already fused, projecting model_score / mismatch still reads extra."""
    result = EvaluatorResult(
        score=7.4,
        reason="ok",
        sub_scores={"准确度": 7.0, "相关性": 8.0},
        extra={
            "dynamic_rubric_opt_in": True,
            "model_score": 99.0,
            "declared_weights": {"准确度": 0.6, "相关性": 0.4},
            "weights": {"准确度": 0.6, "相关性": 0.4},
        },
    )
    projected = project_criterion_score_row(result)
    assert projected["score"] == 7.4
    assert projected["model_score"] == 99.0
    assert projected["computed_fused_score"] == 7.4
    assert projected["score_mismatch"] is True


def test_non_opt_in_judge_ignores_dynamic_fields() -> None:
    result = EvaluatorResult(
        score=9.0,
        sub_scores={"a": 8.0},
        extra={"weights": {"a": 1.0}, "attribution_threshold": 6.0},
    )
    assert maybe_override_fused_score({"custom": result}) is None
    assert extract_dynamic_weights(result) is None
    assert extract_attribution_threshold(result) is None
    assert (
        extract_attribution_threshold_from_criterion_scores(
            {"custom": {"score": 7.0, "attribution_threshold": 6.0}}
        )
        is None
    )


def test_weights_mismatch_keeps_judge_threshold_independent() -> None:
    """Weight fallback does not affect the threshold: a legal threshold returned by the judge takes effect on its own."""
    result = EvaluatorResult(
        score=99.0,
        sub_scores={"a": 8.0, "b": 6.0},
        extra={
            "dynamic_rubric_opt_in": True,
            "declared_weights": {"a": 0.5, "b": 0.5},
            "weights": {"a": 0.9, "b": 0.1},
            "attribution_threshold": 2.0,
            "sub_score_scale": [0.0, 10.0],
        },
    )
    resolved = resolve_dynamic_rubric(result)
    assert resolved.fused_score == 7.0
    assert resolved.attribution_threshold == 2.0
    assert resolved.fallback_reason == "weights_mismatch"


def test_weight_sum_and_threshold_range_are_strictly_validated() -> None:
    result = EvaluatorResult(
        score=99.0,
        sub_scores={"a": 8.0},
        extra={
            "dynamic_rubric_opt_in": True,
            "declared_weights": {"a": 1.0},
            "weights": {"a": 2.0},
            "attribution_threshold": 99.0,
            "sub_score_scale": [0.0, 10.0],
        },
    )
    resolved = resolve_dynamic_rubric(result)
    assert resolved.fused_score == 8.0
    # A threshold outside the declared score scale → invalid, fall back to the business static policy (a registry default no longer fails this path with it)
    assert resolved.attribution_threshold is None
    assert resolved.fallback_reason == "weights_mismatch"
    projected = project_criterion_score_row(result)
    assert "attribution_threshold" not in projected
    assert projected["sub_score_scale"] == [0.0, 10.0]


def test_stamp_fused_grade_writes_top_level_and_fills_missing() -> None:
    from benchmark.runs.common.dynamic_rubric import stamp_fused_grade

    judge_json = {
        "criterion_scores": {
            "example": {"score": 2.9, "reason": "low"},
            "other": {"score": 8.0, "grade": "EXCELLENT"},
        },
        "fused_score": 2.9,
    }
    stamp_fused_grade(judge_json, fused_score=2.9, pass_threshold=6.0)
    assert judge_json["grade"] == "POOR"
    assert judge_json["criterion_scores"]["example"]["grade"] == "POOR"
    # An existing grade is not overwritten
    assert judge_json["criterion_scores"]["other"]["grade"] == "EXCELLENT"
