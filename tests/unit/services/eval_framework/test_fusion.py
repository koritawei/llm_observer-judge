"""Unit tests for ScoreFuser — item-level score fusion across criteria."""

from collections.abc import Callable

import pytest
from benchmark.evaluators.base import EvaluatorResult
from benchmark.reporting.framework.fusion import (
    EvalFailure,
    ScoreFuser,
    is_failure,
)
from benchmark.reporting.framework.schema import (
    AggregationConfig,
    AggregationMethod,
    Criterion,
)


def _criterion(cid: str, weight: float = 1.0) -> Criterion:
    return Criterion(
        id=cid,
        name=cid,
        description="",
        weight=weight,
        evaluator_ref="business.evaluators.example.agent:ExampleAgent",
    )


def _config(method: str, **kwargs: object) -> AggregationConfig:
    return AggregationConfig(method=method, **kwargs)  # type: ignore[arg-type]


def _custom_gate(
    scores: dict[str, EvaluatorResult | EvalFailure],
    rubric: list[Criterion],
    aggregation: AggregationConfig,
    score_of: Callable[[str], float],
) -> float:
    """Test custom fusion: if any dimension is 0 the total is 0 (the gate rule)."""
    del scores, aggregation
    return 0.0 if any(score_of(c.id) == 0.0 for c in rubric) else 1.0


_CUSTOM_GATE_REF = f"{_custom_gate.__module__}:{_custom_gate.__name__}"


def _exploding_fusion(
    scores: dict[str, EvaluatorResult | EvalFailure],
    rubric: list[Criterion],
    aggregation: AggregationConfig,
    score_of: Callable[[str], float],
) -> float:
    raise KeyError("genuine bug")


_EXPLODING_REF = f"{_exploding_fusion.__module__}:{_exploding_fusion.__name__}"

# ════════════════════════════════════════════════════════════════════════════
# EvalFailure marker
# ════════════════════════════════════════════════════════════════════════════


def test_is_failure_distinguishes_marker_from_result() -> None:
    assert is_failure(EvalFailure(RuntimeError("boom"))) is True
    assert is_failure(EvaluatorResult(score=0.0)) is False
    assert is_failure(None) is False


def test_eval_failure_repr_carries_error_type() -> None:
    assert "TimeoutError" in repr(EvalFailure(TimeoutError("slow")))


# ════════════════════════════════════════════════════════════════════════════
# weighted_avg
# ════════════════════════════════════════════════════════════════════════════


def test_weighted_avg_basic() -> None:
    rubric = [_criterion("a", weight=1.0), _criterion("b", weight=3.0)]
    scores = {
        "a": EvaluatorResult(score=1.0),
        "b": EvaluatorResult(score=0.5),
    }
    fused = ScoreFuser.fuse(scores, rubric, _config(AggregationMethod.WEIGHTED_AVG.value))
    assert fused == pytest.approx((1.0 * 1.0 + 0.5 * 3.0) / 4.0)


def test_weighted_avg_failure_counts_zero_but_keeps_weight() -> None:
    """A failure counts as score 0 and its weight stays in the denominator — runs stay comparable, and a failure has a visible penalty."""
    rubric = [_criterion("a"), _criterion("b")]
    scores = {
        "a": EvaluatorResult(score=1.0),
        "b": EvalFailure(RuntimeError("eval exploded")),
    }
    fused = ScoreFuser.fuse(scores, rubric, _config(AggregationMethod.WEIGHTED_AVG.value))
    assert fused == pytest.approx(0.5)


def test_weighted_avg_missing_criterion_scores_zero() -> None:
    rubric = [_criterion("a"), _criterion("b")]
    scores = {"a": EvaluatorResult(score=1.0)}
    fused = ScoreFuser.fuse(scores, rubric, _config(AggregationMethod.WEIGHTED_AVG.value))
    assert fused == pytest.approx(0.5)


def test_weighted_avg_zero_total_weight_returns_zero() -> None:
    rubric = [_criterion("a", weight=0.0)]
    scores = {"a": EvaluatorResult(score=1.0)}
    assert ScoreFuser.fuse(scores, rubric, _config(AggregationMethod.WEIGHTED_AVG.value)) == 0.0


# ════════════════════════════════════════════════════════════════════════════
# all_pass / majority_vote
# ════════════════════════════════════════════════════════════════════════════


def test_all_pass_requires_every_criterion_above_threshold() -> None:
    rubric = [_criterion("a"), _criterion("b")]
    cfg = _config(AggregationMethod.ALL_PASS.value, pass_threshold=0.8)
    assert (
        ScoreFuser.fuse(
            {"a": EvaluatorResult(score=0.9), "b": EvaluatorResult(score=0.8)},
            rubric,
            cfg,
        )
        == 1.0
    )
    assert (
        ScoreFuser.fuse(
            {"a": EvaluatorResult(score=0.9), "b": EvaluatorResult(score=0.7)},
            rubric,
            cfg,
        )
        == 0.0
    )


def test_majority_vote_strictly_more_than_half() -> None:
    """Majority is strictly more than half — 1 of 2 dimensions passing does not pass."""
    rubric = [_criterion("a"), _criterion("b")]
    cfg = _config(AggregationMethod.MAJORITY_VOTE.value, pass_threshold=0.8)
    one_pass = {"a": EvaluatorResult(score=1.0), "b": EvaluatorResult(score=0.0)}
    assert ScoreFuser.fuse(one_pass, rubric, cfg) == 0.0

    three = [_criterion("a"), _criterion("b"), _criterion("c")]
    two_pass = {
        "a": EvaluatorResult(score=1.0),
        "b": EvaluatorResult(score=1.0),
        "c": EvaluatorResult(score=0.0),
    }
    assert ScoreFuser.fuse(two_pass, three, cfg) == 1.0


# ════════════════════════════════════════════════════════════════════════════
# custom / fusion_fn_ref
# ════════════════════════════════════════════════════════════════════════════


def test_custom_method_without_fusion_ref_raises_actionable_error() -> None:
    """method='custom' covers only run-level aggregation; item-level fusion must go through fusion_fn_ref."""
    rubric = [_criterion("a")]
    cfg = _config(AggregationMethod.CUSTOM.value, custom_fn_ref="some.module:fn")
    with pytest.raises(ValueError, match="item-level fusion requires fusion_fn_ref"):
        ScoreFuser.fuse({"a": EvaluatorResult(score=1.0)}, rubric, cfg)


def test_fusion_fn_ref_takes_precedence_over_method() -> None:
    rubric = [_criterion("a")]
    cfg = _config(
        AggregationMethod.WEIGHTED_AVG.value,
        fusion_fn_ref=_CUSTOM_GATE_REF,
    )
    scores = {"a": EvaluatorResult(score=0.5)}
    assert ScoreFuser.fuse(scores, rubric, cfg) == 1.0


def test_fusion_fn_ref_gate_rule_sees_failure_as_zero() -> None:
    rubric = [_criterion("a"), _criterion("b")]
    cfg = _config(
        AggregationMethod.WEIGHTED_AVG.value,
        fusion_fn_ref=_CUSTOM_GATE_REF,
    )
    scores = {
        "a": EvaluatorResult(score=1.0),
        "b": EvalFailure(RuntimeError("boom")),
    }
    assert ScoreFuser.fuse(scores, rubric, cfg) == 0.0


def test_unresolvable_fusion_fn_ref_falls_back_to_method() -> None:
    """When ref resolution fails (the module does not exist), fall back to method instead of failing the whole item."""
    rubric = [_criterion("a")]
    cfg = _config(
        AggregationMethod.WEIGHTED_AVG.value,
        fusion_fn_ref="no.such.module:fn",
    )
    scores = {"a": EvaluatorResult(score=0.6)}
    assert ScoreFuser.fuse(scores, rubric, cfg) == pytest.approx(0.6)


def test_fusion_fn_ref_missing_attr_falls_back_to_method() -> None:
    cfg = _config(
        AggregationMethod.WEIGHTED_AVG.value,
        fusion_fn_ref=f"{_custom_gate.__module__}:no_such_function",
    )
    scores = {"a": EvaluatorResult(score=0.6)}
    assert ScoreFuser.fuse(scores, [_criterion("a")], cfg) == pytest.approx(0.6)


def test_exception_inside_custom_fn_propagates() -> None:
    """An exception raised inside the custom fn is not swallowed — a real bug must stay visible."""
    cfg = _config(
        AggregationMethod.WEIGHTED_AVG.value,
        fusion_fn_ref=_EXPLODING_REF,
    )
    with pytest.raises(KeyError, match="genuine bug"):
        ScoreFuser.fuse({"a": EvaluatorResult(score=1.0)}, [_criterion("a")], cfg)
