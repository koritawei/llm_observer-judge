"""Unit tests for AggregationCalculator — run-level score aggregation."""

from typing import Any

import pytest
from benchmark.reporting.framework.aggregation_calculator import (
    AggregationCalculator,
    EvaluatorResultEntry,
)
from benchmark.reporting.framework.schema import (
    AggregationConfig,
    AggregationMethod,
    Criterion,
    CriterionStats,
)


def _criterion(cid: str, weight: float = 1.0) -> Criterion:
    return Criterion(
        id=cid,
        name=cid,
        description="",
        weight=weight,
        evaluator_ref="business.evaluators.example.agent:ExampleAgent",
    )


def _entry(
    cid: str,
    *,
    mean: float = 0.0,
    pass_rate: float = 0.0,
    sample_count: int = 10,
    attribution: dict[str, Any] | None = None,
    react_rounds: dict[str, Any] | None = None,
) -> EvaluatorResultEntry:
    return EvaluatorResultEntry(
        criterion_id=cid,
        evaluator_ref="ref",
        sample_count=sample_count,
        success_count=sample_count - 1,
        failed_count=1,
        criterion_stats_json={"mean": mean, "pass_rate": pass_rate},
        attribution_stats_json=attribution,
        react_rounds_json=react_rounds,
    )


def _custom_run_score(
    rubric: list[Criterion],
    per_criterion: dict[str, CriterionStats],
    agg_config: AggregationConfig,
) -> float:
    """Test run-level custom aggregation: take the minimum of every dimension's mean."""
    del rubric, agg_config
    return min((s.mean for s in per_criterion.values()), default=0.0)


_CUSTOM_RUN_REF = f"{_custom_run_score.__module__}:{_custom_run_score.__name__}"

# ════════════════════════════════════════════════════════════════════════════
# aggregate — per-criterion stats / attribution / react rounds
# ════════════════════════════════════════════════════════════════════════════


async def test_aggregate_builds_per_criterion_stats() -> None:
    rubric = [_criterion("a"), _criterion("b")]
    entries = [_entry("a", mean=0.9, pass_rate=0.95), _entry("b", mean=0.5, pass_rate=0.4)]
    cfg = AggregationConfig(method=AggregationMethod.WEIGHTED_AVG.value)

    final, per_criterion, metrics = await AggregationCalculator().aggregate(cfg, rubric, entries)

    assert final == pytest.approx(0.7)
    assert per_criterion["a"].mean == 0.9
    assert per_criterion["b"].pass_rate == 0.4
    assert metrics.aggregation_method == "weighted_avg"
    assert metrics.total_samples == 10


async def test_aggregate_missing_stats_json_defaults_to_zero() -> None:
    entry = EvaluatorResultEntry(
        criterion_id="a",
        evaluator_ref="ref",
        sample_count=3,
        success_count=3,
        failed_count=0,
        criterion_stats_json=None,
    )
    _, per_criterion, _ = await AggregationCalculator().aggregate(
        AggregationConfig(method=AggregationMethod.WEIGHTED_AVG.value),
        [_criterion("a")],
        [entry],
    )
    assert per_criterion["a"].mean == 0.0
    assert per_criterion["a"].pass_rate == 0.0


async def test_aggregate_attribution_preserves_value_labels() -> None:
    """value_labels must pass through with the aggregation — report rendering uses them to turn a raw value into a display label."""
    attr = {
        "field_distributions": {"primary_issue_category": {"intent_error": 3}},
        "value_labels": {"primary_issue_category": {"intent_error": "意图理解错误"}},
        "total_attributed": 3,
    }
    _, _, metrics = await AggregationCalculator().aggregate(
        AggregationConfig(method=AggregationMethod.WEIGHTED_AVG.value),
        [_criterion("a")],
        [_entry("a", attribution=attr)],
    )
    stats = metrics.attribution_metrics["a"]
    assert stats.field_distributions == attr["field_distributions"]
    assert stats.value_labels == attr["value_labels"]
    assert stats.total_attributed == 3


async def test_aggregate_without_attribution_leaves_metrics_empty() -> None:
    _, _, metrics = await AggregationCalculator().aggregate(
        AggregationConfig(method=AggregationMethod.WEIGHTED_AVG.value),
        [_criterion("a")],
        [_entry("a")],
    )
    assert metrics.attribution_metrics == {}


async def test_aggregate_react_rounds_handles_plus_suffix_keys() -> None:
    """A key in the "5+" style is folded into 5; an illegal key is skipped and does not blow up."""
    entries = [
        _entry("a", react_rounds={"distribution": {"1": 5, "5+": 2, "bad": 9}}),
        _entry("b", react_rounds={"distribution": {"5+": 3}}),
    ]
    _, _, metrics = await AggregationCalculator().aggregate(
        AggregationConfig(method=AggregationMethod.WEIGHTED_AVG.value),
        [_criterion("a"), _criterion("b")],
        entries,
    )
    assert metrics.react_rounds_summary == {1: 5, 5: 5}


async def test_aggregate_empty_entries_total_samples_zero() -> None:
    final, _, metrics = await AggregationCalculator().aggregate(
        AggregationConfig(method=AggregationMethod.WEIGHTED_AVG.value),
        [_criterion("a")],
        [],
    )
    assert final == 0.0
    assert metrics.total_samples == 0


# ════════════════════════════════════════════════════════════════════════════
# _compute_final_score — four methods
# ════════════════════════════════════════════════════════════════════════════


async def test_weighted_avg_missing_criterion_stats_mean_zero() -> None:
    rubric = [_criterion("a"), _criterion("b")]
    entries = [_entry("a", mean=1.0)]
    final, _, _ = await AggregationCalculator().aggregate(
        AggregationConfig(method=AggregationMethod.WEIGHTED_AVG.value),
        rubric,
        entries,
    )
    assert final == pytest.approx(0.5)


async def test_weighted_avg_zero_total_weight_returns_zero() -> None:
    final, _, _ = await AggregationCalculator().aggregate(
        AggregationConfig(method=AggregationMethod.WEIGHTED_AVG.value),
        [_criterion("a", weight=0.0)],
        [_entry("a", mean=1.0)],
    )
    assert final == 0.0


async def test_all_pass_compares_pass_rate_against_threshold() -> None:
    cfg = AggregationConfig(method=AggregationMethod.ALL_PASS.value, pass_threshold=0.8)
    rubric = [_criterion("a"), _criterion("b")]
    both_pass = [_entry("a", pass_rate=0.9), _entry("b", pass_rate=0.8)]
    final, _, _ = await AggregationCalculator().aggregate(cfg, rubric, both_pass)
    assert final == 1.0

    one_below = [_entry("a", pass_rate=0.9), _entry("b", pass_rate=0.7)]
    final, _, _ = await AggregationCalculator().aggregate(cfg, rubric, one_below)
    assert final == 0.0


async def test_majority_vote_strictly_more_than_half() -> None:
    cfg = AggregationConfig(
        method=AggregationMethod.MAJORITY_VOTE.value,
        pass_threshold=0.8,
    )
    rubric = [_criterion("a"), _criterion("b")]
    one_pass = [_entry("a", pass_rate=1.0), _entry("b", pass_rate=0.0)]
    final, _, _ = await AggregationCalculator().aggregate(cfg, rubric, one_pass)
    assert final == 0.0


async def test_custom_method_delegates_to_custom_fn_ref() -> None:
    cfg = AggregationConfig(
        method=AggregationMethod.CUSTOM.value,
        custom_fn_ref=_CUSTOM_RUN_REF,
    )
    rubric = [_criterion("a"), _criterion("b")]
    entries = [_entry("a", mean=0.9), _entry("b", mean=0.4)]
    final, _, _ = await AggregationCalculator().aggregate(cfg, rubric, entries)
    assert final == pytest.approx(0.4)


def test_unknown_method_raises() -> None:
    """When schema validation is bypassed (the static method is called directly), an unknown method must raise explicitly."""
    bogus = AggregationConfig.__new__(AggregationConfig)
    object.__setattr__(bogus, "method", "nope")
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        AggregationCalculator._compute_final_score(bogus, [_criterion("a")], {})
