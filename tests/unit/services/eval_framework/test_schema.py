"""Unit tests for eval_framework schema — decoration-time config validation."""

import pytest
from benchmark.reporting.framework.schema import (
    AggregatedMetrics,
    AggregationConfig,
    AggregationMethod,
    AttributionStats,
    Criterion,
    CriterionStats,
    DatasetRef,
)
from pydantic import ValidationError

# ════════════════════════════════════════════════════════════════════════════
# Criterion
# ════════════════════════════════════════════════════════════════════════════


def _criterion(**overrides: object) -> Criterion:
    kwargs = {
        "id": "accuracy",
        "name": "准确性",
        "description": "回答是否准确",
        "weight": 1.0,
        "evaluator_ref": "business.evaluators.example.agent:ExampleAgent",
    }
    kwargs.update(overrides)
    return Criterion(**kwargs)  # type: ignore[arg-type]


def test_criterion_valid() -> None:
    c = _criterion()
    assert c.id == "accuracy"
    assert c.weight == 1.0


def test_criterion_empty_id_rejected() -> None:
    with pytest.raises(ValueError, match="id must not be empty"):
        _criterion(id="")


def test_criterion_empty_evaluator_ref_rejected() -> None:
    with pytest.raises(ValueError, match="evaluator_ref must not be empty"):
        _criterion(evaluator_ref="")


def test_criterion_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="weight must be >= 0"):
        _criterion(weight=-0.1)


def test_criterion_zero_weight_allowed() -> None:
    """weight=0 is legal — that dimension is only reported in stats and does not participate in the weighted sum."""
    assert _criterion(weight=0.0).weight == 0.0


def test_criterion_frozen() -> None:
    c = _criterion()
    with pytest.raises(AttributeError):
        c.weight = 2.0  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════════
# AggregationConfig
# ════════════════════════════════════════════════════════════════════════════


def test_aggregation_config_defaults() -> None:
    cfg = AggregationConfig(method=AggregationMethod.WEIGHTED_AVG.value)
    assert cfg.pass_threshold == 0.8
    assert cfg.custom_fn_ref is None
    assert cfg.fusion_fn_ref is None


def test_aggregation_config_unknown_method_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        AggregationConfig(method="not_a_method")


@pytest.mark.parametrize("threshold", [-0.01, -1.0])
def test_aggregation_config_threshold_out_of_range_rejected(threshold: float) -> None:
    """Thresholds use the same scale as business scores (not a normalized assumption): only a negative is rejected, and >1 is legal (a full score of 2)."""
    with pytest.raises(ValueError, match="pass_threshold must be"):
        AggregationConfig(
            method=AggregationMethod.WEIGHTED_AVG.value,
            pass_threshold=threshold,
        )


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_aggregation_config_threshold_boundary_allowed(threshold: float) -> None:
    cfg = AggregationConfig(
        method=AggregationMethod.ALL_PASS.value,
        pass_threshold=threshold,
    )
    assert cfg.pass_threshold == threshold


def test_aggregation_config_custom_method_requires_fn_ref() -> None:
    with pytest.raises(ValueError, match="custom_fn_ref is required"):
        AggregationConfig(method=AggregationMethod.CUSTOM.value)


def test_aggregation_config_custom_method_with_ref_ok() -> None:
    cfg = AggregationConfig(
        method=AggregationMethod.CUSTOM.value,
        custom_fn_ref="some.module:fn",
    )
    assert cfg.custom_fn_ref == "some.module:fn"


def test_aggregation_config_custom_fn_ref_without_colon_rejected() -> None:
    """A ref with no colon fails fast at config time, instead of blowing up deep inside fusion halfway through a run."""
    with pytest.raises(ValueError, match="module:function"):
        AggregationConfig(
            method=AggregationMethod.CUSTOM.value,
            custom_fn_ref="no_colon_ref",
        )


def test_aggregation_config_fusion_fn_ref_without_colon_rejected() -> None:
    with pytest.raises(ValueError, match="module:function"):
        AggregationConfig(
            method=AggregationMethod.WEIGHTED_AVG.value,
            fusion_fn_ref="no_colon_ref",
        )


# ════════════════════════════════════════════════════════════════════════════
# DatasetRef
# ════════════════════════════════════════════════════════════════════════════


def test_dataset_ref_defaults() -> None:
    ref = DatasetRef(dataset_name="ds", project_id="proj")
    assert ref.sample_size is None
    assert ref.sample_seed == 42


# ════════════════════════════════════════════════════════════════════════════
# Pydantic statistics models
# ════════════════════════════════════════════════════════════════════════════


def _criterion_stats(**overrides: object) -> CriterionStats:
    kwargs = {
        "criterion_id": "accuracy",
        "evaluator_ref": "ref",
        "sample_count": 10,
        "success_count": 9,
        "failed_count": 1,
        "mean": 0.75,
        "median": 0.8,
        "p95": 0.95,
        "min": 0.0,
        "max": 1.0,
        "std": 0.2,
        "pass_rate": 0.9,
    }
    kwargs.update(overrides)
    return CriterionStats(**kwargs)  # type: ignore[arg-type]


def test_criterion_stats_negative_count_rejected() -> None:
    with pytest.raises(ValidationError):
        _criterion_stats(sample_count=-1)


def test_attribution_stats_defaults_empty() -> None:
    stats = AttributionStats()
    assert stats.field_distributions == {}
    assert stats.value_labels == {}
    assert stats.total_attributed == 0
    # Legacy fields are kept for deserialization compatibility
    assert stats.triage_level_distribution == {}


def test_aggregated_metrics_requires_method_and_totals() -> None:
    metrics = AggregatedMetrics(
        aggregation_method=AggregationMethod.WEIGHTED_AVG.value,
        total_samples=10,
        total_duration_ms=123,
    )
    assert metrics.criterion_metrics == {}
    assert metrics.attribution_metrics == {}
    assert metrics.react_rounds_summary == {}


def test_aggregated_metrics_negative_total_rejected() -> None:
    with pytest.raises(ValidationError):
        AggregatedMetrics(
            aggregation_method="weighted_avg",
            total_samples=-1,
            total_duration_ms=0,
        )


def test_aggregation_method_enum_values() -> None:
    assert {m.value for m in AggregationMethod} == {
        "weighted_avg",
        "all_pass",
        "majority_vote",
        "custom",
    }


# ════════════════════════════════════════════════════════════════════════════
# aggregated_metrics_from_json (rebuild the report data from the DB source of truth)
# ════════════════════════════════════════════════════════════════════════════


def _full_metrics_json() -> dict:
    metrics = AggregatedMetrics(
        criterion_metrics={
            "intent": CriterionStats(
                criterion_id="intent",
                evaluator_ref="sub_scores",
                sample_count=5,
                success_count=5,
                failed_count=0,
                mean=1.8,
                median=2.0,
                p95=2.0,
                min=1.0,
                max=2.0,
                std=0.4,
                pass_rate=0.8,
            ),
        },
        attribution_metrics={
            "fused": AttributionStats(
                field_distributions={"primary_issue_category": {"tool/timeout": 2}},
                total_attributed=2,
            ),
        },
        react_rounds_summary={1: 3, 2: 2},
        aggregation_method="weighted_avg",
        total_samples=5,
        total_duration_ms=1234,
    )
    return metrics.model_dump()


def test_aggregated_metrics_from_json_roundtrip() -> None:
    """A complete persisted snapshot → rebuilt into an equal AggregatedMetrics."""
    from benchmark.reporting.framework.schema import aggregated_metrics_from_json

    rebuilt = aggregated_metrics_from_json(_full_metrics_json())
    assert rebuilt is not None
    assert rebuilt.criterion_metrics["intent"].mean == 1.8
    assert rebuilt.attribution_metrics["fused"].total_attributed == 2
    assert rebuilt.react_rounds_summary == {1: 3, 2: 2}
    assert rebuilt.total_samples == 5


def test_aggregated_metrics_from_json_string_keys_coerced() -> None:
    """String keys of a JSON-encoded react_rounds_summary → pydantic forces them to int keys."""
    from benchmark.reporting.framework.schema import aggregated_metrics_from_json

    payload = _full_metrics_json()
    payload["react_rounds_summary"] = {"1": 3, "2": 2}
    payload["finalize_source"] = "external_cas_no_callback"  # Extra keys are ignored
    rebuilt = aggregated_metrics_from_json(payload)
    assert rebuilt is not None
    assert rebuilt.react_rounds_summary == {1: 3, 2: 2}


def test_aggregated_metrics_from_json_missing_criterion_metrics() -> None:
    """A mid-run incremental merge fragment (no criterion_metrics) → None, and the caller falls back to memory."""
    from benchmark.reporting.framework.schema import aggregated_metrics_from_json

    assert aggregated_metrics_from_json(None) is None
    assert aggregated_metrics_from_json({}) is None
    assert aggregated_metrics_from_json({"criterion_metrics": {}}) is None
    assert aggregated_metrics_from_json("not-a-dict") is None


def test_aggregated_metrics_from_json_invalid_payload() -> None:
    """criterion_metrics is present but its value is illegal → deserialization fails and returns None (no exception)."""
    from benchmark.reporting.framework.schema import aggregated_metrics_from_json

    bad = _full_metrics_json()
    bad["criterion_metrics"] = {"intent": {"mean": "not-a-number"}}
    assert aggregated_metrics_from_json(bad) is None
