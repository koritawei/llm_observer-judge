"""Persisted aggregation config must preserve the custom callback reference."""

from benchmark.runs.common.config_helpers import load_aggregation_from_config


def test_custom_aggregation_reference_survives_config_reload() -> None:
    config = {
        "aggregation": {
            "method": "custom",
            "pass_threshold": 1.0,
            "custom_fn_ref": "tests.fixtures:custom_aggregation",
        }
    }
    restored = load_aggregation_from_config(config)
    assert restored.method == "custom"
    assert restored.custom_fn_ref == "tests.fixtures:custom_aggregation"
    assert restored.pass_threshold == 1.0


def test_default_aggregation_still_uses_weighted_average() -> None:
    restored = load_aggregation_from_config({})
    assert restored.method == "weighted_avg"
    assert restored.custom_fn_ref is None
