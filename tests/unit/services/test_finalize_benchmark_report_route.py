"""Lock: the external report route of a sync-judge finalize hits only search_eval.

ieval has no ``stat_dimensions`` registration, so even with oss_prefix / parse_interceptor_ref
the report path must not switch.
"""

from benchmark.runs.common.dimension_stats import (
    DimensionStatSpec,
    clear_stat_dimensions,
    register_stat_dimensions,
    uses_external_report_assembly,
)


def test_uses_external_report_when_stat_dimensions_in_config() -> None:
    config = {
        "stat_dimensions": [
            {"key": "intent_understanding", "name": "意图理解", "full_score": 2.0},
        ],
    }
    assert uses_external_report_assembly(config, "search_eval") is True


def test_oss_prefix_alone_does_not_route_ieval() -> None:
    """An ieval OSS submit also has oss_prefix, and the report path must not switch on that alone."""
    clear_stat_dimensions()
    try:
        assert (
            uses_external_report_assembly(
                {"oss_prefix": "oss://bucket/path", "parse_interceptor_ref": "x:Y"},
                "amap_chat",
            )
            is False
        )
    finally:
        clear_stat_dimensions()


def test_plain_ieval_sync_does_not_use_external_report() -> None:
    clear_stat_dimensions()
    try:
        assert uses_external_report_assembly({"rubric": []}, "amap_chat") is False
    finally:
        clear_stat_dimensions()


def test_uses_external_report_via_registry_fallback() -> None:
    clear_stat_dimensions()
    try:
        register_stat_dimensions(
            "search_eval",
            (DimensionStatSpec(key="relevance", name="内容切题"),),
        )
        assert uses_external_report_assembly({}, "search_eval") is True
        assert uses_external_report_assembly({}, "amap_chat") is False
    finally:
        clear_stat_dimensions()
