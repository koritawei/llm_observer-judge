"""Enum lifecycle guard: an enum with zero references on the business side must not exist in data_model.enums.

This stops a later change from adding an enum that was "reserved for the future" back into enums.py.
"""

from benchmark.records import enums


def test_only_live_enums_in_data_model() -> None:
    expected = {
        "TriageLevel",
        "TraceStatus",
        "NotificationTaskJobStatus",
        "BenchmarkStatus",
    }
    actual = set(enums.__all__)
    assert actual == expected, (
        f"data_model.enums 出现未授权枚举: {actual - expected}, 或丢失: {expected - actual}"
    )


def test_triage_level_values() -> None:
    assert {v.value for v in enums.TriageLevel} == {
        "good",
        "watch",
        "mild_bad",
        "severe_bad",
        "no_decision",
        "attribution_overridden",
    }


def test_triage_attribution_schema_uses_shared_enum() -> None:
    """TriageLevel must come from data_model.enums (so it does not fork again)."""
    from benchmark.attribution.service import schema

    assert schema.TriageLevel is enums.TriageLevel


def test_benchmark_schema_uses_shared_enum() -> None:
    """BenchmarkStatus must come from data_model.enums (so it does not fork again)."""
    from benchmark.runs.schema import (
        BenchmarkStatus as SchemaBenchmarkStatus,
    )

    assert SchemaBenchmarkStatus is enums.BenchmarkStatus


def test_benchmark_status_values() -> None:
    assert {v.value for v in enums.BenchmarkStatus} == {
        # accepting: streaming ingest in progress (benchmark-inline-dataset design §4)
        "initializing",
        "accepting",
        "running",
        "aggregating",
        "generating_report",
        "completed",
        "failed",
        "stopped",
    }

