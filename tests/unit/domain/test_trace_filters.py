"""Kafka filter semantics stay on the shared matcher."""

import pytest
from benchmark.domain.schemas import EvalConfigEntry
from benchmark.domain.trace_filters import (
    is_test_adiu,
    matches_trace_filters,
    parse_timestamp,
    should_sample_deterministic,
    should_skip_test_adiu,
)
from benchmark.streaming.schemas import TraceRecord


def test_matches_name_tags_timestamp() -> None:
    config = EvalConfigEntry.model_validate(
        {
            "evaluatorName": "example",
            "sampling": 1.0,
            "name": ["amap"],
            "tags": ["online"],
            "startTime": "2024-01-01T00:00:00Z",
        }
    )
    hit = TraceRecord(
        id="t1",
        project_id="p",
        name="amap",
        tags=["online", "x"],
        timestamp="2024-06-01T00:00:00Z",
    )
    miss_name = hit.model_copy(update={"name": "other"})
    assert matches_trace_filters(config, hit) is True
    assert matches_trace_filters(config, miss_name) is False


def test_parse_timestamp_int_overflow_returns_none() -> None:
    assert parse_timestamp(10**20) is None


def test_sampling_is_deterministic() -> None:
    key = "trace-a:amap_agent"
    assert should_sample_deterministic(1.0, key) is True
    assert should_sample_deterministic(0.0, key) is False
    assert should_sample_deterministic(0.5, key) == should_sample_deterministic(0.5, key)


def test_is_test_adiu_matches_suffix_case_insensitive() -> None:
    assert is_test_adiu("xladadadldadadfadldada_test") is True
    assert is_test_adiu("xxx_TEST") is True
    assert is_test_adiu("  foo_Test  ") is True


def test_is_test_adiu_ignores_non_suffix() -> None:
    assert is_test_adiu("real_user") is False
    assert is_test_adiu("test_prefix") is False
    assert is_test_adiu("foo_test_bar") is False
    assert is_test_adiu("") is False
    assert is_test_adiu(None) is False
    assert is_test_adiu(123) is False


def test_should_skip_test_adiu_from_user_id() -> None:
    assert should_skip_test_adiu(user_id="abc_test") is True
    assert should_skip_test_adiu(user_id="abc") is False
    assert should_skip_test_adiu(user_id=None) is False


def test_should_skip_test_adiu_from_nested_input() -> None:
    assert should_skip_test_adiu(input_value={"adiu": "device_test"}) is True
    assert (
        should_skip_test_adiu(
            input_value={"request_body": {"user": {"adiu": "device_test"}}},
        )
        is True
    )
    assert should_skip_test_adiu(input_value={"adiu": "ok"}) is False


def test_should_skip_test_adiu_decodes_json_string_input() -> None:
    assert should_skip_test_adiu(input_value='{"user": {"adiu": "x_test"}}') is True


@pytest.mark.parametrize(
    ("filters", "name", "expected"),
    [
        ({"traceNameContains": ["vehicle", "vui"]}, "amap.vui_agent", True),
        ({"traceNameContains": ["vehicle", "vui"]}, "amap_agent", False),
        ({"traceNameContains": ["vui"]}, None, False),
        ({"traceNameExcludes": ["vehicle", "vui"]}, "amap.vui_agent", False),
        ({"traceNameExcludes": ["vehicle", "vui"]}, "amap_agent", True),
        ({"traceNameExcludes": ["vui"]}, None, True),
        ({"traceNameContains": ["vui"], "traceNameExcludes": ["debug"]}, "vui.debug", False),
        ({"traceNameContains": [], "traceNameExcludes": []}, None, True),
        ({}, None, True),
        ({"name": ["vui"], "traceNameContains": ["vui"]}, "amap.vui", False),
        ({"name": ["amap"], "traceNameContains": ["vui"]}, "amap", False),
        ({"name": ["amap.vui"], "traceNameContains": ["vui"]}, "amap.vui", True),
        ({"traceNameContains": ["VUI"]}, "vui", False),
    ],
)
def test_trace_substring_name_filters(filters: dict, name: str | None, expected: bool) -> None:
    config = EvalConfigEntry.model_validate(
        {"evaluatorName": "example", "sampling": 1.0, **filters}
    )
    trace = TraceRecord(id="t", project_id="p", name=name)
    assert matches_trace_filters(config, trace) is expected


@pytest.mark.parametrize(
    ("tags", "timestamp", "expected"),
    [
        (["online"], "2024-06-01T00:00:00Z", True),
        (["offline"], "2024-06-01T00:00:00Z", False),
        (["online"], "2023-06-01T00:00:00Z", False),
        (["online"], None, False),
    ],
)
def test_substring_filters_combine_with_tags_and_time(
    tags: list[str],
    timestamp: str | None,
    expected: bool,
) -> None:
    config = EvalConfigEntry.model_validate(
        {
            "evaluatorName": "example",
            "sampling": 1.0,
            "traceNameContains": ["vui"],
            "tags": ["online"],
            "startTime": "2024-01-01T00:00:00Z",
        }
    )
    trace = TraceRecord(id="t", project_id="p", name="amap.vui", tags=tags, timestamp=timestamp)
    assert matches_trace_filters(config, trace) is expected
