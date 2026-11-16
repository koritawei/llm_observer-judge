"""sidecar-trace notify parse + unpack + thin filter."""

from unittest.mock import MagicMock, patch

import pytest
import zstandard as zstd
from benchmark.domain.eval_config import EvalConfigEntry, EvalConfigFile, EvalConfigStore
from benchmark.ingest.sidecar.consumer_group import (
    SidecarConsumerIdentity,
    consumer_group_for,
    iter_sidecar_eval_entries,
    live_sidecar_entries,
)
from benchmark.ingest.sidecar.filter import notify_match_reason, select_sampled_configs
from benchmark.ingest.sidecar.notify import SidecarNotifyParseError, parse_notify
from benchmark.ingest.sidecar.unpack import UnpackError, unpack_observations


def _entry(**kwargs: object) -> EvalConfigEntry:
    payload = {
        "evaluatorName": "amap_agent",
        "sampling": 1.0,
        "ingest": "sidecar",
        **kwargs,
    }
    return EvalConfigEntry.model_validate(payload)


def test_parse_notify_sample_json() -> None:
    notify = parse_notify(
        b'{"trace_id":"t-1","finish_reason":"finish","name":"amap","tags":[],'
        b'"has_error":false,"timestamp":1724486400000,"project_id":"p-1",'
        b'"agent_name":"amap-agent"}'
    )
    assert notify.trace_id == "t-1"
    assert notify.finish_reason == "finish"
    assert notify.project_id == "p-1"
    assert notify.agent_name == "amap-agent"
    assert notify.tags == []
    assert notify.has_error is False


def test_parse_notify_missing_optional_fields() -> None:
    notify = parse_notify({"trace_id": "t-2", "finish_reason": "timeout"})
    assert notify.name is None
    assert notify.tags == []
    assert notify.has_error is False
    assert notify.latency_ms is None


def test_parse_notify_ignores_unknown_keys() -> None:
    notify = parse_notify(
        {
            "trace_id": "t-3",
            "finish_reason": "finish",
            "observations": [{"span_id": "nope"}],
            "extra": 1,
        }
    )
    assert notify.trace_id == "t-3"
    assert not hasattr(notify, "observations") or "observations" not in notify.model_fields_set


@pytest.mark.parametrize("reason", ["done", "", None])
def test_parse_notify_rejects_invalid_finish_reason(reason: object) -> None:
    with pytest.raises(SidecarNotifyParseError):
        parse_notify({"trace_id": "t", "finish_reason": reason})


def test_parse_notify_requires_trace_id() -> None:
    with pytest.raises(SidecarNotifyParseError):
        parse_notify({"trace_id": "  ", "finish_reason": "finish"})


def test_parse_notify_rejects_invalid_utf8() -> None:
    with pytest.raises(SidecarNotifyParseError):
        parse_notify(b"\xff\xfe")


def test_iter_sidecar_eval_entries_skips_kafka() -> None:
    config_file = EvalConfigFile(
        configs={
            "p-a": [_entry(ingest="kafka"), _entry(evaluatorName="example")],
            "p-b": [_entry(ingest="kafka", evaluatorName="example")],
        }
    )
    items = iter_sidecar_eval_entries(config_file)
    assert [(pid, entry.evaluator_name) for pid, entry in items] == [("p-a", "example")]


def test_live_sidecar_entries_uses_current_sampling() -> None:
    stale = _entry(sampling=0.0005)
    current = _entry(sampling=1.0, name=["amap"])
    identity = SidecarConsumerIdentity(project_id="p-a", evaluator_name="amap_agent")
    assert live_sidecar_entries([stale], identity) == [stale]
    found = live_sidecar_entries([current], identity)
    assert len(found) == 1
    assert found[0].sampling == 1.0
    assert found[0].name == ["amap"]


def test_live_sidecar_entries_misses_when_ingest_returns_to_kafka() -> None:
    identity = SidecarConsumerIdentity(project_id="p-a", evaluator_name="amap_agent")
    assert live_sidecar_entries([_entry(ingest="kafka")], identity) == []
    assert live_sidecar_entries([], identity) == []


def test_live_sidecar_entries_returns_all_streams() -> None:
    identity = SidecarConsumerIdentity(project_id="p-a", evaluator_name="amap_agent")
    a = _entry(streamInstanceId="ds-a", sampling=0.1)
    b = _entry(streamInstanceId="ds-b", sampling=0.2)
    found = live_sidecar_entries([a, b], identity)
    assert [item.stream_instance_id for item in found] == ["ds-a", "ds-b"]


def test_consumer_group_is_per_project_evaluator() -> None:
    prefix = "CID_sidecar_trace_notify"
    first = consumer_group_for(
        prefix, SidecarConsumerIdentity(project_id="p-a", evaluator_name="amap_agent")
    )
    second = consumer_group_for(
        prefix, SidecarConsumerIdentity(project_id="p-b", evaluator_name="example")
    )
    # The ACL allowlist uses a fixed short CID, and must not concatenate registration / stream
    same_project = consumer_group_for(
        prefix,
        SidecarConsumerIdentity(project_id="p-a", evaluator_name="amap_agent"),
    )
    assert first == "CID_sidecar_trace_notify_p-a_amap_agent"
    assert second == "CID_sidecar_trace_notify_p-b_example"
    assert first != second
    assert same_project == first


def test_unpack_roundtrip() -> None:
    payload = [{"span_id": "s1", "name": "root"}]
    packed = zstd.ZstdCompressor().compress(b'[{"span_id":"s1","name":"root"}]')
    assert unpack_observations(packed) == payload
    assert unpack_observations(memoryview(packed)) == payload


def test_unpack_rejects_non_list() -> None:
    packed = zstd.ZstdCompressor().compress(b'{"span_id":"s1"}')
    with pytest.raises(UnpackError):
        unpack_observations(packed)


def test_unpack_rejects_corrupt_zstd() -> None:
    with pytest.raises(UnpackError, match="zstd JSON array"):
        unpack_observations(b"not-zstd")


def test_filter_miss_does_not_need_readback() -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_configs.return_value = []
    notify = parse_notify(
        {
            "trace_id": "t-miss",
            "finish_reason": "finish",
            "project_id": "unknown",
            "agent_name": "amap-agent",
        }
    )
    assert select_sampled_configs(notify, store) == []
    store.get_configs.assert_called_once_with("unknown")


def test_filter_hits_name_and_sampling() -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_configs.return_value = [
        _entry(name=["amap"], sampling=1.0),
        _entry(evaluatorName="example", name=["other"], sampling=1.0),
    ]
    notify = parse_notify(
        {
            "trace_id": "t-hit",
            "finish_reason": "finish",
            "project_id": "p-1",
            "agent_name": "amap-agent",
            "name": "amap",
            "environment": "prod",
        }
    )
    matched = select_sampled_configs(notify, store)
    assert [item.evaluator_name for item in matched] == ["amap_agent"]


def test_filter_skips_test_adiu_user_id() -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_configs.return_value = [_entry()]
    notify = parse_notify(
        {
            "trace_id": "t-test-adiu",
            "finish_reason": "finish",
            "project_id": "p-1",
            "agent_name": "amap-agent",
            "user_id": "xladadadldadadfadldada_test",
        }
    )
    assert select_sampled_configs(notify, store) == []


def test_filter_skips_kafka_ingest_entry() -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_configs.return_value = [_entry(ingest="kafka")]
    notify = parse_notify(
        {
            "trace_id": "t-kafka",
            "finish_reason": "finish",
            "project_id": "p-1",
            "agent_name": "amap-agent",
            "name": "amap",
        }
    )
    assert select_sampled_configs(notify, store) == []


@patch("benchmark.ingest.sidecar.filter.get_agent_names_by_project_id", return_value=[])
def test_filter_discards_unknown_agent(_mock: object) -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_configs.return_value = [_entry()]
    notify = parse_notify(
        {
            "trace_id": "t-unk",
            "finish_reason": "finish",
            "project_id": "p-1",
            "agent_name": "unknown",
        }
    )
    assert select_sampled_configs(notify, store) == []


@patch(
    "benchmark.ingest.sidecar.filter.get_agent_names_by_project_id",
    return_value=["amap-agent"],
)
def test_filter_resolves_missing_agent_name_from_registry(_mock: object) -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_configs.return_value = [_entry()]
    notify = parse_notify(
        {
            "trace_id": "t-reg",
            "finish_reason": "finish",
            "project_id": "p-1",
        }
    )
    matched = select_sampled_configs(notify, store)
    assert [item.evaluator_name for item in matched] == ["amap_agent"]


@patch("benchmark.ingest.sidecar.filter.get_agent_names_by_project_id", return_value=[])
def test_filter_missing_agent_name_hits_declared_entry(_mock: object) -> None:
    """Missing name and the entry declares agentName → the owner is the declared value, and it hits directly.

    No registry candidate does not block it either (the declaration is the truth, consistent with the kafka chain's S3' injection semantics).
    """
    store = MagicMock(spec=EvalConfigStore)
    store.get_configs.return_value = [_entry(agentName="amap-agent", name=["amap"])]
    notify = parse_notify(
        {
            "trace_id": "t-declared",
            "finish_reason": "finish",
            "project_id": "p-1",
            "name": "amap",
        }
    )
    matched = select_sampled_configs(notify, store)
    assert [item.evaluator_name for item in matched] == ["amap_agent"]


@patch("benchmark.ingest.sidecar.filter.get_agent_names_by_project_id", return_value=[])
def test_filter_reported_agent_name_mismatched_declared_blocked(_mock: object) -> None:
    """The name hits, the report says vui_agent, and the entry declares amap-agent → block."""
    store = MagicMock(spec=EvalConfigStore)
    store.get_configs.return_value = [
        _entry(agentName="amap-agent", name=["amap-agent.stream.ai_chat.standard"]),
    ]
    notify = parse_notify(
        {
            "trace_id": "t-mismatch",
            "finish_reason": "finish",
            "project_id": "p-1",
            "name": "amap-agent.stream.ai_chat.standard",
            "agent_name": "vui_agent",
        }
    )
    assert select_sampled_configs(notify, store) == []


@patch("benchmark.ingest.sidecar.filter.get_agent_names_by_project_id", return_value=[])
def test_filter_reported_agent_name_hyphen_underscore_equivalent(_mock: object) -> None:
    """When the name hits, an SDK underscore report and a page hyphen declaration are the same agent, and must not filter_agent_name."""
    store = MagicMock(spec=EvalConfigStore)
    store.get_configs.return_value = [
        _entry(agentName="amap-agent", name=["amap-agent.stream.ai_chat.standard"]),
    ]
    notify = parse_notify(
        {
            "trace_id": "t-equiv",
            "finish_reason": "finish",
            "project_id": "p-1",
            "name": "amap-agent.stream.ai_chat.standard",
            "agent_name": "amap_agent",
        }
    )
    matched = select_sampled_configs(notify, store)
    assert [item.evaluator_name for item in matched] == ["amap_agent"]


@pytest.mark.parametrize(
    ("filters", "name", "expected"),
    [
        ({"traceNameContains": ["vehicle", "vui"]}, "amap.vui", "hit"),
        ({"traceNameContains": ["vui"]}, "amap", "filter_name"),
        ({"traceNameContains": ["vui"]}, None, "filter_name"),
        ({"traceNameExcludes": ["vehicle", "vui"]}, "amap.vui", "filter_name"),
        ({"traceNameExcludes": ["vui"]}, None, "hit"),
        ({"traceNameContains": [], "traceNameExcludes": []}, None, "hit"),
        (
            {"traceNameContains": ["vui"], "traceNameExcludes": ["debug"]},
            "vui.debug",
            "filter_name",
        ),
        ({"name": ["other"], "traceNameContains": ["vui"]}, "amap.vui", "filter_name"),
        (
            {
                "traceNameContains": ["other"],
                "tags": ["online"],
                "startTime": "2024-01-01T00:00:00Z",
            },
            "amap.vui",
            "filter_name",
        ),
        (
            {"traceNameContains": ["vui"], "tags": ["online"], "startTime": "2024-01-01T00:00:00Z"},
            "amap.vui",
            "filter_tags",
        ),
        (
            {"traceNameContains": ["vui"], "startTime": "2024-01-01T00:00:00Z"},
            "amap.vui",
            "filter_time",
        ),
    ],
)
def test_sidecar_substring_filters_and_reason_precedence(
    filters: dict,
    name: str | None,
    expected: str,
) -> None:
    notify = parse_notify(
        {
            "trace_id": "t",
            "finish_reason": "finish",
            "project_id": "p-1",
            "agent_name": "amap-agent",
            "name": name,
        }
    )
    assert notify_match_reason(notify, _entry(**filters), project_id="p-1") == expected
