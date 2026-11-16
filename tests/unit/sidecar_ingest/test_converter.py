"""Sidecar notify + observation tree → EvaluatorRequest."""

from unittest.mock import patch

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.ingest.sidecar.converter import SidecarIdentityError, to_sidecar_evaluator_request
from benchmark.ingest.sidecar.notify import SidecarTraceNotify

_TREE = [
    {
        "span_id": "gen-1",
        "parent_span_id": "root",
        "name": "AmapChatModel.chat",
        "type": "GENERATION",
        "level": "DEFAULT",
        "end_time": 300,
        "input": [{"role": "user", "content": "内部 messages"}],
        "output": {"text": "generation 输出"},
        "metadata": {"span": "generation"},
    },
    {
        "span_id": "root",
        "parent_span_id": "",
        "name": "amap-agent.stream.ai_chat.standard",
        "type": "SPAN",
        "level": "DEFAULT",
        "end_time": 200,
        "input": {"query": "附近咖啡"},
        "output": {"text": "去星巴克"},
        "metadata": {"request_id": "r-1", "scene": "navi"},
    },
]


def test_converter_uses_io_root_input_output_metadata() -> None:
    notify = SidecarTraceNotify.model_validate(
        {
            "trace_id": "t-1",
            "finish_reason": "finish",
            "project_id": "p-1",
            "agent_name": "amap-agent",
            "name": "chat",
            "user_id": "u-1",
            "session_id": "s-1",
            "environment": "prod",
            "timestamp": 1724486400000,
        }
    )
    request = to_sidecar_evaluator_request(
        notify,
        _TREE,
        EvalConfigEntry.model_validate({"evaluatorName": "amap_agent", "sampling": 1.0}),
    )
    assert request.input == {"query": "附近咖啡"}
    assert request.output == {"text": "去星巴克"}
    assert request.metadata["request_id"] == "r-1"
    assert request.metadata["scene"] == "navi"
    assert request.metadata["agent_name"] == "amap-agent"
    assert request.metadata["pipeline_source"] == "sidecar"
    assert request.agent_name == "amap-agent"
    assert request.trace_id == "t-1"
    assert request.trace_timestamp == 1724486400000


@patch(
    "benchmark.ingest.sidecar.filter.get_agent_names_by_project_id",
    return_value=["from-registry"],
)
def test_converter_stamps_registry_agent_name(_mock: object) -> None:
    notify = SidecarTraceNotify.model_validate(
        {
            "trace_id": "t-reg",
            "finish_reason": "finish",
            "project_id": "p-1",
        }
    )
    tree = [
        {
            "span_id": "root",
            "parent_span_id": "",
            "input": {"q": 1},
            "output": "a",
        }
    ]
    request = to_sidecar_evaluator_request(
        notify,
        tree,
        EvalConfigEntry.model_validate({"evaluatorName": "amap_agent", "sampling": 1.0}),
    )
    assert request.agent_name == "from-registry"
    assert request.metadata["agent_name"] == "from-registry"


@patch("benchmark.ingest.sidecar.filter.get_agent_names_by_project_id", return_value=[])
def test_converter_identity_error_when_agent_name_unresolved(_mock: object) -> None:
    notify = SidecarTraceNotify.model_validate(
        {
            "trace_id": "t-miss",
            "finish_reason": "finish",
            "project_id": "p-1",
        }
    )
    with pytest.raises(SidecarIdentityError):
        to_sidecar_evaluator_request(
            notify,
            [{"span_id": "root", "parent_span_id": ""}],
            EvalConfigEntry.model_validate({"evaluatorName": "amap_agent", "sampling": 1.0}),
        )


@patch(
    "benchmark.ingest.sidecar.filter.get_agent_names_by_project_id",
    return_value=["from-registry"],
)
def test_converter_declared_agent_name_wins_over_registry(_mock: object) -> None:
    """When the name hits, the entry's agentName wins over the reported value and the registry fallback."""
    notify = SidecarTraceNotify.model_validate(
        {
            "trace_id": "t-declared",
            "finish_reason": "finish",
            "project_id": "p-1",
            "name": "amap-agent.stream.ai_chat.standard",
        }
    )
    request = to_sidecar_evaluator_request(
        notify,
        [{"span_id": "root", "parent_span_id": ""}],
        EvalConfigEntry.model_validate(
            {
                "evaluatorName": "amap_agent",
                "sampling": 1.0,
                "agentName": "amap-agent",
                "name": ["amap-agent.stream.ai_chat.standard"],
            }
        ),
    )
    assert request.agent_name == "amap-agent"
    assert request.metadata["agent_name"] == "amap-agent"
