"""Name-first resolution: after a name hit, cross projects and force-replace project_id."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.domain.schemas import EvalConfigFile
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorDeps,
    resolve_streaming_kafka_configs,
)
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.libs.mq.consumer import ConsumedMessage

_SEARCH_PROJECT = "cmpwln0qs04fo6j8eryfdgznb"
_PLATFORM_PROJECT = "cmt2emw9z0001bn3454gcjt4q"
_TRACE_NAME = "search_ai_card"


def _entry(**kwargs: object) -> EvalConfigEntry:
    base: dict[str, object] = {
        "evaluatorName": "search_eval",
        "sampling": 1.0,
        "name": [_TRACE_NAME],
    }
    base.update(kwargs)
    return EvalConfigEntry.model_validate(base)


def test_resolve_prefers_name_match_across_projects() -> None:
    """When the message's project is not registered, a name can still hit an entry under another project."""
    configured = _entry()
    store = MagicMock()
    store.get_configs.return_value = []
    store.get_config_file.return_value = EvalConfigFile(
        configs={_PLATFORM_PROJECT: [configured]},
    )

    configs, source, resolved = resolve_streaming_kafka_configs(
        store,
        project_id=_SEARCH_PROJECT,
        trace_name=_TRACE_NAME,
    )
    assert source == "name"
    assert configs == [configured]
    assert resolved == _PLATFORM_PROJECT
    store.get_configs.assert_not_called()


def test_resolve_falls_back_to_project_when_no_name_hit() -> None:
    open_cfg = _entry(name=None, evaluatorName="example")
    store = MagicMock()
    store.get_configs.return_value = [open_cfg]
    store.get_config_file.return_value = EvalConfigFile(
        configs={_PLATFORM_PROJECT: [_entry()]},
    )

    configs, source, resolved = resolve_streaming_kafka_configs(
        store,
        project_id=_PLATFORM_PROJECT,
        trace_name="unrelated_trace",
    )
    assert source == "project"
    assert configs == [open_cfg]
    assert resolved == _PLATFORM_PROJECT


def test_resolve_none_when_no_name_and_project_empty() -> None:
    store = MagicMock()
    store.get_configs.return_value = []
    store.get_config_file.return_value = EvalConfigFile(
        configs={_PLATFORM_PROJECT: [_entry()]},
    )

    configs, source, resolved = resolve_streaming_kafka_configs(
        store,
        project_id=_SEARCH_PROJECT,
        trace_name="unrelated_trace",
    )
    assert source == "none"
    assert configs == []
    assert resolved is None


def test_resolve_name_keeps_first_matched_project_only() -> None:
    """When several projects hit the same name, keep only the entries under the first registered project."""
    first = _entry(evaluatorName="search_eval_a")
    second = _entry(evaluatorName="search_eval_b")
    store = MagicMock()
    store.get_config_file.return_value = EvalConfigFile(
        configs={
            _PLATFORM_PROJECT: [first],
            _SEARCH_PROJECT: [second],
        },
    )

    configs, source, resolved = resolve_streaming_kafka_configs(
        store,
        project_id="unrelated",
        trace_name=_TRACE_NAME,
    )
    assert source == "name"
    assert resolved == _PLATFORM_PROJECT
    assert configs == [first]


@pytest.mark.asyncio
async def test_consumer_forces_project_id_from_name_match() -> None:
    """When the Kafka project ≠ the registered project, a name hit forces request.project_id to the registered side."""
    configured = _entry(agentName="search_eval")
    config_store = MagicMock()
    config_store.get_configs.return_value = []
    config_store.get_config_file.return_value = EvalConfigFile(
        configs={_PLATFORM_PROJECT: [configured]},
    )
    evaluator_service = AsyncMock()
    evaluator_service.evaluate.return_value = MagicMock(score=1.0, reason="ok", sub_scores=None)
    seen_project_ids: list[str | None] = []

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        obs_project = getattr(observation, "project_id", None)
        seen_project_ids.append(obs_project)
        return EvaluatorRequest(
            input={"q": "x"},
            output={"a": "y"},
            metadata={"agent_name": "search_eval"},
            skip_trace_resolve=True,
            project_id=obs_project,
            trace_id="t-cross",
            agent_name="search_eval",
        )

    evaluator = BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=evaluator_service,
            request_converter=_converter,
        ),
    )
    record = TraceRecord(
        id="t-cross",
        project_id=_SEARCH_PROJECT,
        name=_TRACE_NAME,
        environment="prod",
        input={"q": "x"},
        output={"a": "y"},
        metadata={"agent_name": "search_eval"},
        tags=["prod"],
    )
    msg = ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="trace", record=record.model_dump()),
        key=f"{_SEARCH_PROJECT}:t-cross",
        offset=1,
        partition=0,
    )
    await evaluator(msg)
    evaluator_service.evaluate.assert_awaited()
    assert seen_project_ids == [_PLATFORM_PROJECT]
    request = evaluator_service.evaluate.await_args.args[1]
    assert request.project_id == _PLATFORM_PROJECT
    assert evaluator._funnel.filter_no_config == 0
    assert evaluator._funnel.sample_hit >= 1
