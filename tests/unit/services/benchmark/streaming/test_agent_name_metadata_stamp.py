"""Ensure agent_name is always present in request.metadata for Judge / LLM."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorDeps,
)
from benchmark.streaming.metadata import AGENT_NAME_METADATA_KEY
from benchmark.streaming.runner import to_evaluator_request
from benchmark.streaming.schemas import (
    ObservationForEval,
    RecordStreamMessage,
    TraceRecord,
)
from benchmark.libs.mq.consumer import ConsumedMessage


def _obs(**kwargs: object) -> ObservationForEval:
    defaults: dict[str, object] = {
        "span_id": "t-1",
        "trace_id": "t-1",
        "project_id": "proj-1",
        "type": "TRACE",
        "name": "chat",
        "environment": "prod",
        "metadata": {},
    }
    defaults.update(kwargs)
    return ObservationForEval(**defaults)  # type: ignore[arg-type]


def _config() -> EvalConfigEntry:
    return EvalConfigEntry(evaluatorName="example", sampling=1.0)


class TestToEvaluatorRequestAgentNameStamp:
    def test_stamps_snake_case_from_metadata(self) -> None:
        req = to_evaluator_request(
            _obs(metadata={"agent_name": "amap_agent"}),
            _config(),
        )
        assert req.agent_name == "amap_agent"
        assert req.metadata[AGENT_NAME_METADATA_KEY] == "amap_agent"

    def test_stamps_registration_id_for_slot_logs(self) -> None:
        req = to_evaluator_request(
            _obs(metadata={"agent_name": "amap_agent"}),
            EvalConfigEntry(
                evaluatorName="example",
                agentRegistrationId="example--biz--default--v56",
                sampling=1.0,
            ),
        )
        assert req.metadata["agent_registration_id"] == "example--biz--default--v56"

    def test_normalizes_legacy_agentName_into_metadata(self) -> None:
        req = to_evaluator_request(
            _obs(metadata={"agentName": "vehicle-agent"}),
            _config(),
        )
        assert req.agent_name == "vehicle-agent"
        assert req.metadata[AGENT_NAME_METADATA_KEY] == "vehicle-agent"
        assert "agentName" not in req.metadata

    def test_model_copy_stamps_when_top_level_only(self) -> None:
        """Simulate converter that set top-level agent_name without metadata."""
        # Build a request as if agent_name were only on the top-level field.
        bare = EvaluatorRequest(
            agent_name="top-only-agent",
            metadata={"pipeline_source": "kafka"},
            skip_trace_resolve=True,
        )
        stamped_meta = dict(bare.metadata)
        stamped_meta[AGENT_NAME_METADATA_KEY] = bare.agent_name  # type: ignore[assignment]
        stamped = bare.model_copy(update={"metadata": stamped_meta})
        assert stamped.metadata[AGENT_NAME_METADATA_KEY] == "top-only-agent"
        assert stamped.agent_name == "top-only-agent"


@pytest.mark.asyncio
async def test_registry_resolve_uses_model_copy_on_observation() -> None:
    """B1.5 must leave agent_name on observation.metadata via model_copy."""
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(evaluatorName="example", sampling=1.0),
    ]
    captured_obs: list[ObservationForEval] = []
    evaluator_service = AsyncMock()

    def _converter(observation: ObservationForEval, config: object) -> EvaluatorRequest:
        captured_obs.append(observation)
        return to_evaluator_request(observation, config)  # type: ignore[arg-type]

    evaluator = BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=evaluator_service,
            request_converter=_converter,
        ),
    )
    record = TraceRecord(
        id="t-reg-stamp",
        project_id="proj-1",
        environment="prod",
        metadata={},
        input={"q": "test"},
        output={"a": "resp"},
    )
    msg = ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="trace", record=record.model_dump()),
        key="proj-1:t-reg-stamp",
        offset=1,
        partition=0,
    )
    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["registry-agent"],
    ):
        await evaluator(msg)

    assert len(captured_obs) == 1
    assert captured_obs[0].metadata[AGENT_NAME_METADATA_KEY] == "registry-agent"
    assert evaluator_service.evaluate.call_count == 1
    request = evaluator_service.evaluate.call_args.args[1]
    assert request.agent_name == "registry-agent"
    assert request.metadata[AGENT_NAME_METADATA_KEY] == "registry-agent"


@pytest.mark.asyncio
async def test_trace_evaluator_stamps_top_level_agent_into_metadata() -> None:
    """When converter only sets Request.agent_name, BenchmarkStreamingEvaluator stamps metadata."""
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(evaluatorName="example", sampling=1.0),
    ]
    captured: list[EvaluatorRequest] = []
    evaluator_service = AsyncMock()

    async def _capture(name: str, request: EvaluatorRequest) -> None:
        captured.append(request)

    evaluator_service.evaluate.side_effect = _capture

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(
            agent_name="top-level-only",
            metadata={},
            skip_trace_resolve=True,
        )

    evaluator = BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=evaluator_service,
            request_converter=_converter,
        ),
    )
    record = TraceRecord(
        id="t-top",
        project_id="proj-1",
        environment="prod",
        metadata={"agent_name": "top-level-only"},
        input={"q": "test"},
        output={"a": "resp"},
    )
    msg = ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="trace", record=record.model_dump()),
        key="proj-1:t-top",
        offset=2,
        partition=0,
    )
    await evaluator(msg)

    assert len(captured) == 1
    assert captured[0].agent_name == "top-level-only"
    assert captured[0].metadata[AGENT_NAME_METADATA_KEY] == "top-level-only"
    assert "kafka" in captured[0].metadata
