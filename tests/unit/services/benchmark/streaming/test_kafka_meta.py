"""Unit tests for Kafka message metadata propagation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    KAFKA_MESSAGE_META_KEY,
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorDeps,
    extract_kafka_message_meta,
)
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.libs.mq.consumer import ConsumedMessage


def test_extract_kafka_message_meta_includes_position_and_headers() -> None:
    msg = ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="trace", record={"id": "t"}),
        key="proj:t1",
        headers=[("x-request-id", b"abc"), ("plain", b"v")],
        partition=3,
        offset=99,
        timestamp_ms=1_700_000_000_000,
    )

    meta = extract_kafka_message_meta(msg)

    assert meta == {
        "topic": "langfuse.records",
        "partition": 3,
        "offset": 99,
        "key": "proj:t1",
        "timestamp_ms": 1_700_000_000_000,
        "headers": {"x-request-id": "abc", "plain": "v"},
    }


@pytest.mark.asyncio
async def test_trace_evaluator_injects_kafka_meta_into_request() -> None:
    """B5 evaluate receives request.metadata['kafka'] from ConsumedMessage."""
    config = EvalConfigEntry(evaluatorName="example", sampling=1.0)
    config_store = MagicMock()
    config_store.get_configs.return_value = [config]

    captured: list[EvaluatorRequest] = []

    async def _capture_evaluate(name: str, request: EvaluatorRequest) -> None:
        captured.append(request)

    evaluator_service = AsyncMock()
    evaluator_service.evaluate.side_effect = _capture_evaluate

    def _converter(observation: object, cfg: object) -> EvaluatorRequest:
        return EvaluatorRequest(
            input="q",
            output="a",
            metadata={"agent_name": "agent-a", "pipeline_source": "kafka"},
            trace_id="trace-001",
            agent_name="agent-a",
            project_id="proj-001",
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
        id="trace-001",
        project_id="proj-001",
        environment="prod",
        metadata={"agent_name": "agent-a"},
        input="q",
        output="a",
    )
    payload = RecordStreamMessage(record_type="trace", record=record.model_dump())
    msg = ConsumedMessage(
        topic="langfuse.records",
        payload=payload,
        key="proj-001:trace-001",
        headers=[("h1", b"v1")],
        partition=2,
        offset=7,
        timestamp_ms=123,
    )

    await evaluator(msg)

    assert len(captured) == 1
    kafka_meta = captured[0].metadata[KAFKA_MESSAGE_META_KEY]
    assert kafka_meta["topic"] == "langfuse.records"
    assert kafka_meta["partition"] == 2
    assert kafka_meta["offset"] == 7
    assert kafka_meta["key"] == "proj-001:trace-001"
    assert kafka_meta["timestamp_ms"] == 123
    assert kafka_meta["headers"] == {"h1": "v1"}
