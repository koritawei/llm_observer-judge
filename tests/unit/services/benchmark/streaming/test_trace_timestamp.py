"""Trace request timestamp propagation tests for the Kafka judge path."""

from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.streaming.consumer import trace_to_observation
from benchmark.streaming.runner import to_evaluator_request
from benchmark.streaming.schemas import TraceRecord


def test_trace_timestamp_propagates_to_evaluator_request() -> None:
    timestamp = "2026-08-03T06:52:03.968Z"
    record = TraceRecord(
        id="trace-1",
        project_id="project-1",
        name="amap-agent.stream.ai_chat.standard",
        environment="prod",
        metadata={"agent_name": "amap-agent"},
        timestamp=timestamp,
    )

    observation = trace_to_observation(record)
    request = to_evaluator_request(
        observation,
        EvalConfigEntry(evaluatorName="amap_agent", sampling=1.0),
    )

    assert observation.trace_timestamp == timestamp
    assert request.trace_timestamp == timestamp
