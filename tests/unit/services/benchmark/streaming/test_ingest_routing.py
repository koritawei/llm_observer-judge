"""Kafka Path B skips sidecar ingest entries."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorDeps,
)
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.libs.mq.consumer import ConsumedMessage


def _evaluator(configs: list[EvalConfigEntry]) -> tuple[BenchmarkStreamingEvaluator, AsyncMock]:
    config_store = MagicMock()
    config_store.get_configs.return_value = configs
    evaluator_service = AsyncMock()

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(metadata={}, skip_trace_resolve=True)

    return (
        BenchmarkStreamingEvaluator(
            deps=BenchmarkStreamingEvaluatorDeps(
                config_store=config_store,
                evaluator_service=evaluator_service,
                request_converter=_converter,
            ),
        ),
        evaluator_service,
    )


def _msg(trace_id: str = "t-1") -> ConsumedMessage:
    record = TraceRecord(
        id=trace_id,
        project_id="proj-1",
        name="chat",
        environment="prod",
        input={"q": "test"},
        output={"a": "resp"},
        metadata={"agent_name": "amap_agent"},
        tags=["prod"],
    )
    return ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="trace", record=record.model_dump()),
        key=f"proj-1:{trace_id}",
        offset=1,
        partition=0,
    )


@pytest.mark.asyncio
async def test_sidecar_ingest_entry_is_not_sampled() -> None:
    evaluator, svc = _evaluator(
        [
            EvalConfigEntry(
                evaluatorName="example",
                sampling=1.0,
                ingest="sidecar",
            )
        ]
    )
    await evaluator(_msg())
    svc.evaluate.assert_not_awaited()
    assert evaluator._funnel.filter_no_config == 1
    assert evaluator._funnel.sample_hit == 0


@pytest.mark.asyncio
async def test_mixed_ingest_only_samples_kafka_entry() -> None:
    evaluator, svc = _evaluator(
        [
            EvalConfigEntry(evaluatorName="example", sampling=1.0),
            EvalConfigEntry(
                evaluatorName="amap_agent",
                sampling=1.0,
                ingest="sidecar",
            ),
        ]
    )
    await evaluator(_msg())
    svc.evaluate.assert_awaited_once()
    assert svc.evaluate.await_args.args[0] == "example"
