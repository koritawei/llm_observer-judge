"""Unit tests: Kafka BenchmarkStreamingEvaluator discards test adiu suffix."""

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


def _make_msg(record: TraceRecord) -> ConsumedMessage[RecordStreamMessage]:
    payload = RecordStreamMessage(
        record_type="trace",
        record=record.model_dump(),
    )
    return ConsumedMessage(
        topic="langfuse.records",
        payload=payload,
        key=f"{record.project_id}:{record.id}",
        offset=100,
        partition=0,
    )


def _make_evaluator(
    evaluator_service: AsyncMock, config_store: MagicMock
) -> BenchmarkStreamingEvaluator:
    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(metadata={}, skip_trace_resolve=True)

    return BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=evaluator_service,
            request_converter=_converter,
        ),
    )


def _ok_record(**updates: object) -> TraceRecord:
    payload: dict[str, object] = {
        "id": "t-adiu",
        "project_id": "proj-1",
        "environment": "prod",
        "metadata": {"agent_name": "amap_agent"},
        "input": {"query": "附近咖啡"},
        "output": {"text": "去星巴克"},
    }
    payload.update(updates)
    return TraceRecord.model_validate(payload)


@pytest.mark.asyncio
async def test_skips_user_id_with_test_suffix() -> None:
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(evaluatorName="amap_agent", sampling=1.0),
    ]
    evaluator_service = AsyncMock()
    evaluator = _make_evaluator(evaluator_service, config_store)

    await evaluator(
        _make_msg(_ok_record(user_id="xladadadldadadfadldada_test")),
    )

    evaluator_service.evaluate.assert_not_called()
    assert evaluator._funnel.filter_test_adiu == 1
    assert evaluator._funnel.eligible == 0
    assert evaluator._funnel.sample_hit == 0


@pytest.mark.asyncio
async def test_skips_nested_input_test_adiu() -> None:
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(evaluatorName="amap_agent", sampling=1.0),
    ]
    evaluator_service = AsyncMock()
    evaluator = _make_evaluator(evaluator_service, config_store)

    await evaluator(
        _make_msg(
            _ok_record(
                input={
                    "query": "附近咖啡",
                    "request_body": {"user": {"adiu": "device_test"}},
                },
            ),
        ),
    )

    evaluator_service.evaluate.assert_not_called()
    assert evaluator._funnel.filter_test_adiu == 1


@pytest.mark.asyncio
async def test_real_adiu_continues_to_evaluate() -> None:
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(evaluatorName="amap_agent", sampling=1.0),
    ]
    evaluator_service = AsyncMock()
    evaluator = _make_evaluator(evaluator_service, config_store)

    await evaluator(_make_msg(_ok_record(user_id="real_user")))

    assert evaluator_service.evaluate.call_count == 1
    assert evaluator._funnel.filter_test_adiu == 0
    assert evaluator._funnel.eval_ok == 1
