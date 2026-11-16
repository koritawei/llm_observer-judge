"""Unit tests for Kafka sampling-hit detail logging helpers."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorDeps,
    _preview_for_log,
)
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.libs.mq.consumer import ConsumedMessage


class TestPreviewForLog:
    def test_none_and_small_object(self) -> None:
        assert _preview_for_log(None) == "null"
        assert _preview_for_log({"q": "加油站"}) == '{"q": "加油站"}'

    def test_truncates_long_payload(self) -> None:
        big = {"text": "x" * 5000}
        preview = _preview_for_log(big, max_chars=100)
        assert "truncated" in preview
        assert len(preview) < 5000
        assert preview.startswith("{")


@pytest.mark.asyncio
async def test_sampling_hit_logs_trace_details(caplog: pytest.LogCaptureFixture) -> None:
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(evaluatorName="example", sampling=1.0),
    ]
    evaluator_service = AsyncMock()

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(metadata={}, skip_trace_resolve=True)

    evaluator = BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=evaluator_service,
            request_converter=_converter,
        ),
    )
    record = TraceRecord(
        id="t-hit",
        project_id="proj-1",
        name="chat",
        environment="prod",
        input={"query": "附近加油站"},
        output={"text": "推荐壳牌"},
        metadata={"agent_name": "amap_agent", "scene": "nav"},
        tags=["prod"],
    )
    msg = ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="trace", record=record.model_dump()),
        key="proj-1:t-hit",
        offset=42,
        partition=1,
    )

    import logging

    with caplog.at_level(logging.INFO, logger="agent_eval.benchmark.streaming.consumer"):
        await evaluator(msg)

    hit_logs = [r.message for r in caplog.records if "Sampling hit:" in r.message]
    assert len(hit_logs) == 1
    hit = hit_logs[0]
    assert "trace_id=t-hit" in hit
    assert "slots=['example']" in hit
    assert "agent_name=" not in hit
    assert "附近加油站" in hit
    assert "推荐壳牌" in hit
    assert "offset=42" in hit
