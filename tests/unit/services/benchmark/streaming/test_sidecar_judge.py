"""Sidecar hits are judged on the benchmark evaluator, with Langfuse kept."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorConfig,
    BenchmarkStreamingEvaluatorDeps,
)


def _entry() -> EvalConfigEntry:
    return EvalConfigEntry.model_validate(
        {"evaluatorName": "amap_agent", "sampling": 1.0, "ingest": "sidecar"},
    )


def _evaluator(service: MagicMock) -> BenchmarkStreamingEvaluator:
    return BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=MagicMock(),
            evaluator_service=service,
            request_converter=lambda _observation, _config: None,
        ),
        config=BenchmarkStreamingEvaluatorConfig(),
    )


@pytest.mark.asyncio
async def test_judge_ingested_keeps_trace_and_skips_evaluate() -> None:
    service = MagicMock()
    service.evaluate = AsyncMock()
    service.judge_and_record = AsyncMock(return_value=SimpleNamespace(score=0.8))
    request = SimpleNamespace(
        agent_name="amap-agent",
        trace_id="t-1",
        project_id="p-1",
        input={"query": "附近咖啡"},
        output={"text": "去星巴克"},
        metadata={"pipeline_source": "sidecar"},
    )
    observations = [{"span_id": "root"}]
    with (
        patch("benchmark.services.evaluation.eval_report_root") as report,
        patch("benchmark.services.evaluation.update_request_context"),
    ):
        result = await _evaluator(service).judge_ingested(_entry(), request, observations)

    assert result.score == 0.8
    service.evaluate.assert_not_awaited()
    service.judge_and_record.assert_awaited_once()
    options = service.judge_and_record.await_args.kwargs["options"]
    assert options.resolve_trace is False
    assert options.sidecar_observations == observations
    assert options.enable_trace_report is True
    report.assert_called_once()
    assert report.call_args.kwargs["name"]
    assert report.call_args.kwargs["input_value"]["trace_id"] == "t-1"
