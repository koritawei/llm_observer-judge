"""Unit tests for per-entry score_writeback passthrough on streaming (S8 → EvalRunOptions).

Migration gap this fills: the old path (``app/judge/consumer.py``) passed the RemoteConfig entry
``evaluators[].scoreWriteback`` to ``EvaluatorService.evaluate`` through ``EvalRunOptions.score_writeback``
(judge scores written back to the original business Langfuse trace); the streaming
path dropped that passthrough when it was merged to main. This file locks the behavior after the fix:

- an entry that configures ``scoreWriteback`` → ``evaluate`` receives options carrying the policy;
- an entry that does not → options stay ``None`` (the old path's default signature, options not passed).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.domain.schemas import ScoreWritebackPolicy
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorConfig,
    BenchmarkStreamingEvaluatorDeps,
)
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.libs.mq.consumer import ConsumedMessage


def _msg() -> ConsumedMessage[RecordStreamMessage]:
    record = TraceRecord(
        id="t-writeback",
        project_id="proj-1",
        environment="prod",
        metadata={"agent_name": "amap_agent"},
        input={"q": "test"},
        output={"a": "resp"},
    )
    return ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="trace", record=record.model_dump()),
        key="proj-1:t-writeback",
        offset=1,
        partition=0,
    )


def _deps(
    evaluator_service: AsyncMock, entry: EvalConfigEntry
) -> BenchmarkStreamingEvaluatorDeps:
    config_store = MagicMock()
    config_store.get_configs.return_value = [entry]

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(metadata={}, skip_trace_resolve=True)

    return BenchmarkStreamingEvaluatorDeps(
        config_store=config_store,
        evaluator_service=evaluator_service,
        request_converter=_converter,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("attribution_execution", [None, "none"])
async def test_score_writeback_policy_forwarded_to_evaluate(
    attribution_execution: str | None,
) -> None:
    """An entry that configures scoreWriteback → evaluate receives options carrying the policy."""
    policy = ScoreWritebackPolicy(enabled=True, scoreName="agent-eval")
    evaluator_service = AsyncMock()
    entry = EvalConfigEntry(
        evaluatorName="example",
        sampling=1.0,
        scoreWriteback=policy,
        attributionExecution=attribution_execution,
    )
    evaluator = BenchmarkStreamingEvaluator(
        deps=_deps(evaluator_service, entry),
        config=BenchmarkStreamingEvaluatorConfig(),
    )

    await evaluator(_msg())

    evaluator_service.evaluate.assert_awaited_once()
    options = evaluator_service.evaluate.await_args.kwargs.get("options")
    assert options is not None
    assert options.score_writeback is policy
    # The write-back policy and per-entry skip-attribution can both be in effect.
    assert options.skip_triage is (attribution_execution == "none")


@pytest.mark.asyncio
async def test_score_writeback_absent_keeps_none_options() -> None:
    """An entry with no scoreWriteback → options=None (the old path's default signature is unchanged)."""
    evaluator_service = AsyncMock()
    entry = EvalConfigEntry(evaluatorName="example", sampling=1.0)
    evaluator = BenchmarkStreamingEvaluator(
        deps=_deps(evaluator_service, entry),
        config=BenchmarkStreamingEvaluatorConfig(),
    )

    await evaluator(_msg())

    evaluator_service.evaluate.assert_awaited_once()
    assert evaluator_service.evaluate.await_args.kwargs.get("options") is None
