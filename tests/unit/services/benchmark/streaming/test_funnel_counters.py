"""Unit tests for hourly Judge funnel counters."""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    _UTC_PLUS_8,
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorConfig,
    BenchmarkStreamingEvaluatorDeps,
    _hour_key_cn,
)
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.utils.exceptions import METADATA_MISSING_CODE, EvaluatorError
from benchmark.libs.mq.consumer import ConsumedMessage


def _make_evaluator(
    *,
    sampling: float = 1.0,
    environment: list[str] | None = None,
    evaluate_timeout_s: int = 0,
    evaluator_service: AsyncMock | None = None,
) -> BenchmarkStreamingEvaluator:
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(
            evaluatorName="example",
            sampling=sampling,
            environment=environment,
        ),
    ]
    svc = evaluator_service or AsyncMock()

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(metadata={}, skip_trace_resolve=True)

    return BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=svc,
            request_converter=_converter,
        ),
        config=BenchmarkStreamingEvaluatorConfig(evaluate_timeout_s=evaluate_timeout_s),
    )


def _make_msg(
    trace_id: str,
    *,
    offset: int = 0,
    project_id: str = "proj-1",
    environment: str = "prod",
    agent_name: str | None = "amap_agent",
    record_type: str = "trace",
    is_deleted: int = 0,
    invalid_record: bool = False,
) -> ConsumedMessage:
    if invalid_record:
        record: dict = {"not_a_valid_trace": True}
    else:
        metadata = {"agent_name": agent_name} if agent_name is not None else {}
        record = TraceRecord(
            id=trace_id,
            project_id=project_id,
            name="chat",
            environment=environment,
            input={"q": "test"},
            output={"a": "resp"},
            metadata=metadata,
            tags=["prod"],
            is_deleted=is_deleted,
        ).model_dump()
    return ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type=record_type, record=record),
        key=f"{project_id}:{trace_id}",
        offset=offset,
        partition=0,
    )


def test_hour_key_cn_uses_plus_eight() -> None:
    dt = datetime(2026, 8, 7, 11, 30, tzinfo=_UTC_PLUS_8)
    assert _hour_key_cn(dt) == "2026-08-07T11+08:00"


@pytest.mark.asyncio
async def test_funnel_counts_filter_and_sampling_stages() -> None:
    """The received → filter → eligible → sample_hit/miss funnel can be reconciled."""
    evaluator = _make_evaluator(sampling=1.0, environment=["prod"])

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=[],
    ):
        await evaluator(_make_msg("t-hit", offset=1))
        await evaluator(_make_msg("t-no-agent", offset=2, agent_name=None))
        await evaluator(_make_msg("t-env", offset=3, environment="dev"))
        await evaluator(_make_msg("t-hit", offset=4))

    empty_ev = _make_evaluator()
    empty_msg = _make_msg("t-empty-io", offset=6)
    empty_msg.payload.record["input"] = None
    empty_msg.payload.record["output"] = None
    await empty_ev(empty_msg)

    miss_ev = _make_evaluator(sampling=0.0)
    await miss_ev(_make_msg("t-miss", offset=5))

    f = evaluator._funnel
    assert f.msg_total == 4
    assert f.skip_record_type == 0
    assert f.skip_invalid == 0
    assert f.skip_deleted == 0
    assert f.received == 4
    assert f.dedup_skip == 1
    assert f.filter_agent_name == 1
    assert f.filter_env == 1
    assert empty_ev._funnel.filter_empty_io == 1
    assert empty_ev._funnel.eligible == 0
    assert empty_ev._funnel.sample_hit == 0
    assert f.eligible == 1
    assert f.sample_hit == 1
    assert f.sample_miss == 0
    assert f.eval_ok == 1

    assert miss_ev._funnel.eligible == 1
    assert miss_ev._funnel.sample_miss == 1
    assert miss_ev._funnel.sample_hit == 0
    assert miss_ev._funnel.eval_ok == 0


@pytest.mark.asyncio
async def test_funnel_counts_ingress_skips() -> None:
    """Three cuts at the entrance: record_type / invalid / deleted are all counted in the funnel."""
    evaluator = _make_evaluator()

    await evaluator(_make_msg("t-obs", offset=1, record_type="observation"))
    await evaluator(_make_msg("t-bad", offset=2, invalid_record=True))
    await evaluator(_make_msg("t-del", offset=3, is_deleted=1))
    await evaluator(_make_msg("t-ok", offset=4))

    f = evaluator._funnel
    assert f.msg_total == 4
    assert f.skip_record_type == 1
    assert f.skip_invalid == 1
    assert f.skip_deleted == 1
    assert f.received == 1
    assert f.sample_hit == 1
    assert f.eval_ok == 1
    assert f.skip_record_type + f.skip_invalid + f.skip_deleted + f.received == f.msg_total


@pytest.mark.asyncio
async def test_funnel_counts_eval_outcomes() -> None:
    """B5: ok / timeout / permanent / transient are counted separately."""
    # permanent
    perm_svc = AsyncMock()
    perm_svc.evaluate.side_effect = EvaluatorError(
        code=400,
        error_type=METADATA_MISSING_CODE,
        message="missing agent_name",
    )
    perm_ev = _make_evaluator(evaluator_service=perm_svc)
    await perm_ev(_make_msg("t-perm", offset=1))
    assert perm_ev._funnel.sample_hit == 1
    assert perm_ev._funnel.eval_fail_permanent == 1
    assert perm_ev._funnel.eval_ok == 0

    # transient (EvaluatorError 5xx) — skip, no raise (fan-out isolation)
    trans_svc = AsyncMock()
    trans_svc.evaluate.side_effect = EvaluatorError(
        code=502,
        error_type="llm_failed",
        message="llm down",
    )
    trans_ev = _make_evaluator(evaluator_service=trans_svc)
    await trans_ev(_make_msg("t-trans", offset=2))
    assert trans_ev._funnel.eval_fail_transient == 1
    assert trans_ev._funnel.eval_ok == 0

    # timeout — inject TimeoutError (same path as asyncio.wait_for expiry)
    timeout_svc = AsyncMock()
    timeout_svc.evaluate.side_effect = TimeoutError("LLM timed out")
    timeout_ev = _make_evaluator(
        evaluator_service=timeout_svc,
        evaluate_timeout_s=5,
    )
    await timeout_ev(_make_msg("t-timeout", offset=3))
    assert timeout_ev._funnel.eval_fail_timeout == 1
    assert timeout_ev._funnel.eval_ok == 0


@pytest.mark.asyncio
async def test_funnel_flushes_on_hour_rollover(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the clock crosses the top of an hour in UTC+8, emit the previous hour's Judge funnel line."""
    with patch(
        "benchmark.streaming.consumer._hour_key_cn",
        return_value="2026-08-07T11+08:00",
    ):
        evaluator = _make_evaluator()
        await evaluator(_make_msg("t-1", offset=1))
    assert evaluator._funnel.received == 1

    with (
        caplog.at_level(logging.WARNING, logger="agent_eval.benchmark.streaming.funnel"),
        patch(
            "benchmark.streaming.consumer._hour_key_cn",
            return_value="2026-08-07T12+08:00",
        ),
    ):
        await evaluator(_make_msg("t-2", offset=2))

    funnel_logs = [r.message for r in caplog.records if "Judge funnel:" in r.message]
    assert len(funnel_logs) == 1
    assert "hour=2026-08-07T11+08:00" in funnel_logs[0]
    assert "msg_total=" in funnel_logs[0]
    assert "eval_ok=" in funnel_logs[0]
    assert evaluator._funnel_hour == "2026-08-07T12+08:00"
    assert evaluator._funnel.received == 1


@pytest.mark.asyncio
async def test_flush_funnel_emits_current_bucket(
    caplog: pytest.LogCaptureFixture,
) -> None:
    evaluator = _make_evaluator()
    await evaluator(_make_msg("t-flush", offset=1))
    with caplog.at_level(logging.WARNING, logger="agent_eval.benchmark.streaming.funnel"):
        evaluator.flush_funnel()
    logs = [r.message for r in caplog.records if "Judge funnel:" in r.message]
    assert len(logs) == 1
    assert "sample_hit=1" in logs[0]
    assert "eval_ok=1" in logs[0]
    assert "msg_total=1" in logs[0]
    assert evaluator._funnel.received == 0
