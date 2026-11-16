"""Unit tests for in-process trace_id dedup (LRU + TTL)."""

import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorDeps,
    _is_empty_io_value,
)
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from cachetools import TTLCache
from benchmark.libs.mq.consumer import ConsumedMessage


def _make_evaluator() -> BenchmarkStreamingEvaluator:
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(evaluatorName="example", sampling=1.0),
    ]
    evaluator_service = AsyncMock()

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(metadata={}, skip_trace_resolve=True)

    return BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=evaluator_service,
            request_converter=_converter,
        ),
    )


def _make_msg(trace_id: str, offset: int = 0, project_id: str = "proj-1") -> ConsumedMessage:
    record = TraceRecord(
        id=trace_id,
        project_id=project_id,
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
        key=f"{project_id}:{trace_id}",
        offset=offset,
        partition=0,
    )


@pytest.mark.asyncio
async def test_duplicate_trace_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """Same trace_id consumed twice → second is skipped."""
    evaluator = _make_evaluator()
    msg = _make_msg("trace-dup-1", offset=10)

    with caplog.at_level(logging.DEBUG, logger="agent_eval.benchmark.streaming.consumer"):
        await evaluator(msg)
        await evaluator(msg)

    eval_calls = evaluator._evaluator_service.evaluate.call_count
    assert eval_calls == 1, "evaluator should be called exactly once for a duplicate"

    skip_logs = [r.message for r in caplog.records if "Skipping duplicate trace" in r.message]
    assert len(skip_logs) == 1
    assert "trace_id=trace-dup-1" in skip_logs[0]
    assert "project_id=proj-1" in skip_logs[0]
    assert evaluator._funnel.received == 2
    assert evaluator._funnel.dedup_skip == 1
    assert evaluator._funnel.sample_hit == 1


@pytest.mark.asyncio
async def test_different_traces_both_processed() -> None:
    """Two distinct trace_ids → both are evaluated."""
    evaluator = _make_evaluator()

    await evaluator(_make_msg("trace-A", offset=1))
    await evaluator(_make_msg("trace-B", offset=2))

    assert evaluator._evaluator_service.evaluate.call_count == 2


@pytest.mark.asyncio
async def test_dedup_cache_is_instance_level() -> None:
    """Each BenchmarkStreamingEvaluator has its own dedup cache."""
    ev1 = _make_evaluator()
    ev2 = _make_evaluator()
    msg = _make_msg("trace-shared")

    await ev1(msg)
    await ev2(msg)

    assert ev1._evaluator_service.evaluate.call_count == 1
    assert ev2._evaluator_service.evaluate.call_count == 1


@pytest.mark.asyncio
async def test_ttl_expiry_allows_reprocessing() -> None:
    """After TTL expires, the same trace_id is processed again."""
    evaluator = _make_evaluator()
    # Replace with a short-TTL cache for fast testing.
    evaluator._seen_trace_ids = TTLCache(maxsize=100, ttl=1)
    msg = _make_msg("trace-ttl")

    await evaluator(msg)
    assert evaluator._evaluator_service.evaluate.call_count == 1

    # Wait for TTL to expire.
    time.sleep(1.2)

    await evaluator(msg)
    assert evaluator._evaluator_service.evaluate.call_count == 2


@pytest.mark.asyncio
async def test_same_trace_id_different_project_both_processed() -> None:
    """Same trace_id under different project_id → both are evaluated."""
    evaluator = _make_evaluator()
    trace_id = "trace-shared-id"

    await evaluator(_make_msg(trace_id, offset=1, project_id="proj-A"))
    await evaluator(_make_msg(trace_id, offset=2, project_id="proj-B"))

    assert evaluator._evaluator_service.evaluate.call_count == 2


@pytest.mark.asyncio
async def test_filter_failure_does_not_burn_dedup_slot() -> None:
    """The first message lacks agent_name, is not filtered, and claims the slot; a later complete message can still be evaluated once."""
    from unittest.mock import patch

    evaluator = _make_evaluator()
    incomplete = _make_msg("trace-late-complete", offset=1)
    # Drop agent_name so B1.6 filters out the first record.
    incomplete.payload.record["metadata"] = {}
    complete = _make_msg("trace-late-complete", offset=2)

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=[],
    ):
        await evaluator(incomplete)
        assert evaluator._evaluator_service.evaluate.call_count == 0
        assert "trace-late-complete:proj-1" not in evaluator._seen_trace_ids

        await evaluator(complete)

    assert evaluator._evaluator_service.evaluate.call_count == 1
    assert evaluator._funnel.filter_agent_name == 1
    assert evaluator._funnel.sample_hit == 1
    assert evaluator._funnel.dedup_skip == 0

    # Third delivery of the same complete key is deduped.
    await evaluator(_make_msg("trace-late-complete", offset=3))
    assert evaluator._evaluator_service.evaluate.call_count == 1
    assert evaluator._funnel.dedup_skip == 1


@pytest.mark.asyncio
async def test_name_filter_failure_allows_later_matching_update() -> None:
    """The first message's name does not match and does not claim the slot; a later one changed to the configured name can enter sampling."""
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(
            evaluatorName="example",
            sampling=1.0,
            name=["amap-agent.stream.ai_chat.standard"],
        ),
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

    early = _make_msg("trace-name-evolve", offset=1)
    early.payload.record["name"] = "search_agent.stream.overview"
    late = _make_msg("trace-name-evolve", offset=2)
    late.payload.record["name"] = "amap-agent.stream.ai_chat.standard"

    await evaluator(early)
    assert evaluator_service.evaluate.call_count == 0
    assert evaluator._funnel.filter_attrs == 1
    assert "trace-name-evolve:proj-1" not in evaluator._seen_trace_ids

    await evaluator(late)
    assert evaluator_service.evaluate.call_count == 1
    assert evaluator._funnel.sample_hit == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ("  ", True),
        ({}, False),  # Streaming semantics (S5): an empty container is not treated as empty — do not widen the interception surface
        ([], False),
        ({"q": "hi"}, False),
        ("hello", False),
        ([1], False),
        (0, False),
    ],
)
def test_is_empty_io_value_semantics(value: object, expected: bool) -> None:
    """The empty-value check locks streaming semantics: only None and blank strings are empty; an empty dict/list is legal."""
    assert _is_empty_io_value(value) is expected


@pytest.mark.asyncio
async def test_none_input_does_not_burn_dedup_slot() -> None:
    """The first message has input/output of None and does not claim the slot; a later complete message can still be evaluated once (S5)."""
    evaluator = _make_evaluator()
    incomplete = _make_msg("trace-empty-io", offset=1)
    incomplete.payload.record["input"] = None
    incomplete.payload.record["output"] = None
    incomplete.payload.record["timestamp"] = 1789892318964
    complete = _make_msg("trace-empty-io", offset=2)

    await evaluator(incomplete)
    assert evaluator._evaluator_service.evaluate.call_count == 0
    assert "trace-empty-io:proj-1" not in evaluator._seen_trace_ids
    assert evaluator._funnel.filter_empty_io == 1

    await evaluator(complete)
    assert evaluator._evaluator_service.evaluate.call_count == 1
    assert evaluator._funnel.sample_hit == 1
    assert evaluator._funnel.dedup_skip == 0


@pytest.mark.asyncio
async def test_none_output_with_input_is_evaluated_as_complete() -> None:
    """Streaming semantics (S5, AND check): input is set and only output=None is a legal complete
    message → evaluate normally and claim the slot; the difference from the old judge pipeline (OR empty-check that drops) is locked here."""
    evaluator = _make_evaluator()
    partial = _make_msg("trace-empty-output", offset=1)
    partial.payload.record["output"] = None

    await evaluator(partial)
    assert evaluator._evaluator_service.evaluate.call_count == 1
    assert "trace-empty-output:proj-1" in evaluator._seen_trace_ids
    assert evaluator._funnel.filter_empty_io == 0

    follow = _make_msg("trace-empty-output", offset=2)
    await evaluator(follow)
    assert evaluator._evaluator_service.evaluate.call_count == 1
    assert evaluator._funnel.dedup_skip == 1


@pytest.mark.asyncio
async def test_empty_container_io_is_evaluated_as_complete() -> None:
    """Streaming semantics (S5): an empty dict / blank-string IO is a legal complete message → evaluate normally and claim the slot.

    The difference from the old judge pipeline (empty-check that drops) is locked here: do not widen the interception surface,
    so a legal empty structure is not killed by mistake (for example a control message with no input slot).
    """
    evaluator = _make_evaluator()
    container_msg = _make_msg("trace-empty-container", offset=1)
    container_msg.payload.record["input"] = {}
    container_msg.payload.record["output"] = "  "

    await evaluator(container_msg)
    # An empty container is not treated as empty → treat it as a complete message, evaluate it, and claim the slot
    assert evaluator._evaluator_service.evaluate.call_count == 1
    assert "trace-empty-container:proj-1" in evaluator._seen_trace_ids
    assert evaluator._funnel.filter_empty_io == 0

    follow = _make_msg("trace-empty-container", offset=2)
    await evaluator(follow)
    assert evaluator._evaluator_service.evaluate.call_count == 1
    assert evaluator._funnel.dedup_skip == 1


@pytest.mark.asyncio
async def test_non_trace_record_type_not_cached() -> None:
    """Non-trace messages (observation/score) do not pollute the dedup cache."""
    evaluator = _make_evaluator()
    record = TraceRecord(
        id="trace-obs-test",
        project_id="proj-1",
        name="chat",
        environment="prod",
        metadata={"agent_name": "amap_agent"},
    )
    # Send an observation message with the same id — should be skipped by
    # record_type filter and NOT added to dedup cache.
    msg_obs = ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="observation", record=record.model_dump()),
        key="proj-1:trace-obs-test",
        offset=0,
        partition=0,
    )
    await evaluator(msg_obs)

    # Now send a trace message with the same id — should be processed
    # (not blocked by the observation message).
    msg_trace = _make_msg("trace-obs-test", offset=1)
    await evaluator(msg_trace)

    assert evaluator._evaluator_service.evaluate.call_count == 1
