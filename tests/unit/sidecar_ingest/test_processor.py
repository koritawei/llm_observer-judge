"""sidecar processor: filter miss skips readback; evaluate uses local observations."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import zstandard as zstd
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.ingest.sidecar.notify import parse_notify
from benchmark.ingest.sidecar.observations import _SIDECAR_OBSERVATIONS_KEY
from benchmark.ingest.sidecar.processor import PermanentSkip, SidecarProcessor
from benchmark.ingest.sidecar.trace_reader import SidecarTraceRecord


def _entry(**kwargs: object) -> EvalConfigEntry:
    payload = {
        "evaluatorName": "amap_agent",
        "sampling": 1.0,
        "ingest": "sidecar",
        **kwargs,
    }
    return EvalConfigEntry.model_validate(payload)


def _packed(obs: list[dict[str, Any]]) -> bytes:
    import json

    return zstd.ZstdCompressor().compress(json.dumps(obs).encode("utf-8"))


@pytest.mark.asyncio
async def test_filter_miss_zero_readbacks() -> None:
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=MagicMock(),
    )
    with pytest.raises(PermanentSkip):
        await processor.handle(
            {
                "trace_id": "t-1",
                "finish_reason": "finish",
                "project_id": "missing",
                "agent_name": "amap-agent",
            },
            config=_entry(),
            project_id="p-1",
        )
    reader.fetch_by_trace_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_kafka_ingest_skips_without_readback() -> None:
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=MagicMock(),
    )
    with pytest.raises(PermanentSkip):
        await processor.handle(
            {
                "trace_id": "t-kafka",
                "finish_reason": "finish",
                "project_id": "p-1",
                "agent_name": "amap-agent",
            },
            config=_entry(ingest="kafka"),
            project_id="p-1",
        )
    reader.fetch_by_trace_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_test_adiu_user_id_skips_without_readback() -> None:
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=MagicMock(),
    )
    with pytest.raises(PermanentSkip):
        await processor.handle(
            {
                "trace_id": "t-test-adiu",
                "finish_reason": "finish",
                "project_id": "p-1",
                "agent_name": "amap-agent",
                "user_id": "xladadadldadadfadldada_test",
            },
            config=_entry(),
            project_id="p-1",
        )
    reader.fetch_by_trace_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_nested_input_test_adiu_skips_after_readback() -> None:
    observations = [
        {
            "span_id": "root",
            "project_id": "p-1",
            "agent_name": "amap-agent",
            "name": "amap",
            "type": "SPAN",
            "input": {"request_body": {"user": {"adiu": "device_test"}}},
            "output": {"text": "去星巴克"},
        }
    ]
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock(
        return_value=SidecarTraceRecord(
            finish_reason="finish",
            observations=_packed(observations),
        )
    )
    evaluator = MagicMock()
    evaluator.judge_ingested = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=evaluator,
    )
    with pytest.raises(PermanentSkip):
        await processor.handle(
            {
                "trace_id": "t-nested-test",
                "finish_reason": "finish",
                "project_id": "p-1",
                "agent_name": "amap-agent",
                "name": "amap",
            },
            config=_entry(),
            project_id="p-1",
        )
    reader.fetch_by_trace_id.assert_awaited_once()
    evaluator.judge_ingested.assert_not_awaited()


@pytest.mark.asyncio
async def test_hit_evaluates_and_second_delivery_is_idempotent() -> None:
    observations = [
        {
            "span_id": "root",
            "project_id": "p-1",
            "agent_name": "amap-agent",
            "name": "amap",
            "type": "SPAN",
            "input": {"query": "附近咖啡"},
            "output": {"text": "去星巴克"},
        }
    ]
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock(
        return_value=SidecarTraceRecord(finish_reason="finish", observations=_packed(observations))
    )
    evaluator = MagicMock()
    evaluator.judge_ingested = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=evaluator,
    )
    body = {
        "trace_id": "t-dup",
        "finish_reason": "finish",
        "project_id": "p-1",
        "agent_name": "amap-agent",
        "name": "amap",
    }
    await processor.handle(body, config=_entry(), project_id="p-1")
    await processor.handle(body, config=_entry(), project_id="p-1")
    assert evaluator.judge_ingested.await_count == 2
    first = evaluator.judge_ingested.await_args_list[0]
    request = first.args[1]
    assert request.metadata["pipeline_source"] == "sidecar"
    assert _SIDECAR_OBSERVATIONS_KEY not in request.metadata
    assert first.args[2] == observations
    assert request.input == {"query": "附近咖啡"}
    assert request.output == {"text": "去星巴克"}
    assert request.skip_trace_resolve is True


@pytest.mark.asyncio
async def test_invalid_notify_is_permanent_skip() -> None:
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=MagicMock(),
    )
    with pytest.raises(PermanentSkip):
        await processor.handle(b"not-json", config=_entry(), project_id="p-1")
    reader.fetch_by_trace_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_utf8_notify_is_permanent_skip() -> None:
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=MagicMock(),
    )
    with pytest.raises(PermanentSkip):
        await processor.handle(b"\xff\xfe", config=_entry(), project_id="p-1")
    reader.fetch_by_trace_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrupt_observations_are_permanent_skip() -> None:
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock(
        return_value=SidecarTraceRecord(finish_reason="finish", observations=b"not-zstd")
    )
    evaluator = MagicMock()
    evaluator.judge_ingested = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=evaluator,
    )
    with pytest.raises(PermanentSkip):
        await processor.handle(
            {
                "trace_id": "t-bad",
                "finish_reason": "finish",
                "project_id": "p-1",
                "agent_name": "amap-agent",
            },
            config=_entry(),
            project_id="p-1",
        )
    evaluator.judge_ingested.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_observations_are_permanent_skip() -> None:
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock(
        return_value=SidecarTraceRecord(finish_reason="finish", observations=_packed([]))
    )
    evaluator = MagicMock()
    evaluator.judge_ingested = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=evaluator,
    )
    with pytest.raises(PermanentSkip):
        await processor.handle(
            {
                "trace_id": "t-empty",
                "finish_reason": "finish",
                "project_id": "p-1",
                "agent_name": "amap-agent",
            },
            config=_entry(),
            project_id="p-1",
        )
    evaluator.judge_ingested.assert_not_awaited()


def test_parse_does_not_use_timestamp_as_partition_key() -> None:
    notify = parse_notify(
        {
            "trace_id": "t-ts",
            "finish_reason": "finish",
            "timestamp": 1,
        }
    )
    assert notify.trace_id == "t-ts"
    assert notify.timestamp == 1
