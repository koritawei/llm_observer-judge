"""Sidecar hourly funnel: filter reasons and read failures are WARNING totals."""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.ingest.sidecar.funnel import SidecarFunnel, _hour_key_cn
from benchmark.ingest.sidecar.processor import PermanentSkip, SidecarProcessor


def _entry(**kwargs: object) -> EvalConfigEntry:
    payload = {
        "evaluatorName": "amap_agent",
        "sampling": 1.0,
        "ingest": "sidecar",
        **kwargs,
    }
    return EvalConfigEntry.model_validate(payload)


def _notify(**kwargs: object) -> dict[str, object]:
    body: dict[str, object] = {
        "trace_id": "t-1",
        "finish_reason": "finish",
        "project_id": "p-1",
        "agent_name": "amap-agent",
        "environment": "prod",
        "name": "amap",
    }
    body.update(kwargs)
    return body


def _processor(funnel: SidecarFunnel) -> tuple[SidecarProcessor, MagicMock]:
    reader = MagicMock()
    reader.fetch_by_trace_id = AsyncMock()
    processor = SidecarProcessor(
        reader=reader,
        benchmark_judge=MagicMock(),
        funnel=funnel,
    )
    return processor, reader


@pytest.mark.asyncio
async def test_env_mismatch_counts_filter_env_without_read() -> None:
    funnel = SidecarFunnel()
    processor, reader = _processor(funnel)
    with pytest.raises(PermanentSkip):
        await processor.handle(
            _notify(environment="dev"),
            config=_entry(environment=["prod"]),
            project_id="p-1",
        )
    reader.fetch_by_trace_id.assert_not_awaited()
    assert funnel._counters.filter_env == 1
    assert funnel._counters.received == 1
    assert funnel._counters.sample_hit == 0
    assert funnel._counters.read_fail == 0


@pytest.mark.asyncio
async def test_name_mismatch_counts_filter_name() -> None:
    funnel = SidecarFunnel()
    processor, reader = _processor(funnel)
    with pytest.raises(PermanentSkip):
        await processor.handle(
            _notify(name="other"),
            config=_entry(name=["amap-agent.stream.ai_chat.standard"]),
            project_id="p-1",
        )
    reader.fetch_by_trace_id.assert_not_awaited()
    assert funnel._counters.filter_name == 1
    assert funnel._counters.eligible == 0


@pytest.mark.asyncio
async def test_sample_miss_is_eligible_and_does_not_read() -> None:
    funnel = SidecarFunnel()
    processor, reader = _processor(funnel)
    with pytest.raises(PermanentSkip):
        await processor.handle(
            _notify(),
            config=_entry(sampling=0.0),
            project_id="p-1",
        )
    reader.fetch_by_trace_id.assert_not_awaited()
    assert funnel._counters.sample_miss == 1
    assert funnel._counters.eligible == 1
    assert funnel._counters.sample_hit == 0


@pytest.mark.asyncio
async def test_lindorm_miss_counts_read_fail() -> None:
    funnel = SidecarFunnel()
    processor, reader = _processor(funnel)
    reader.fetch_by_trace_id = AsyncMock(return_value=None)
    with pytest.raises(LookupError):
        await processor.handle(_notify(), config=_entry(), project_id="p-1")
    assert funnel._counters.sample_hit == 1
    assert funnel._counters.read_fail == 1
    assert funnel._counters.eval_ok == 0


@pytest.mark.asyncio
async def test_hour_rollover_emits_sidecar_funnel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with patch(
        "benchmark.ingest.sidecar.funnel._hour_key_cn",
        return_value="2026-10-13T16+08:00",
    ):
        funnel = SidecarFunnel()
        processor, _reader = _processor(funnel)
        with pytest.raises(PermanentSkip):
            await processor.handle(
                _notify(environment="dev"),
                config=_entry(environment=["prod"]),
                project_id="p-1",
            )
    with (
        caplog.at_level(logging.WARNING, logger="agent_eval.sidecar_ingest.funnel"),
        patch(
            "benchmark.ingest.sidecar.funnel._hour_key_cn",
            return_value="2026-10-13T17+08:00",
        ),
    ):
        with pytest.raises(PermanentSkip):
            await processor.handle(b"not-json", config=_entry(), project_id="p-1")
    lines = [r.message for r in caplog.records if "Sidecar funnel:" in r.message]
    assert len(lines) == 1
    assert "hour=2026-10-13T16+08:00" in lines[0]
    assert "filter_env=1" in lines[0]
    assert "read_fail=0" in lines[0]
    assert funnel._counters.skip_invalid == 1
    assert funnel._counters.filter_env == 0


def test_hour_key_cn_uses_plus_eight() -> None:
    dt = datetime(2026, 10, 13, 16, 30, tzinfo=timezone(timedelta(hours=8)))
    assert _hour_key_cn(dt) == "2026-10-13T16+08:00"


@pytest.mark.asyncio
async def test_filter_sample_logs_once_per_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    funnel = SidecarFunnel()
    processor, _reader = _processor(funnel)
    with caplog.at_level(logging.WARNING, logger="agent_eval.sidecar_ingest.processor"):
        for _ in range(2):
            with pytest.raises(PermanentSkip):
                await processor.handle(
                    _notify(name="other"),
                    config=_entry(name=["amap-agent.stream.ai_chat.standard"]),
                    project_id="p-1",
                )
    lines = [r.message for r in caplog.records if "sidecar.filter_sample" in r.message]
    assert len(lines) == 1
    assert "reason=filter_name" in lines[0]
    assert "notify_name=other" in lines[0]
    assert "amap-agent.stream.ai_chat.standard" in lines[0]


def test_progress_line_emits_without_reset(caplog: pytest.LogCaptureFixture) -> None:
    funnel = SidecarFunnel()
    funnel._last_progress = 0.0
    with (
        caplog.at_level(logging.WARNING, logger="agent_eval.sidecar_ingest.funnel"),
        patch("benchmark.ingest.sidecar.funnel._PROGRESS_INTERVAL_S", 0.0),
    ):
        funnel.bump("received")
    lines = [r.message for r in caplog.records if "Sidecar funnel progress:" in r.message]
    assert len(lines) == 1
    assert "received=1" in lines[0]
    assert funnel._counters.received == 1
