"""Kafka consume holdback: wait until message timestamp is 120s old."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from benchmark.streaming.holdback import (
    KAFKA_MIN_MESSAGE_AGE_S,
    kafka_message_holdback_s,
    wait_kafka_message_holdback,
)

_NOW_MS = 1_700_000_120_000


def test_holdback_constant_is_120s() -> None:
    assert KAFKA_MIN_MESSAGE_AGE_S == 120.0


def test_holdback_zero_when_timestamp_missing() -> None:
    assert kafka_message_holdback_s(None, now_ms=_NOW_MS) == 0.0


def test_holdback_zero_when_timestamp_negative() -> None:
    assert kafka_message_holdback_s(-1, now_ms=_NOW_MS) == 0.0


def test_holdback_zero_when_old_enough() -> None:
    ts = _NOW_MS - int(KAFKA_MIN_MESSAGE_AGE_S * 1000)
    assert kafka_message_holdback_s(ts, now_ms=_NOW_MS) == 0.0


def test_holdback_remaining_when_fresh() -> None:
    ts = _NOW_MS - 10_000
    assert kafka_message_holdback_s(ts, now_ms=_NOW_MS) == pytest.approx(110.0)


def test_holdback_clamped_when_timestamp_in_future() -> None:
    ts = _NOW_MS + 60_000
    assert kafka_message_holdback_s(ts, now_ms=_NOW_MS) == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_wait_skips_when_old_enough() -> None:
    ts = _NOW_MS - int(KAFKA_MIN_MESSAGE_AGE_S * 1000)
    with patch(
        "benchmark.streaming.holdback.asyncio.wait_for", new_callable=AsyncMock
    ) as wait_for:
        waited = await wait_kafka_message_holdback(ts, asyncio.Event(), now_ms=_NOW_MS)
    assert waited == 0.0
    wait_for.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_uses_remaining_timeout() -> None:
    ts = _NOW_MS - 10_000

    async def _timeout_wait_for(aw: object, timeout: float | None = None) -> None:
        if hasattr(aw, "close"):
            aw.close()
        raise TimeoutError

    with patch(
        "benchmark.streaming.holdback.asyncio.wait_for",
        new_callable=AsyncMock,
        side_effect=_timeout_wait_for,
    ) as wait_for:
        waited = await wait_kafka_message_holdback(ts, asyncio.Event(), now_ms=_NOW_MS)
    assert waited == pytest.approx(110.0)
    wait_for.assert_awaited_once()
    assert wait_for.await_args.kwargs["timeout"] == pytest.approx(110.0)


@pytest.mark.asyncio
async def test_wait_returns_immediately_when_stop_event_set() -> None:
    stop_event = asyncio.Event()
    stop_event.set()
    ts = _NOW_MS - 10_000
    waited = await asyncio.wait_for(
        wait_kafka_message_holdback(ts, stop_event, now_ms=_NOW_MS),
        timeout=1.0,
    )
    assert waited == pytest.approx(110.0)


@pytest.mark.asyncio
async def test_wait_sleeps_when_stop_event_is_not_asyncio_event() -> None:
    ts = _NOW_MS - 10_000
    with (
        patch(
            "benchmark.streaming.holdback.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
        patch(
            "benchmark.streaming.holdback.asyncio.wait_for", new_callable=AsyncMock
        ) as wait_for,
    ):
        waited = await wait_kafka_message_holdback(ts, object(), now_ms=_NOW_MS)
    assert waited == pytest.approx(110.0)
    sleep.assert_awaited_once()
    assert sleep.await_args.args[0] == pytest.approx(110.0)
    wait_for.assert_not_awaited()
