"""Hold evaluate until the Kafka message timestamp is at least 120s old."""

import asyncio
import time

#: Kafka record timestamp must be this old before TraceEvaluator runs.
KAFKA_MIN_MESSAGE_AGE_S = 120.0


def kafka_message_holdback_s(
    timestamp_ms: int | None,
    *,
    now_ms: int | None = None,
) -> float:
    """Seconds to wait so ``timestamp_ms`` is at least 120s behind now.

    Missing / negative timestamps skip the wait. Future timestamps clamp to
    ``KAFKA_MIN_MESSAGE_AGE_S``.
    """
    if timestamp_ms is None or timestamp_ms < 0:
        return 0.0
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    age_s = (now_ms - timestamp_ms) / 1000.0
    wait_s = KAFKA_MIN_MESSAGE_AGE_S - age_s
    if wait_s <= 0:
        return 0.0
    return min(wait_s, KAFKA_MIN_MESSAGE_AGE_S)


async def wait_kafka_message_holdback(
    timestamp_ms: int | None,
    stop_event: object | None = None,
    *,
    now_ms: int | None = None,
) -> float:
    """Sleep the holdback; ``asyncio.Event`` ``stop_event`` interrupts.

    Returns the delay that was scheduled (0 if skipped). Non-Event stop
    values are ignored and the wait uses ``asyncio.sleep``.
    """
    delay_s = kafka_message_holdback_s(timestamp_ms, now_ms=now_ms)
    if delay_s <= 0:
        return 0.0
    event = stop_event if isinstance(stop_event, asyncio.Event) else None
    if event is None:
        await asyncio.sleep(delay_s)
        return delay_s
    try:
        await asyncio.wait_for(event.wait(), timeout=delay_s)
    except TimeoutError:
        return delay_s
    return delay_s
