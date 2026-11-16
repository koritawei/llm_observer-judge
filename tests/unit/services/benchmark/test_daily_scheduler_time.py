"""Unit tests for daily_scheduler HH:MM parsing and due-window helpers."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from benchmark.workers.daily_scheduler import (
    in_schedule_due_window,
    parse_hh_mm,
    schedule_slot_elapsed,
    seconds_until_cutoff,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("06:00", (6, 0)),
        ("02:00", (2, 0)),
        ("23:59", (23, 59)),
        ("0600", (6, 0)),
        (" 06:00 ", (6, 0)),
    ],
)
def test_parse_hh_mm_ok(value: str, expected: tuple[int, int]) -> None:
    assert parse_hh_mm(value) == expected


@pytest.mark.parametrize("value", ["", "6", "6:0:0", "ab:cd", "06-00"])
def test_parse_hh_mm_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        parse_hh_mm(value)


def test_in_schedule_due_window_accepts_1ms_early_wake() -> None:
    """asyncio.sleep waking 1ms early is still inside the due window, but the slot must not be consumed early.

    The early-wake slack applies only to the due window; slot consumption uses ``candidate <= now`` — otherwise
    when now is in [target-5s, target) that day's slot is skipped to the next matching day, and the benchmark's
    schedule for the whole day is lost (_trigger_once has already finished, so _bm_due_now never gets another chance to judge due).
    """
    tz = ZoneInfo("Asia/Shanghai")
    target = datetime(2026, 9, 21, 5, 0, 0, tzinfo=tz)
    early = datetime(2026, 9, 21, 4, 59, 59, 999000, tzinfo=tz)
    assert in_schedule_due_window(early, target) is True
    assert schedule_slot_elapsed(early, target) is False


def test_in_schedule_due_window_rejects_outside_early_slack() -> None:
    """Earlier than EARLY_SLACK (5s) is not due, and the sleep calculation has not consumed the slot yet."""
    tz = ZoneInfo("Asia/Shanghai")
    target = datetime(2026, 9, 21, 5, 0, 0, tzinfo=tz)
    too_early = datetime(2026, 9, 21, 4, 59, 54, tzinfo=tz)
    assert in_schedule_due_window(too_early, target) is False
    assert schedule_slot_elapsed(too_early, target) is False


def test_schedule_slot_elapsed_at_exact_target() -> None:
    """The exact scheduled time arrives: due inside the window, and the slot is treated as reached (so it does not immediately sleep back to the same second)."""
    tz = ZoneInfo("Asia/Shanghai")
    target = datetime(2026, 9, 21, 5, 0, 0, tzinfo=tz)
    assert in_schedule_due_window(target, target) is True
    assert schedule_slot_elapsed(target, target) is True


def test_seconds_until_cutoff_before_and_at_target() -> None:
    """Seconds left until cutoff: positive before it, and 0 at the exact time."""
    tz = ZoneInfo("Asia/Shanghai")
    cutoff_at = datetime(2026, 9, 21, 6, 0, 0, tzinfo=tz)
    earlier = datetime(2026, 9, 21, 5, 30, 0, tzinfo=tz)

    assert seconds_until_cutoff(earlier, "06:00") == 30 * 60
    assert seconds_until_cutoff(cutoff_at, "06:00") == 0
