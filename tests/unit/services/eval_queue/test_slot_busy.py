"""Unit tests for slot_busy — aligned with design doc §6.5 / D15.

The slot is busy: a jittered delay, a longer delay when the probe is rate-limited, and retry_count is not incremented.
bullmq-py: scripts.moveToDelayed + WaitingChildrenError.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.queue.slot_busy import (
    SlotBusyDelay,
    SlotProbePolicy,
    delay_for_slot_busy,
)
from bullmq import WaitingChildrenError


def _job_with_scripts() -> MagicMock:
    job = MagicMock()
    job.id = "run1:item1"
    job.scripts = MagicMock()
    job.scripts.moveToDelayed = AsyncMock()
    return job


@pytest.mark.asyncio
async def test_delay_for_slot_busy_moves_to_delayed_without_probe() -> None:
    job = _job_with_scripts()

    with (
        patch("benchmark.queue.slot_busy.random.randint", return_value=100),
        pytest.raises(WaitingChildrenError),
    ):
        await delay_for_slot_busy(
            job,
            SlotBusyDelay(run_id="run1", base_delay_ms=3000, token="tok-1"),
        )

    job.scripts.moveToDelayed.assert_awaited_once()
    args = job.scripts.moveToDelayed.await_args.args
    # (job_id, timestamp, delay_ms, token)
    assert args[0] == "run1:item1"
    assert args[2] == 3100
    assert args[3] == "tok-1"
    assert args[1] > 0


@pytest.mark.asyncio
async def test_delay_for_slot_busy_probe_limited_triples_delay() -> None:
    job = _job_with_scripts()

    with (
        patch("benchmark.queue.slot_busy.random.randint", return_value=0),
        patch(
            "benchmark.queue.slot_busy.acquire_slot",
            new_callable=AsyncMock,
            return_value=False,
        ) as acquire,
        patch(
            "benchmark.queue.slot_busy.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch("benchmark.queue.slot_busy.time.time", return_value=100.0),
        pytest.raises(WaitingChildrenError),
    ):
        await delay_for_slot_busy(
            job,
            SlotBusyDelay(
                run_id="run1",
                base_delay_ms=1000,
                token="tok-1",
                probe=SlotProbePolicy("probe:node:run1", max_probes=2),
            ),
        )

    acquire.assert_awaited_once()
    release.assert_not_awaited()  # The probe is not held
    # delay = (1000+0)*3 = 3000 → timestamp = 100000 + 3000
    args = job.scripts.moveToDelayed.await_args.args
    assert args[1] == 103_000
    assert args[2] == 3000


@pytest.mark.asyncio
async def test_delay_for_slot_busy_releases_held_probe() -> None:
    job = _job_with_scripts()

    with (
        patch("benchmark.queue.slot_busy.random.randint", return_value=0),
        patch(
            "benchmark.queue.slot_busy.acquire_slot",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "benchmark.queue.slot_busy.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch("benchmark.queue.slot_busy.time.time", return_value=100.0),
        pytest.raises(WaitingChildrenError),
    ):
        await delay_for_slot_busy(
            job,
            SlotBusyDelay(
                run_id="run1",
                base_delay_ms=1000,
                token="tok-1",
                probe=SlotProbePolicy("probe:node:run1", max_probes=2),
            ),
        )

    release.assert_awaited_once_with("probe:node:run1")
    args = job.scripts.moveToDelayed.await_args.args
    assert args[1] == 101_000
    assert args[2] == 1000


@pytest.mark.asyncio
async def test_delay_for_slot_busy_reraises_move_failure() -> None:
    job = _job_with_scripts()
    job.scripts.moveToDelayed = AsyncMock(side_effect=RuntimeError("lock lost"))

    with pytest.raises(RuntimeError, match="lock lost"):
        await delay_for_slot_busy(
            job,
            SlotBusyDelay(run_id="run1", base_delay_ms=1000, token="tok-1"),
        )
