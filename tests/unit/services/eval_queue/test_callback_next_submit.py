"""Unit tests for callback_next_submit — a latch race + promote the next one."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.queue.callback_next_submit import (
    JudgeSubmitGate,
    SlotReleaseRequest,
    promote_delayed_for_run,
    release_slot_and_promote_next,
)


@pytest.mark.asyncio
async def test_gate_waits_until_complete() -> None:
    gate = JudgeSubmitGate()
    await gate.register("run1", "item1")

    done = False

    async def _waiter() -> None:
        nonlocal done
        await gate.wait("run1", "item1")
        done = True

    task = asyncio.create_task(_waiter())
    await asyncio.sleep(0)
    assert done is False
    await gate.complete("run1", "item1")
    await task
    assert done is True


@pytest.mark.asyncio
async def test_gate_complete_before_register_does_not_deadlock() -> None:
    """The callback arrives before register: an early credit → wait returns immediately after register."""
    gate = JudgeSubmitGate()
    await gate.complete("run1", "item1")
    await gate.register("run1", "item1")
    await gate.wait("run1", "item1")  # must not hang


@pytest.mark.asyncio
async def test_promote_delayed_for_run_only_matching() -> None:
    job_ok = MagicMock()
    job_ok.data = {"run_id": "run1"}
    job_ok.id = "run1:a"
    job_ok.promote = AsyncMock()
    job_other = MagicMock()
    job_other.data = {"run_id": "run2"}
    job_other.id = "run2:b"
    job_other.promote = AsyncMock()

    wait_q = MagicMock()
    wait_q.getJobs = AsyncMock(return_value=[job_ok, job_other])

    async def _list(q: MagicMock, types: list[str], **_: object) -> list[object]:
        return await q.getJobs(types, 0, -1)

    with patch(
        "benchmark.queue.callback_next_submit.list_queue_jobs",
        side_effect=_list,
    ):
        n = await promote_delayed_for_run(wait_q, "run1", limit=1)
    assert n == 1
    job_ok.promote.assert_awaited_once()
    job_other.promote.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_slot_and_promote_next_full_path() -> None:
    running_job = MagicMock()
    running_job.data = {
        "run_id": "run1",
        "item_id": "item1",
        "slot_key": "slot:run1",
        "cluster_kind": "judge",
    }
    running_job.remove = AsyncMock()

    running_q = MagicMock()
    running_q.getJob = AsyncMock(return_value=running_job)
    wait_q = MagicMock()
    wait_q.getJobs = AsyncMock(return_value=[])

    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    gate = JudgeSubmitGate()
    await gate.register("run1", "item1")

    with (
        patch(
            "benchmark.queue.callback_next_submit.judge_submit_gate",
            gate,
        ),
        patch(
            "benchmark.queue.callback_next_submit.promote_delayed_for_run",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch(
            "benchmark.queue.callback_next_submit.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.queue.callback_next_submit.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
    ):
        result = await release_slot_and_promote_next(
            SlotReleaseRequest(
                queue_manager=qm,
                notification_job_id="aone-1",
                run_id="run1",
                item_id="item1",
            ),
        )

    assert result["gate_completed"] is True
    assert result["running_removed"] is True
    assert result["slot_released"] is True
    release.assert_awaited_once_with("slot:run1")
    release_cluster.assert_awaited_once_with("judge")
    running_job.remove.assert_awaited_once()


@pytest.mark.asyncio
async def test_release_slot_when_running_already_removed() -> None:
    """When requeue/force_fail has already deleted running, the limit must still be released by the explicit slot_key."""
    running_q = MagicMock()
    running_q.getJob = AsyncMock(return_value=None)
    wait_q = MagicMock()
    wait_q.getJobs = AsyncMock(return_value=[])

    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )

    with (
        patch(
            "benchmark.queue.callback_next_submit.get_queue_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "benchmark.queue.callback_next_submit.promote_delayed_for_run",
            new_callable=AsyncMock,
            return_value=1,
        ),
        patch(
            "benchmark.queue.callback_next_submit.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.queue.callback_next_submit.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
    ):
        result = await release_slot_and_promote_next(
            SlotReleaseRequest(
                queue_manager=qm,
                notification_job_id="aone-gone",
                run_id="run1",
                item_id="item1",
                slot_key="slot:run1",
                cluster_kind="judge",
            ),
        )

    assert result["running_removed"] is False
    assert result["slot_released"] is True
    assert result["promoted"] == 1
    release.assert_awaited_once_with("slot:run1")
    release_cluster.assert_awaited_once_with("judge")


@pytest.mark.asyncio
async def test_enqueue_external_cases_does_not_stamp_wait_clock() -> None:
    from benchmark.queue.callback_next_submit import enqueue_external_cases

    wait_q = MagicMock()
    wait_q.add = AsyncMock()
    qm = MagicMock()
    qm.get_queue = MagicMock(return_value=wait_q)

    items = [
        {
            "item_id": "i1",
            "input": {"query": "q"},
            "output": {"response_text": "a"},
            "metadata": {"csv_row": {"task_id": "i1", "query": "q"}},
        },
        {"id": "i2", "metadata": {}},
        {"no_id": True},  # skipped
    ]
    n = await enqueue_external_cases(
        queue_manager=qm,
        run_id="runX",
        items=items,
        ds="20260101",
    )
    assert n == 2
    assert wait_q.add.await_count == 2

    first_call = wait_q.add.await_args_list[0]
    assert first_call.args[0] == "case"
    payload = first_call.args[1]
    assert payload["run_id"] == "runX"
    assert payload["item_id"] == "i1"
    assert payload["retry_count"] == 0
    assert "first_enqueued_at" not in payload
    assert payload["metadata"]["csv_row"]["query"] == "q"
    assert first_call.args[2]["jobId"] == "runX:i1"

    second = wait_q.add.await_args_list[1].args[1]
    assert second["item_id"] == "i2"
    assert "first_enqueued_at" not in second
