"""Unit tests for the slot-gap sweep — clamp + the healer aligned with external/benchmark-running."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.queue.queue_factory import (
    BENCHMARK_RUNNING,
    EXTERNAL_RUNNING,
)
from benchmark.workers.running_healer import (
    RunningHealer,
    _running_job_slot_kind,
)


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "sync_slot_heal_interval_s": 60,
        "external_slot_heal_interval_s": 60,
        "eval_queue_lock_duration_ms": 1_200_000,
        "external_wait_lock_duration_ms": 300_000,
        "external_running_ttl_ms": 86_400_000,
        "eval_queue_max_attempts": 3,
        "sync_retry_max_duration_ms": 7_200_000,
        "external_retry_max_duration_ms": 7_200_000,
        "eval_queue_backoff_initial_ms": 5000,
        "eval_queue_backoff_max_ms": 25000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _job(*, name: str, data: dict[str, Any]) -> MagicMock:
    job = MagicMock()
    job.id = "j1"
    job.name = name
    job.data = data
    job.timestamp = 1
    return job


def _bind_named_queues(qm: MagicMock) -> None:
    queues: dict[str, MagicMock] = {}

    def get_queue(name: str) -> MagicMock:
        queue = queues.get(name)
        if queue is None:
            queue = MagicMock()
            queue.queue_name = name
            queues[name] = queue
        return queue

    qm.get_queue = MagicMock(side_effect=get_queue)


def _list_jobs_by_queue(
    jobs_by_queue: dict[str, list[MagicMock]],
) -> AsyncMock:
    async def list_jobs(queue: MagicMock, _types: object) -> list[MagicMock]:
        return jobs_by_queue.get(getattr(queue, "queue_name", ""), [])

    return AsyncMock(side_effect=list_jobs)


@pytest.mark.asyncio
async def test_heal_slot_counters_clamps_leak_when_running_empty() -> None:
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    _bind_named_queues(qm)

    healer = RunningHealer(queue_manager=qm, settings=_settings())

    with (
        patch(
            "benchmark.workers.running_healer.list_queue_jobs",
            new=_list_jobs_by_queue({}),
        ),
        patch(
            "benchmark.workers.running_healer.build_cluster_slot_key",
            return_value="pre:bullmq:inflight:cluster:judge",
        ),
        patch(
            "benchmark.workers.running_healer.scan_slot_keys",
            new_callable=AsyncMock,
            return_value=["pre:bullmq:inflight:judge:node1:run1"],
        ),
        patch(
            "benchmark.workers.running_healer.clamp_slot_count",
            new_callable=AsyncMock,
            side_effect=[4, 2, 0, 0],
        ) as clamp,
    ):
        n = await healer._heal_slot_counters()

    # judge kind: 4 + 2 = 6; sync kind: 0 + 0 = 0; total = 6
    assert n == 6
    assert clamp.await_count == 4
    clamp.assert_any_await(
        "pre:bullmq:inflight:cluster:judge",
        expected=0,
        ttl_s=86400,
    )
    clamp.assert_any_await(
        "pre:bullmq:inflight:judge:node1:run1",
        expected=0,
        ttl_s=86400,
    )


@pytest.mark.asyncio
async def test_heal_slot_counters_keeps_expected_from_running() -> None:
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    _bind_named_queues(qm)
    healer = RunningHealer(queue_manager=qm, settings=_settings())
    job = _job(
        name="pending_callback",
        data={
            "run_id": "run1",
            "slot_key": "pre:bullmq:inflight:judge:node1:run1",
            "cluster_kind": "judge",
        },
    )

    with (
        patch(
            "benchmark.workers.running_healer.list_queue_jobs",
            new=_list_jobs_by_queue({EXTERNAL_RUNNING: [job]}),
        ),
        patch(
            "benchmark.workers.running_healer.build_cluster_slot_key",
            return_value="pre:bullmq:inflight:cluster:judge",
        ),
        patch(
            "benchmark.workers.running_healer.scan_slot_keys",
            new_callable=AsyncMock,
            return_value=["pre:bullmq:inflight:judge:node1:run1"],
        ),
        patch(
            "benchmark.workers.running_healer.clamp_slot_count",
            new_callable=AsyncMock,
            return_value=0,
        ) as clamp,
    ):
        n = await healer._heal_slot_counters()

    assert n == 0
    clamp.assert_any_await(
        "pre:bullmq:inflight:cluster:judge",
        expected=1,
        ttl_s=86400,
    )
    clamp.assert_any_await(
        "pre:bullmq:inflight:judge:node1:run1",
        expected=1,
        ttl_s=86400,
    )


@pytest.mark.asyncio
async def test_heal_slot_counters_counts_benchmark_running_judge() -> None:
    """A unified-queue pending_callback counts toward the judge slot, not sync."""
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    _bind_named_queues(qm)
    healer = RunningHealer(queue_manager=qm, settings=_settings())
    job = _job(
        name="pending_callback",
        data={
            "run_id": "ieval_external-1",
            "slot_key": "pre:bullmq:inflight:judge:node1:ieval_external-1",
            "cluster_kind": "judge",
        },
    )

    cluster_keys = {
        "judge": "pre:bullmq:inflight:cluster:judge",
        "sync": "pre:bullmq:inflight:cluster:sync",
    }

    with (
        patch(
            "benchmark.workers.running_healer.list_queue_jobs",
            new=_list_jobs_by_queue({BENCHMARK_RUNNING: [job]}),
        ),
        patch(
            "benchmark.workers.running_healer.build_cluster_slot_key",
            side_effect=lambda kind: cluster_keys[kind],
        ),
        patch(
            "benchmark.workers.running_healer.scan_slot_keys",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "benchmark.workers.running_healer.clamp_slot_count",
            new_callable=AsyncMock,
            return_value=0,
        ) as clamp,
    ):
        n = await healer._heal_slot_counters()

    assert n == 0
    clamp.assert_any_await(
        "pre:bullmq:inflight:cluster:judge",
        expected=1,
        ttl_s=86400,
    )
    clamp.assert_any_await(
        "pre:bullmq:inflight:judge:node1:ieval_external-1",
        expected=1,
        ttl_s=86400,
    )
    clamp.assert_any_await(
        "pre:bullmq:inflight:cluster:sync",
        expected=0,
        ttl_s=1200,
    )


def test_running_job_slot_kind_keeps_unified_queue_judge_and_sync_apart() -> None:
    judge_job = _job(
        name="pending_callback",
        data={"cluster_kind": "judge"},
    )
    assert (
        _running_job_slot_kind(
            judge_job,
            queue_name=BENCHMARK_RUNNING,
            default_kind="sync",
        )
        == "judge"
    )

    sync_job = _job(name="inflight", data={})
    assert (
        _running_job_slot_kind(
            sync_job,
            queue_name=BENCHMARK_RUNNING,
            default_kind="judge",
        )
        == "sync"
    )
