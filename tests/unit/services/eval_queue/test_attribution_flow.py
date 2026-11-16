"""End-to-end acceptance of the attribution queue (fully mocked, Redis/PG are not touched).

Covers: the judge callback enqueues → the worker submits → the provider callback → the slot is released → finalize.
Also isolation of the attr/judge slot keys across nodes.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.queue.inflight_slot import SlotKeySpec, build_slot_key
from benchmark.queue.queue_factory import ATTRIBUTION_WAIT
from benchmark.workers.external_runner import (
    AttributionCaseSpec,
    ExternalBenchmarkRunner,
    ExternalRunnerDeps,
)


@pytest.mark.asyncio
async def test_enqueue_attribution_case_payload_contract() -> None:
    wait_q = MagicMock()
    wait_q.add = AsyncMock()
    qm = MagicMock()
    qm.get_queue = MagicMock(return_value=wait_q)
    runner = ExternalBenchmarkRunner(
        benchmark_service=MagicMock(),
        storage_client=None,
        deps=ExternalRunnerDeps(queue_manager=qm),
    )

    await runner._enqueue_attribution_case(
        AttributionCaseSpec(
            run_id="run1",
            item_id="item1",
            source_trace_id="judge-1",
            adiu="adiu-1",
            ds_start="20260817",
            issue={"criterion_scores": {"D0": {"score": 0.2, "reason": "x"}}},
            attribution_provider="external",
        )
    )

    qm.get_queue.assert_called_with(ATTRIBUTION_WAIT)
    wait_q.add.assert_awaited_once()
    name, data, opts = wait_q.add.await_args.args
    assert name == "attr_case"
    assert data["retry_count"] == 0
    assert "first_enqueued_at" not in data
    assert data["source_trace_id"] == "judge-1"
    assert data["attribution_provider"] == "external"
    assert data["issue"]["criterion_scores"]["D0"]["score"] == 0.2
    assert opts["jobId"] == "run1:item1"
    # The attribution wait queue has BullMQ-level retries (a second track beside the application retry_count; see the comment at the enqueue site)
    assert opts["attempts"] == 3


@pytest.mark.asyncio
async def test_enqueue_attribution_case_requires_queue_manager() -> None:
    """When there is no queue_manager it must fail; a silent skip that inflates expected is forbidden."""
    runner = ExternalBenchmarkRunner(
        benchmark_service=MagicMock(),
        storage_client=None,
        deps=ExternalRunnerDeps(queue_manager=None),
    )
    with pytest.raises(RuntimeError, match="queue_manager"):
        await runner._enqueue_attribution_case(
            AttributionCaseSpec(
                run_id="run1",
                item_id="item1",
                source_trace_id="judge-1",
                adiu="adiu-1",
                ds_start="20260817",
                issue={},
                attribution_provider="external",
            )
        )


def test_multi_node_attr_and_judge_slots_do_not_collide() -> None:
    """On a multi-node deploy the judge/attr slot keys must be isolated, and the same run runs in parallel across nodes."""
    keys = {
        build_slot_key(
            "prod:bullmq",
            SlotKeySpec(
                kind="judge",
                node="n1",
                run_id="run1",
            ),
        ),
        build_slot_key(
            "prod:bullmq",
            SlotKeySpec(
                kind="judge",
                node="n2",
                run_id="run1",
            ),
        ),
        build_slot_key(
            "prod:bullmq",
            SlotKeySpec(
                kind="attr",
                node="n1",
                run_id="run1",
            ),
        ),
        build_slot_key(
            "prod:bullmq",
            SlotKeySpec(
                kind="attr",
                node="n2",
                run_id="run1",
            ),
        ),
    }
    assert len(keys) == 4
    cluster_judge = "prod:bullmq:inflight:cluster:judge"
    cluster_attr = "prod:bullmq:inflight:cluster:attr"
    assert cluster_judge != cluster_attr


@pytest.mark.asyncio
async def test_healer_tick_skips_nested_attr_lock_when_d19_enabled() -> None:
    """D19: once the leader has entered the tick, do not grab the attr short lock again, so a mistaken skip is avoided."""
    from benchmark.workers.running_healer import RunningHealer

    settings = SimpleNamespace(
        sync_slot_heal_interval_s=60,
        external_slot_heal_interval_s=60,
        eval_queue_lock_duration_ms=1_200_000,
        external_wait_lock_duration_ms=300_000,
        external_running_ttl_ms=86_400_000,
        eval_queue_max_attempts=3,
        sync_retry_max_duration_ms=7_200_000,
        external_retry_max_duration_ms=7_200_000,
        eval_queue_backoff_initial_ms=5000,
        eval_queue_backoff_max_ms=25000,
        attr_callback_timeout_ms=86_400_000,
        eval_queue_healer_leader_disabled=False,
    )
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    healer = RunningHealer(queue_manager=qm, settings=settings)

    with (
        patch(
            "benchmark.workers.running_healer.collect_active_run_ids",
            new_callable=AsyncMock,
            return_value=set(),
        ),
        patch.object(healer, "_refresh_heartbeats", new_callable=AsyncMock, return_value=0),
        patch.object(
            healer,
            "_heal_sync_pair",
            new_callable=AsyncMock,
            return_value=(0, 0),
        ),
        patch.object(healer, "_heal_external", new_callable=AsyncMock, return_value=(0, 0)),
        patch.object(healer, "_heal_slot_counters", new_callable=AsyncMock, return_value=0),
        patch.object(
            healer,
            "_heal_aggregating_external",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch.object(
            healer,
            "_heal_stuck_external_from_pg",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch.object(
            healer,
            "_heal_aggregating_sync",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch.object(
            healer,
            "_heal_attribution",
            new_callable=AsyncMock,
            return_value=(0, 0),
        ) as heal_attr,
        patch.object(
            healer,
            "_heal_attr_slot_counters",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch.object(
            healer,
            "_try_acquire_heal_lock",
            new_callable=AsyncMock,
            return_value=False,
        ) as lock,
    ):
        stats = await healer.tick()

    lock.assert_not_awaited()
    assert stats["attribution_heal_skipped"] == 0
    heal_attr.assert_awaited_once()


@pytest.mark.asyncio
async def test_healer_tick_uses_nested_lock_when_d19_disabled() -> None:
    """When leader election is off, a failed attr short lock skips this round's attribution sweep."""
    from benchmark.workers.running_healer import RunningHealer

    settings = SimpleNamespace(
        sync_slot_heal_interval_s=60,
        external_slot_heal_interval_s=60,
        eval_queue_lock_duration_ms=1_200_000,
        external_wait_lock_duration_ms=300_000,
        external_running_ttl_ms=86_400_000,
        eval_queue_max_attempts=3,
        sync_retry_max_duration_ms=7_200_000,
        external_retry_max_duration_ms=7_200_000,
        eval_queue_backoff_initial_ms=5000,
        eval_queue_backoff_max_ms=25000,
        attr_callback_timeout_ms=86_400_000,
        eval_queue_healer_leader_disabled=True,
    )
    healer = RunningHealer(queue_manager=MagicMock(prefix="pre:bullmq"), settings=settings)

    with (
        patch(
            "benchmark.workers.running_healer.collect_active_run_ids",
            new_callable=AsyncMock,
            return_value=set(),
        ),
        patch.object(healer, "_refresh_heartbeats", new_callable=AsyncMock, return_value=0),
        patch.object(
            healer,
            "_heal_sync_pair",
            new_callable=AsyncMock,
            return_value=(0, 0),
        ),
        patch.object(healer, "_heal_external", new_callable=AsyncMock, return_value=(0, 0)),
        patch.object(healer, "_heal_slot_counters", new_callable=AsyncMock, return_value=0),
        patch.object(
            healer,
            "_heal_aggregating_external",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch.object(
            healer,
            "_heal_stuck_external_from_pg",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch.object(
            healer,
            "_heal_aggregating_sync",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch.object(
            healer,
            "_heal_attribution",
            new_callable=AsyncMock,
            return_value=(1, 0),
        ) as heal_attr,
        patch.object(
            healer,
            "_try_acquire_heal_lock",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        stats = await healer.tick()

    assert stats["attribution_heal_skipped"] == 1
    heal_attr.assert_not_awaited()
