"""Unit tests for callback-timeout heal — a completed job with a lost callback is requeued; in-flight / already landed are skipped."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.workers.running_healer import (
    HealerCallbacks,
    HealJobContext,
    RunningHealer,
)


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "sync_slot_heal_interval_s": 60,
        "external_slot_heal_interval_s": 60,
        "eval_queue_lock_duration_ms": 1_200_000,
        "external_wait_lock_duration_ms": 300_000,
        "external_running_ttl_ms": 86_400_000,
        "external_callback_timeout_ms": 600_000,
        "eval_queue_max_attempts": 3,
        "sync_retry_max_duration_ms": 7_200_000,
        "external_retry_max_duration_ms": 3_600_000,
        "eval_queue_backoff_initial_ms": 5000,
        "eval_queue_backoff_max_ms": 25000,
        "eval_queue_healer_leader_disabled": True,
        "attr_callback_timeout_ms": 86_400_000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _job(
    *,
    job_id: str = "judge-1",
    name: str = "pending_callback",
    data: dict[str, Any] | None = None,
    timestamp: int = 0,
) -> MagicMock:
    job = MagicMock()
    job.id = job_id
    job.name = name
    job.data = data or {
        "run_id": "ieval_external-1",
        "item_id": "1",
        "notification_job_id": "aone-1",
        "slot_key": "slot:judge:node1:ieval_external-1",
        "cluster_kind": "judge",
        "retry_count": 0,
        "first_enqueued_at": 1_000_000,
    }
    job.timestamp = timestamp
    job.remove = AsyncMock()
    job.updateData = AsyncMock()
    return job


def _external_response(status: str) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.error = None
    return resp


def _make_healer(
    *,
    settings: SimpleNamespace | None = None,
    external_client: MagicMock | None = None,
    mark_failed: AsyncMock | None = None,
    benchmark_run_service: MagicMock | None = None,
    callback_handler: AsyncMock | None = None,
) -> RunningHealer:
    qm = MagicMock()
    qm.prefix = "pre"
    healer = RunningHealer(
        queue_manager=qm,
        settings=settings or _settings(),
        callbacks=HealerCallbacks(
            external_judge_client=external_client,
            external_mark_failed=mark_failed,
            benchmark_run_service=benchmark_run_service,
            external_callback_handler=callback_handler,
        ),
    )
    return healer


# ── judge callback timeout ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_judge_timeout_skips_completed_when_callback_in_flight() -> None:
    """Aone is completed but PG has not written it, and a full observation round has not finished → skip, so an in-flight callback is not requeued by mistake."""
    now = 2_000_000
    job = _job(timestamp=now - 700_000)
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=_external_response("completed"),
    )
    bench = MagicMock()
    bench.is_external_judge_item_terminal = AsyncMock(return_value=False)

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release_slot,
    ):
        healer = _make_healer(
            external_client=external_client,
            mark_failed=AsyncMock(),
            benchmark_run_service=bench,
        )
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is False
    requeue.assert_not_called()
    release_slot.assert_not_called()
    job.updateData.assert_awaited_once()
    stamped = job.updateData.await_args.args[0]
    assert stamped["callback_completed_seen_at"] == now


@pytest.mark.asyncio
async def test_judge_timeout_skips_completed_when_pg_already_applied() -> None:
    """Aone is completed and PG is already terminal → a normal completed, and do not requeue."""
    now = 2_000_000
    job = _job(timestamp=now - 700_000)
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=_external_response("completed"),
    )
    bench = MagicMock()
    bench.is_external_judge_item_terminal = AsyncMock(return_value=True)

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
        patch(
            "benchmark.workers.running_healer.force_fail_from_running",
            new_callable=AsyncMock,
        ) as force_fail,
    ):
        healer = _make_healer(
            external_client=external_client,
            mark_failed=AsyncMock(),
            benchmark_run_service=bench,
        )
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is False
    external_client.query_job_status.assert_not_called()
    requeue.assert_not_called()
    force_fail.assert_not_called()


@pytest.mark.asyncio
async def test_judge_timeout_requeues_when_notification_completed() -> None:
    """Aone is completed, the observation window has elapsed, and PG still has not written it → a lost callback; release the slot and requeue onto wait."""
    now = 2_000_000
    job = _job(
        timestamp=now - 700_000,
        data={
            "run_id": "ieval_external-1",
            "item_id": "1",
            "notification_job_id": "aone-1",
            "slot_key": "slot:judge:node1:ieval_external-1",
            "cluster_kind": "judge",
            "retry_count": 0,
            "first_enqueued_at": 1_000_000,
            "callback_completed_seen_at": now - 600_000,
        },
    )
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=_external_response("completed"),
    )
    mark_failed = AsyncMock()
    wait_q = MagicMock()
    running_q = MagicMock()
    bench = MagicMock()
    bench.is_external_judge_item_terminal = AsyncMock(return_value=False)

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release_slot,
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
        patch(
            "benchmark.workers.running_healer.force_fail_from_running",
            new_callable=AsyncMock,
        ) as force_fail,
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="requeued",
        ) as requeue,
    ):
        healer = _make_healer(
            external_client=external_client,
            mark_failed=mark_failed,
            benchmark_run_service=bench,
        )
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=running_q,
                wait_q=wait_q,
                data=dict(job.data),
            ),
        )

    assert result is True
    external_client.query_job_status.assert_awaited_once_with("aone-1")
    release_slot.assert_awaited_once()
    release_cluster.assert_awaited_once()
    force_fail.assert_not_called()
    requeue.assert_awaited_once()
    args = requeue.await_args.args
    assert args[1] == "judge-1"
    deps = args[0]
    assert deps.running_queue is running_q
    assert deps.wait_queue is wait_q
    policy = deps.policy
    assert policy.max_attempts == 3
    assert policy.retry_max_duration_ms == 3_600_000


@pytest.mark.asyncio
async def test_judge_timeout_skips_lost_completed_if_pg_lands_during_grace() -> None:
    """After the observation window, a second PG lookup is already terminal → still treat it as a normal completed, and do not requeue."""
    now = 2_000_000
    job = _job(
        timestamp=now - 700_000,
        data={
            "run_id": "ieval_external-1",
            "item_id": "1",
            "notification_job_id": "aone-1",
            "slot_key": "slot:judge:node1:ieval_external-1",
            "cluster_kind": "judge",
            "retry_count": 0,
            "first_enqueued_at": 1_000_000,
            "callback_completed_seen_at": now - 600_000,
        },
    )
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=_external_response("completed"),
    )
    bench = MagicMock()
    bench.is_external_judge_item_terminal = AsyncMock(side_effect=[False, True])

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release_slot,
    ):
        healer = _make_healer(
            external_client=external_client,
            mark_failed=AsyncMock(),
            benchmark_run_service=bench,
        )
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is False
    requeue.assert_not_called()
    release_slot.assert_not_called()
    assert bench.is_external_judge_item_terminal.await_count == 2


@pytest.mark.asyncio
async def test_judge_timeout_callback_lost_terminal_fail() -> None:
    """Requeue decides on a terminal failure (3 times or past 1h) → outcome=failed, and the slot is still released."""
    now = 2_000_000
    job = _job(
        timestamp=now - 700_000,
        data={
            "run_id": "ieval_external-1",
            "item_id": "1",
            "notification_job_id": "aone-1",
            "slot_key": "slot:judge:node1:ieval_external-1",
            "cluster_kind": "judge",
            "retry_count": 0,
            "first_enqueued_at": 1_000_000,
            "callback_completed_seen_at": now - 600_000,
        },
    )
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=_external_response("completed"),
    )
    mark_failed = AsyncMock()

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release_slot,
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="failed",
        ) as requeue,
    ):
        healer = _make_healer(external_client=external_client, mark_failed=mark_failed)
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is True
    release_slot.assert_awaited_once()
    requeue.assert_awaited_once()
    assert requeue.await_args is not None
    deps = requeue.await_args.args[0]
    policy = deps.policy
    assert policy.max_attempts == 3
    assert policy.retry_max_duration_ms == 3_600_000


@pytest.mark.asyncio
@pytest.mark.parametrize("notification_status", ["running", "pending"])
async def test_judge_timeout_skips_when_notification_in_flight(notification_status: str) -> None:
    """Aone pending/running → skip force_fail, and leave the job on the queue waiting for the callback."""
    now = 2_000_000
    job = _job(timestamp=now - 700_000)
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=_external_response(notification_status),
    )

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.force_fail_from_running",
            new_callable=AsyncMock,
        ) as force_fail,
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
    ):
        healer = _make_healer(external_client=external_client)
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is False
    force_fail.assert_not_called()
    requeue.assert_not_called()
    job.remove.assert_not_called()


@pytest.mark.asyncio
async def test_judge_timeout_force_fail_when_case_window_exceeded() -> None:
    """More than 1h since first_enqueued_at → force_fail even if Aone is still running."""
    now = 5_000_000
    job = _job(
        timestamp=now - 700_000,
        data={
            "run_id": "ieval_external-1",
            "item_id": "1",
            "notification_job_id": "aone-1",
            "slot_key": "slot:judge:node1:ieval_external-1",
            "cluster_kind": "judge",
            "retry_count": 0,
            "first_enqueued_at": now - 3_600_001,
        },
    )
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=_external_response("running"),
    )
    mark_failed = AsyncMock()

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.force_fail_from_running",
            new_callable=AsyncMock,
            return_value="failed",
        ) as force_fail,
    ):
        healer = _make_healer(external_client=external_client, mark_failed=mark_failed)
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is True
    external_client.query_job_status.assert_not_called()
    force_fail.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_timeout_requeues_when_notification_failed() -> None:
    """Aone failed → requeue onto wait (keep first_enqueued_at), and do not force_fail immediately."""
    now = 2_000_000
    job = _job(timestamp=now - 700_000)
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=_external_response("failed"),
    )
    mark_failed = AsyncMock()

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.force_fail_from_running",
            new_callable=AsyncMock,
        ) as force_fail,
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="requeued",
        ) as requeue,
    ):
        healer = _make_healer(external_client=external_client, mark_failed=mark_failed)
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is True
    force_fail.assert_not_called()
    requeue.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_timeout_force_fail_when_ttl_exceeded() -> None:
    """Past external_running_ttl_ms (24h) → force_fail regardless of the Aone status."""
    now = 100_000_000
    job = _job(timestamp=now - 87_000_000)  # age > 24h ttl
    external_client = MagicMock()
    # Even if Aone reports completed, force_fail once the TTL is reached
    external_client.query_job_status = AsyncMock(
        return_value=_external_response("completed"),
    )
    mark_failed = AsyncMock()

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.force_fail_from_running",
            new_callable=AsyncMock,
            return_value="failed",
        ) as force_fail,
    ):
        healer = _make_healer(
            external_client=external_client,
            mark_failed=mark_failed,
            settings=_settings(external_running_ttl_ms=86_400_000),
        )
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is True
    # When the TTL is exceeded, do not query Aone, and force_fail directly
    external_client.query_job_status.assert_not_called()
    force_fail.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_timeout_requeues_when_no_client() -> None:
    """No Aone client → treat it as not running, and requeue onto wait."""
    now = 2_000_000
    job = _job(timestamp=now - 700_000)
    mark_failed = AsyncMock()

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.force_fail_from_running",
            new_callable=AsyncMock,
        ) as force_fail,
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="requeued",
        ) as requeue,
    ):
        healer = _make_healer(external_client=None, mark_failed=mark_failed)
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is True
    force_fail.assert_not_called()
    requeue.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_timeout_requeues_when_query_raises() -> None:
    """An Aone query exception → requeue onto wait."""
    now = 2_000_000
    job = _job(timestamp=now - 700_000)
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(side_effect=RuntimeError("network"))
    mark_failed = AsyncMock()

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.release_cluster_slot",
            new_callable=AsyncMock,
        ),
        patch(
            "benchmark.workers.running_healer.force_fail_from_running",
            new_callable=AsyncMock,
        ) as force_fail,
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
            return_value="requeued",
        ) as requeue,
    ):
        healer = _make_healer(external_client=external_client, mark_failed=mark_failed)
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is True
    force_fail.assert_not_called()
    requeue.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_timeout_replays_callback_from_get_result() -> None:
    """GET completed and the score can be restored → go through handle_callback, and do not requeue onto wait."""
    from benchmark.runs.common.external_client import (
        ExternalJobStatusResponse,
    )

    now = 2_000_000
    job = _job(timestamp=now - 700_000)
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=ExternalJobStatusResponse.model_validate(
            {
                "job_id": "aone-1",
                "status": "completed",
                "params": {
                    "run_id": "ieval_external-1",
                    "task_id": "1",
                },
                "result": {
                    "records": [
                        {
                            "criterion_scores": {
                                "D0": {
                                    "score": 1.0,
                                    "reason": "ok",
                                    "sub_scores": None,
                                },
                            },
                        },
                    ],
                },
            },
        ),
    )
    handler = AsyncMock()
    bench = MagicMock()
    bench.is_external_judge_item_terminal = AsyncMock(return_value=False)

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.get_queue_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
        patch(
            "benchmark.workers.running_healer.release_slot",
            new_callable=AsyncMock,
        ) as release_slot,
    ):
        healer = _make_healer(
            external_client=external_client,
            mark_failed=AsyncMock(),
            benchmark_run_service=bench,
            callback_handler=handler,
        )
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is True
    handler.assert_awaited_once()
    payload = handler.await_args.args[0]
    assert payload.run_id == "ieval_external-1"
    assert payload.criterion_scores["D0"].score == 1.0
    requeue.assert_not_called()
    release_slot.assert_not_called()
    job.updateData.assert_not_called()


@pytest.mark.asyncio
async def test_judge_timeout_empty_completed_get_still_uses_grace() -> None:
    """GET completed but there is no score → still record the grace point, and do not take the callback path."""
    from benchmark.runs.common.external_client import (
        ExternalJobStatusResponse,
    )

    now = 2_000_000
    job = _job(timestamp=now - 700_000)
    external_client = MagicMock()
    external_client.query_job_status = AsyncMock(
        return_value=ExternalJobStatusResponse.model_validate(
            {
                "job_id": "aone-1",
                "status": "completed",
                "result": {"metadata": {"score": {"success": 1, "failed": 0}}},
            },
        ),
    )
    handler = AsyncMock()
    bench = MagicMock()
    bench.is_external_judge_item_terminal = AsyncMock(return_value=False)

    with (
        patch(
            "benchmark.workers.running_healer.time.time",
            return_value=now / 1000,
        ),
        patch(
            "benchmark.workers.running_healer.requeue_or_fail_from_running",
            new_callable=AsyncMock,
        ) as requeue,
    ):
        healer = _make_healer(
            external_client=external_client,
            mark_failed=AsyncMock(),
            benchmark_run_service=bench,
            callback_handler=handler,
        )
        result = await healer._heal_external_callback_timeout(
            HealJobContext(
                job=job,
                running_q=MagicMock(),
                wait_q=MagicMock(),
                data=dict(job.data),
            ),
        )

    assert result is False
    handler.assert_not_called()
    requeue.assert_not_called()
    job.updateData.assert_awaited_once()
