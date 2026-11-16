"""callback_next_submit — release the next submit as soon as an async callback finishes (design §8 / D8).

Two coordination mechanisms:
1. **In-process latch** (legacy path ``submit_batch``): when the previous callback arrives the
   ``Future`` completes and the submit loop continues with the next item immediately
   (replacing ``asyncio.sleep(submit_interval)``).
2. **BullMQ slot + promote** (queue path): after the callback ``release_slot``,
   ``promote`` the same run's delayed wait job so the Worker takes the slot and submits at once.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Queue

from benchmark.libs.jobqueue.job_listing import get_queue_job, list_queue_jobs
from benchmark.queue.inflight_slot import (
    SlotKeySpec,
    build_slot_key,
    release_cluster_slot,
    release_slot,
)
from benchmark.queue.queue_factory import (
    BENCHMARK_CASES,
    BENCHMARK_RUNNING,
    EXTERNAL_RUNNING,
    EXTERNAL_WAIT,
    EvalQueueRegistry,
)


def queue_names_for_pair(queue_pair: str) -> tuple[str, str]:
    """A run's ``queue_pair`` flag → (wait queue, running queue) (§8.3 queue merge).

    - ``"benchmark"``: the unified queue pair (default for new runs after the merge);
    - anything else / missing: the legacy ``external-wait/running`` pair (unchanged during compatibility).
    """
    if queue_pair == "benchmark":
        return BENCHMARK_CASES, BENCHMARK_RUNNING
    return EXTERNAL_WAIT, EXTERNAL_RUNNING


logger = get_logger("agent_eval.eval_queue.callback_next_submit")


class JudgeSubmitGate:
    """Per-run latch: wait for the callback before submitting the next item.

    Usage: ``register(item_id)`` before submit → submit → ``await wait(item_id)`` (not the last item);
    the callback calls ``complete(item_id)``. If the callback arrives before ``register``, record an
    early credit so the Future is already completed at ``register`` and the waiter does not hang.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pending: dict[str, dict[str, asyncio.Future[None]]] = {}
        self._early: dict[str, set[str]] = {}

    async def register(self, run_id: str, item_id: str) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            early = self._early.get(run_id) or set()
            if item_id in early:
                early.discard(item_id)
                if not early:
                    self._early.pop(run_id, None)
                else:
                    self._early[run_id] = early
                fut.set_result(None)
            bucket = self._pending.setdefault(run_id, {})
            existing = bucket.get(item_id)
            if existing is not None and not existing.done():
                return
            bucket[item_id] = fut

    async def wait(self, run_id: str, item_id: str) -> None:
        async with self._lock:
            fut = self._pending.get(run_id, {}).get(item_id)
        if fut is None:
            return
        await fut

    async def complete(self, run_id: str, item_id: str | None) -> bool:
        """Callback finished: release the submit loop waiting on this item. Returns whether the latch was hit."""
        if not item_id:
            async with self._lock:
                bucket = self._pending.get(run_id) or {}
                for iid, fut in list(bucket.items()):
                    if not fut.done():
                        fut.set_result(None)
                        bucket.pop(iid, None)
                        logger.info(
                            "submit_gate.complete_any run_id=%s item_id=%s",
                            run_id,
                            iid,
                        )
                        return True
            return False

        async with self._lock:
            bucket = self._pending.get(run_id) or {}
            fut = bucket.pop(item_id, None)
            if not bucket and run_id in self._pending:
                del self._pending[run_id]
            if fut is None:
                self._early.setdefault(run_id, set()).add(item_id)
                logger.info(
                    "submit_gate.complete_early run_id=%s item_id=%s",
                    run_id,
                    item_id,
                )
                return True
        if not fut.done():
            fut.set_result(None)
        logger.info(
            "submit_gate.complete run_id=%s item_id=%s",
            run_id,
            item_id,
        )
        return True

    async def cancel_run(self, run_id: str) -> None:
        async with self._lock:
            bucket = self._pending.pop(run_id, {})
            self._early.pop(run_id, None)
        for fut in bucket.values():
            if not fut.done():
                fut.cancel()


#: Process-level singleton latch (shared by ExternalBenchmarkRunner and the callback)
judge_submit_gate = JudgeSubmitGate()


async def promote_delayed_for_run(
    wait_queue: Queue,
    run_id: str,
    *,
    limit: int = 1,
) -> int:
    """Promote the same run's delayed wait job back to waiting (a Worker can consume it immediately)."""
    try:
        delayed = await list_queue_jobs(wait_queue, ["delayed"])
    except Exception:
        logger.warning(
            "promote.list_delayed_failed run_id=%s",
            run_id,
            exc_info=True,
        )
        return 0

    promoted = 0
    for job in delayed:
        if job is None:
            continue
        if (job.data or {}).get("run_id") != run_id:
            continue
        try:
            await job.promote()
            promoted += 1
            logger.info(
                "promote.delayed_job run_id=%s job_id=%s",
                run_id,
                job.id,
            )
        except Exception:
            logger.warning(
                "promote.failed run_id=%s job_id=%s",
                run_id,
                job.id,
                exc_info=True,
            )
        if promoted >= limit:
            break
    return promoted


@dataclass(frozen=True)
class SlotReleaseRequest:
    """Slot release and promote request after a callback
    (parameter object for ``release_slot_and_promote_next``).

    ``slot_key`` / ``cluster_kind`` may be passed explicitly: if the failure path already
    deleted the running job via ``requeue_or_fail`` / ``force_fail``, the rate limit must
    still be released, or the slot leaks and the next item cannot be submitted.
    """

    queue_manager: EvalQueueRegistry | None
    notification_job_id: str | None
    run_id: str
    item_id: str | None = None
    promote_limit: int = 1
    slot_key: str | None = None
    cluster_kind: str = "judge"
    wait_queue_name: str = EXTERNAL_WAIT
    """Wait queue name (§8.3 merge: unified run = ``benchmark-cases``, legacy = external-wait)."""
    running_queue_name: str = EXTERNAL_RUNNING
    """Running queue name (§8.3 merge: unified run = ``benchmark-running``)."""


async def release_slot_and_promote_next(req: SlotReleaseRequest) -> dict[str, Any]:
    """After the callback: remove pending_callback → release the slot → promote the next item."""
    queue_manager = req.queue_manager
    notification_job_id = req.notification_job_id
    run_id = req.run_id
    item_id = req.item_id
    result: dict[str, Any] = {
        "running_removed": False,
        "slot_released": False,
        "promoted": 0,
        "gate_completed": False,
    }

    result["gate_completed"] = await judge_submit_gate.complete(run_id, item_id)

    if queue_manager is None:
        return result

    running_q = queue_manager.get_queue(req.running_queue_name)
    wait_q = queue_manager.get_queue(req.wait_queue_name)

    resolved_slot = req.slot_key
    resolved_kind = req.cluster_kind
    job = None
    if notification_job_id:
        try:
            job = await get_queue_job(running_q, notification_job_id)
        except Exception:
            logger.warning(
                "callback_next.get_running_failed job_id=%s",
                notification_job_id,
                exc_info=True,
            )

    if job is not None:
        data = dict(job.data or {})
        if not resolved_slot:
            resolved_slot = data.get("slot_key")
            if not resolved_slot and data.get("submitter_node_id") and data.get("run_id"):
                resolved_slot = build_slot_key(
                    queue_manager.prefix,
                    SlotKeySpec(
                        kind="judge",
                        node=str(data["submitter_node_id"]),
                        run_id=str(data["run_id"]),
                    ),
                )
        if data.get("cluster_kind"):
            resolved_kind = str(data["cluster_kind"])
        try:
            await job.remove()
            result["running_removed"] = True
        except Exception:
            logger.warning(
                "callback_next.remove_running_failed job_id=%s",
                notification_job_id,
                exc_info=True,
            )

    if resolved_slot:
        await release_slot(str(resolved_slot))
        result["slot_released"] = True
        await release_cluster_slot(resolved_kind)
    elif notification_job_id:
        logger.warning(
            "callback_next.slot_key_missing run_id=%s item_id=%s "
            "notification_job_id=%s (slot may leak until TTL)",
            run_id,
            item_id,
            notification_job_id,
        )

    result["promoted"] = await promote_delayed_for_run(
        wait_q,
        run_id,
        limit=req.promote_limit,
    )
    logger.info(
        "callback_next.done run_id=%s item_id=%s notification_job_id=%s %s",
        run_id,
        item_id,
        notification_job_id,
        result,
    )
    return result


async def enqueue_external_cases(
    *,
    queue_manager: EvalQueueRegistry,
    run_id: str,
    items: list[dict[str, Any]],
    ds: str = "",
) -> int:
    """Enqueue every case onto external-wait (``retry_count=0``).

    Pass the item body through (including ``metadata.csv_row``) so the Worker can ``build_record``.
    Do **not** write ``first_enqueued_at`` here: that timestamp is set when the case first enters
    running, so time spent queued on wait does not consume the per-case 1h budget.
    """
    wait_q = queue_manager.get_queue(EXTERNAL_WAIT)
    job_opts = {
        "attempts": 1,
        "removeOnComplete": True,
        "removeOnFail": True,
    }
    count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id") or item.get("id") or "")
        if not item_id:
            continue
        # Spread the item first, then overwrite queue-contract fields, so columns such as csv_row are not dropped
        payload: dict[str, Any] = {
            **item,
            "eval_type": "external",
            "run_id": run_id,
            "item_id": item_id,
            "id": str(item.get("id") or item_id),
            "retry_count": 0,
            "ds": ds or str(item.get("ds") or ""),
            "metadata": item.get("metadata") or {},
            "case_id": item.get("case_id") or item_id,
        }
        payload.pop("first_enqueued_at", None)
        await wait_q.add(
            "case",
            payload,
            {"jobId": f"{run_id}:{item_id}", **job_opts},
        )
        count += 1
    logger.info(
        "external.enqueued_wait run_id=%s count=%d",
        run_id,
        count,
    )
    return count


__all__ = [
    "JudgeSubmitGate",
    "SlotReleaseRequest",
    "enqueue_external_cases",
    "judge_submit_gate",
    "promote_delayed_for_run",
    "queue_names_for_pair",
    "release_slot_and_promote_next",
]
