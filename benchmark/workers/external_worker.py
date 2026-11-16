"""external_worker — async-callback eval BullMQ Worker (design doc §8).

Three processors:
1. **external-wait processor**: consume wait-queue case jobs → one-step Aone judge submit
   → write run_items(running) → move case data to the running queue
2. **attr_submit processor**: submit Aone attribution
3. **attr_local processor**: local LLM attribution → write run_items → increment counters

v2.1: ``process_external_wait_case`` uses the shared skeleton ``process_sync_eval_case``,
adapted to async-callback differences via optional ``SyncEvalHandlers`` fields:
- ``cluster_slot_kind="judge"``
- ``hold_slot_on_success=True`` (slot kept for the Aone callback / TTL)
- ``remove_running_on_success=False`` (upgrade inflight → pending_callback)
- ``handle_exception`` custom submit-failure classification
"""

import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Job

from benchmark.settings.setting import Settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.libs.jobqueue.job_listing import get_queue_job
from benchmark.libs.jobqueue.worker import Worker
from benchmark.runs.common.execution_classes import (
    AsyncJudgeImpl,
    AsyncJudgeSubmitRequest,
)
from benchmark.runs.common.utils import extract_ds
class ExternalJudgeClient:
    pass
IevalExternalJudge = None
from benchmark.runs.run_service import BenchmarkService
from benchmark.runs.run_stop import is_stop_requested
from benchmark.runs.schema import ExternalJudgeItemResult
from benchmark.queue.inflight_slot import (
    release_cluster_slot,
    release_slot,
)
from benchmark.queue.queue_factory import (
    EXTERNAL_RUNNING,
    EXTERNAL_WAIT,
    EvalQueueRegistry,
)
from benchmark.queue.retry_requeue import (
    RequeueDeps,
    RequeueOpts,
    force_fail_from_running,
    requeue_or_fail_from_running,
    retry_policy_from_settings,
)
from benchmark.queue.submit_failure import (
    RunningJobWrite,
    add_running_with_retry,
    is_retryable_submit_failure,
    should_release_on_submit_failure,
)
from benchmark.queue.worker_skeleton import (
    SKIP_HOLD_SLOT_FLAG,
    ExceptionContext,
    JudgeWriteCall,
    SlotConfig,
    SyncEvalHandlers,
    SyncEvalRuntime,
    process_sync_eval_case,
)
from benchmark.workers.finalize_trigger import tick_and_maybe_finalize
from benchmark.workers.fused_row import (
    FusedRunningPrewrite,
    prewrite_running_fused_item,
)

logger = get_logger("agent_eval.eval_workers.external_worker")

#: Node identity (slot-key isolation). On k8s a container restart does not change the Pod hostname;
#: stale slot counts from a rebuilt Pod self-heal via slot TTL (TID251 forbids reading env vars directly).
WORKER_NAME = socket.gethostname()

#: ODPS partition day ``YYYYMMDD`` length.
_ODPS_DS_LENGTH = 8


def _resolve_external_ds(run_id: str, ds_from_payload: object) -> str:
    """Resolve the ODPS partition day for an Aone submit: prefer payload.ds, else extract it from run_id."""
    raw = str(ds_from_payload or "").strip()
    if len(raw) == _ODPS_DS_LENGTH and raw.isdigit():
        return raw
    return extract_ds(run_id)


_TERMINAL_STATUSES = frozenset(
    {
        BenchmarkStatus.COMPLETED.value,
        BenchmarkStatus.FAILED.value,
        BenchmarkStatus.STOPPED.value,
    }
)


async def _is_run_terminal(run_id: str, run_service: BenchmarkService) -> bool:
    run = await run_service.get_run(run_id)
    if run is None:
        return True
    if run.status in _TERMINAL_STATUSES:
        return True
    return is_stop_requested(getattr(run, "config_json", None))


@dataclass(frozen=True)
class _ExternalJudgeSubmit:
    """Arguments for ``_submit_judge_to_aone`` (FUNC-PARAMS-001)."""

    run_id: str
    item_id: str
    data: dict[str, Any]
    dims: str | None = None
    prompt_version: str | None = None


@dataclass(frozen=True)
class _PendingCallbackUpgrade:
    """Arguments for ``_upgrade_external_pending_callback`` (FUNC-PARAMS-001)."""

    run_id: str
    item_id: str
    data: dict[str, Any]
    notification_job_id: str | None
    running_queue_name: str = EXTERNAL_RUNNING


@dataclass(frozen=True)
class _AcceptedFailureHandle:
    """Arguments for ``_handle_accepted_failure`` (FUNC-PARAMS-001)."""

    notification_job_id: str | None
    exc: Exception
    mark_failed: Callable[..., Awaitable[None]]
    slot_was_held: bool
    cluster_was_held: bool


async def _submit_judge_to_external(
    external_judge_client: ExternalJudgeClient | None,
    submit: _ExternalJudgeSubmit,
    judge_impl: AsyncJudgeImpl | None = None,
) -> str:
    """Submit one judge via the async judge class and return notification_job_id (core submit step).

    ``judge_impl`` comes from the run snapshot ``judge.class_ref``; the default (legacy runs
    with no class_ref / external-wait drain) is :class:`IevalExternalJudge`.
    ``dims`` / ``prompt_version`` are per-run profile config (§5.1).
    """
    run_id = submit.run_id
    item_id = submit.item_id
    if external_judge_client is None:
        raise RuntimeError("Aone judge client not configured for external evaluation")

    impl = judge_impl or IevalExternalJudge()
    notification_job_id = await impl.submit(
        AsyncJudgeSubmitRequest(
            run_id=run_id,
            item_id=item_id,
            data=submit.data,
            ds=_resolve_external_ds(run_id, submit.data.get("ds")),
            client=external_judge_client,
            dims=submit.dims,
            prompt_version=submit.prompt_version,
        ),
    )
    logger.info(
        "external.aone_submitted run_id=%s item=%s notification_job_id=%s impl=%s",
        run_id,
        item_id,
        notification_job_id,
        type(impl).__name__,
    )
    return notification_job_id


async def _upgrade_external_pending_callback(
    queue_manager: EvalQueueRegistry,
    upgrade: _PendingCallbackUpgrade,
) -> None:
    """Upgrade inflight → pending_callback (jobId=notification_job_id).

    ``running_queue_name``: §8.3 queue merge — legacy external runs use
    ``external-running``; unified runs use ``benchmark-running``.
    """
    run_id = upgrade.run_id
    item_id = upgrade.item_id
    data = upgrade.data
    notification_job_id = upgrade.notification_job_id
    running_queue_name = upgrade.running_queue_name
    case_job_id = f"{run_id}:{item_id}"
    running_q = queue_manager.get_queue(running_queue_name)
    old = await get_queue_job(running_q, case_job_id)
    logger.info(
        "bullmq.running_get_job job_id=%s found=%s run_id=%s item=%s",
        case_job_id,
        old is not None,
        run_id,
        item_id,
    )
    old_data = old.data if old else data
    payload = {
        **old_data,
        "notification_job_id": notification_job_id,
        "submitter_node_id": WORKER_NAME,
        "slot_key": old_data.get("slot_key", ""),
        "cluster_kind": "judge",
    }
    if old is not None:
        await old.remove()
        logger.info(
            "bullmq.running_removed job_id=%s run_id=%s item=%s",
            case_job_id,
            run_id,
            item_id,
        )
    await add_running_with_retry(
        running_q,
        RunningJobWrite(
            job_id=notification_job_id,
            name="pending_callback",
            data=payload,
            attempts=5,
        ),
    )
    logger.info(
        "bullmq.pending_callback_add queue=external-running job_id=%s run_id=%s item=%s",
        notification_job_id,
        run_id,
        item_id,
    )


async def _recover_external_pending_callback(
    ctx: ExceptionContext,
    notification_job_id: str,
) -> bool:
    """Submit failed after accept → rewrite pending_callback (slot kept for callback/TTL).

    Returns:
        Whether recovery succeeded. On failure the caller must terminate deterministically
        (mark failed + release the held slot) — without pending_callback neither the callback
        nor the healer has a trail, and the task waits forever for a callback that does not exist.
    """
    try:
        await add_running_with_retry(
            ctx.running_q,
            RunningJobWrite(
                job_id=notification_job_id,
                name="pending_callback",
                data={
                    **ctx.data,
                    "notification_job_id": notification_job_id,
                    "submitter_node_id": WORKER_NAME,
                    "slot_key": ctx.slot_key,
                    "cluster_kind": "judge",
                },
                attempts=3,
            ),
        )
        return True
    except Exception as recover_exc:
        logger.error(
            "external.pending_callback_recover_failed run_id=%s item=%s notification_job_id=%s err=%s",
            ctx.data["run_id"],
            ctx.data["item_id"],
            notification_job_id,
            recover_exc,
        )
        return False


async def _handle_accepted_failure(
    ctx: ExceptionContext,
    handle: _AcceptedFailureHandle,
) -> None:
    """Submit-failure handling after Aone has accepted: keep the slot + recover pending_callback.

    On recover failure, terminate deterministically (mark the item failed + release the held
    slot) — without pending_callback neither the callback nor the healer has a trail, and the
    task waits forever for a callback that does not exist. A late Aone callback hits the
    idempotent terminal guard.
    """
    notification_job_id = handle.notification_job_id
    exc = handle.exc
    mark_failed = handle.mark_failed
    slot_was_held = handle.slot_was_held
    cluster_was_held = handle.cluster_was_held
    logger.error(
        "external.submit_post_accept_fail run_id=%s item=%s "
        "notification_job_id=%s err=%s (slot kept for callback/TTL)",
        ctx.data["run_id"],
        ctx.data["item_id"],
        notification_job_id,
        exc,
    )
    if not notification_job_id:
        return
    recovered = await _recover_external_pending_callback(ctx, notification_job_id)
    if recovered:
        return
    await mark_failed({"run_id": ctx.data["run_id"], "item_id": ctx.data["item_id"]})
    if slot_was_held:
        await release_slot(ctx.slot_key)
    if cluster_was_held:
        await release_cluster_slot(ctx.cluster_slot_kind)


# ════════════════════════════════════════════════════════════════════════════
# 1. external-wait processor (submit Aone judge + move to the running queue)
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ExternalWorkerDeps:
    """Service dependency bundle for the external worker (parameter object for ``process_external_wait_case``).

    Judge submit depends on the Aone client; ``queue_manager`` is a required skeleton argument and stays outside.
    """

    run_service: BenchmarkService
    external_judge_client: ExternalJudgeClient | None
    settings: Settings


async def process_external_wait_case(
    job: Job,
    token: str,
    *,
    queue_manager: EvalQueueRegistry,
    deps: ExternalWorkerDeps,
) -> None:
    """Consume legacy case jobs on the external-wait queue (§8.3 compatibility drain).

    Implementation moved to :func:`build_async_judge_handlers` (shared with the unified
    case worker); this entry remains only to drain in-flight external-wait/running jobs.
    New runs always enter ``benchmark-cases`` through the unified submit pipeline (§8.3).
    """
    await process_sync_eval_case(
        job,
        token,
        SyncEvalRuntime(
            handlers=build_async_judge_handlers(
                AsyncJudgeHandlersBuild(
                    run_service=deps.run_service,
                    settings=deps.settings,
                    external_judge_client=deps.external_judge_client,
                    queue_manager=queue_manager,
                ),
            ),
            queue_manager=queue_manager,
            deps=deps,
        ),
    )


@dataclass(frozen=True)
class AsyncJudgeOverrides:
    """Variable parts of build_async_judge_handlers: queue pair + judge override config.

    Default = legacy external-wait drain behavior; the unified case worker passes the
    benchmark-cases queue pair + per-run judge config.
    """

    wait_queue_name: str = EXTERNAL_WAIT
    running_queue_name: str = EXTERNAL_RUNNING
    judge_dims: str | None = None
    judge_prompt_version: str | None = None
    #: Execution class loaded from the run snapshot judge.class_ref; None = IevalExternalJudge
    judge_impl: AsyncJudgeImpl | None = None


@dataclass(frozen=True)
class AsyncJudgeHandlersBuild:
    """Arguments for ``build_async_judge_handlers`` (FUNC-PARAMS-001; public: reused by benchmark_worker)."""

    run_service: BenchmarkService
    settings: Settings
    external_judge_client: ExternalJudgeClient | None
    queue_manager: EvalQueueRegistry
    overrides: AsyncJudgeOverrides | None = None


async def _prewrite_or_skip_hold(
    write: FusedRunningPrewrite,
    data: dict[str, Any],
    *,
    log_event: str,
    extra: str = "",
) -> bool:
    """Prewrite the fused running row; if already terminal, log skip_hold and return False."""
    still_open = await prewrite_running_fused_item(write)
    if still_open:
        return True
    data[SKIP_HOLD_SLOT_FLAG] = True
    logger.info(
        "%s run_id=%s item=%s%s",
        log_event,
        write.run_id,
        write.item_id,
        extra,
    )
    return False


def build_async_judge_handlers(
    build: AsyncJudgeHandlersBuild,
) -> SyncEvalHandlers:
    """Build async-judge (Aone submit + hold slot for callback) handlers (§8.2 / §8.3).

    The unified case worker (``benchmark-cases``, judge.execution=async) and the
    external-wait legacy drain worker share this async judge implementation; they differ
    only in queue name / client instance / dims. Finalize uses the external counter model
    (async-judge runs write ``judge_expected_count`` at submit time).

    Key differences from sync workers:
    - ``cluster_slot_kind="judge"``
    - ``hold_slot_on_success=True`` (slot kept for the Aone callback / TTL)
    - ``remove_running_on_success=False`` (upgrade inflight → pending_callback)
    - ``handle_exception`` is custom (submit-failure classification)
    """
    run_service, settings, external_judge_client, queue_manager = (
        build.run_service,
        build.settings,
        build.external_judge_client,
        build.queue_manager,
    )
    ov = build.overrides or AsyncJudgeOverrides()

    # Per-job shared state (handlers are built per job): passes "whether Aone has
    # accepted" between judge_and_write and _handle_exception
    notification_accepted = False
    notification_job_id: str | None = None

    async def _judge_and_write(
        call: JudgeWriteCall,
    ) -> None:
        nonlocal notification_accepted, notification_job_id
        run_id, item_id, case, data = call.run_id, call.item_id, call.case, call.data

        if not await _prewrite_or_skip_hold(
            FusedRunningPrewrite(
                run_id=run_id,
                item_id=item_id,
                item_input=case.input,
                item_output=case.output,
            ),
            data,
            log_event="bullmq.skip_submit_terminal",
        ):
            return

        # 4) One-step Aone judge submit
        notification_job_id = await _submit_judge_to_external(
            external_judge_client,
            _ExternalJudgeSubmit(
                run_id=run_id,
                item_id=item_id,
                data=data,
                dims=ov.judge_dims,
                prompt_version=ov.judge_prompt_version,
            ),
            ov.judge_impl,
        )
        notification_accepted = True

        # 5) Write back source_trace_id; do not overwrite status if already terminal (callback landed first).
        if not await _prewrite_or_skip_hold(
            FusedRunningPrewrite(
                run_id=run_id,
                item_id=item_id,
                source_trace_id=notification_job_id,
                item_input=case.input,
                item_output=case.output,
            ),
            data,
            log_event="bullmq.skip_pending_callback_terminal",
            extra=f" source_trace_id={notification_job_id}",
        ):
            return
        logger.info(
            "bullmq.run_item_written run_id=%s item=%s source_trace_id=%s status=running",
            run_id,
            item_id,
            notification_job_id,
        )

        # 6) Upgrade inflight → pending_callback (jobId=notification_job_id)
        await _upgrade_external_pending_callback(
            queue_manager,
            _PendingCallbackUpgrade(
                run_id=run_id,
                item_id=item_id,
                data=data,
                notification_job_id=notification_job_id,
                running_queue_name=ov.running_queue_name,
            ),
        )

    async def _mark_failed_from_wait(
        *,
        data: dict[str, Any],
        reason: str = "retry_deadline",
    ) -> None:
        """Terminal fail on the wait path (never entered running): advance failed + counters in one transaction.

        The job payload carries case input/output; write them to PG on terminal fail so the
        report case detail is not entirely empty (never entered running → no prewrite row →
        input/output NULL).
        """
        await run_service.cas_advance_external_judge_item(
            result=ExternalJudgeItemResult(
                run_id=data["run_id"],
                item_id=data["item_id"],
                failed=True,
                score=0.0,
                judge_json={"error": reason, "reason": reason},
                error_message=reason,
                item_input=data.get("input"),
                item_output=data.get("output"),
            ),
        )

    async def _mark_failed_from_running(
        *,
        data: dict[str, Any],
        exc: Exception,
    ) -> None:
        err_msg = str(exc)[:2000]
        await run_service.cas_advance_external_judge_item(
            result=ExternalJudgeItemResult(
                run_id=data["run_id"],
                item_id=data["item_id"],
                failed=True,
                score=0.0,
                judge_json={"error": err_msg, "reason": err_msg},
                error_message=err_msg,
            ),
        )

    async def _handle_exception(ctx: ExceptionContext) -> None:
        """Classify submit failure (§8.2).

        - Aone already accepted → keep the slot + recover pending_callback
        - Retryable failure → requeue_or_fail
        - Non-retryable failure → force_fail
        """
        nonlocal notification_accepted, notification_job_id

        # §8.2: if Aone already accepted, keep the slot; otherwise release, then requeue / force_fail
        slot_was_held = ctx.state.held
        cluster_was_held = ctx.state.cluster_held
        do_release = should_release_on_submit_failure(
            ctx.exc,
            notification_accepted=notification_accepted,
        )
        if ctx.state.held and do_release:
            await release_slot(ctx.slot_key)
        if ctx.state.cluster_held and do_release:
            await release_cluster_slot(ctx.cluster_slot_kind)

        # Whether released or not, mark not held so finally does not release again
        ctx.state.held = False
        ctx.state.cluster_held = False

        err_msg = str(ctx.exc)[:2000]
        run_id = ctx.data["run_id"]
        item_id = ctx.data["item_id"]

        async def _mark_failed(
            d: dict[str, Any],
            *,
            retry_count: int = 0,
            **kw: object,
        ) -> None:
            await run_service.cas_advance_external_judge_item(
                result=ExternalJudgeItemResult(
                    run_id=d["run_id"],
                    item_id=d["item_id"],
                    failed=True,
                    score=0.0,
                    judge_json={"error": err_msg, "reason": err_msg},
                    error_message=err_msg,
                ),
            )

        if notification_accepted:
            # Already accepted: do not requeue (no double submit) or force_fail; keep the slot for callback / TTL / healer
            await _handle_accepted_failure(
                ctx,
                _AcceptedFailureHandle(
                    notification_job_id=notification_job_id,
                    exc=ctx.exc,
                    mark_failed=_mark_failed,
                    slot_was_held=slot_was_held,
                    cluster_was_held=cluster_was_held,
                ),
            )
            return

        if is_retryable_submit_failure(ctx.exc):
            outcome = await requeue_or_fail_from_running(
                RequeueDeps(
                    running_queue=ctx.running_q,
                    wait_queue=ctx.wait_q,
                    policy=retry_policy_from_settings(
                        settings,
                        retry_max_duration_ms=settings.external_retry_max_duration_ms,
                    ),
                    mark_failed=_mark_failed,
                ),
                ctx.case_job_id,
                RequeueOpts(
                    wait_job=ctx.wait_job,
                    token=ctx.token or None,
                ),
            )
            if outcome == "failed":
                logger.warning(
                    "external.submit_terminal_fail run_id=%s item=%s err=%s",
                    run_id,
                    item_id,
                    ctx.exc,
                )
            else:
                logger.info(
                    "external.submit_requeued run_id=%s item=%s err=%s",
                    run_id,
                    item_id,
                    ctx.exc,
                )
        else:
            await force_fail_from_running(
                running_queue=ctx.running_q,
                running_job_id=ctx.case_job_id,
                mark_failed=_mark_failed,
            )
            logger.warning(
                "external.submit_force_fail run_id=%s item=%s err=%s",
                run_id,
                item_id,
                ctx.exc,
            )

    async def _finalize(run_id: str) -> None:
        await tick_and_maybe_finalize(
            run_id,
            "external",
            benchmark_run_service=run_service,
        )

    return SyncEvalHandlers(
        eval_type="external",
        wait_queue_name=ov.wait_queue_name,
        running_queue_name=ov.running_queue_name,
        slot_config=SlotConfig(
            retry_max_duration_ms=settings.external_retry_max_duration_ms,
            cluster_soft_cap=settings.external_cluster_notification_soft_cap,
            slots_per_run_per_node=settings.external_slots_per_run_per_node,
            slot_busy_delay_ms=settings.external_slot_busy_delay_ms,
            slot_ttl_s=settings.external_slot_ttl_s,
        ),
        is_run_terminal=lambda rid: _is_run_terminal(rid, run_service),
        heartbeat=run_service.heartbeat,
        judge_and_write=_judge_and_write,
        mark_failed_from_wait=_mark_failed_from_wait,
        mark_failed_from_running=_mark_failed_from_running,
        finalize_fn=_finalize,
        cluster_slot_kind="judge",
        hold_slot_on_success=True,
        remove_running_on_success=False,
        handle_exception=_handle_exception,
    )


# ════════════════════════════════════════════════════════════════════════════
# 2. attr_local processor (local LLM attribution)
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# Worker factory function
# ════════════════════════════════════════════════════════════════════════════


def create_external_wait_worker(
    *,
    queue_manager: EvalQueueRegistry,
    run_service: BenchmarkService,
    external_judge_client: ExternalJudgeClient | None,
    settings: Settings,
) -> Worker:
    """Create the BullMQ Worker for the external-wait queue.

    concurrency comes from YAML ``slots.external.worker_concurrency`` (D14).
    Backpressure is driven by ``slots_per_run_per_node`` + callback ``release_slot``/``promote``
    (submit the next item as soon as the previous callback completes); no fixed-time limiter.
    """
    queue = queue_manager.get_queue(EXTERNAL_WAIT)
    opts = queue_manager.build_worker_opts(
        queue_name=EXTERNAL_WAIT,
        lock_duration_ms=settings.external_wait_lock_duration_ms,
        stalled_interval_ms=settings.eval_queue_stalled_interval_ms,
    )
    opts["concurrency"] = settings.external_worker_concurrency
    # Backpressure is driven by slots_per_run_per_node + callback release/promote;
    # no fixed duration limiter (otherwise the next submit still waits out the interval after a callback).

    async def processor(job: Job, token: str) -> None:
        await process_external_wait_case(
            job,
            token,
            queue_manager=queue_manager,
            deps=ExternalWorkerDeps(
                run_service=run_service,
                external_judge_client=external_judge_client,
                settings=settings,
            ),
        )

    worker = Worker(queue.name, processor, opts)
    logger.info(
        "external_wait_worker.started queue=%s concurrency=%d (slot-backed; no time limiter)",
        queue.name,
        settings.external_worker_concurrency,
    )
    return worker


__all__ = [
    "AsyncJudgeOverrides",
    "ExternalWorkerDeps",
    "build_async_judge_handlers",
    "create_external_wait_worker",
    "process_external_wait_case",
]
