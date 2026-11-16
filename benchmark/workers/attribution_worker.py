"""attribution_worker — attribution-submit BullMQ Worker.

Reuses the shared skeleton ``process_sync_eval_case``, adapted via ``SyncEvalHandlers``:
- ``cluster_slot_kind="attr"``
- ``hold_slot_on_success=True`` (slot kept for the provider callback / TTL)
- ``remove_running_on_success=False`` (upgrade inflight → pending_callback)
"""

import asyncio
import socket
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Job

from benchmark.settings.setting import Settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.libs.jobqueue.job_listing import get_queue_job
from benchmark.libs.jobqueue.worker import Worker
class AttributionProviderRegistry:
    pass
AttributionSubmitRequest = None
class AttributionSubmitResult:
    pass
split_attribution_issue_payload = None
from benchmark.runs.run_service import BenchmarkService
from benchmark.runs.run_stop import is_stop_requested
from benchmark.queue.callback_next_submit import promote_delayed_for_run
from benchmark.queue.inflight_slot import (
    release_cluster_slot,
    release_slot,
)
from benchmark.queue.queue_factory import (
    ATTRIBUTION_RUNNING,
    ATTRIBUTION_WAIT,
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
    ExceptionContext,
    JudgeWriteCall,
    SlotConfig,
    SyncEvalHandlers,
    SyncEvalRuntime,
    process_sync_eval_case,
)
from benchmark.workers.finalize_trigger import tick_and_maybe_finalize

logger = get_logger("agent_eval.eval_workers.attribution_worker")

#: Node identity (slot-key isolation)
WORKER_NAME = socket.gethostname()

_TERMINAL_STATUSES = frozenset(
    {
        BenchmarkStatus.COMPLETED.value,
        BenchmarkStatus.FAILED.value,
        BenchmarkStatus.STOPPED.value,
    }
)


@dataclass(frozen=True)
class _AttributionProviderSubmit:
    """Arguments for ``_submit_attribution_to_provider`` (FUNC-PARAMS-001)."""

    run_id: str
    item_id: str
    data: dict[str, Any]


@dataclass(frozen=True)
class _AttributionPendingUpgrade:
    """Arguments for ``_upgrade_attribution_pending_callback`` (FUNC-PARAMS-001)."""

    run_id: str
    item_id: str
    data: dict[str, Any]
    attribution_job_id: str | None


async def _is_run_terminal(run_id: str, run_service: BenchmarkService) -> bool:
    try:
        run = await asyncio.wait_for(
            run_service.get_run(run_id),
            timeout=10,
        )
    except TimeoutError:
        logger.warning(
            "attribution.run_terminal_check_timeout run_id=%s",
            run_id,
        )
        return False
    if run is None:
        return True
    if run.status in _TERMINAL_STATUSES:
        return True
    return is_stop_requested(getattr(run, "config_json", None))


async def _submit_attribution_to_provider(
    registry: AttributionProviderRegistry | None,
    settings: Settings,
    submit: "_AttributionProviderSubmit",
) -> AttributionSubmitResult:
    """Submit attribution to the provider and check the returned item_id (core worker submit step).

    Steps 3-4 of ``process_attribution_wait_case._submit_and_write``:
    build ``AttributionSubmitRequest`` (§4.2 business fields) → POST the provider
    attribution API → check that ``response_item_id`` matches the submitted task_id.
    """
    run_id = submit.run_id
    item_id = submit.item_id
    data = submit.data
    if registry is None:
        raise RuntimeError("Attribution provider registry not configured")

    provider_name = data.get("attribution_class_ref") or data.get("attribution_provider", "external")
    provider = registry.resolve_for_job(data)
    if provider is None:
        raise ValueError(
            f"Unknown attribution provider: {provider_name}",
        )

    source_trace_id = data.get("source_trace_id", "")

    # Attribution inputs (aligned with async-attribution-design.md §4.2 params business fields)
    adiu = data.get("adiu", "")
    ds_start = data.get("ds_start", "")
    ds_end = data.get("ds_end")
    trace_id = data.get("trace_id")
    do_cross_verify = data.get("do_cross_verify", True)
    issue_raw = data.get("issue") or {}
    # Required pipeline business fields (§4.2)
    attr_query = data.get("query", "")
    attr_response = data.get("response", "")
    attr_tool_calls_json = data.get("full_tool_calls_json")

    # Required-field precheck: missing query/response in upstream Job Data leaves attribution
    # input empty, and submitting to the Provider would only produce a useless attribution.
    # Raise ValueError (classified non-retryable by is_retryable_submit_failure) → the skeleton
    # force_fail terminates the job, with no retry and no empty submit.
    missing_fields = [
        field_name
        for field_name, field_value in (("query", attr_query), ("response", attr_response))
        if not field_value or not str(field_value).strip()
    ]
    if missing_fields:
        logger.error(
            "attribution.submit_missing_fields run_id=%s item_id=%s missing=%s",
            run_id,
            item_id,
            missing_fields,
        )
        raise ValueError(f"attribution job data missing required fields: {missing_fields}")

    # Full judge → judge_result; CaseIssue's five fields → issue (hint only for failing dims)
    issue, judge_result = split_attribution_issue_payload(
        issue_raw if isinstance(issue_raw, dict) else {},
    )

    logger.info(
        "attribution.worker_submit_start run_id=%s item_id=%s "
        "provider=%s judge_job_id=%s adiu=%s ds_start=%s "
        "issue_keys=%s verdict=%s dimension=%s issue_hint=%.100s "
        "judge_result_keys=%s issue_dump=%.500s",
        run_id,
        item_id,
        provider_name,
        source_trace_id,
        adiu,
        ds_start,
        list(issue.keys()),
        issue.get("verdict"),
        issue.get("dimension"),
        issue.get("issue_hint"),
        list(judge_result.keys()),
        issue,
    )

    # BullMQ-driven model: no rate_limiter; backpressure is cluster slot +
    # per-node slot (same as external_worker _judge_and_write)
    result: AttributionSubmitResult = await provider.submit(
        AttributionSubmitRequest(
            run_id=run_id,
            item_id=item_id,
            job_id=source_trace_id or "",
            adiu=adiu,
            ds_start=ds_start,
            ds_end=ds_end,
            trace_id=trace_id,
            do_cross_verify=do_cross_verify,
            issue=issue,
            judge_result=judge_result,
            callback_url=settings.attr_callback_url,
            query=attr_query,
            response=attr_response,
            full_tool_calls_json=attr_tool_calls_json,
            # per-run attribution target (§6.4, run config snapshot → enqueue payload passthrough)
            target_url=data.get("attribution_target_url") or None,
        )
    )

    logger.info(
        "attribution.submitted run_id=%s item_id=%s "
        "task_id=%s judge_job_id=%s provider=%s "
        "response_item_id=%s cached=%s deduplicated=%s",
        run_id,
        item_id,
        result.job_id,
        source_trace_id,
        provider_name,
        result.response_item_id,
        result.cached,
        result.deduplicated,
    )

    # Check: the item_id echoed by Aone must match the submitted task_id.
    # Aone's response item_id field echoes params.task_id (= f"{run_id}:{item_id}"),
    # not the database item_id column. A mismatch means job_id may belong to the wrong case.
    expected_task_id = f"{run_id}:{item_id}" if item_id else run_id
    if result.response_item_id and result.response_item_id != expected_task_id:
        logger.error(
            "attribution.submit_item_id_mismatch "
            "run_id=%s item_id=%s expected=%s response_item_id=%s "
            "cached=%s deduplicated=%s "
            "— job_id=%s may not correspond to this case",
            run_id,
            item_id,
            expected_task_id,
            result.response_item_id,
            result.cached,
            result.deduplicated,
            result.job_id,
        )

    return result


async def _upgrade_attribution_pending_callback(
    queue_manager: EvalQueueRegistry,
    upgrade: "_AttributionPendingUpgrade",
) -> str:
    """Upgrade inflight → pending_callback (jobId=case_job_id) and return slot_key.

    Use case_job_id (= {run_id}:{item_id}) as the BullMQ jobId, not the
    attribution_job_id returned by Aone. case_job_id is unique by construction, so even if
    Aone returns the same job_id for different cases they do not overwrite each other in BullMQ.
    """
    run_id = upgrade.run_id
    item_id = upgrade.item_id
    data = upgrade.data
    attribution_job_id = upgrade.attribution_job_id
    case_job_id = f"{run_id}:{item_id}"
    running_q = queue_manager.get_queue(ATTRIBUTION_RUNNING)
    old = await get_queue_job(running_q, case_job_id)

    old_data = old.data if old else data
    slot_key = old_data.get("slot_key") or ""
    payload = {
        **old_data,
        "attribution_job_id": attribution_job_id,
        "submitter_node_id": WORKER_NAME,
        "slot_key": slot_key,
        "cluster_kind": "attr",
    }
    if old is not None:
        await old.remove()
    await add_running_with_retry(
        running_q,
        RunningJobWrite(
            job_id=case_job_id,
            name="pending_callback",
            data=payload,
            attempts=5,
        ),
    )

    logger.info(
        "attribution.pending_callback_add queue=attribution-running job_id=%s run_id=%s item=%s",
        attribution_job_id,
        run_id,
        item_id,
    )
    return slot_key


@dataclass(frozen=True)
class EarlyCallbackContext:
    """Early-callback cleanup context (parameter object for ``_handle_early_attribution_callback``).

    Bundles service deps (run_service/queue_manager/settings) with case location
    (run_id/item_id/attribution_job_id/slot_key).
    """

    run_service: BenchmarkService
    queue_manager: EvalQueueRegistry
    settings: Settings
    run_id: str
    item_id: str
    attribution_job_id: str | None
    slot_key: str


async def _handle_early_attribution_callback(ctx: EarlyCallbackContext) -> None:
    """Early-callback cleanup: the provider already callback'd and counted before pending_callback was written.

    Clean up only while this Worker still holds pending; if the callback already deleted pending,
    do not DECR again (that would double-release with the callback and break multi-node limits).
    """
    if not await ctx.run_service.is_attribution_received(ctx.run_id, ctx.item_id):
        return
    running_q = ctx.queue_manager.get_queue(ATTRIBUTION_RUNNING)
    pending = await get_queue_job(running_q, f"{ctx.run_id}:{ctx.item_id}")
    if pending is None:
        logger.info(
            "attribution.early_callback_already_handled run_id=%s item=%s task_id=%s",
            ctx.run_id,
            ctx.item_id,
            ctx.attribution_job_id,
        )
        return
    await pending.remove()
    if ctx.slot_key:
        await release_slot(str(ctx.slot_key))
    if ctx.settings.attr_cluster_soft_cap > 0:
        await release_cluster_slot("attr")
    wait_q = ctx.queue_manager.get_queue(ATTRIBUTION_WAIT)
    await promote_delayed_for_run(wait_q, ctx.run_id, limit=1)
    logger.info(
        "attribution.early_callback_complete run_id=%s item=%s task_id=%s",
        ctx.run_id,
        ctx.item_id,
        ctx.attribution_job_id,
    )


async def _recover_attribution_pending_callback(
    ctx: ExceptionContext,
    attribution_task_id: str,
) -> None:
    """Submit failed after accept → rewrite pending_callback (slot kept for callback/TTL)."""
    try:
        await add_running_with_retry(
            ctx.running_q,
            RunningJobWrite(
                job_id=attribution_task_id,
                name="pending_callback",
                data={
                    **ctx.data,
                    "attribution_job_id": attribution_task_id,
                    "submitter_node_id": WORKER_NAME,
                    "slot_key": ctx.slot_key,
                    "cluster_kind": "attr",
                },
                attempts=3,
            ),
        )
    except Exception as recover_exc:
        logger.error(
            "attribution.pending_callback_recover_failed run_id=%s item=%s task_id=%s err=%s",
            ctx.data["run_id"],
            ctx.data["item_id"],
            attribution_task_id,
            recover_exc,
        )


@dataclass(frozen=True)
class AttributionWorkerDeps:
    """Service dependency bundle for the attribution worker (parameter object for ``process_attribution_wait_case``).

    Submit depends on the provider registry; ``queue_manager`` is a required skeleton argument and stays outside.
    """

    run_service: BenchmarkService
    attribution_provider_registry: AttributionProviderRegistry | None
    settings: Settings


async def process_attribution_wait_case(
    job: Job,
    token: str,
    *,
    queue_manager: EvalQueueRegistry,
    deps: AttributionWorkerDeps,
) -> None:
    """Consume an ATTRIBUTION_WAIT case job → submit async attribution → move to the running queue.

    Flow (reuses the process_sync_eval_case skeleton):
    1. Terminal guard → retry deadline → cluster slot → per-node slot
    2. register ATTRIBUTION_RUNNING (inflight)
    3. POST provider attribution API → response.job_id = task_id
    4. update PG: benchmark_run_items.attribution_job_id = task_id
    5. promote ATTRIBUTION_RUNNING: inflight → pending_callback (jobId=task_id)
    """
    run_service, attribution_provider_registry, settings = (
        deps.run_service,
        deps.attribution_provider_registry,
        deps.settings,
    )
    attribution_accepted = False
    attribution_task_id: str | None = None

    async def _submit_and_write(
        call: JudgeWriteCall,
    ) -> None:
        nonlocal attribution_accepted, attribution_task_id
        run_id, item_id, data = call.run_id, call.item_id, call.data

        result = await _submit_attribution_to_provider(
            attribution_provider_registry,
            settings,
            _AttributionProviderSubmit(
                run_id=run_id,
                item_id=item_id,
                data=data,
            ),
        )
        attribution_task_id = result.job_id
        attribution_accepted = True

        # 5) Write PG: attribution_job_id = job_id
        # WHERE uses the local item_id (the DB item_id column stores item_id,
        # not the task_id echoed by Aone = f"{run_id}:{item_id}").
        await run_service.update_attribution_job_id(
            run_id=run_id,
            item_id=item_id,
            attribution_job_id=attribution_task_id,
        )

        # 6) Upgrade inflight → pending_callback (jobId=case_job_id) + early-callback cleanup
        slot_key = await _upgrade_attribution_pending_callback(
            queue_manager,
            _AttributionPendingUpgrade(
                run_id=run_id,
                item_id=item_id,
                data=data,
                attribution_job_id=attribution_task_id,
            ),
        )
        await _handle_early_attribution_callback(
            EarlyCallbackContext(
                run_service=run_service,
                queue_manager=queue_manager,
                settings=settings,
                run_id=run_id,
                item_id=item_id,
                attribution_job_id=attribution_task_id,
                slot_key=slot_key,
            ),
        )

    async def _mark_failed_from_wait(
        *,
        data: dict[str, Any],
        reason: str = "retry_deadline",
    ) -> None:
        """Terminal fail on the wait path: increment attribution_received_count (avoid a stuck finalize)."""
        await run_service.increment_attribution_received_once(
            data["run_id"],
            data["item_id"],
        )
        logger.warning(
            "attribution.mark_failed_from_wait run_id=%s item=%s reason=%s",
            data["run_id"],
            data["item_id"],
            reason,
        )

    async def _mark_failed_from_running(
        *,
        data: dict[str, Any],
        exc: Exception,
    ) -> None:
        """Terminal fail on the running path: increment attribution_received_count.

        Signature aligned with external_worker._mark_failed_from_running (*, data, exc).
        Attribution terminal failure does not write cas_advance_external_judge_item (judge
        result is already written); it only increments the counter so finalize does not stick.
        """
        await run_service.increment_attribution_received_once(
            data["run_id"],
            data["item_id"],
        )

    async def _handle_exception(ctx: ExceptionContext) -> None:
        """Classify submit failure (same pattern as judge)."""
        nonlocal attribution_accepted, attribution_task_id

        do_release = should_release_on_submit_failure(
            ctx.exc,
            notification_accepted=attribution_accepted,
        )
        if ctx.state.held and do_release:
            await release_slot(ctx.slot_key)
        if ctx.state.cluster_held and do_release:
            await release_cluster_slot("attr")

        ctx.state.held = False
        ctx.state.cluster_held = False

        run_id = ctx.data["run_id"]
        item_id = ctx.data["item_id"]

        async def _mark_failed(
            d: dict[str, Any],
            *,
            retry_count: int = 0,
            **kw: object,
        ) -> None:
            await run_service.increment_attribution_received_once(
                d["run_id"],
                d["item_id"],
            )

        if attribution_accepted:
            # Already accepted: do not requeue; keep the slot and wait for the callback
            logger.error(
                "attribution.submit_post_accept_fail run_id=%s item=%s "
                "task_id=%s err=%s (slot kept for callback/TTL)",
                run_id,
                item_id,
                attribution_task_id,
                ctx.exc,
            )
            if attribution_task_id:
                await _recover_attribution_pending_callback(
                    ctx,
                    attribution_task_id,
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
                    wait_job_name="attr_case",
                    wait_job=ctx.wait_job,
                    token=ctx.token or None,
                ),
            )
            if outcome == "failed":
                logger.warning(
                    "attribution.submit_terminal_fail run_id=%s item=%s err=%s",
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
                "attribution.submit_force_fail run_id=%s item=%s err=%s",
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

    handlers = SyncEvalHandlers(
        eval_type="attribution",
        wait_queue_name=ATTRIBUTION_WAIT,
        running_queue_name=ATTRIBUTION_RUNNING,
        slot_config=SlotConfig(
            retry_max_duration_ms=settings.external_retry_max_duration_ms,
            cluster_soft_cap=settings.attr_cluster_soft_cap,
            slots_per_run_per_node=settings.attr_slots_per_run_per_node,
            slot_busy_delay_ms=settings.external_slot_busy_delay_ms,
            slot_ttl_s=settings.external_slot_ttl_s,
        ),
        is_run_terminal=lambda rid: _is_run_terminal(rid, run_service),
        heartbeat=run_service.heartbeat,
        judge_and_write=_submit_and_write,
        mark_failed_from_wait=_mark_failed_from_wait,
        mark_failed_from_running=_mark_failed_from_running,
        finalize_fn=_finalize,
        cluster_slot_kind="attr",
        hold_slot_on_success=True,
        remove_running_on_success=False,
        handle_exception=_handle_exception,
    )

    await process_sync_eval_case(
        job,
        token,
        SyncEvalRuntime(
            handlers=handlers,
            queue_manager=queue_manager,
            deps=deps,
        ),
    )


def create_attribution_wait_worker(
    *,
    queue_manager: EvalQueueRegistry,
    run_service: BenchmarkService,
    attribution_provider_registry: AttributionProviderRegistry | None,
    settings: Settings,
) -> Worker:
    """Create the BullMQ Worker for the attribution-wait queue."""
    queue = queue_manager.get_queue(ATTRIBUTION_WAIT)
    opts = queue_manager.build_worker_opts(
        queue_name=ATTRIBUTION_WAIT,
        lock_duration_ms=settings.external_wait_lock_duration_ms,
        stalled_interval_ms=settings.eval_queue_stalled_interval_ms,
    )
    opts["concurrency"] = settings.attr_worker_concurrency

    async def processor(job: Job, token: str) -> None:
        await process_attribution_wait_case(
            job,
            token,
            queue_manager=queue_manager,
            deps=AttributionWorkerDeps(
                run_service=run_service,
                attribution_provider_registry=attribution_provider_registry,
                settings=settings,
            ),
        )

    worker = Worker(queue.name, processor, opts)
    logger.info(
        "attribution_wait_worker.started queue=%s concurrency=%d",
        queue.name,
        settings.attr_worker_concurrency,
    )
    return worker


__all__ = [
    "ATTRIBUTION_RUNNING",
    "ATTRIBUTION_WAIT",
    "AttributionWorkerDeps",
    "create_attribution_wait_worker",
    "process_attribution_wait_case",
]
