"""worker_skeleton — eval case 公共 processor 骨架。

benchmark_worker、dataset_worker 和 external_worker 的 processor 流程 ~90% 相同：
终态守卫 → retry deadline → cluster soft_cap → per-node slot →
register running → judge callback → 异常处理 → finalize。

本模块提取公共骨架，各 worker 仅实现差异化的 judge / mark_failed / exception 回调。
external_worker 的异步回调差异通过 ``hold_slot_on_success`` /
``remove_running_on_success`` / ``handle_exception`` 三个可选字段适配。
"""

import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Job, Queue, WaitingChildrenError

from benchmark.libs.jobqueue.job_listing import get_queue_job
from benchmark.queue.eval_case import EvalCase
from benchmark.queue.inflight_slot import (
    SlotKeySpec,
    acquire_cluster_slot,
    acquire_slot,
    build_probe_key,
    build_slot_key,
    release_cluster_slot,
    release_slot,
)
from benchmark.queue.retry_requeue import (
    RequeueDeps,
    RequeueOpts,
    is_retry_window_exhausted,
    requeue_or_fail_from_running,
    retry_policy_from_settings,
    stamp_first_running_at,
)
from benchmark.queue.slot_busy import (
    SlotBusyDelay,
    SlotProbePolicy,
    delay_for_slot_busy,
)

if TYPE_CHECKING:
    from typing import Protocol

    from benchmark.settings.setting import Settings
    from benchmark.queue.queue_factory import EvalQueueRegistry

    class WorkerDepsLike(Protocol):
        """``deps`` 参数最小协议：骨架仅读取 ``settings``，其余字段不解读。

        ``settings`` 声明为只读 property：各 worker 的 deps 均为 frozen
        dataclass（只读字段），可变属性 Protocol 会被 mypy 拒绝（LSP）。
        """

        @property
        def settings(self) -> "Settings": ...


logger = get_logger("agent_eval.eval_queue.worker_skeleton")

#: 5xx 视为服务端瞬时错误（4xx 归为不可恢复的业务错误）。
_HTTP_SERVER_ERROR_MIN = 500

#: 节点标识（槽位键隔离）
WORKER_NAME = socket.gethostname()

#: judge_and_write 发现 fused 行已终态时写入 job data，骨架释放槽位而非 hold。
SKIP_HOLD_SLOT_FLAG = "skip_hold_slot"


# ── Slot 配置 ─────────────────────────────────────────────────────────────


@dataclass
class SlotConfig:
    """各 worker 差异化的槽位设置。

    sync workers 从 ``settings.sync_*`` 构造；
    external worker 从 ``settings.external_*`` 构造。
    """

    retry_max_duration_ms: int
    """重试时间窗上限（``settings.sync_retry_max_duration_ms`` 等）。"""

    cluster_soft_cap: int
    """集群级软顶（≤0 禁用）。"""

    slots_per_run_per_node: int
    """每节点每 run 并发槽位上限（D7）。"""

    slot_busy_delay_ms: int
    """槽位忙时 wait delay 基准（D15）。"""

    slot_ttl_s: int
    """Redis 槽位 key TTL（秒）。

    sync 用 ``eval_queue_lock_duration_ms // 1000``；
    external 用 ``external_running_ttl_ms // 1000``。
    """


# ── 可变槽位状态 ──────────────────────────────────────────────────────────


@dataclass
class SlotState:
    """可变槽位状态 — 骨架与自定义异常处理器共享。

    ``held`` / ``cluster_held`` 在骨架内部用局部变量管理，
    但自定义 ``handle_exception`` 回调需要修改这些标志，
    因此用 dataclass 封装为可变引用。
    """

    held: bool = False
    cluster_held: bool = False


# ── 异常处理上下文 ─────────────────────────────────────────────────────────


@dataclass
class ExceptionContext:
    """自定义异常处理器的上下文。

    传入 ``SyncEvalHandlers.handle_exception`` 时，骨架将异常信息和
    槽位 / 队列 / 状态全部打包到此 dataclass，回调可按需修改
    ``state.held`` / ``state.cluster_held`` 以控制 finally 释放行为。
    """

    exc: Exception
    slot_key: str
    cluster_slot_kind: str
    running_q: Any
    wait_q: Any
    case_job_id: str
    data: dict[str, Any]
    settings: Any
    state: SlotState
    wait_job: Any = None
    """当前正在处理的 wait ``Job``（回插须 ``moveToDelayed``）。"""
    token: str = ""
    """Worker lock token，与 ``wait_job`` 成对传给 ``requeue_or_fail``。"""


# ── Handler 定义 ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgeWriteCall:
    """``SyncEvalHandlers.judge_and_write`` 入参（FUNC-PARAMS-001 参数收敛）。

    ``deps`` 为各 worker 自定义的服务依赖包（``BenchmarkWorkerDeps`` /
    ``DatasetWorkerDeps``），骨架不解读、只透传。
    """

    run_id: str
    item_id: str
    case: EvalCase
    data: dict[str, Any]
    deps: Any


@dataclass
class SyncEvalHandlers:
    """Worker-specific callbacks for eval case processing.

    Each worker constructs an instance with its own judge logic,
    failure handling, and queue names. The common skeleton calls
    these callbacks at the appropriate phases.
    """

    eval_type: str
    """``"benchmark"`` / ``"dataset"`` / ``"external"`` — for logging and slot key."""

    wait_queue_name: str
    """BullMQ wait queue name (e.g. ``BENCHMARK_CASES``)."""

    running_queue_name: str
    """BullMQ running queue name (e.g. ``BENCHMARK_RUNNING``)."""

    slot_config: SlotConfig
    """Slot settings — each worker constructs from its own settings prefix."""

    is_run_terminal: Callable[[str], Awaitable[bool]]
    """Check if the run is in terminal status."""

    heartbeat: Callable[[str], Awaitable[None]]
    """Send heartbeat for the run."""

    judge_and_write: Callable[..., Awaitable[None]]
    """Execute judge logic and write results to PG.

    Called with a :class:`JudgeWriteCall` argument
    （``deps`` 为各 worker 自定义的服务依赖包，骨架不解读、只透传）。
    """

    mark_failed_from_wait: Callable[..., Awaitable[None]]
    """Mark item as failed from wait path (before entering running).

    Writes input/output to PG so report details aren't empty.
    Called with: ``data``, ``reason``.
    """

    mark_failed_from_running: Callable[..., Awaitable[None]]
    """Mark item as failed from running path (after judge exception).

    Called with: ``data``, ``exc``.
    """

    finalize_fn: Callable[[str], Awaitable[None]]
    """Call ``tick_and_maybe_finalize`` for the run."""

    cluster_slot_kind: str = "sync"
    """Cluster slot kind (``"sync"`` for benchmark/dataset, ``"judge"`` for external)."""

    hold_slot_on_success: bool = False
    """If True, on success keep slots in Redis (set held=False without releasing).

    External worker uses this to keep slots for Aone callback / TTL.
    """

    remove_running_on_success: bool = True
    """If False, don't remove the running job on success.

    External worker upgrades inflight → pending_callback instead of removing.
    """

    handle_exception: Callable[[ExceptionContext], Awaitable[None]] | None = field(
        default=None,
    )
    """Custom exception handler.

    If provided, called instead of the default requeue_or_fail logic.
    The callback receives an :class:`ExceptionContext` with all needed state.
    Modify ``ctx.state.held`` / ``ctx.state.cluster_held`` to control
    whether the finally block releases slots.

    If None, the skeleton uses the default sync behavior:
    unconditional slot release + ``requeue_or_fail_from_running``.
    """


# ── 公共 processor 骨架 ───────────────────────────────────────────────────


@dataclass(frozen=True)
class SyncEvalRuntime:
    """``process_sync_eval_case`` 运行时组件（FUNC-PARAMS-001 参数收敛）。

    ``deps`` 为各 worker 自定义的服务依赖包（benchmark/dataset 传
    评测三服务 + settings，external/attribution 传 Aone/registry
    + run_service + settings）：骨架仅读取 ``deps.settings``，
    其余字段透传给 ``handlers.judge_and_write``。
    """

    handlers: SyncEvalHandlers
    queue_manager: "EvalQueueRegistry"
    deps: "WorkerDepsLike"


@dataclass
class _CaseContext:
    """``process_sync_eval_case`` 的流程上下文（辅助函数参数打包）。

    ``state`` 为可变槽位状态，与 :class:`ExceptionContext.state`
    共享同一实例，finally 释放行为由 ``state`` 标志驱动。
    """

    run_id: str
    item_id: str
    case_job_id: str
    data: dict[str, Any]
    slot_key: str
    probe_key: str
    cluster_kind: str
    token: str
    state: SlotState


async def _fail_from_wait(
    handlers: SyncEvalHandlers,
    ctx: _CaseContext,
    reason: str,
) -> None:
    """wait 路径失败收尾：``mark_failed_from_wait`` + ``finalize_fn``。"""
    await handlers.mark_failed_from_wait(data=ctx.data, reason=reason)
    await handlers.finalize_fn(ctx.run_id)


async def _should_skip_for_terminal_run(
    handlers: SyncEvalHandlers,
    ctx: _CaseContext,
) -> bool:
    """run 终态守卫：终态（或终态检查失败）时返回 True，调用方直接 return。"""
    eval_type = handlers.eval_type
    try:
        is_terminal = await handlers.is_run_terminal(ctx.run_id)
    except Exception as terminal_exc:
        logger.warning(
            "%s.run_terminal_check_failed run_id=%s item=%s err=%s",
            eval_type,
            ctx.run_id,
            ctx.item_id,
            terminal_exc,
        )
        await _fail_from_wait(
            handlers,
            ctx,
            reason=f"run_terminal_check_failed: {terminal_exc}",
        )
        return True

    if is_terminal:
        logger.info(
            "%s.case_skipped_terminal run_id=%s item=%s",
            eval_type,
            ctx.run_id,
            ctx.item_id,
        )
        return True
    return False


@dataclass(frozen=True)
class _SlotBusyAction:
    """``_delay_or_fail_on_slot_busy`` 动作参数（FUNC-PARAMS-001 参数收敛）。

    ``busy_kind`` 为 ``"cluster_slot_busy"`` / ``"slot_busy"``，用作
    日志与 failed reason 前缀；``max_probes`` 为探测限流上限
    （``settings.max_slot_probes_per_run``）。
    """

    busy_kind: str
    max_probes: int


async def _delay_or_fail_on_slot_busy(
    handlers: SyncEvalHandlers,
    job: Job,
    ctx: _CaseContext,
    action: _SlotBusyAction,
) -> None:
    """槽位忙时的等待/失败处理（cluster 与 per-node 共用）。

    所有路径处理完毕后调用方直接 return
    （``WaitingChildrenError`` 原样冒泡给 BullMQ）。
    """
    eval_type = handlers.eval_type
    sc = handlers.slot_config
    busy_kind = action.busy_kind
    if is_retry_window_exhausted(
        ctx.data,
        max_duration_ms=sc.retry_max_duration_ms,
    ):
        logger.info(
            "%s.%s_retry_deadline run_id=%s item=%s",
            eval_type,
            busy_kind,
            ctx.run_id,
            ctx.item_id,
        )
        await _fail_from_wait(handlers, ctx, reason="retry_deadline")
        return

    try:
        await delay_for_slot_busy(
            job,
            SlotBusyDelay(
                run_id=ctx.run_id,
                base_delay_ms=sc.slot_busy_delay_ms,
                token=ctx.token,
                probe=SlotProbePolicy(
                    ctx.probe_key,
                    max_probes=action.max_probes,
                ),
            ),
        )
    except WaitingChildrenError:
        raise
    except Exception as delay_exc:
        # moveToDelayed 失败：先写 PG failed 行，避免 removeOnFail 删光任务导致丢数据
        logger.warning(
            "%s.%s_delay_failed run_id=%s item=%s err=%s",
            eval_type,
            busy_kind,
            ctx.run_id,
            ctx.item_id,
            delay_exc,
        )
        await _fail_from_wait(
            handlers,
            ctx,
            reason=f"{busy_kind}_delay_failed: {delay_exc}",
        )


async def _heartbeat_or_warn(
    handlers: SyncEvalHandlers,
    ctx: _CaseContext,
) -> None:
    """发送 run 心跳，失败仅告警（不中断 case 处理）。"""
    try:
        await handlers.heartbeat(ctx.run_id)
    except Exception as hb_exc:
        logger.warning(
            "%s.heartbeat_failed run_id=%s item=%s err=%s",
            handlers.eval_type,
            ctx.run_id,
            ctx.item_id,
            hb_exc,
        )


async def _register_running(
    job: Job,
    handlers: SyncEvalHandlers,
    queue_manager: "EvalQueueRegistry",
    ctx: _CaseContext,
) -> tuple[Queue, Queue]:
    """注册 inflight running 记录，返回 ``(running_q, wait_q)``。"""
    eval_type = handlers.eval_type
    # 该 case 首次进 running 才打 first_enqueued_at；回插 wait 不得改写。
    stamp_first_running_at(ctx.data)
    try:
        await job.updateData(ctx.data)
    except Exception:
        logger.warning(
            "%s.stamp_first_running_update_failed run_id=%s item=%s",
            eval_type,
            ctx.run_id,
            ctx.item_id,
            exc_info=True,
        )
    running_q = queue_manager.get_queue(handlers.running_queue_name)
    wait_q = queue_manager.get_queue(handlers.wait_queue_name)
    await running_q.add(
        "inflight",
        {
            **ctx.data,
            "submitter_node_id": WORKER_NAME,
            "slot_key": ctx.slot_key,
            "cluster_kind": ctx.cluster_kind,
        },
        {"jobId": ctx.case_job_id, "removeOnComplete": False, "removeOnFail": False},
    )
    logger.info(
        "bullmq.running_add queue=%s job_id=%s run_id=%s item=%s",
        handlers.running_queue_name,
        ctx.case_job_id,
        ctx.run_id,
        ctx.item_id,
    )
    return running_q, wait_q


async def _finish_success(
    handlers: SyncEvalHandlers,
    running_q: Queue,
    ctx: _CaseContext,
) -> None:
    """成功路径：移除 running 记录（可选）+ 保留槽位（可选）+ 完成日志。"""
    eval_type = handlers.eval_type
    if ctx.data.get(SKIP_HOLD_SLOT_FLAG):
        # 行已终态（回调抢先落地）：清 inflight，交给 finally 释放本 worker 槽位。
        rj = await get_queue_job(running_q, ctx.case_job_id)
        if rj is not None:
            await rj.remove()
        logger.info(
            "%s.skip_hold_already_terminal run_id=%s item=%s",
            eval_type,
            ctx.run_id,
            ctx.item_id,
        )
        return
    if handlers.remove_running_on_success:
        rj = await get_queue_job(running_q, ctx.case_job_id)
        logger.info(
            "bullmq.running_get_job job_id=%s found=%s run_id=%s item=%s",
            ctx.case_job_id,
            rj is not None,
            ctx.run_id,
            ctx.item_id,
        )
        if rj is not None:
            await rj.remove()
            logger.info(
                "bullmq.running_removed job_id=%s run_id=%s item=%s",
                ctx.case_job_id,
                ctx.run_id,
                ctx.item_id,
            )

    if handlers.hold_slot_on_success:
        # Keep slots in Redis for callback / TTL — mark as not held
        # so the finally block doesn't release them.
        ctx.state.held = False
        ctx.state.cluster_held = False

    logger.info(
        "%s.case_completed run_id=%s item=%s",
        eval_type,
        ctx.run_id,
        ctx.item_id,
    )


async def _handle_judge_exception(
    handlers: SyncEvalHandlers,
    ctx: ExceptionContext,
) -> None:
    """judge 异常处理：自定义 handler，或默认「释放槽位 + requeue_or_fail」。"""
    data = ctx.data
    run_id: str = data["run_id"]
    item_id: str = data["item_id"]
    eval_type = handlers.eval_type
    state = ctx.state

    if handlers.handle_exception is not None:
        # Custom exception handler (e.g. external_worker submit failure).
        # 失败回退默认逻辑：放任异常外泄会让 running inflight 作业无人善后
        # （僵尸直到 TTL 过期，finally 只释放槽位不动 running 队列）。
        try:
            await handlers.handle_exception(ctx)
            return
        except Exception as handler_exc:
            logger.warning(
                "%s.custom_handle_exception_failed run_id=%s item=%s "
                "judge_err=%s handler_err=%s fallback=default",
                eval_type,
                run_id,
                item_id,
                ctx.exc,
                handler_exc,
            )
            # fall through：默认释放槽位 + requeue_or_fail（报 ctx.exc 原始异常）

    # Default sync behavior: unconditional release + requeue_or_fail
    if state.held:
        await release_slot(ctx.slot_key)
        state.held = False
    if state.cluster_held:
        await release_cluster_slot(ctx.cluster_slot_kind)
        state.cluster_held = False

    # Permanent error (4xx EvaluatorError etc.) — skip requeue, mark failed directly.
    # Requeueing a metadata_missing or similar 4xx error creates an infinite
    # retry loop because the root cause (e.g. empty agent_name) never changes.
    exc_code = getattr(ctx.exc, "code", None)
    if isinstance(exc_code, int) and exc_code < _HTTP_SERVER_ERROR_MIN:
        logger.warning(
            "%s.case_permanent_fail run_id=%s item=%s error_code=%s err=%s",
            eval_type,
            run_id,
            item_id,
            exc_code,
            ctx.exc,
        )
        await handlers.mark_failed_from_running(data=data, exc=ctx.exc)
        return

    # 异常经 ``ctx.exc``（dataclass 属性）持有引用——``except ... as`` 变量
    # 在块外被隐式 del，闭包经属性读取才能在块外安全调用。
    async def _mark_failed(
        d: dict[str, Any],
        *,
        retry_count: int = 0,
        **_kw: object,
    ) -> None:
        await handlers.mark_failed_from_running(data=d, exc=ctx.exc)

    # wait_job+token：moveToDelayed，避免同 jobId Queue.add 静默丢任务
    outcome = await requeue_or_fail_from_running(
        RequeueDeps(
            running_queue=ctx.running_q,
            wait_queue=ctx.wait_q,
            policy=retry_policy_from_settings(
                ctx.settings,
                retry_max_duration_ms=handlers.slot_config.retry_max_duration_ms,
            ),
            mark_failed=_mark_failed,
        ),
        # 同值双名：wait 侧 case_job_id ≡ running 侧 running_job_id
        # （两队列共用 jobId={run_id}:{item_id}，§3 D9 ID 归一）
        ctx.case_job_id,
        RequeueOpts(wait_job=ctx.wait_job, token=ctx.token),
    )
    if outcome == "failed":
        logger.warning(
            "%s.case_terminal_fail run_id=%s item=%s err=%s",
            eval_type,
            run_id,
            item_id,
            ctx.exc,
        )


async def process_sync_eval_case(
    job: Job,
    token: str,
    runtime: SyncEvalRuntime,
) -> None:
    """Common processor for sync (benchmark/dataset) and external eval cases.

    Flow:
        1. run terminal guard
        2. retry deadline → fail from wait (writes input/output)
        3. cluster soft_cap (if ``slot_config.cluster_soft_cap > 0``)
        4. per-node slot
        5. register running (inflight)
        6. judge callback (worker-specific)
        7. success: remove from running (if ``remove_running_on_success``)
        8. exception → custom handler or default requeue_or_fail
        9. finalize

    v2.1 retry model: ``attempts=1`` + payload ``retry_count`` +
    running→wait requeue.

    ``runtime.deps`` 为各 worker 自定义的服务依赖包，骨架仅读取
    ``deps.settings``，其余字段透传给 ``handlers.judge_and_write``。
    """
    handlers = runtime.handlers
    queue_manager = runtime.queue_manager
    deps = runtime.deps
    data = job.data
    run_id: str = data["run_id"]
    item_id: str = data["item_id"]
    eval_type = handlers.eval_type
    cluster_kind = handlers.cluster_slot_kind
    sc = handlers.slot_config
    slot_key = build_slot_key(
        queue_manager.prefix,
        SlotKeySpec(
            kind=cluster_kind,
            node=WORKER_NAME,
            run_id=run_id,
            eval_type=eval_type,
        ),
    )
    probe_key = build_probe_key(
        queue_manager.prefix,
        kind=cluster_kind,
        node=WORKER_NAME,
        run_id=run_id,
    )
    state = SlotState()
    ctx = _CaseContext(
        run_id=run_id,
        item_id=item_id,
        case_job_id=f"{run_id}:{item_id}",
        data=data,
        slot_key=slot_key,
        probe_key=probe_key,
        cluster_kind=cluster_kind,
        token=token,
        state=state,
    )

    logger.info(
        "bullmq.job_received queue=%s job_id=%s run_id=%s item=%s retry_count=%s",
        handlers.wait_queue_name,
        job.id,
        run_id,
        item_id,
        data.get("retry_count", 0),
    )

    # 1) run terminal guard (DB query — protect against transient failures)
    if await _should_skip_for_terminal_run(handlers, ctx):
        return

    # 1.5) retry deadline → fail from wait (writes input/output to PG)
    if is_retry_window_exhausted(
        data,
        max_duration_ms=sc.retry_max_duration_ms,
    ):
        logger.info(
            "%s.case_retry_deadline run_id=%s item=%s first_enqueued_at=%s",
            eval_type,
            run_id,
            item_id,
            data.get("first_enqueued_at"),
        )
        await _fail_from_wait(handlers, ctx, reason="retry_deadline")
        return

    # 2) cluster soft_cap (D14)
    soft_cap = sc.cluster_soft_cap
    if soft_cap > 0:
        if not await acquire_cluster_slot(
            cluster_kind,
            soft_cap=soft_cap,
            ttl_s=sc.slot_ttl_s,
        ):
            logger.info(
                "%s.cluster_slot_busy run_id=%s item=%s soft_cap=%d",
                eval_type,
                run_id,
                item_id,
                soft_cap,
            )
            await _delay_or_fail_on_slot_busy(
                handlers,
                job,
                ctx,
                _SlotBusyAction(
                    busy_kind="cluster_slot_busy",
                    max_probes=deps.settings.max_slot_probes_per_run,
                ),
            )
            return
        state.cluster_held = True

    # 3) per-node slot (D7)
    if not await acquire_slot(
        slot_key,
        limit=sc.slots_per_run_per_node,
        ttl_s=sc.slot_ttl_s,
    ):
        if state.cluster_held:
            await release_cluster_slot(cluster_kind)
            state.cluster_held = False
        await _delay_or_fail_on_slot_busy(
            handlers,
            job,
            ctx,
            _SlotBusyAction(
                busy_kind="slot_busy",
                max_probes=deps.settings.max_slot_probes_per_run,
            ),
        )
        return
    state.held = True

    await _heartbeat_or_warn(handlers, ctx)

    # 4) register running (inflight)
    running_q, wait_q = await _register_running(job, handlers, queue_manager, ctx)
    case = EvalCase.from_payload(data)

    try:
        # 5) judge callback (worker-specific)
        await handlers.judge_and_write(
            JudgeWriteCall(
                run_id=run_id,
                item_id=item_id,
                case=case,
                data=data,
                deps=deps,
            ),
        )

        # 6) success path
        await _finish_success(handlers, running_q, ctx)

    except Exception as exc:
        await _handle_judge_exception(
            handlers,
            ExceptionContext(
                exc=exc,
                slot_key=slot_key,
                cluster_slot_kind=cluster_kind,
                running_q=running_q,
                wait_q=wait_q,
                case_job_id=ctx.case_job_id,
                data=data,
                settings=deps.settings,
                state=state,
                wait_job=job,
                token=token,
            ),
        )

    finally:
        if state.held:
            await release_slot(slot_key)
        if state.cluster_held:
            await release_cluster_slot(cluster_kind)
        await handlers.finalize_fn(run_id)


__all__ = [
    "ExceptionContext",
    "SKIP_HOLD_SLOT_FLAG",
    "SlotConfig",
    "SlotState",
    "SyncEvalHandlers",
    "SyncEvalRuntime",
    "process_sync_eval_case",
]
