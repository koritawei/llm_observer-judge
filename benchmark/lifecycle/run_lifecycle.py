"""run 级生命周期公共件 — benchmark_runs 上的同构操作。

run 表拥有相同的生命周期字段（``status`` / ``owner`` /
``heartbeat_at`` / ``created_at`` / ``finished_at`` / ``error_message``），
本模块收敛其上的逐行同构操作：心跳、判失败、owner 在途清扫、stale 恢复。

测试契约约束：``model`` / 状态枚举值（``failed_status`` /
``non_terminal_statuses``）与 ``logger`` 由业务侧绑定于
:class:`RunLifecycleBinding` 传入；日志 event 字符串由调用方提供，
保持各侧契约不变。
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.agenthost.observability.logger.factory import LazyLoggerProxy
from benchmark.libs.models import BaseModel

from benchmark.constant import RUN_STALE_TIMEOUT_S

logger = get_logger("agent_eval.eval_runtime.run_lifecycle")

#: recover_stale_runs 写入的 error_message（两侧文案一致）
STALE_RECOVERY_REASON = "recovered as stale (heartbeat timeout / crash / deploy incomplete)"


class _RunFailService(Protocol):
    async def fail_run(self, *, run_id: str, reason: str) -> object: ...


@dataclass(frozen=True)
class RunLifecycleBinding:
    """业务侧 run 表生命周期绑定（fail / recover 系列共用参数对象）。

    ``model`` 由业务侧模块命名空间读取后绑定；状态枚举值与
    ``logger`` 随绑定一次提供，调用方只传每次变化的
    run_id / reason / 日志 event 字符串。
    """

    model: type[BaseModel]
    failed_status: str
    non_terminal_statuses: list[str]
    logger: LazyLoggerProxy


async def heartbeat_run(
    model: type[BaseModel],
    run_id: str,
    *,
    logger: LazyLoggerProxy,
    no_row_log_message: str,
) -> None:
    """Update ``heartbeat_at`` to current UTC time（无行时 warn）。"""
    updated = await model.filter(run_id=run_id).update(
        heartbeat_at=datetime.now(UTC),
    )
    if updated == 0:
        logger.warning(no_row_log_message, run_id)


@dataclass(frozen=True)
class FailRunSpec:
    """``fail_run`` 失败参数（FUNC-PARAMS-001 参数收敛）。

    日志 event 字符串由调用方提供，保持各侧契约不变。
    """

    reason: str
    no_row_log_message: str
    failed_log_message: str


async def fail_run(
    binding: RunLifecycleBinding,
    run_id: str,
    spec: FailRunSpec,
) -> None:
    """Mark a run as FAILED with an error message（幂等：无条件 UPDATE）。"""
    reason = spec.reason
    updated = await binding.model.filter(run_id=run_id).update(
        status=binding.failed_status,
        error_message=reason,
        finished_at=datetime.now(UTC),
    )
    if updated == 0:
        binding.logger.error(spec.no_row_log_message, run_id, reason)
        return
    binding.logger.warning(spec.failed_log_message, run_id, reason)


async def fail_inflight_runs(
    binding: RunLifecycleBinding,
    *,
    owner: str,
    reason: str,
    log_message: str,
) -> list[str]:
    """Immediately mark all non-terminal runs for ``owner`` as FAILED.

    Used on graceful shutdown / drain cancel so queued or mid-flight runs
    are not left as ``initializing`` / ``running`` after the process exits.

    Returns:
        List of failed ``run_id`` values (caller can use to fail orphaned
        run items via ``_fail_run_items``).
    """
    runs = [
        r
        async for r in binding.model.filter(
            owner=owner,
            status__in=binding.non_terminal_statuses,
        )
    ]
    now = datetime.now(UTC)
    failed_ids: list[str] = []
    for run in runs:
        await binding.model.filter(run_id=run.run_id).update(
            status=binding.failed_status,
            error_message=reason,
            finished_at=now,
        )
        binding.logger.warning(log_message, run.run_id, owner, reason)
        failed_ids.append(run.run_id)
    return failed_ids


async def recover_stale_runs(
    binding: RunLifecycleBinding,
    *,
    stale_timeout_s: int = RUN_STALE_TIMEOUT_S,
    log_message: str,
) -> list[str]:
    """Mark stale non-terminal runs as FAILED (timeout / crash / deploy).

    A run is considered stale if:
    - ``heartbeat_at`` older than ``stale_timeout_s``, OR
    - ``heartbeat_at`` is NULL and ``created_at`` older than cutoff

    两个查询按 ``heartbeat_at`` 非空 / 为空互斥，结果集不相交，无需去重。
    Covers force-kill and rolling deploy: the old process dies, heartbeats
    stop, and the next startup recovers leftover rows. Does NOT re-run.

    Returns:
        List of recovered ``run_id`` values (caller can use to fail orphaned
        run items via ``_fail_run_items``).
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=stale_timeout_s)

    heartbeat_qs = binding.model.filter(
        heartbeat_at__lt=cutoff,
        status__in=binding.non_terminal_statuses,
    )
    never_heartbeat_qs = binding.model.filter(
        heartbeat_at__isnull=True,
        created_at__lt=cutoff,
        status__in=binding.non_terminal_statuses,
    )

    stale_runs = []
    for qs in (heartbeat_qs, never_heartbeat_qs):
        async for r in qs:
            stale_runs.append(r)

    now = datetime.now(UTC)
    recovered_ids: list[str] = []
    for run in stale_runs:
        # Save the pre-recovery status first: update_from_dict mutates the
        # instance in-place, so run.status already reads failed_status after it.
        prev_status = run.status
        await run.update_from_dict(
            {
                "status": binding.failed_status,
                "error_message": STALE_RECOVERY_REASON,
                "finished_at": now,
            }
        ).save()
        binding.logger.warning(
            log_message,
            run.run_id,
            run.owner,
            run.heartbeat_at,
            prev_status,
        )
        recovered_ids.append(run.run_id)

    return recovered_ids


async def fail_eval_run(
    run_service: _RunFailService,
    run_id: str,
    reason: str,
    *,
    log_prefix: str,
) -> None:
    """Best-effort mark run as failed — never raises（runner 异常善后共用）。"""
    try:
        await run_service.fail_run(run_id=run_id, reason=reason)
    except Exception:
        logger.exception("%s.fail_run_failed run_id=%s", log_prefix, run_id)


__all__ = [
    "RUN_STALE_TIMEOUT_S",
    "STALE_RECOVERY_REASON",
    "RunLifecycleBinding",
    "fail_eval_run",
    "fail_inflight_runs",
    "fail_run",
    "heartbeat_run",
    "recover_stale_runs",
]
