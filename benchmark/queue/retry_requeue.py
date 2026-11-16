"""retry_requeue — 失败回插与终败入口（设计文档 §6.4 D4/D17/D18）。

**不变式：只有失败任务可以从 running 插入 wait；成功路径禁止回插。**

终败条件（先触达者生效）：``retry_count >= max_attempts`` **或**
``now >= first_enqueued_at + retry_max_duration_ms``。

三个公开函数：
- ``requeue_or_fail_from_running``：可重试失败 → 校验 running 存在 →
  ``retry_count += 1`` → 次数∨时窗任一触顶则 PG failed / 否则回插 wait
- ``force_fail_from_running``：不可重试失败 → 校验 running 存在 → remove → PG failed（不入 wait）
- ``is_retry_window_exhausted``：Worker 消费前 / slot_busy 路径检测时窗是否耗尽

Worker 路径回插（``wait_job`` + ``token`` 已传）：
对**当前 active wait job** ``updateData`` + ``moveToDelayed`` + 抛
``WaitingChildrenError``（对齐 ``slot_busy``）。禁止对同 jobId 再
``Queue.add``——BullMQ 在 job 仍存在时会静默跳过 add，随后 Worker
``moveToCompleted`` + ``removeOnComplete`` 会把任务删光。

Healer / HTTP 回调路径（无 active wait job）：仍用 ``wait_queue.add``。
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Job, Queue, WaitingChildrenError

from benchmark.settings.setting import Settings
from benchmark.libs.jobqueue.job_listing import get_queue_job

logger = get_logger("agent_eval.eval_queue.retry_requeue")

#: ``mark_failed`` 回调类型：``(data, *, retry_count) -> None``
MarkFailed = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    """失败重试策略（``requeue_or_fail_from_running`` 用）。

    ``max_attempts`` 与退避参数全局唯一（``eval_queue_*`` 配置），
    ``retry_max_duration_ms`` 按队列族指定（sync / external / attr）。
    """

    max_attempts: int
    retry_max_duration_ms: int
    backoff_initial_ms: int
    backoff_max_ms: int

    def backoff_ms(self, retry_count: int) -> int:
        """按当前 retry_count 计算回插 delay（指数退避，钳制在 max_ms）。"""
        return eval_backoff_ms(
            retry_count,
            initial_ms=self.backoff_initial_ms,
            max_ms=self.backoff_max_ms,
        )


def retry_policy_from_settings(
    settings: Settings,
    *,
    retry_max_duration_ms: int,
) -> RetryPolicy:
    """从 Settings 派生重试策略（各 worker / healer 调用点共用）。"""
    return RetryPolicy(
        max_attempts=settings.eval_queue_max_attempts,
        retry_max_duration_ms=retry_max_duration_ms,
        backoff_initial_ms=settings.eval_queue_backoff_initial_ms,
        backoff_max_ms=settings.eval_queue_backoff_max_ms,
    )


@dataclass(frozen=True)
class RequeueDeps:
    """失败回插执行依赖（``requeue_or_fail_from_running`` 参数对象）。

    队列对 + 重试策略 + 终败回调在同一 worker / healer 上下文中
    相对固定，收拢为一个依赖包；每次变化的 ``running_job_id`` 与
    Worker 路径参数（``wait_job`` + ``token``）保持独立传入。
    """

    running_queue: Queue
    wait_queue: Queue
    policy: RetryPolicy
    mark_failed: MarkFailed


@dataclass(frozen=True)
class RequeueOpts:
    """``requeue_or_fail_from_running`` 可选参数（FUNC-PARAMS-001）。

    ``wait_job`` + ``token`` 成对传入走 Worker ``moveToDelayed`` 路径；
    均缺省时走 Healer/回调 ``wait_queue.add`` 路径。
    """

    wait_job_name: str = "case"
    wait_job: Job | None = None
    token: str | None = None


# ── 时窗辅助 ──────────────────────────────────────────────────────────────


def _retry_deadline_ms(data: dict[str, Any], *, max_duration_ms: int) -> int:
    """计算重试截止时刻（ms epoch）。

    ``first_enqueued_at + max_duration_ms``。
    """
    return int(data["first_enqueued_at"]) + max_duration_ms


def stamp_first_running_at(
    data: dict[str, Any],
    *,
    now_ms: int | None = None,
) -> int:
    """首次进入 running 时写入 ``first_enqueued_at``；已有则不改写。

    时钟按 **该 case 第一次进 running** 起算，wait 排队不占用 1h 预算。
    """
    existing = data.get("first_enqueued_at")
    if existing:
        return int(existing)
    ts = int(now_ms if now_ms is not None else time.time() * 1000)
    data["first_enqueued_at"] = ts
    return ts


def is_retry_window_exhausted(
    data: dict[str, Any],
    *,
    max_duration_ms: int,
) -> bool:
    """检测重试时间窗是否已耗尽（D18）。

    自该 case ``first_enqueued_at``（首次进 running）起算。
    字段缺失表示尚未进 running，时钟未开始，返回 False。
    """
    first_enqueued = data.get("first_enqueued_at")
    if first_enqueued is None:
        return False
    now_ms = int(time.time() * 1000)
    return now_ms >= _retry_deadline_ms(data, max_duration_ms=max_duration_ms)


# ── 失败回插 ──────────────────────────────────────────────────────────────


async def requeue_or_fail_from_running(
    deps: RequeueDeps,
    running_job_id: str,
    opts: "RequeueOpts | None" = None,
) -> Literal["requeued", "failed"]:
    """失败回插 wait 或直接终败（§6.4 D4/D17/D18）。

    流程：
    1. ``get_queue_job(running_queue, running_job_id)`` — 无 running 登记则返回 ``"failed"``
       （禁止凭空入 wait）
    2. ``retry_count += 1`` 写回 payload
    3. 回插保留 ``first_enqueued_at``（缺失则按 running job 时间补打点）
    4. ``remove()`` running job
    5. 若 ``retry_count >= max_attempts`` **或** 已逾时窗 → ``mark_failed`` → ``"failed"``
    6. delay 钳制在截止时刻内；``remaining <= 0`` → ``mark_failed`` → ``"failed"``
    7. 回插 wait：
       - Worker 路径（``wait_job`` + ``token``）：``updateData`` + ``moveToDelayed``，
         然后抛 ``WaitingChildrenError``（不返回）
       - Healer/回调路径：``wait_queue.add``（同 case jobId）→ ``"requeued"``

    Args:
        deps: 回插执行依赖（running/wait 队列 + 重试策略 + 终败回调）。
        running_job_id: running job ID（``{run_id}:{item_id}`` 或异步 ``{notification_job_id}``）。
            同值双名：调用侧 wait 语境变量名 ``case_job_id``，此处按 running
            队列语境命名 —— wait/running 两队列共用同一 jobId（§3 D9 ID 归一）。
        wait_job_name: healer/回调路径 ``Queue.add`` 的 job name（``opts`` 字段）。
        wait_job: Worker 正在处理的 wait job（与 ``token`` 成对传入；``opts`` 字段）。
        token: Worker lock token；与 ``wait_job`` 成对传入时走 ``moveToDelayed``
            （``opts`` 字段）。

    Returns:
        ``"requeued"`` 如果已 ``Queue.add`` 回插；``"failed"`` 如果已终败或无 running。
        Worker ``moveToDelayed`` 成功时抛 ``WaitingChildrenError``，不返回。
    """
    o = opts or RequeueOpts()
    wait_job_name = o.wait_job_name
    wait_job = o.wait_job
    token = o.token
    job = await get_queue_job(deps.running_queue, running_job_id)
    if job is None:
        # 无 running 登记则不得入 wait（防止凭空重试）
        logger.warning(
            "requeue.no_running_job job_id=%s → failed (no running)",
            running_job_id,
        )
        return "failed"

    data = dict(job.data)
    now_ms = int(time.time() * 1000)
    retry_count = int(data.get("retry_count", 0)) + 1
    data["retry_count"] = retry_count

    # 回插不得改写 first_enqueued_at；缺失则用 running job 时间补打点
    if not data.get("first_enqueued_at"):
        raw_ts = getattr(job, "timestamp", None)
        fallback = int(raw_ts) if isinstance(raw_ts, (int, float)) and raw_ts > 0 else now_ms
        stamp_first_running_at(data, now_ms=fallback)

    await job.remove()

    # 终败条件：次数超限 ∨ 时窗耗尽（先触达者，D18）
    policy = deps.policy
    over_attempts = retry_count >= policy.max_attempts
    over_deadline = now_ms >= _retry_deadline_ms(
        data,
        max_duration_ms=policy.retry_max_duration_ms,
    )
    if over_attempts or over_deadline:
        reason = "max_attempts" if over_attempts else "retry_deadline"
        await deps.mark_failed(data, retry_count=retry_count, reason=reason)
        logger.info(
            "requeue.terminal_failed job_id=%s retry_count=%d reason=%s "
            "max_attempts=%d retry_max_duration_ms=%d",
            running_job_id,
            retry_count,
            reason,
            policy.max_attempts,
            policy.retry_max_duration_ms,
        )
        return "failed"

    # delay 钳制：不得越过截止时刻（D18）
    delay = policy.backoff_ms(retry_count)
    remaining = (
        _retry_deadline_ms(
            data,
            max_duration_ms=policy.retry_max_duration_ms,
        )
        - now_ms
    )
    if remaining <= 0:
        await deps.mark_failed(
            data,
            retry_count=retry_count,
            reason="retry_deadline",
        )
        logger.info(
            "requeue.terminal_failed job_id=%s retry_count=%d reason=retry_deadline (remaining<=0)",
            running_job_id,
            retry_count,
        )
        return "failed"
    delay = min(delay, remaining)

    case_job_id = f"{data['run_id']}:{data['item_id']}"

    # Worker 路径：当前 wait job 仍 active，同 jobId 的 Queue.add 会被静默丢弃。
    if wait_job is not None and token is not None:
        await wait_job.updateData(data)
        timestamp = now_ms + delay
        await wait_job.scripts.moveToDelayed(
            wait_job.id,
            timestamp,
            delay,
            token,
        )
        logger.info(
            "requeue.delayed job_id=%s → wait retry_count=%d delay=%dms "
            "first_enqueued_at=%d remaining=%dms",
            case_job_id,
            retry_count,
            delay,
            int(data["first_enqueued_at"]),
            remaining,
        )
        # 阻止 Worker moveToCompleted + removeOnComplete 删光任务
        raise WaitingChildrenError()

    # Healer / 回调路径：wait 上无同 id active job，可安全 add
    await deps.wait_queue.add(
        wait_job_name,
        data,
        {
            "jobId": case_job_id,
            "attempts": 1,
            "delay": delay,
            "removeOnComplete": True,
            "removeOnFail": True,
        },
    )
    logger.info(
        "requeue.requeued job_id=%s → wait retry_count=%d delay=%dms "
        "first_enqueued_at=%d remaining=%dms",
        running_job_id,
        retry_count,
        delay,
        int(data["first_enqueued_at"]),
        remaining,
    )
    return "requeued"


async def force_fail_from_running(
    *,
    running_queue: Queue,
    running_job_id: str,
    mark_failed: MarkFailed,
) -> Literal["failed"]:
    """不可重试失败：校验 running 存在 → remove → PG failed（不入 wait）。

    用于 submit 不可重试失败、回调不可重试失败等场景（§6.4）。

    Args:
        running_queue: running 队列实例。
        running_job_id: running job ID。
        mark_failed: 终败回调。

    Returns:
        ``"failed"``
    """
    job = await get_queue_job(running_queue, running_job_id)
    if job is not None:
        data = dict(job.data)
        await job.remove()
        await mark_failed(data, retry_count=int(data.get("retry_count", 0)))
        logger.info(
            "force_fail.done job_id=%s retry_count=%d",
            running_job_id,
            int(data.get("retry_count", 0)),
        )
    else:
        logger.warning(
            "force_fail.no_running_job job_id=%s",
            running_job_id,
        )
    return "failed"


def eval_backoff_ms(
    retry_count: int,
    *,
    initial_ms: int = 5000,
    max_ms: int = 25000,
) -> int:
    """指数退避 delay（非 BullMQ attempts backoff）。

    ``delay = initial * 2^(retry_count - 1)``，上限 ``max_ms``。

    Args:
        retry_count: 当前失败次数（≥1）。
        initial_ms: 初始 delay（``settings.eval_queue_backoff_initial_ms``）。
        max_ms: delay 上限（``settings.eval_queue_backoff_max_ms``）。

    Returns:
        delay 毫秒数。
    """
    delay = initial_ms * (2 ** (retry_count - 1))
    return min(delay, max_ms)


__all__ = [
    "RequeueDeps",
    "RequeueOpts",
    "RetryPolicy",
    "eval_backoff_ms",
    "force_fail_from_running",
    "is_retry_window_exhausted",
    "requeue_or_fail_from_running",
    "retry_policy_from_settings",
    "stamp_first_running_at",
]
