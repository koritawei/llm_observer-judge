"""BullMQ job 列表辅助 — 规避空队列时上游 ``getJobs`` 崩溃。

``bullmq.Queue.getJobs`` 在 ``job_ids`` 为空时仍调用 ``asyncio.wait([])``，
触发 ``ValueError: Set of Tasks/Futures is empty.``（空队列 / 新 hash-tag
命名空间下 healer 启动巡检常见）。本模块先 ``getRanges``，无 id 直接返回 []。

另：bullmq-py 的 ``Queue`` **没有** Node 版 ``getJob(id)``，统一走 ``Job.fromId``。
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from bullmq import Job, Queue
from bullmq.utils import extract_result


@dataclass(frozen=True)
class JobListWindow:
    """``list_queue_jobs`` 分页窗口（FUNC-PARAMS-001 参数收敛）。"""

    start: int = 0
    end: int = -1
    asc: bool = False


async def get_queue_job(queue: Queue, job_id: str) -> Job | None:
    """按 jobId 取单个 Job；不存在则返回 ``None``。

    bullmq-py ``Queue`` 无 Node 版 ``getJob``，真实队列统一走 ``Job.fromId``。
    单测里的 ``MagicMock`` 队列仍可通过 ``queue.getJob`` 注入。
    """
    if isinstance(queue, Queue):
        return await Job.fromId(queue, job_id)
    get_job = getattr(queue, "getJob", None)
    if callable(get_job):
        return await get_job(job_id)
    return await Job.fromId(queue, job_id)


async def list_queue_jobs(
    queue: Queue,
    types: list[str],
    window: "JobListWindow | None" = None,
) -> list[Any]:
    """列出指定状态的 job；无 job 时返回空列表（不抛错）。"""
    start = window.start if window is not None else 0
    end = window.end if window is not None else -1
    asc = window.asc if window is not None else False
    current_types = queue.sanitizeJobTypes(types)
    job_ids = await queue.scripts.getRanges(current_types, start, end, asc)
    if not job_ids:
        return []

    tasks = [asyncio.create_task(Job.fromId(queue, job_id)) for job_id in job_ids]
    job_set, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
    jobs = [extract_result(task, queue.emit) for task in job_set]
    jobs = [job for job in jobs if job is not None]

    by_id = {job.id: job for job in jobs}
    return [by_id[job_id] for job_id in job_ids if job_id in by_id]


__all__ = ["JobListWindow", "get_queue_job", "list_queue_jobs"]
