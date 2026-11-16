"""stop-run "truly in flight" predicate, shared by the console inventory and the healer fallback.

Truly in flight =
- BullMQ ``active`` (a worker is executing it), or
- job.name = ``inflight`` (sync / pre-submit placeholder; see worker_skeleton._register_running), or
- job.name = ``pending_callback`` (submitted externally, waiting for the callback result)

Same rule as the running_healer slot sweep: both ``pending_callback`` and ``inflight`` still
hold a slot. Other parked names (unknown orphans) are not in flight and may be cleared on stop.
"""

from collections.abc import Sequence
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.libs.jobqueue.job_listing import list_queue_jobs
from benchmark.queue.queue_factory import (
    ATTRIBUTION_RUNNING,
    BENCHMARK_RUNNING,
    EXTERNAL_RUNNING,
)

logger = get_logger("agent_eval.benchmark.stop_inflight")

#: BullMQ is currently executing this job on a worker.
JOB_TYPES_ACTIVE: list[str] = ["active"]
#: Not yet claimed, or parked, on the running queue (placeholder names: ``STOP_INFLIGHT_JOB_NAMES``).
JOB_TYPES_PARKED: list[str] = ["waiting", "delayed", "paused", "prioritized"]
#: Sync / pre-submit placeholder (worker_skeleton writes it to the running queue).
INFLIGHT_PLACEHOLDER_JOB_NAME = "inflight"
#: External-callback placeholder name: not active, but still waiting for a result.
PENDING_CALLBACK_JOB_NAME = "pending_callback"
#: job.name values that still count as truly in flight while parked (aligned with the healer slot rule).
STOP_INFLIGHT_JOB_NAMES: frozenset[str] = frozenset(
    {
        INFLIGHT_PLACEHOLDER_JOB_NAME,
        PENDING_CALLBACK_JOB_NAME,
    },
)

STOP_RUNNING_JUDGE_QUEUES: tuple[str, ...] = (EXTERNAL_RUNNING, BENCHMARK_RUNNING)
STOP_RUNNING_ATTR_QUEUES: tuple[str, ...] = (ATTRIBUTION_RUNNING,)
STOP_RUNNING_ALL_QUEUES: tuple[str, ...] = (
    *STOP_RUNNING_JUDGE_QUEUES,
    *STOP_RUNNING_ATTR_QUEUES,
)


def job_belongs_to_run(job: Any, run_id: str) -> bool:  # noqa: ANN401 — duck-typed BullMQ job
    data = dict(job.data or {})
    return str(data.get("run_id") or "") == run_id


def is_stop_inflight_job(job: Any, *, listed_as_active: bool) -> bool:  # noqa: ANN401
    """Truly in flight: BullMQ active, or an ``inflight`` / ``pending_callback`` placeholder."""
    if listed_as_active:
        return True
    return str(getattr(job, "name", "") or "") in STOP_INFLIGHT_JOB_NAMES


async def count_stop_inflight_in_queues(
    qm: Any,  # noqa: ANN401 — duck-typed queue manager
    run_id: str,
    queue_names: Sequence[str],
    *,
    fail_closed: bool = True,
) -> int | None:
    """Count truly in-flight jobs for ``run_id`` in the given queues (read-only).

    ``fail_closed=True`` (default): if ``list_queue_jobs`` fails for any queue, return
    ``None`` (unknown). That avoids treating the count as 0, which would make the console
    think nothing is in flight and ``finalize_as_stopped`` immediately. ``False`` skips the
    failed queue (fail-open; use with care).
    """
    total = 0
    for queue_name in queue_names:
        queue = qm.get_queue(queue_name)
        try:
            active_jobs = await list_queue_jobs(queue, JOB_TYPES_ACTIVE)
            parked_jobs = await list_queue_jobs(queue, JOB_TYPES_PARKED)
        except Exception:
            logger.warning(
                "stop_inflight.count_failed queue=%s run_id=%s fail_closed=%s",
                queue_name,
                run_id,
                fail_closed,
                exc_info=True,
            )
            if fail_closed:
                return None
            continue
        for job in active_jobs:
            if job is None or not job_belongs_to_run(job, run_id):
                continue
            if is_stop_inflight_job(job, listed_as_active=True):
                total += 1
        for job in parked_jobs:
            if job is None or not job_belongs_to_run(job, run_id):
                continue
            if is_stop_inflight_job(job, listed_as_active=False):
                total += 1
    return total


async def count_stop_inflight_jobs(
    qm: Any,  # noqa: ANN401
    run_id: str,
    *,
    fail_closed: bool = True,
) -> tuple[int, int] | None:
    """Count truly in-flight jobs on the running queues → (judge, attribution); return ``None`` when unknown."""
    judge = await count_stop_inflight_in_queues(
        qm,
        run_id,
        STOP_RUNNING_JUDGE_QUEUES,
        fail_closed=fail_closed,
    )
    if judge is None:
        return None
    attr = await count_stop_inflight_in_queues(
        qm,
        run_id,
        STOP_RUNNING_ATTR_QUEUES,
        fail_closed=fail_closed,
    )
    if attr is None:
        return None
    return judge, attr


async def run_has_stop_inflight(
    qm: Any,  # noqa: ANN401
    run_id: str,
    *,
    fail_closed: bool = True,
) -> bool:
    """Whether any job is still truly in flight.

    ``fail_closed=True`` (healer default): a list failure counts as still in flight,
    so the target is not dropped by mistake.
    """
    for queue_name in STOP_RUNNING_ALL_QUEUES:
        queue = qm.get_queue(queue_name)
        try:
            active_jobs = await list_queue_jobs(queue, JOB_TYPES_ACTIVE)
            parked_jobs = await list_queue_jobs(queue, JOB_TYPES_PARKED)
        except Exception:
            logger.warning(
                "stop_inflight.list_failed queue=%s run_id=%s",
                queue_name,
                run_id,
                exc_info=True,
            )
            if fail_closed:
                return True
            continue
        for job in active_jobs:
            if job is None or not job_belongs_to_run(job, run_id):
                continue
            if is_stop_inflight_job(job, listed_as_active=True):
                return True
        for job in parked_jobs:
            if job is None or not job_belongs_to_run(job, run_id):
                continue
            if is_stop_inflight_job(job, listed_as_active=False):
                return True
    return False


__all__ = [
    "INFLIGHT_PLACEHOLDER_JOB_NAME",
    "JOB_TYPES_ACTIVE",
    "JOB_TYPES_PARKED",
    "PENDING_CALLBACK_JOB_NAME",
    "STOP_INFLIGHT_JOB_NAMES",
    "STOP_RUNNING_ALL_QUEUES",
    "STOP_RUNNING_ATTR_QUEUES",
    "STOP_RUNNING_JUDGE_QUEUES",
    "count_stop_inflight_in_queues",
    "count_stop_inflight_jobs",
    "is_stop_inflight_job",
    "job_belongs_to_run",
    "run_has_stop_inflight",
]
