"""**agent-eval execution side** of WebUI scheduled evals (APScheduler).

Responsibility split (plan A):
- **WebUI / console**: only writes tasks to the DB (``kind=eval_task``); no cron, no lock.
- **This module (agent-eval)**: loads/re-registers cron from the DB in-process, leader ``fire``
  at the slot, and catch-up after restart; it does not depend on another service process to schedule.

Execution surface:

1. Every 60s (and startup ``run_now``) ``sync_jobs``: read DB → add/update/delete APScheduler
   cron (daily 02:00 / weekly Monday 02:00 / custom 5-field cron, Asia/Shanghai).
2. **At the end of the same tick** ``catch_up_missed_tasks``: only fill "**today** past the slot
   and still no run" (same-day missed-fire recovery). **Next-day on-time fire must use cron**,
   not catch_up and not a LocalEval / other-module fallback. After today's run is done, any
   restart before the next-day slot only re-registers cron (``next_run`` = next slot); catch_up
   no-ops because a run already exists or the slot has not arrived.
3. At the slot ``fire_task``: reuse ``_accept_external_run`` (after accept, pick the judge from
   the capability chain; LocalEval is a sync executor, not a schedule fallback); run_id =
   ``{benchmark}-{yyyymmdd}``; same agent same day uses ``reuse_existing=True``.
4. Realtime sampling datasets are not re-activated on a schedule.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from benchmark.libs.agenthost.observability.logger import get_logger
from fastapi.responses import JSONResponse

from benchmark.records.models.benchmark_run import BenchmarkRun
from benchmark.domain.benchmark_console import (
    find_agent_document_by_id,
    find_dataset_document,
    find_task_document,
    list_schedulable_tasks,
    record_task_run,
)
from benchmark.domain.report_storage_config import get_default_download_prefix
from benchmark.runs.common.config_helpers import resolve_oss_prefix
from benchmark.runs.common.oss_date_macro import (
    DEFAULT_BENCHMARK_CSV,
    prepare_oss_prefix_for_load,
)
from benchmark.runs.manual_run_id import (
    resolve_external_data_date,
    scheduled_run_id,
)
from benchmark.runs.run_lifecycle import ListRunsQuery
from benchmark.runs.schema import ExternalBenchmarkRequest
from benchmark.workers.scheduler_hub import (
    TICK_SECONDS,
    SchedulerHub,
    TickJobOptions,
)

if TYPE_CHECKING:
    from benchmark.libs.agenthost.app import AppRuntime
    from apscheduler.triggers.cron import CronTrigger  # noqa: TID251

    from benchmark.runs.run_service import BenchmarkService
    from benchmark.workers.external_runner import ExternalBenchmarkRunner

logger = get_logger("agent_eval.eval_workers.webui_task_scheduler")

#: Fixed slot for daily / weekly tasks (page copy: every day 02:00, every Monday 02:00)
_DEFAULT_TRIGGER_HOUR = 2
_DEFAULT_TRIGGER_MINUTE = 0

#: Schedule timezone: the page is fixed to a Chinese context, so cron is interpreted in UTC+8
_SCHEDULE_TZ_NAME = "Asia/Shanghai"
_SCHEDULE_TZ_FALLBACK = ZoneInfo("UTC")

_TASK_JOB_PREFIX = "webui-eval-task:"
_SYNC_JOB_ID = "webui-eval-task-sync"
_CATCHUP_JOB_ID = "webui-eval-task-startup-catchup"


def _schedule_tz() -> ZoneInfo:
    try:
        return ZoneInfo(_SCHEDULE_TZ_NAME)
    except (ZoneInfoNotFoundError, ValueError):  # pragma: no cover - when system tzdata is missing
        return _SCHEDULE_TZ_FALLBACK


def build_task_trigger(
    trigger: str,
    cron_expression: str | None,
) -> "CronTrigger | None":
    """Map a task document's trigger into an APScheduler CronTrigger (None if illegal)."""
    # APScheduler trigger types share the same TID251 waiver (see the scheduler_hub comment).
    from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415, TID251

    tz = _schedule_tz()
    if trigger == "daily":
        return CronTrigger(
            hour=_DEFAULT_TRIGGER_HOUR,
            minute=_DEFAULT_TRIGGER_MINUTE,
            timezone=tz,
        )
    if trigger == "weekly":
        return CronTrigger(
            day_of_week="mon",
            hour=_DEFAULT_TRIGGER_HOUR,
            minute=_DEFAULT_TRIGGER_MINUTE,
            timezone=tz,
        )
    if trigger == "cron":
        expression = (cron_expression or "").strip()
        if not expression:
            return None
        try:
            return CronTrigger.from_crontab(expression, timezone=tz)
        except ValueError:
            logger.error(
                "webui_task_scheduler.invalid_cron trigger=cron expression=%r",
                expression,
            )
            return None
    return None


def _today_fire_already_due(
    trigger: "CronTrigger",
    now: datetime,
    tz: ZoneInfo,
) -> bool:
    """Whether today's schedule slot has already passed (startup catch-up; yesterday's slot excluded)."""
    start_of_today = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    # Count from "1 second before today 00:00" to get the first fire today (or later)
    next_fire = trigger.get_next_fire_time(None, start_of_today - timedelta(seconds=1))
    if next_fire is None or next_fire > now:
        return False
    return next_fire.astimezone(tz).date() == now.astimezone(tz).date()


@dataclass(frozen=True)
class WebuiTaskSchedulerDeps:
    """Constructor dependencies for ``WebuiTaskScheduler`` (FUNC-PARAMS-001)."""

    hub: SchedulerHub
    benchmark_service: "BenchmarkService"
    runtime: "AppRuntime"
    external_runner: "ExternalBenchmarkRunner"


@dataclass(frozen=True)
class _FireTarget:
    """Fire context after ``fire_task`` prechecks pass."""

    task: Any
    dataset: Any
    agent: Any


@dataclass(frozen=True)
class _FireStorageTarget:
    """Data day and already-expanded OSS location for a scheduled fire (prevent a second accept offset)."""

    data_date: str
    oss_prefix: str | None = None
    oss_object_key: str | None = None


def _resolve_fire_storage(dataset: object, *, now: datetime) -> _FireStorageTarget | None:
    """The schedule day is the macro base; the expanded literal partition day drives ``data_date`` / ``run_id``.

    Same order as ``daily_scheduler``: expand ``${yyyymmdd±n}`` from the UTC+8 schedule day,
    then take the partition day from the literal path. Write the expanded prefix/key onto the
    request so accept does not expand again from the offset day after the partition day is locked.
    Upload-style ``oss_partition_date`` drives the date directly and does not pre-expand the prefix
    (the accept chain ``_apply_dataset_oss_path`` builds the key).

    Returns:
        The parsed result; ``None`` when the prefix/macro is illegal (the caller skips this fire).
    """
    schedule_day = now.strftime("%Y%m%d")
    partition = getattr(dataset, "oss_partition_date", None)
    if partition:
        date = resolve_external_data_date(
            None,
            now=now,
            storage_path=str(partition),
        )
        return _FireStorageTarget(data_date=date)

    raw_prefix = (getattr(dataset, "oss_prefix", None) or "").strip()
    if not raw_prefix:
        return _FireStorageTarget(data_date=schedule_day)

    try:
        normalized = resolve_oss_prefix(
            raw_prefix,
            default_prefix=get_default_download_prefix(),
        )
        effective, object_key = prepare_oss_prefix_for_load(
            normalized,
            base_date=schedule_day,
            file_name=DEFAULT_BENCHMARK_CSV,
        )
    except ValueError as exc:
        logger.error(
            "webui_task_scheduler.oss_prefix_invalid dataset=%s error=%s",
            getattr(dataset, "name", None) or getattr(dataset, "id", None),
            exc,
        )
        return None

    date = resolve_external_data_date(
        schedule_day,
        now=now,
        storage_path=object_key or effective,
    )
    return _FireStorageTarget(
        data_date=date,
        oss_prefix=effective,
        oss_object_key=object_key,
    )


class WebuiTaskScheduler:
    """Register cron jobs from webUI task documents and fire eval runs at the slot."""

    def __init__(self, deps: WebuiTaskSchedulerDeps) -> None:
        self._hub = deps.hub
        self._benchmark_service = deps.benchmark_service
        self._runtime = deps.runtime
        self._external_runner = deps.external_runner
        #: job_id → signature (trigger/cron/agent/dataset); rebuild the job only when the signature changes
        self._signatures: dict[str, str] = {}

    async def start(self) -> None:
        """Start: the hub must already have been started by lifespan after QM is ready.

        sync tick ``run_now=True`` → immediately re-register every cron from the DB; catch_up is
        embedded in the tick. Also attach a one-shot catchup so startup can fill within seconds
        (skipped when not leader; a later sync covers it).
        """
        await self._hub.start()
        self._hub.add_tick(
            _SYNC_JOB_ID,
            self.sync_jobs,
            TickJobOptions(seconds=TICK_SECONDS, run_now=True),
        )
        self._hub.add_once(_CATCHUP_JOB_ID, self.catch_up_missed_tasks)
        logger.warning(
            "webui_task_scheduler.started tick_s=%d "
            "(db=eval_task; sync+catchup on leader; webui writes db only)",
            TICK_SECONDS,
        )

    async def stop(self) -> None:
        for job_id in list(self._signatures):
            self._hub.remove(job_id)
        self._signatures.clear()
        self._hub.remove(_SYNC_JOB_ID)
        self._hub.remove(_CATCHUP_JOB_ID)
        logger.info("webui_task_scheduler.stopped")

    # ── task → job sync ─────────────────────────────────────────────

    async def sync_jobs(self) -> None:
        """Reload schedulable tasks from the DB into APScheduler, and catch up today's missed fires.

        Only the Redis leader enters this method (hub ``_wrap``). After a process restart any
        instance that takes the lock finishes "re-register + catch_up" within ≤60s; no long-lived leader.
        """
        try:
            tasks = await list_schedulable_tasks()
        except Exception:
            logger.exception("webui_task_scheduler.list_tasks_failed")
            return
        desired: dict[str, tuple[str, CronTrigger | None]] = {}
        for task in tasks:
            job_id = f"{_TASK_JOB_PREFIX}{task.id}"
            signature = (
                f"{task.trigger}|{task.cron_expression or ''}|{task.agent_id}|{task.dataset_id}"
            )
            desired[job_id] = (signature, build_task_trigger(task.trigger, task.cron_expression))
        for job_id in list(self._signatures):
            if job_id not in desired:
                self._hub.remove(job_id)
                self._signatures.pop(job_id, None)
                logger.info("webui_task_scheduler.job_removed job_id=%s", job_id)
        for job_id, (signature, trigger) in desired.items():
            # Illegal/unsupported trigger: remove the old cron so it does not keep firing on a stale expression
            if trigger is None:
                if self._signatures.pop(job_id, None) is not None:
                    self._hub.remove(job_id)
                    logger.warning("webui_task_scheduler.job_disabled job_id=%s", job_id)
                continue
            # Signature unchanged and next_run still set → keep the next slot APScheduler already computed,
            # avoiding replace jitter; if next_run is missing (paused), force a re-register.
            if (
                self._signatures.get(job_id) == signature
                and self._hub.next_run_time(job_id) is not None
            ):
                continue
            task_id = job_id[len(_TASK_JOB_PREFIX) :]
            self._hub.add_cron(
                job_id,
                partial(self.fire_task, task_id),
                trigger,
            )
            self._signatures[job_id] = signature
            next_run = self._hub.next_run_time(job_id)
            logger.warning(
                "webui_task_scheduler.job_registered task_id=%s trigger=%s next_run=%s",
                task_id,
                signature,
                next_run.isoformat() if next_run is not None else None,
            )
            if next_run is None:
                logger.error(
                    "webui_task_scheduler.job_paused task_id=%s "
                    "(next_run_time=None; cron will not fire until re-registered)",
                    task_id,
                )
        # Plan A: each leader sync ends by filling "today" missed fires; it no-ops before the next-day
        # slot. Next-day on-time fire still depends on the cron registered above (including misfire_grace).
        await self.catch_up_missed_tasks()

    # ── startup catch-up ────────────────────────────────────────────

    async def catch_up_missed_tasks(self) -> None:
        """Catch-up: only batch tasks whose "today slot has passed" and that still have no run today.

        Judge by "the next fire from today 00:00", not from now-24h — otherwise yesterday 11:00
        looks already due, and a restart before today 11:00 would falsely fire today's run.

        Idempotent: skip when a same-day run exists (including failed); only the leader enters.
        """
        try:
            tasks = await list_schedulable_tasks()
        except Exception:
            logger.exception("webui_task_scheduler.catchup_list_failed")
            return
        tz = _schedule_tz()
        now = datetime.now(tz)
        schedule_day = now.strftime("%Y%m%d")
        fired = 0
        for task in tasks:
            if task.status == "stopped":
                continue
            trigger = build_task_trigger(task.trigger, task.cron_expression)
            if trigger is None:
                continue
            if not _today_fire_already_due(trigger, now, tz):
                continue
            try:
                dataset = await find_dataset_document(task.dataset_id)
            except Exception:
                logger.exception(
                    "webui_task_scheduler.catchup_dataset_failed task_id=%s",
                    task.id,
                )
                continue
            if dataset is None or dataset.source == "realtime":
                continue
            storage = _resolve_fire_storage(dataset, now=now)
            if storage is None:
                continue
            # Same rule as fire_task: dedupe by the expected partition day (${yyyymmdd-1} → T-1)
            if await self._has_run_for_date(task.id, storage.data_date):
                continue
            logger.warning(
                "webui_task_scheduler.catchup_fire task_id=%s data_date=%s schedule_day=%s",
                task.id,
                storage.data_date,
                schedule_day,
            )
            await self.fire_task(task.id)
            fired += 1
        if fired:
            logger.warning(
                "webui_task_scheduler.catchup_done schedule_day=%s fired=%d",
                schedule_day,
                fired,
            )

    async def _has_run_for_date(self, task_id: str, data_date: str) -> bool:
        """Any run today (including failed) counts as already executed; do not auto-retry at startup."""
        runs = await self._benchmark_service.list_runs(ListRunsQuery(task_id=task_id, limit=100))
        for run in runs:
            config = run.config_json if isinstance(run.config_json, dict) else {}
            if str(config.get("data_date") or "") == data_date:
                return True
            if f"-{data_date}" in run.run_id:
                return True
        return False

    # ── fire ────────────────────────────────────────────────────────

    async def _batch_execution_aliases(self, benchmark_name: str) -> dict[str, str]:
        """Batch judge / attribution aliases declared by the component (including streaming→evaluator_service_judge fallback).

        Same lookup as webui manualRun (``batch_trigger_execution_aliases``).
        When the component declares no capability chain (page-dynamic agent), do not guess a
        default; keep the resolver's original behavior.
        """
        from benchmark.runs.common.agent_capabilities import (  # noqa: PLC0415
            batch_trigger_execution_aliases,
        )
        from benchmark.runs.common.dynamic_component import (  # noqa: PLC0415
            ensure_dynamic_benchmark_component,
        )

        try:
            services = await ensure_dynamic_benchmark_component(benchmark_name, self._runtime)
        except (KeyError, ValueError):
            # Let _accept_external_run report 404/400 uniformly (the log carries the full reason)
            return {}
        declared = getattr(type(services), "benchmark_capabilities", None) or ()
        aliases = batch_trigger_execution_aliases(declared)
        if aliases:
            logger.info(
                "webui_task_scheduler.execution_aliases benchmark=%s aliases=%s",
                benchmark_name,
                aliases,
            )
        return aliases

    async def _resolve_fire_target(self, task_id: str) -> _FireTarget | None:
        """Load a fireable task/dataset/agent; log and return None when it cannot fire."""
        task = await find_task_document(task_id)
        if task is None:
            logger.info("webui_task_scheduler.task_gone task_id=%s", task_id)
            return None
        if task.status == "stopped":
            logger.info("webui_task_scheduler.task_stopped task_id=%s", task_id)
            return None
        dataset = await find_dataset_document(task.dataset_id)
        if dataset is None:
            logger.error(
                "webui_task_scheduler.dataset_missing task_id=%s dataset_id=%s",
                task_id,
                task.dataset_id,
            )
            return None
        if dataset.source == "realtime":
            # Realtime sampling is a resident pipeline (one activation keeps consuming); a scheduled re-activation only double-registers
            logger.info(
                "webui_task_scheduler.skip_realtime task_id=%s dataset=%s",
                task_id,
                dataset.name,
            )
            return None
        agent = await find_agent_document_by_id(task.agent_id)
        if agent is None:
            logger.error(
                "webui_task_scheduler.agent_missing task_id=%s agent_id=%s",
                task_id,
                task.agent_id,
            )
            return None
        return _FireTarget(task=task, dataset=dataset, agent=agent)

    async def fire_task(self, task_id: str) -> None:
        """Fire one eval run at the slot (run_id = ``{benchmark}-{date}``, aligned with RemoteConfig)."""
        target = await self._resolve_fire_target(task_id)
        if target is None:
            return
        task, dataset, agent = target.task, target.dataset, target.agent

        now = datetime.now(_schedule_tz())
        storage = _resolve_fire_storage(dataset, now=now)
        if storage is None:
            return
        date = storage.data_date
        # Execution-class aliases match webui immediate fire: use the component's batch judge/attribution aliases.
        # Without them the default async judge is used, and a benchmark whose rubric declares
        # judge_execution_policy=local (such as search_eval) is rejected by the resolver with 400.
        aliases = await self._batch_execution_aliases(agent.name)
        payload = ExternalBenchmarkRequest(
            benchmark_name=agent.name,
            date=date,
            dataset_id=task.dataset_id,
            dataset_name=dataset.name,
            eval_task_id=task.id,
            agent_registration_id=task.agent_id,
            generate_report=True,
            oss_prefix=storage.oss_prefix,
            oss_object_key=storage.oss_object_key,
            **aliases,
        )
        # Same shape as RemoteConfig external-daily / scheduled_task daily schedules; reuse within the day.
        run_id = scheduled_run_id(agent.name, date)
        logger.info(
            "webui_task_scheduler.fire task_id=%s benchmark=%s dataset=%s date=%s run_id=%s",
            task.id,
            agent.name,
            dataset.name,
            date,
            run_id,
        )
        try:
            accepted = await self._accept_run(run_id, payload=payload, date=date)
        except Exception:
            logger.exception("webui_task_scheduler.fire_failed task_id=%s", task.id)
            return
        if accepted:
            await self._record_run(task.id, run_id)

    async def _accept_run(
        self,
        run_id: str,
        *,
        payload: ExternalBenchmarkRequest,
        date: str,
    ) -> bool:
        """Reuse external accept; reuse within the day. failed/4xx JSONResponse → False (no write-back)."""
        from benchmark.routes.benchmark import (  # noqa: PLC0415
            AcceptExternalRunInput,
            _accept_external_run,
        )

        result: Any = await _accept_external_run(
            payload,
            AcceptExternalRunInput(
                benchmark_service=self._benchmark_service,
                runtime=self._runtime,
                external_runner=self._external_runner,
                assigned_run_id=run_id,
                data_date=date,
                reuse_existing=True,
            ),
        )
        if isinstance(result, JSONResponse):
            logger.warning(
                "webui_task_scheduler.accept_rejected task_id=%s run_id=%s status=%s",
                payload.eval_task_id,
                run_id,
                result.status_code,
            )
            return False
        return True

    async def _record_run(self, task_id: str, run_id: str) -> None:
        """Write back the task document's latest run (basis for task-list status / report linkage)."""
        try:
            run = await BenchmarkRun.filter(run_id=run_id, deleted_at__isnull=True).first()
            if run is None:
                logger.warning(
                    "webui_task_scheduler.run_not_found task_id=%s run_id=%s",
                    task_id,
                    run_id,
                )
                return
            config = run.config_json if isinstance(run.config_json, dict) else {}
            owned_task = str(config.get("eval_task_id") or "").strip()
            if owned_task and owned_task != task_id:
                logger.warning(
                    "webui_task_scheduler.record_run_ownership_mismatch "
                    "task_id=%s run_id=%s owned_task=%s",
                    task_id,
                    run_id,
                    owned_task,
                )
                return
            await record_task_run(task_id, run_id=run_id, run_pk=str(run.id))
        except Exception:
            # Write-back failure does not affect the run; the next page sync can still group by run.taskId
            logger.exception(
                "webui_task_scheduler.record_run_failed task_id=%s run_id=%s",
                task_id,
                run_id,
            )


__all__ = ["WebuiTaskScheduler", "build_task_trigger"]
