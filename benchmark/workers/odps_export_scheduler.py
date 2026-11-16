"""OdpsExportScheduler — daily scheduler for ODPS sampled export.

In-process asyncio scheduler: fire at each RemoteConfig ``jobs[]`` entry's
``schedule_time`` / ``schedule_timezone`` (``id`` optional, default ``default``).
Legacy top-level ``sql_template`` / ``schedule_*`` remain compatible.

Multi-host dedupe and idempotency (every key carries ``queue_prefix``; daily/pre/prod isolation):

- Execution lock ``{prefix}:odps_export:{job_id}:{oss_ds}`` (``SET NX EX``, TTL 6h) —
  one node in the cluster runs a given job on a given day; DEL on failure allows retry;
  if it crashes mid-run, another node can take over after TTL expiry.
- Done marker ``{prefix}:odps_export:done:{job_id}:{oss_ds}`` (TTL 48h) —
  written after a successful export; checked before firing, **do not run again if already done today**
  (covers a restart/release startup compensation that would otherwise re-export after the lock expires).
- Lock-held retry: a slot that sees lock held does not sleep until tomorrow; recheck every 30min until
  the lock TTL expires (harmless if the holder is running normally; if it crashed without release,
  takeover happens on TTL expiry — otherwise the day's export is lost entirely, the 20260923 incident).
- Redis unavailable: skip the whole round (fail-safe).

In-flight task lifetime: the startup compensation task is tracked (``_compensation_task``) and
``stop()`` cancels it together with the main loop; cancel during export releases the execution lock
(another node's startup compensation can take over immediately, without waiting for TTL).

Date convention: ODPS partition ``odps_ds`` = **T-1**; OSS path ``oss_ds`` = **T**.
"""

import asyncio
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.odps_export_config import (
    OdpsExportConfig,
    OdpsExportJob,
    fetch_odps_export_config,
)
from benchmark.reporting.framework.odps_export import (
    OdpsExportRequest,
    default_odps_ds,
    default_oss_ds,
    run_odps_export,
)
from benchmark.workers.daily_scheduler import (
    in_schedule_due_window,
    parse_hh_mm,
    schedule_slot_elapsed,
)
from benchmark.workers.scheduler_hub import TICK_SECONDS, SchedulerHub, TickJobOptions

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger("agent_eval.eval_workers.odps_export_scheduler")

_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

#: Execution-lock TTL: covers the run window (crash mid-run; another node can rerun after TTL expiry).
_LOCK_TTL_S = 6 * 3600

#: Done-marker TTL: same-day idempotency across restart/release (key includes oss_ds, so the next day is a new key).
#: Separate from the execution lock: the lock dedupes hosts (short TTL, takeover); done means "already ran, skip" (long TTL).
_DONE_TTL_S = 48 * 3600

#: Failure-backoff TTL: aligned with the lock recheck interval. Under ``schedule_slot_elapsed`` + a 60s tick,
#: a job that keeps failing would rerun ODPS every minute without backoff; a short-TTL fail key throttles retries.
_FAIL_BACKOFF_TTL_S = 30 * 60

#: Lock-held recheck interval (aligned with daily_scheduler's 30min no-data recheck cadence)
_LOCK_HELD_RETRY_INTERVAL_S = 30 * 60
#: Lock-held recheck window: takeover is possible once the execution-lock TTL expires; past the window, wait for tomorrow's slot
_LOCK_HELD_RETRY_WINDOW_S = _LOCK_TTL_S + _LOCK_HELD_RETRY_INTERVAL_S

#: Node identity (lock value, for debugging)
_WORKER_NAME = socket.gethostname()

#: APScheduler job id (unique within the process)
_TICK_JOB_ID = "odps-export-tick"

#: Only the holder may release the execution lock (compare-and-delete, same pattern as the in-repo ieval dedupe Lua):
#: after TTL expiry and takeover, the original node's release does not delete the new holder's lock
#: (value mismatch → return 0 and do not delete).
_RELEASE_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


@dataclass(frozen=True)
class _ExportTask:
    """Per-job arguments for ``_export_one_job`` (FUNC-PARAMS-001)."""

    job: OdpsExportJob
    odps_ds: str
    oss_ds: str
    reason: str


class OdpsExportScheduler:
    """Daily scheduler for ODPS sampled export (multiple jobs + Redis execution lock)."""

    def __init__(
        self,
        *,
        config_fetcher: Callable[..., OdpsExportConfig] = fetch_odps_export_config,
        config_refresher: Callable[[], Awaitable[None]] | None = None,
        hub: SchedulerHub | None = None,
    ) -> None:
        self._config_fetcher = config_fetcher
        self._config_refresher = config_refresher
        #: Scheduled fires all go through APScheduler (60s tick reuses the existing due check)
        self._hub = hub or SchedulerHub()
        self._owns_hub = hub is None
        self._compensation_task: asyncio.Task[None] | None = None
        self._stopped = False

    async def _reload(self) -> None:
        if self._config_refresher is not None:
            await self._config_refresher()

    async def start(self) -> None:
        """Start the scheduler: APScheduler tick + startup compensation."""
        await self._reload()
        config = self._fetch_config()
        if config.enabled is False:
            logger.info("odps_export_scheduler.disabled (enabled=false)")
            return
        if not config.scheduler_enabled():
            logger.info("odps_export_scheduler.waiting (no runnable jobs yet)")

        self._stopped = False
        await self._hub.start()
        self._hub.add_tick(_TICK_JOB_ID, self._tick, TickJobOptions(seconds=TICK_SECONDS))
        jobs = config.enabled_jobs()
        if not jobs:
            return
        logger.info(
            "odps_export_scheduler.started jobs=%s tick_s=%d (apscheduler)",
            [f"{j.id}@{j.schedule_time}/{j.schedule_timezone}" for j in jobs],
            TICK_SECONDS,
        )
        # Track the compensation task so stop() cancels it too (an orphan is killed when the script
        # process exits and leaves the execution lock held, blocking other nodes; the 20260923 incident)
        self._compensation_task = asyncio.create_task(
            self._guarded_retry_locked_exports(reason_first="startup_compensation"),
            name="odps-export-startup-compensation",
        )

    async def stop(self) -> None:
        """Stop the scheduler: remove the tick job + cancel in-flight compensation (also close a self-owned hub)."""
        self._stopped = True
        self._hub.remove(_TICK_JOB_ID)
        if self._compensation_task is not None:
            self._compensation_task.cancel()
            try:
                await self._compensation_task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Guard: if the task ends with an exception other than CancelledError, do not abort shutdown
                logger.exception("odps_export_scheduler.stop_task_error")
            self._compensation_task = None
        if self._owns_hub:
            await self._hub.shutdown()
        logger.info("odps_export_scheduler.stopped")

    def _fetch_config(self) -> OdpsExportConfig:
        return self._config_fetcher(use_cache=False)

    async def _tick(self) -> None:
        """APScheduler interval tick: reload the DB snapshot and run one due check."""
        await self._reload()
        config = self._fetch_config()
        if config.enabled is False:
            logger.info("odps_export_scheduler.disabled at runtime, skip tick")
            return
        if not config.scheduler_enabled():
            return
        await self._retry_locked_exports(reason_first="scheduled")

    async def _retry_locked_exports(self, *, reason_first: str) -> None:
        """Fire exports; when the execution lock is held, recheck on a short period until TTL expiry and do not drop today's window.

        Holder running normally → recheck is harmless (done marker / lock contention covers it); crash
        without release → the only recovery is takeover after lock TTL expiry. The old code slept until
        the next day's slot on lock held, so a dead holder lost the whole day's export — the 20260923
        incident: an ops-script node took the lock then was killed on process exit (no done marker), the
        resident node skipped at the 10:30 slot (lock held) and slept 24h, that day's search-eval CSV was
        missing, and every external trigger raised DailyNoDataError.

        Recheck scope is narrow: only jobs skipped for the lock in the first due batch (``job_ids``
        filter), and the due window stays closed — a job outside its due window is not exported early
        by a recheck (its T-1 partition may not be ready; an early success would write a wrong done
        marker and skip the real slot that day).
        """
        pending = await self._trigger_once(reason=reason_first)
        deadline = time.monotonic() + _LOCK_HELD_RETRY_WINDOW_S
        while pending and not self._stopped and time.monotonic() < deadline:
            logger.info(
                "odps_export_scheduler.lock_held_retry next_in=%ds window_s=%d",
                _LOCK_HELD_RETRY_INTERVAL_S,
                _LOCK_HELD_RETRY_WINDOW_S,
            )
            await asyncio.sleep(_LOCK_HELD_RETRY_INTERVAL_S)
            if self._stopped:
                return
            pending = await self._trigger_once(reason="lock_retry", job_ids=pending)
        if pending and not self._stopped:
            logger.warning(
                "odps_export_scheduler.lock_held_retry_exhausted window_s=%d "
                "— give up, wait for next scheduled trigger",
                _LOCK_HELD_RETRY_WINDOW_S,
            )

    async def _guarded_retry_locked_exports(self, *, reason_first: str) -> None:
        """Exception guard for startup compensation (same try/except Exception semantics as the old ``_startup_compensation``):
        transient faults such as the config center are only logged and not propagated, and the exception
        is not stored on the task — so a non-CancelledError does not escape shutdown when stop() awaits.
        """
        try:
            await self._retry_locked_exports(reason_first=reason_first)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("odps_export_scheduler.compensation_error")

    @staticmethod
    def _zone(tz_name: str) -> tzinfo:
        try:
            return ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning(
                "odps_export_scheduler.invalid_timezone tz=%s fallback=Asia/Shanghai",
                tz_name,
            )
            return _SHANGHAI_TZ

    def _seconds_until_job_trigger(self, job: OdpsExportJob) -> float:
        hour, minute = parse_hh_mm(job.schedule_time)
        now = datetime.now(self._zone(job.schedule_timezone))
        next_trigger = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if schedule_slot_elapsed(now, next_trigger):
            next_trigger += timedelta(days=1)
        return max((next_trigger - now).total_seconds(), 1.0)

    def _calc_sleep_until_next_trigger(self, config: OdpsExportConfig) -> float:
        """Sleep seconds until the nearest slot among all enabled jobs."""
        jobs = config.enabled_jobs()
        if not jobs:
            # Fallback: top-level schedule (should not enter the loop with no jobs, but keep it testable)
            hour, minute = parse_hh_mm(config.schedule_time)
            now = datetime.now(self._zone(config.schedule_timezone))
            next_trigger = now.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )
            if schedule_slot_elapsed(now, next_trigger):
                next_trigger += timedelta(days=1)
            return max((next_trigger - now).total_seconds(), 1.0)
        return min(self._seconds_until_job_trigger(job) for job in jobs)

    def _job_due_now(self, job: OdpsExportJob, *, reason: str) -> bool:
        """Full startup compensation; scheduled runs jobs whose slot has arrived.

        Early wake (left edge of the window) or a slot already passed (late today) both count as due:
        a strict 2-minute window lets an earlier job's export duration push a later job out of the window,
        and ``_seconds_until_job_trigger`` then defers it to the next day, losing the whole day's export
        (the same class of bug as trace_report / daily_scheduler).
        """
        if reason != "scheduled":
            return True
        hour, minute = parse_hh_mm(job.schedule_time)
        now = datetime.now(self._zone(job.schedule_timezone))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return in_schedule_due_window(now, target) or schedule_slot_elapsed(now, target)

    @staticmethod
    def _prefixed(key: str) -> str:
        """Join the environment prefix (YAML ``queue_prefix``, isolating daily/pre/prod)."""
        from benchmark.libs.queue_runtime import get_prefix  # noqa: PLC0415

        prefix = get_prefix()
        return f"{prefix}:{key}" if prefix else key

    @classmethod
    def _lock_key(cls, oss_ds: str, job_id: str = "default") -> str:
        """Today's execution-lock key: ``…:odps_export:{job_id}:{oss_ds}``."""
        safe_job = (job_id or "default").strip() or "default"
        return cls._prefixed(f"odps_export:{safe_job}:{oss_ds}")

    @classmethod
    def _done_key(cls, oss_ds: str, job_id: str = "default") -> str:
        """Today's done-marker key: ``…:odps_export:done:{job_id}:{oss_ds}``."""
        safe_job = (job_id or "default").strip() or "default"
        return cls._prefixed(f"odps_export:done:{safe_job}:{oss_ds}")

    @classmethod
    def _fail_key(cls, oss_ds: str, job_id: str = "default") -> str:
        """Failure-backoff key: ``…:odps_export:fail:{job_id}:{oss_ds}`` (short TTL)."""
        safe_job = (job_id or "default").strip() or "default"
        return cls._prefixed(f"odps_export:fail:{safe_job}:{oss_ds}")

    async def _get_redis(self) -> "Redis | None":
        from benchmark.libs.queue_runtime import get_redis  # noqa: PLC0415

        try:
            return get_redis()
        except Exception:
            logger.warning(
                "odps_export_scheduler.redis_unavailable — skip this tick "
                "(fail-safe: no lock, no multi-node execution)",
                exc_info=True,
            )
            return None

    async def _acquire_lock(self, redis: "Redis", lock_key: str) -> bool:
        try:
            acquired = await redis.set(lock_key, _WORKER_NAME, nx=True, ex=_LOCK_TTL_S)
        except Exception:
            logger.warning(
                "odps_export_scheduler.lock_acquire_error key=%s",
                lock_key,
                exc_info=True,
            )
            return False
        return bool(acquired)

    async def _release_lock(self, redis: "Redis", lock_key: str) -> None:
        try:
            await redis.eval(_RELEASE_LOCK_LUA, 1, lock_key, _WORKER_NAME)
        except Exception:
            logger.warning(
                "odps_export_scheduler.lock_release_failed key=%s",
                lock_key,
                exc_info=True,
            )

    async def _is_done(self, redis: "Redis", done_key: str) -> bool:
        """Look up today's done marker; treat a Redis error as not done (the execution lock still dedupes)."""
        try:
            return bool(await redis.exists(done_key))
        except Exception:
            logger.warning(
                "odps_export_scheduler.done_check_error key=%s (treat as not done)",
                done_key,
                exc_info=True,
            )
            return False

    async def _mark_done(self, redis: "Redis", done_key: str) -> None:
        """Write today's done marker (48h); failure is warning-only (OSS same-key overwrite is idempotent; cost is a rare rerun)."""
        try:
            await redis.set(done_key, _WORKER_NAME, ex=_DONE_TTL_S)
        except Exception:
            logger.warning(
                "odps_export_scheduler.done_mark_failed key=%s",
                done_key,
                exc_info=True,
            )

    async def _mark_fail_backoff(self, redis: "Redis", fail_key: str) -> None:
        """Write the failure-backoff marker (30min); stop the tick from rerunning ODPS every minute for a job that keeps failing."""
        try:
            await redis.set(fail_key, _WORKER_NAME, ex=_FAIL_BACKOFF_TTL_S)
        except Exception:
            logger.warning(
                "odps_export_scheduler.fail_backoff_mark_failed key=%s",
                fail_key,
                exc_info=True,
            )

    async def _trigger_once(
        self,
        *,
        reason: str,
        job_ids: set[str] | None = None,
    ) -> set[str]:
        """Run one round: refresh config → acquire locks and export due jobs serially.

        :param job_ids: only this subset (lock-retry semantics: jobs skipped because the lock was
            held in the first due batch); ``None`` = every enabled job (first-round semantics).
            A recheck does not open the due window — a job outside its due window is not exported early.

        Returns:
            Job ids skipped because the execution lock was held (retry semantics in
            :meth:`_retry_locked_exports`).
        """
        await self._reload()
        config = self._fetch_config()
        if not config.scheduler_enabled():
            logger.info("odps_export_scheduler.trigger_skip reason=%s (disabled)", reason)
            return set()

        enabled = config.enabled_jobs()
        if job_ids is not None:
            enabled = [j for j in enabled if j.id in job_ids]

        # scheduled drains in rounds: a job that becomes due while the first round is exporting
        # is picked up on the next round, without waiting for the next wake. ``attempted`` stops an
        # already-handled job (including one whose slot has arrived) from exporting again across rounds
        # (the done marker is a second guard).
        lock_held_ids: set[str] = set()
        attempted: set[str] = set()
        while not self._stopped:
            jobs = [
                j for j in enabled if j.id not in attempted and self._job_due_now(j, reason=reason)
            ]
            if not jobs:
                if not attempted:
                    logger.info(
                        "odps_export_scheduler.trigger_skip reason=%s (no due jobs)",
                        reason,
                    )
                break

            # "Today" uses the first due job's timezone (multiple timezones are rare; one config is usually consistent)
            now = datetime.now(self._zone(jobs[0].schedule_timezone))
            odps_ds = default_odps_ds(now)
            oss_ds = default_oss_ds(now)
            logger.info(
                "odps_export_scheduler.trigger reason=%s odps_ds=%s(T-1) oss_ds=%s(T) jobs=%s",
                reason,
                odps_ds,
                oss_ds,
                [j.id for j in jobs],
            )

            redis = await self._get_redis()
            if redis is None:
                return lock_held_ids

            for job in jobs:
                attempted.add(job.id)
                if await self._export_one_job(
                    redis,
                    _ExportTask(
                        job=job,
                        odps_ds=odps_ds,
                        oss_ds=oss_ds,
                        reason=reason,
                    ),
                ):
                    lock_held_ids.add(job.id)
            if reason != "scheduled":
                break
        return lock_held_ids

    async def _export_one_job(
        self,
        redis: "Redis",
        task: "_ExportTask",
    ) -> bool:
        """Export one job; return whether it was skipped because the execution lock was held.

        CancelledError (service shutdown) → release the execution lock then re-raise: another node's
        startup compensation can take over immediately, instead of waiting out the 6h lock TTL.
        """
        job = task.job
        odps_ds = task.odps_ds
        oss_ds = task.oss_ds
        reason = task.reason

        # 1. Done-marker filter: a job already exported successfully today does not run again (idempotent across restart/release)
        done_key = self._done_key(oss_ds, job.id)
        if await self._is_done(redis, done_key):
            logger.info(
                "odps_export_scheduler.skip reason=%s job=%s oss_ds=%s (already done)",
                reason,
                job.id,
                oss_ds,
            )
            return False

        # 1b. Failure backoff: throttle a job that keeps failing so the 60s tick does not rerun ODPS every minute
        fail_key = self._fail_key(oss_ds, job.id)
        if await self._is_done(redis, fail_key):
            logger.info(
                "odps_export_scheduler.skip reason=%s job=%s oss_ds=%s (fail backoff)",
                reason,
                job.id,
                oss_ds,
            )
            return False

        # 2. Execution lock: dedupe across hosts (crash mid-run; another node can take over after TTL expiry)
        lock_key = self._lock_key(oss_ds, job.id)
        if not await self._acquire_lock(redis, lock_key):
            logger.info(
                "odps_export_scheduler.skip reason=%s job=%s oss_ds=%s (lock held)",
                reason,
                job.id,
                oss_ds,
            )
            return True

        logger.info(
            "odps_export_scheduler.job_start reason=%s job=%s limit=%s prefix=%s",
            reason,
            job.id,
            job.limit,
            job.oss_prefix,
        )
        try:
            result = await run_odps_export(
                OdpsExportRequest(
                    sql_template=job.sql_template,
                    odps_ds=odps_ds,
                    oss_ds=oss_ds,
                    limit=int(job.limit or 200),
                    oss_prefix=job.oss_prefix,
                    object_name=job.object_name,
                    job_id=job.id,
                    project_id=job.project_id,
                ),
            )
        except asyncio.CancelledError:
            logger.warning(
                "odps_export_scheduler.export_cancelled reason=%s job=%s oss_ds=%s "
                "— lock released for takeover",
                reason,
                job.id,
                oss_ds,
            )
            await self._release_lock(redis, lock_key)
            raise
        except Exception as exc:
            logger.warning(
                "odps_export_scheduler.export_failed reason=%s job=%s odps_ds=%s oss_ds=%s err=%s",
                reason,
                job.id,
                odps_ds,
                oss_ds,
                exc,
                exc_info=True,
            )
            await self._mark_fail_backoff(redis, fail_key)
            await self._release_lock(redis, lock_key)
            return False

        # 3. Export succeeded → write the done marker (later fires / restart compensation skip it)
        await self._mark_done(redis, done_key)

        logger.info(
            "odps_export_scheduler.exported reason=%s job=%s odps_ds=%s oss_ds=%s "
            "rows=%d key=%s url=%s",
            reason,
            job.id,
            result.odps_ds,
            result.oss_ds,
            result.rows,
            result.object_key,
            result.oss_url,
        )
        return False


__all__ = ["OdpsExportScheduler"]
