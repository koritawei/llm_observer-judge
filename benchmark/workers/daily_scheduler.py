"""DailyExternalBenchmarkScheduler — batch scheduled-eval scheduler (§5.1).

Config source: **DB ``runtime_config_records`` ``kind=scheduled_task``** (see
:mod:`benchmark.domain.scheduled_task`). RemoteConfig ``agent-eval.benchmark.external-daily``
is fully disabled and is not read at runtime. Firing is driven by the APScheduler hub's 60s tick
(reusing the due check in this module), with no long-lived ``asyncio.sleep`` loop of its own.

When it fires:
1. One compensation check at service startup (catch up only benchmarks that should have fired today,
   covering a restart/release that missed the slot)
2. On the trigger day for the schedule frequency (schedule_frequency: daily/weekly/monthly), fire at
   schedule_time (default 02:00, interpreted in schedule_timezone)
3. No-data recheck: if the day's OSS file is missing at the slot, probe every 30 minutes;
   a benchmark-level ``no_data_cutoff`` keeps probing until the cutoff (on timeout mark failed,
   reason "startup failed"); without one, probe once more after 30 minutes and stop. Once the file
   appears and the run starts, stop probing.
"""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun
from benchmark.domain.external_daily_config import (
    ExternalDailyBenchmarkConfig,
    ExternalDailyConfig,
    fetch_external_daily_config,
)
from benchmark.domain.report_storage_config import get_default_download_prefix
from benchmark.domain.service_switches import is_benchmark_disabled
from benchmark.runs.common.config_helpers import resolve_oss_prefix
from benchmark.runs.common.oss_date_macro import prepare_oss_prefix_for_load
DailyDataLoader = None
DailyLoadSpec = None
DailyNoDataError = None
from benchmark.runs.manual_run_id import resolve_external_data_date
from benchmark.runs.registry import BENCHMARK_SERVICES_CLASSES
from benchmark.runs.run_service import BenchmarkRunSpec
from benchmark.runs.schema import ExternalBenchmarkRequest
from benchmark.workers.scheduler_hub import TICK_SECONDS, SchedulerHub, TickJobOptions

if TYPE_CHECKING:
    from benchmark.reporting.framework.base import BaseBenchmarkServices
    from benchmark.workers.external_runner import ExternalBenchmarkRunner

logger = get_logger("agent_eval.eval_workers.daily_scheduler")


@dataclass(frozen=True)
class DailySchedulerOptions:
    """Optional dependencies for ``DailyExternalBenchmarkScheduler`` (FUNC-PARAMS-001)."""

    benchmark_registry: object | None = None
    hub: SchedulerHub | None = None


#: Length of the compact ``HHMM`` format (for example ``0930``).
_HHMM_COMPACT_LEN = 4

# Asia/Shanghai timezone (fallback when ``schedule_timezone`` is illegal)
_SHANGHAI_TZ = timezone(timedelta(hours=8))

#: Lateness window allowed after the slot (sleep drift / startup jitter), aligned with OdpsExportScheduler
_DUE_WINDOW = timedelta(minutes=2)

#: asyncio.sleep may wake a few ms early; allow entering the window slightly early to avoid the "not due yet + today's slot already passed" double miss
_EARLY_SLACK = timedelta(seconds=5)

#: Forward-scan bound (days) before a frequency match: covers the longest gap for rare monthly days (such as the 31st)
_FREQUENCY_SCAN_DAYS = 400

#: Fallback sleep when the frequency still has no match inside the scan bound (defensive, avoids a busy wait)
_FALLBACK_SLEEP_SECONDS = 24 * 3600.0

#: No-data recheck interval (minutes): loop while a cutoff is set; probe only once more when it is not
_NO_DATA_RETRY_INTERVAL_MIN = 30

#: Failure reason written on the run when cutoff arrives with no data fetched / the run never started
_START_FAILED_REASON = "启动失败"

#: APScheduler job id (unique within the process)
_TICK_JOB_ID = "daily-external-benchmark-tick"


def parse_hh_mm(value: str) -> tuple[int, int]:
    """Parse ``HH:MM`` (preferred) or compact ``HHMM`` into ``(hour, minute)``.

    RemoteConfig / config defaults use the ``"06:00"`` form; the old ``int(cutoff[2:])`` treated
    ``":00"`` as the minute and raised ``ValueError``.
    """
    text = value.strip()
    if ":" in text:
        hour_s, minute_s = text.split(":", 1)
        return int(hour_s), int(minute_s)
    if len(text) == _HHMM_COMPACT_LEN and text.isdigit():
        return int(text[:2]), int(text[2:])
    raise ValueError(f"invalid HH:MM value: {value!r}")


def in_schedule_due_window(now: datetime, target: datetime) -> bool:
    """Whether the slot is inside the fireable window ``[target - EARLY_SLACK, target + DUE_WINDOW)``."""
    return target - _EARLY_SLACK <= now < target + _DUE_WINDOW


def schedule_slot_elapsed(now: datetime, candidate: datetime) -> bool:
    """Whether the slot has truly arrived (``candidate <= now``, early-wake slack excluded).

    Early-wake slack is only for the due-window check in ``in_schedule_due_window``; folding it
    into slot consumption makes ``now ∈ [target-5s, target)`` skip today's slot to the next
    matching day — and the benchmark has not fired yet (``_trigger_once`` already finished, and
    ``_bm_due_now`` gets no further chance to mark it due), so that day's schedule is lost entirely.
    Keep the ``candidate <= now`` meaning: sleep at most 1s more before the slot, then one idempotent fire.
    """
    return candidate <= now


def seconds_until_cutoff(now: datetime, cutoff: str) -> float:
    """Seconds left until today's ``cutoff`` (``HH:MM``); 0 or negative if it has passed."""
    hour, minute = parse_hh_mm(cutoff)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return (target - now).total_seconds()


@dataclass(frozen=True)
class _RetrySchedule:
    """Retry policy for ``_retry_until_cutoff`` (cutoff / business date / timezone).

    ``cutoff`` is ``None``: probe once more after 30 minutes and stop; otherwise probe every 30 minutes until the cutoff.
    """

    cutoff: str | None
    date: str
    tz: tzinfo


class DailyExternalBenchmarkScheduler:
    """On the schedule frequency (daily/weekly/monthly), GET the OSS data file directly and fire an external benchmark.

    When it fires:
    1. One compensation check at service startup (catch up only benchmarks that should have fired today)
    2. On the trigger day, fire at schedule_time (default 02:00, interpreted in schedule_timezone)

    Each fire walks the benchmark list in the schedule config and, for each one:
    1. Compute the data date = today (yyyymmdd in the schedule_timezone)
    2. Build run_id = f"{benchmark_name}-{date}"
    3. Status-aware idempotency check (§5.6.2)
    4. OSS file precheck (§5.2)
    5. Create a placeholder run (initializing) → asyncio.create_task(initialize_and_submit(...))
    """

    def __init__(
        self,
        *,
        runner: "ExternalBenchmarkRunner",
        config: ExternalDailyConfig | None = None,
        options: "DailySchedulerOptions | None" = None,
    ) -> None:
        opts = options or DailySchedulerOptions()
        self._runner = runner
        self._config_injected = config is not None
        self._config = config or fetch_external_daily_config()
        self._benchmark_registry = opts.benchmark_registry
        #: Scheduled fires all go through APScheduler (60s tick reuses the existing due check)
        self._hub = opts.hub or SchedulerHub()
        self._owns_hub = opts.hub is None
        self._stopped = False
        #: Startup compensation task (one-shot, not periodic; kept as an asyncio task so shutdown can cancel it)
        self._compensation_task: asyncio.Task[None] | None = None

    @property
    def external_config(self) -> ExternalDailyConfig:
        """Load the external_daily config section."""
        return self._config

    @property
    def enabled(self) -> bool:
        return self.external_config.enabled

    @property
    def benchmarks(self) -> list[ExternalDailyBenchmarkConfig]:
        return self.external_config.benchmarks

    async def start(self) -> None:
        """Start the scheduler: APScheduler interval tick + startup compensation."""
        if not self.enabled:
            logger.info("daily_scheduler.disabled (external_daily.enabled=false)")
            return

        self._stopped = False
        await self._hub.start()
        self._hub.add_tick(_TICK_JOB_ID, self._tick, TickJobOptions(seconds=TICK_SECONDS))
        logger.info(
            "daily_scheduler.started schedule_time=%s timezone=%s benchmarks=%d "
            "tick_s=%d (apscheduler)",
            self.external_config.schedule_time,
            self.external_config.schedule_timezone,
            len(self.benchmarks),
            TICK_SECONDS,
        )

        # Startup compensation: fire once immediately (one-shot task, cancellable on shutdown)
        self._compensation_task = asyncio.create_task(
            self._trigger_once(reason="startup_compensation"),
            name="daily-scheduler-startup-compensation",
        )

    async def stop(self) -> None:
        """Stop the scheduler: remove the tick job and cancel the compensation task (also close a self-owned hub)."""
        self._stopped = True
        self._hub.remove(_TICK_JOB_ID)
        if self._compensation_task is not None:
            self._compensation_task.cancel()
            try:
                await self._compensation_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("daily_scheduler.compensation_task_error", exc_info=True)
            self._compensation_task = None
        if self._owns_hub:
            await self._hub.shutdown()
        logger.info("daily_scheduler.stopped")

    def _refresh_config(self) -> None:
        """Hot-reload the config snapshot (injected-config tests/embedders do not refresh).

        ``runtime_config_store`` rebuilds the snapshot from DB ``kind=scheduled_task`` every 30s;
        tasks added or retimed at runtime take effect within about a minute.
        """
        if not self._config_injected:
            self._config = fetch_external_daily_config()

    async def _tick(self) -> None:
        """APScheduler interval tick: reload config and run one due check.

        Config is reloaded every round, so benchmarks added or retimed at runtime take effect within
        about a minute; ``_bm_due_now`` has a 2-minute due window, so a 60s poll will not miss a slot.
        """
        self._refresh_config()
        await self._trigger_once(reason="scheduled")

    async def _trigger_once(self, *, reason: str) -> None:
        """Run one schedule fire: walk every benchmark config whose slot has arrived.

        scheduled drains in rounds: a benchmark that becomes due while the first round is running
        (a no-data recheck can take tens of minutes) is picked up on the next round, without waiting
        for the next wake. ``attempted`` stops the same benchmark from being selected again across rounds.
        """
        self._refresh_config()
        attempted: set[int] = set()
        while not self._stopped:
            due = [
                (idx, bm)
                for idx, bm in enumerate(self.benchmarks)
                if idx not in attempted and self._bm_due_now(bm, reason=reason)
            ]
            if not due:
                break
            logger.info(
                "daily_scheduler.trigger reason=%s benchmarks=%d due=%d",
                reason,
                len(self.benchmarks),
                len(due),
            )
            for idx, bm_config in due:
                attempted.add(idx)
                try:
                    await self._trigger_one_benchmark(bm_config, reason=reason)
                except Exception:
                    logger.exception(
                        "daily_scheduler.trigger_failed benchmark=%s",
                        bm_config.benchmark_name or "?",
                    )
            if reason != "scheduled":
                break

    async def _skip_existing_run(self, run_id: str, *, reason: str) -> bool:
        """Status-aware idempotency check (§5.6.2): in progress → skip; FAILED → CAS rebuild.

        Returns:
            True means this fire should skip (in progress / CAS not won);
            False means there is no existing run or it was CAS-rebuilt, so the fire may continue.
        """
        run = await BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            return False

        if run.status in (
            BenchmarkStatus.INITIALIZING.value,
            BenchmarkStatus.RUNNING.value,
            BenchmarkStatus.AGGREGATING.value,
            BenchmarkStatus.GENERATING_REPORT.value,
            BenchmarkStatus.COMPLETED.value,
        ):
            logger.info(
                "daily_scheduler.skip run_id=%s status=%s reason=%s",
                run_id,
                run.status,
                reason,
            )
            return True

        if run.status == BenchmarkStatus.FAILED.value:
            # CAS rebuild (§5.6.2)
            max_attempts = self.external_config.max_pull_attempts
            # Obtain the BenchmarkService instance
            benchmark_service = self._runner._benchmark_service  # type: ignore[attr-defined]
            won = await benchmark_service.cas_reset_failed_run(
                run_id,
                max_attempts=max_attempts,
            )
            if not won:
                logger.error(
                    "daily_scheduler.cas_reset_failed run_id=%s (max_attempts=%d or not won)",
                    run_id,
                    max_attempts,
                )
                return True
            logger.info("daily_scheduler.cas_reset_won run_id=%s", run_id)
        return False

    async def _trigger_one_benchmark(  # noqa: PLR0912
        self,
        bm_config: ExternalDailyBenchmarkConfig,
        *,
        reason: str,
    ) -> None:
        """Fire the schedule for a single benchmark."""
        benchmark_name = bm_config.benchmark_name
        if self._should_skip_trigger(bm_config, reason=reason):
            return

        # 1. Default schedule day → expand OSS macros/literal path → if the path has a partition day, override the run_id date
        tz = self._schedule_tz(bm_config.schedule_timezone or None)
        default_date = self._compute_date(tz)
        resolved = self._resolve_oss_prefix_or_skip(
            bm_config, benchmark_name, date=default_date
        )
        if resolved is None:
            return
        oss_prefix, oss_object_key = resolved
        date = resolve_external_data_date(
            default_date,
            storage_path=oss_object_key or oss_prefix,
        )
        run_id_suffix = bm_config.run_id_suffix
        if run_id_suffix:
            run_id = f"{benchmark_name}-{date}-{run_id_suffix}"
        else:
            run_id = f"{benchmark_name}-{date}"

        # 2. Status-aware idempotency check (§5.6.2)
        if await self._skip_existing_run(run_id, reason=reason):
            return

        # 3. Build ExternalBenchmarkRequest (schedule-item profile extension fields pass through, §9)
        payload = ExternalBenchmarkRequest(
            benchmark_name=benchmark_name,
            judge_class=bm_config.judge_class,
            attribution_class=bm_config.attribution_class,
            rubric_class=bm_config.rubric_class,
            date=date,
            oss_prefix=oss_prefix,
            oss_object_key=oss_object_key,
            dataset_name=bm_config.dataset_name,
            generate_report=bm_config.generate_report,
            mode=bm_config.mode,
            rubric=bm_config.rubric,
            parse_interceptor_ref=bm_config.parse_interceptor_ref,
            limit=bm_config.limit,
            require_case_id=bm_config.require_case_id,
            effective_date=date,
            effective_oss_prefix=oss_prefix,
        )

        # 4. OSS file precheck + no-data recheck (cutoff is benchmark-level only)
        no_data_cutoff = self._effective_no_data_cutoff(bm_config)

        # First precheck
        file_exists = await self._check_storage_file_exists(payload)
        if not file_exists:
            file_exists = await self._retry_until_cutoff(
                payload,
                _RetrySchedule(
                    cutoff=no_data_cutoff,
                    date=date,
                    tz=tz,
                ),
            )

        if not file_exists:
            if self._stopped:
                # A shutdown-interrupted recheck (_retry_until_cutoff returns False because _stopped) is not
                # "cutoff reached": do not mark failed — otherwise shutdown creates a placeholder run and
                # writes the startup-failure reason, which the next _skip_existing_run CAS reset then wastes
                # max_pull_attempts.
                logger.info(
                    "daily_scheduler.no_data_interrupted run_id=%s date=%s (shutdown)",
                    run_id,
                    date,
                )
            elif no_data_cutoff:
                logger.warning(
                    "daily_scheduler.no_data run_id=%s date=%s (cutoff reached)",
                    run_id,
                    date,
                )
                await self._mark_start_failed(run_id, bm_config, payload)
            else:
                logger.warning(
                    "daily_scheduler.no_data run_id=%s date=%s (oneshot retry exhausted)",
                    run_id,
                    date,
                )
            return

        # 5. Create a placeholder run + run it in the background
        benchmark_services = self._lookup_benchmark_services(benchmark_name)

        if benchmark_services is None:
            logger.error(
                "daily_scheduler.benchmark_not_found name=%s",
                benchmark_name,
            )
            return

        # Create the placeholder run (if it does not exist yet)
        benchmark_service = self._runner._benchmark_service  # type: ignore[attr-defined]
        created = await benchmark_service.create_placeholder_run(
            run_id,
            BenchmarkRunSpec(
                benchmark_name=benchmark_name,
                service_name=benchmark_services.get_metadata().service_name,
                dataset_name=bm_config.dataset_name,
                config_json={
                    "benchmark_services": benchmark_name,
                    "judge_class": payload.judge_class,
                    "attribution_class": payload.attribution_class,
                    "rubric_class": payload.rubric_class,
                    "generate_report": payload.generate_report,
                    # Same as the accept chain: an init failure can still locate OSS from the snapshot (including macro expansion)
                    "data_date": date,
                    "oss_prefix": oss_prefix,
                    "oss_object_key": oss_object_key,
                },
            ),
        )

        if not created:
            logger.info(
                "daily_scheduler.placeholder_exists run_id=%s (skip)",
                run_id,
            )
            return

        # Run initialize_and_submit in the background
        asyncio.create_task(
            self._run_initialize(run_id, payload, benchmark_services),
            name=f"external-init-{run_id}",
        )

    async def _run_initialize(
        self,
        run_id: str,
        payload: "ExternalBenchmarkRequest",
        benchmark_services: "BaseBenchmarkServices",
    ) -> None:
        """Run initialize_and_submit, and mark the run FAILED on failure."""
        benchmark_service = self._runner._benchmark_service  # type: ignore[attr-defined]
        try:
            await self._runner.initialize_and_submit(
                run_id=run_id,
                payload=payload,
                benchmark_services=benchmark_services,
            )
        except DailyNoDataError as exc:
            logger.warning(
                "daily_scheduler.oss_no_data run_id=%s err=%s",
                run_id,
                exc,
            )
            await benchmark_service.fail_run(
                run_id=run_id,
                reason=f"oss no data: {exc}",
            )
        except Exception as exc:
            logger.exception(
                "daily_scheduler.initialize_failed run_id=%s",
                run_id,
            )
            await benchmark_service.fail_run(
                run_id=run_id,
                reason=f"initialize failed: {exc}",
            )

    def _effective_no_data_cutoff(
        self,
        bm: ExternalDailyBenchmarkConfig,
    ) -> str | None:
        """Benchmark-level cutoff; ``None`` when unset (one probe after 30min)."""
        text = (bm.no_data_cutoff or "").strip()
        return text or None

    def _should_skip_trigger(
        self,
        bm_config: ExternalDailyBenchmarkConfig,
        *,
        reason: str,
    ) -> bool:
        """Pre-fire guard: execution-class name completeness + the benchmark master switch (RemoteConfig).

        Do not fire when a class name is missing (any of ``judge_class`` / ``attribution_class`` /
        ``rubric_class`` is empty) or the benchmark master switch is on; an empty benchmark_name is not
        logged (or-short-circuit semantics stay as they are).
        """
        benchmark_name = bm_config.benchmark_name
        class_aliases = (
            bm_config.judge_class,
            bm_config.attribution_class,
            bm_config.rubric_class,
        )
        if not benchmark_name or not all(class_aliases):
            if benchmark_name:
                logger.error(
                    "daily_scheduler.execution_class_missing benchmark=%s "
                    "judge_class=%r attribution_class=%r rubric_class=%r",
                    benchmark_name,
                    *class_aliases,
                )
            return True
        # When the benchmark switch is on, fire no benchmark schedule (existing runs are unaffected)
        if is_benchmark_disabled():
            logger.info(
                "daily_scheduler.disabled_skip benchmark=%s reason=%s "
                "(RemoteConfig benchmark_disabled=true)",
                benchmark_name,
                reason,
            )
            return True
        return False

    def _resolve_oss_prefix_or_skip(
        self,
        bm: ExternalDailyBenchmarkConfig,
        benchmark_name: str,
        *,
        date: str,
    ) -> tuple[str, str | None] | None:
        """Normalize the OSS prefix and expand date macros (D4).

        ``..`` / empty segment / illegal macro → error log and return ``None`` (skip).
        On success return ``(effective_prefix, oss_object_key | None)``.
        """
        try:
            normalized = resolve_oss_prefix(
                bm.oss_prefix,
                default_prefix=get_default_download_prefix(),
            )
            return prepare_oss_prefix_for_load(
                normalized,
                base_date=date,
                file_name=DailyDataLoader.FILE_NAME,
            )
        except ValueError as exc:
            logger.error(
                "daily_scheduler.oss_prefix_invalid benchmark=%s error=%s",
                benchmark_name,
                exc,
            )
            return None

    def _lookup_benchmark_services(
        self,
        benchmark_name: str,
    ) -> "BaseBenchmarkServices | None":
        for cls in BENCHMARK_SERVICES_CLASSES:
            meta = cls.get_metadata()
            if meta.name == benchmark_name:
                return cls(name=benchmark_name)
        return None

    async def _mark_start_failed(
        self,
        run_id: str,
        bm_config: ExternalDailyBenchmarkConfig,
        payload: "ExternalBenchmarkRequest",
    ) -> None:
        """Cutoff arrived and the run never started: create a placeholder run, then mark it failed.

        ``fail_run`` only when this call actually created the placeholder: the recheck window can last
        hours, and the same run_id may have been created and reached RUNNING/COMPLETED by another path
        (another replica found the file and started / ops fired it via the external-run API). An
        unconditional fail would cascade-destroy a healthy run and its non-terminal items (aligned with
        the idempotent-skip contract of ``_trigger_one_benchmark`` and the API entry).
        """
        benchmark_services = self._lookup_benchmark_services(bm_config.benchmark_name)
        if benchmark_services is None:
            logger.error(
                "daily_scheduler.benchmark_not_found name=%s",
                bm_config.benchmark_name,
            )
            return
        benchmark_service = self._runner._benchmark_service  # type: ignore[attr-defined]
        created = await benchmark_service.create_placeholder_run(
            run_id,
            BenchmarkRunSpec(
                benchmark_name=bm_config.benchmark_name,
                service_name=benchmark_services.get_metadata().service_name,
                dataset_name=bm_config.dataset_name,
                config_json={
                    "benchmark_services": bm_config.benchmark_name,
                    "judge_class": payload.judge_class,
                    "attribution_class": payload.attribution_class,
                    "rubric_class": payload.rubric_class,
                    "generate_report": payload.generate_report,
                },
            ),
        )
        if not created:
            logger.info(
                "daily_scheduler.mark_start_failed_skip run_id=%s (run exists)",
                run_id,
            )
            return
        await benchmark_service.fail_run(run_id=run_id, reason=_START_FAILED_REASON)

    async def _check_storage_file_exists(self, payload: "ExternalBenchmarkRequest") -> bool:
        """Check whether the storage file exists (direct GET precheck).

        Same rule as the real fetch: a pinned ``oss_object_key`` is used verbatim, so a macro-expanded
        or literal date-partition prefix is not concatenated with ``effective_date`` again.
        """
        loader = DailyDataLoader(self._runner._storage_client)  # type: ignore[attr-defined]
        try:
            await loader.load_daily(
                DailyLoadSpec(
                    run_id="precheck",
                    oss_prefix=payload.effective_oss_prefix or "",
                    date=payload.effective_date or "",
                    object_key=payload.oss_object_key,
                ),
            )
            return True
        except DailyNoDataError:
            return False
        except Exception:
            logger.exception("daily_scheduler.oss_precheck_error")
            return False

    async def _retry_until_cutoff(
        self,
        payload: "ExternalBenchmarkRequest",
        schedule: _RetrySchedule,
    ) -> bool:
        """No-data recheck: with a cutoff, probe every 30min until it; without one, probe once more after 30min.

        Return True as soon as the file is found (the caller then starts the run and stops probing).
        """
        interval_s = _NO_DATA_RETRY_INTERVAL_MIN * 60
        cutoff = (schedule.cutoff or "").strip() or None
        date = schedule.date
        tz = schedule.tz

        if cutoff is None:
            logger.info(
                "daily_scheduler.no_data_retry date=%s next_in=%dmin oneshot=true",
                date,
                _NO_DATA_RETRY_INTERVAL_MIN,
            )
            await asyncio.sleep(interval_s)
            if self._stopped:
                return False
            return await self._check_storage_file_exists(payload)

        while not self._stopped:
            now = datetime.now(tz)
            remaining_s = seconds_until_cutoff(now, cutoff)
            if remaining_s <= 0:
                return False
            sleep_s = min(interval_s, remaining_s)
            logger.info(
                "daily_scheduler.no_data_retry date=%s next_in=%.0fs cutoff=%s",
                date,
                sleep_s,
                cutoff,
            )
            await asyncio.sleep(sleep_s)
            if self._stopped:
                return False
            if await self._check_storage_file_exists(payload):
                return True
        return False

    def _schedule_tz(self, tz_name: str | None = None) -> tzinfo:
        """Schedule timezone: benchmark-level first, else the top level; an illegal value falls back to Shanghai and warns."""
        name = tz_name or self.external_config.schedule_timezone
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning(
                "daily_scheduler.invalid_timezone tz=%s fallback=Asia/Shanghai",
                name,
            )
            return _SHANGHAI_TZ

    def _compute_date(self, tz: tzinfo | None = None) -> str:
        """Compute today's data date (yyyymmdd in the effective schedule timezone)."""
        now = datetime.now(tz or self._schedule_tz())
        return now.strftime("%Y%m%d")

    def _date_matches_frequency(self, bm: ExternalDailyBenchmarkConfig, day: date) -> bool:
        """Frequency match: ``daily`` is always true; ``weekly`` uses weekdays (0=Monday); ``monthly`` uses month_days.

        ``schedule_frequency`` is already normalized at the config layer (illegal values fall back to daily); only three branches are needed here.
        """
        frequency = bm.schedule_frequency
        if frequency == "weekly":
            return day.weekday() in (bm.schedule_weekdays or [0])
        if frequency == "monthly":
            return day.day in (bm.schedule_month_days or [1])
        return True

    def _bm_due_now(self, bm: ExternalDailyBenchmarkConfig, *, reason: str) -> bool:
        """Frequency match + slot-window check.

        - Frequency mismatch (weekly/monthly not a trigger day) → never due;
        - startup full compensation: catch up only what should have fired today (daily semantics unchanged);
        - scheduled: trigger day + slot reached (from ``target - EARLY_SLACK``,
          including sleep early-wake slack; a slot already passed is still due the same day, with no right edge).
        """
        now = datetime.now(self._schedule_tz(bm.schedule_timezone or None))
        if not self._date_matches_frequency(bm, now.date()):
            return False
        if reason != "scheduled":
            return True
        time_text = bm.schedule_time or self.external_config.schedule_time
        hour, minute = parse_hh_mm(time_text)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # Early-wake slack (left edge) or a slot already passed (late today) both count as due: a strict
        # 2-minute window lets an earlier benchmark's recheck/run push a later one out of the window,
        # and ``_seconds_until_bm_trigger`` then defers it to the next frequency day (a whole week/month
        # lost for weekly/monthly). Duplicate fires are covered by the run idempotency check and
        # ``_trigger_once``'s attempted drain.
        return in_schedule_due_window(now, target) or schedule_slot_elapsed(now, target)

    def _seconds_until_bm_trigger(self, bm: ExternalDailyBenchmarkConfig) -> float:
        """Seconds until a single benchmark's next valid fire (frequency-matching day + slot)."""
        time_text = bm.schedule_time or self.external_config.schedule_time
        hour, minute = parse_hh_mm(time_text)
        now = datetime.now(self._schedule_tz(bm.schedule_timezone or None))
        base = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        for offset in range(_FREQUENCY_SCAN_DAYS):
            candidate = base + timedelta(days=offset)
            if not self._date_matches_frequency(bm, candidate.date()):
                continue
            if schedule_slot_elapsed(now, candidate):
                continue
            return max((candidate - now).total_seconds(), 1.0)
        # Guard: a legal config must match within 400 days; on anomaly fall back to 1 day to avoid a busy wait
        return _FALLBACK_SLEEP_SECONDS

    def _calc_sleep_until_next_trigger(self) -> float:
        """Seconds until the next fire (the minimum of the benchmark-level slots)."""
        if not self.benchmarks:
            now = datetime.now(self._schedule_tz())
            hour, minute = parse_hh_mm(self.external_config.schedule_time)
            next_trigger = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if schedule_slot_elapsed(now, next_trigger):
                next_trigger += timedelta(days=1)
            return max((next_trigger - now).total_seconds(), 1.0)
        return min(self._seconds_until_bm_trigger(bm) for bm in self.benchmarks)


__all__ = [
    "DailyExternalBenchmarkScheduler",
    "DailySchedulerOptions",
    "in_schedule_due_window",
    "parse_hh_mm",
    "schedule_slot_elapsed",
    "seconds_until_cutoff",
]
