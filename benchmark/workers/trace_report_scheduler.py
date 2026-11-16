"""TraceDailyReportScheduler — daily (T+1) scheduler for trace-pipeline reports.

**Independent path**: it only aggregates existing eval results in ``trace_pipeline_records``
over a T-1 window using RemoteConfig filters (source / agent_name / project_id). It does **not**
fetch OSS data files, run judge / attribution, or consume Kafka — existing benchmark /
dataset / kafka behavior is unchanged.

**Shared report flow**: aggregated results go through
:func:`benchmark.reporting.framework.report_spec.generate_benchmark_report`
(static unified template first, webdev-agent LLM fallback). Case detail is rebuilt from real
trace records by ``load_trace_daily_case_results`` (daily-report field adapter).

When it fires (same shape as DailyExternalBenchmarkScheduler):
1. One compensation via hub ``add_once`` at service startup (**also through the Redis leader**;
   if today's slot has passed, catch up T-1, covering a restart/release that missed the slot)
2. Every day at ``schedule_time`` (item default 09:00; RemoteConfig top level may override,
   interpreted in ``schedule_timezone``) via hub ``add_tick`` — **do not** pass
   ``next_run_time=None`` (APScheduler would pause the job)

Each fire walks the RemoteConfig ``reports`` list and, for each item:
1. Data date = trigger day T-1 (item timezone), run_id = ``{name}-{yyyymmdd}``
2. Idempotency: skip if a run with that name exists (failed needs a manual run delete then retry,
   same semantics as an external trigger)
3. Sample the T-1 window of ``trace_pipeline_records`` (no data → warning and skip,
   do not create a placeholder run)
4. Aggregate per-criterion stats + attribution distribution → ``benchmark_runs.metrics_json``
5. Shared report flow → complete_run

The daily trend serves only this path: the window ends on the data day, history is replayed from
``trace_pipeline_records`` (not ``benchmark_run_items``), and the result is written to
``benchmark_runs.metrics_json.daily_trend_point`` for the next read.
Other benchmarks' trend logic does not pass through here.
"""

import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun
from benchmark.domain.service_switches import is_benchmark_disabled
from benchmark.domain.trace_report_config import (
    TRACE_REPORT_NAMESPACE_CONFIG_KEY,
    TRACE_REPORT_ORIGIN_AUTO_DEFAULT,
    TraceReportConfig,
    TraceReportItemConfig,
    fetch_trace_report_config,
    list_db_trace_report_items,
    registered_dataset_name_from_report_name,
)
from benchmark.runs.common.rubric_base import BaseRubric
from benchmark.runs.common.rubric_classes import (
    RubricClassResolutionError,
    resolve_rubric_class,
)
from benchmark.runs.registry import BENCHMARK_SERVICES_CLASSES
from benchmark.runs.run_helpers import CompleteRunWrite, TraceRecordSampleQuery
from benchmark.runs.run_service import BenchmarkRunSpec
from benchmark.runs.run_stop import (
    STOP_KIND_MANUAL,
    STOP_REASON,
    partial_report_meta,
    streaming_stopped_run_id,
)
from benchmark.runs.trace_daily_report_adapter import (
    build_weight_dimension_section,
)
from benchmark.runs.trace_daily_trend import (
    TraceDailySampleRequest,
    trace_daily_sample_snapshot,
)
from benchmark.reporting.framework.report_spec import ReportJobInput, generate_benchmark_report
from benchmark.reporting.framework.schema import (
    AggregatedMetrics,
    AggregationConfig,
    AttributionStats,
    Criterion,
    CriterionStats,
)
from benchmark.plugins.class_loader import load_plugin_class
from benchmark.workers.daily_scheduler import (
    parse_hh_mm,
    schedule_slot_elapsed,
)
from benchmark.workers.scheduler_hub import (
    TICK_SECONDS,
    SchedulerHub,
    TickJobOptions,
)

if TYPE_CHECKING:
    from benchmark.runs.run_service import BenchmarkService
    from benchmark.reporting.framework.aggregation_calculator import AggregationCalculator
    from benchmark.reporting.framework.base import BenchmarkComponentMetadata
    from benchmark.reporting.framework.webdev_client import WebdevReportClient
    from benchmark.reporting.publish import ReportSiteReportPublisher

logger = get_logger("agent_eval.eval_workers.trace_report_scheduler")

# Asia/Shanghai timezone (fallback when ``schedule_timezone`` is illegal)
_SHANGHAI_TZ = timezone(timedelta(hours=8))

#: APScheduler job id (unique within the process)
_TICK_JOB_ID = "trace-daily-report-tick"
_COMPENSATION_JOB_ID = "trace-daily-report-startup-compensation"

#: Days to keep fire records (``_fired``), to avoid slow memory growth
_FIRED_RETENTION_DAYS = 7

#: Tick liveness heartbeat interval (seconds). Prod defaults to WARNING, so the heartbeat uses WARNING to stay visible.
_TICK_HEARTBEAT_INTERVAL_S = 600.0


@dataclass(frozen=True)
class _TraceReportJob:
    """Arguments for ``_generate_report`` (FUNC-PARAMS-001)."""

    run_id: str
    rubric: list[Criterion]
    metrics: AggregatedMetrics
    final_score: float
    config: dict[str, Any]
    trace_ids: list[str]
    pass_threshold: float = 0.5


@dataclass(frozen=True)
class TraceReportRunSpec:
    """Arguments for ``_run_item`` (FUNC-PARAMS-001; includes an optional stop-window override)."""

    item: TraceReportItemConfig
    data_date: date
    tz: tzinfo
    run_id_override: str | None = None
    mark_stop: bool = False


@dataclass(frozen=True)
class TraceReportSchedulerDeps:
    """Constructor dependencies for ``TraceDailyReportScheduler`` (FUNC-PARAMS-001)."""

    benchmark_service: "BenchmarkService"
    aggregation_calculator: "AggregationCalculator"
    webdev: "WebdevReportClient | None" = None
    pages_publisher: "AonePagesReportPublisher | None" = None
    config: TraceReportConfig | None = None
    hub: SchedulerHub | None = None


class TraceDailyReportScheduler:
    """Scheduler that aggregates T-1 data from ``trace_pipeline_records`` at a daily slot and emits a report.

    The data date is not configurable — it is fixed to T-1 of the trigger day (item timezone);
    filters (source / agent_name / project_id / sample_size) all come from
    RemoteConfig ``agent-eval.benchmark.trace-daily-report``.
    """

    def __init__(self, deps: TraceReportSchedulerDeps) -> None:
        self._benchmark_service = deps.benchmark_service
        self._aggregation = deps.aggregation_calculator
        self._webdev = deps.webdev
        self._pages_publisher = deps.pages_publisher
        self._config = deps.config or fetch_trace_report_config()
        #: Scheduled fires all go through APScheduler (60s tick reuses the existing due check)
        self._hub = deps.hub or SchedulerHub()
        self._owns_hub = deps.hub is None
        self._stopped = False
        # In-process fire record {(report_name, data_date_iso)}: prevents a same-day double fire from
        # sleep drift / startup jitter; after a process restart the run idempotency check covers it.
        self._fired: set[tuple[str, str]] = set()
        #: Last tick heartbeat (monotonic)
        self._last_heartbeat_at = 0.0

    @property
    def report_config(self) -> TraceReportConfig:
        """Load the current schedule config (hot-reloaded before each fire)."""
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def reports(self) -> list[TraceReportItemConfig]:
        return self._config.reports

    # ═══════════════════════════════════════════════════════════════════════
    # lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Start the scheduler (no-op when RemoteConfig enabled=false): APScheduler tick + compensation.

        Startup compensation and the interval tick **both** go through :meth:`SchedulerHub.add_once` /
        ``add_tick`` (via Redis leader election). Do not ``asyncio.create_task`` straight into
        ``_trigger_once`` — a non-leader would also emit a compensation report, and if the slot tick
        were misconfigured as paused there would be no logs all day, which is very hard to debug.
        """
        if not self.enabled:
            logger.info("trace_report_scheduler.disabled (trace_daily_report.enabled=false)")
            return

        self._stopped = False
        await self._hub.start()
        self._hub.add_tick(_TICK_JOB_ID, self._tick, TickJobOptions(seconds=TICK_SECONDS))
        # Startup compensation: same path as the tick (leader gate); if today's slot has passed, catch up T-1
        self._hub.add_once(_COMPENSATION_JOB_ID, self._startup_compensation)
        logger.info(
            "trace_report_scheduler.started schedule_time=%s timezone=%s "
            "remote_config_reports=%d tick_s=%d (apscheduler; DB items merged per tick)",
            self._config.schedule_time,
            self._config.schedule_timezone,
            len(self.reports),
            TICK_SECONDS,
        )

    async def stop(self) -> None:
        """Stop the scheduler: remove the tick / compensation jobs (also close a self-owned hub)."""
        self._stopped = True
        self._hub.remove(_TICK_JOB_ID)
        self._hub.remove(_COMPENSATION_JOB_ID)
        if self._owns_hub:
            await self._hub.shutdown()
        logger.info("trace_report_scheduler.stopped")

    # ═══════════════════════════════════════════════════════════════════════
    # schedule loop
    # ═══════════════════════════════════════════════════════════════════════

    async def _startup_compensation(self) -> None:
        """Startup one-shot compensation (called after the hub leader wrapper)."""
        await self._trigger_once(reason="startup_compensation")

    async def _tick(self) -> None:
        """APScheduler interval tick: heartbeat + due check (``_trigger_once`` reloads config internally)."""
        await self._emit_tick_heartbeat()
        await self._trigger_once(reason="scheduled")

    async def _emit_tick_heartbeat(self) -> None:
        """Periodic WARNING heartbeat proving the slot tick is still running (prod does not show INFO by default).

        The first tick always logs one line; after that, log every ``_TICK_HEARTBEAT_INTERVAL_S``.
        """
        now = time.monotonic()
        if (
            self._last_heartbeat_at > 0.0
            and now - self._last_heartbeat_at < _TICK_HEARTBEAT_INTERVAL_S
        ):
            return
        self._last_heartbeat_at = now
        try:
            merged = await self._merged_report_items()
            n_items = len(merged)
        except Exception:
            n_items = -1
        next_run = self._hub.next_run_time(_TICK_JOB_ID)
        logger.warning(
            "trace_report_scheduler.tick_alive remote_config_enabled=%s remote_config_schedule=%s "
            "merged_items=%d fired=%d next_tick=%s",
            self._config.enabled,
            self._config.schedule_time,
            n_items,
            len(self._fired),
            next_run.isoformat() if next_run is not None else None,
        )

    def _item_tz(self, item: TraceReportItemConfig) -> tzinfo:
        """Item-level timezone override; an illegal value falls back to Asia/Shanghai."""
        text = (item.schedule_timezone or self._config.schedule_timezone).strip()
        try:
            return ZoneInfo(text)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning(
                "trace_report_scheduler.invalid_timezone %r — fallback Asia/Shanghai",
                text,
            )
            return _SHANGHAI_TZ

    def _item_schedule_hh_mm(self, item: TraceReportItemConfig) -> tuple[int, int]:
        """Item-level slot override; an illegal value falls back to the default slot 09:00."""
        text = (item.schedule_time or self._config.schedule_time).strip()
        try:
            return parse_hh_mm(text)
        except ValueError:
            logger.warning(
                "trace_report_scheduler.invalid_schedule_time %r — fallback 09:00",
                text,
            )
            return (9, 0)

    async def _merged_report_items(self) -> list[TraceReportItemConfig]:
        """Merge report items from two sources: RemoteConfig ``reports[]`` + DB registrations.

        Name clash: a DB entry overrides RemoteConfig (a repeated page submit keeps the latest), but if
        the DB item is only a system default registration (origin=auto_default), the user's same-name
        RemoteConfig customization wins — the default only fills in, it does not override a customization.
        A DB read failure degrades to RemoteConfig-only (DB jitter must not take down existing report schedules).
        """
        try:
            db_items = await list_db_trace_report_items()
        except Exception:
            logger.exception("trace_report_scheduler.db_items_unavailable — RemoteConfig only")
            db_items = []
        merged: dict[str, TraceReportItemConfig] = {
            item.name: item for item in self._config.reports if item.name
        }
        for db_item in db_items:
            if not db_item.name:
                continue
            # User customization overrides the default registration: when the DB item is a system default
            # (origin=auto_default) and RemoteConfig has a same-name customization, RemoteConfig wins;
            # other DB items (page/user registrations) keep the existing "override RemoteConfig" semantics.
            if db_item.name in merged and db_item.origin == TRACE_REPORT_ORIGIN_AUTO_DEFAULT:
                continue
            merged[db_item.name] = db_item
        return list(merged.values())

    async def _trigger_once(self, *, reason: str) -> None:
        """Fire one round: walk RemoteConfig report items and emit a T-1 report for each whose slot today has passed."""
        # Config hot reload: refetch RemoteConfig on every fire (parse-cache TTL 300s)
        self._config = fetch_trace_report_config()
        if not self._config.enabled:
            logger.info("trace_report_scheduler.disabled (trace_daily_report.enabled=false)")
            return
        # When the benchmark master switch is off, fire no report schedule (existing runs are unaffected)
        if is_benchmark_disabled():
            logger.info(
                "trace_report_scheduler.disabled_skip (RemoteConfig benchmark_disabled=true)",
            )
            return

        self._prune_fired()
        # Two-source merge of report items (the wake-time calculation shares this result; see
        # ``_merged_report_items``): RemoteConfig ``reports[]`` plus ``kind=trace_report`` rows written
        # automatically by realtime-stream registration (a webUI registration reports on its own,
        # with no hand-written RemoteConfig).
        merged_items = await self._merged_report_items()
        now = datetime.now(UTC)
        for item in merged_items:
            if self._stopped:
                break
            if not item.name or (not item.benchmark_name and not item.rubric_class):
                logger.warning(
                    "trace_report_scheduler.invalid_item name=%r benchmark_name=%r "
                    "rubric_class=%r — skipped",
                    item.name,
                    item.benchmark_name,
                    item.rubric_class,
                )
                continue
            tz = self._item_tz(item)
            local_now = now.astimezone(tz)
            hour, minute = self._item_schedule_hh_mm(item)
            today_target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if not schedule_slot_elapsed(local_now, today_target):
                # Today's slot has not arrived: log once, then skip. A silent skip used to make
                # "woke at 02:00, the 09:00 item was skipped with no log" impossible to debug.
                logger.info(
                    "trace_report_scheduler.schedule_not_due name=%s "
                    "schedule_time=%02d:%02d reason=%s",
                    item.name,
                    hour,
                    minute,
                    reason,
                )
                continue
            data_date = local_now.date() - timedelta(days=1)
            fire_key = (item.name, data_date.isoformat())
            if fire_key in self._fired:
                continue
            self._fired.add(fire_key)
            logger.info(
                "trace_report_scheduler.trigger run=%s-%s reason=%s source=%s",
                item.name,
                data_date.strftime("%Y%m%d"),
                reason,
                item.source,
            )
            try:
                await self._run_item(item, data_date=data_date, tz=tz)
            except Exception:
                # _run_item already called fail_run; this guard stops one item's exception from taking down the others
                logger.exception(
                    "trace_report_scheduler.item_failed name=%s",
                    item.name,
                )

    def _prune_fired(self) -> None:
        """Drop fire records past the retention window.

        Use the schedule clock ``datetime.now``, the same source as the fire check. ``date.today``
        would, after a test injects a clock, delete not-yet-expired dedupe keys by the real date.
        """
        today = datetime.now(UTC).date()
        cutoff = (today - timedelta(days=_FIRED_RETENTION_DAYS)).isoformat()
        self._fired = {key for key in self._fired if key[1] >= cutoff}

    # ═══════════════════════════════════════════════════════════════════════
    # Single-report execution: sample → aggregate → report (shared flow)
    # ═══════════════════════════════════════════════════════════════════════

    async def find_report_item(
        self,
        *,
        registration_id: str,
        stream_instance_id: str | None = None,
    ) -> TraceReportItemConfig | None:
        """Exact-match a registration (and optional stream) in the RemoteConfig+DB merged items.

        An API stop-window report must use this method; do not read only ``reports`` (RemoteConfig alone).
        """
        reg = (registration_id or "").strip()
        if not reg:
            return None
        si = (stream_instance_id or "").strip()
        for item in await self._merged_report_items():
            item_reg = str(getattr(item, "agent_registration_id", "") or "").strip()
            item_si = str(getattr(item, "stream_instance_id", "") or "").strip()
            if not item_reg or item_reg != reg:
                continue
            if si and item_si != si:
                continue
            return item
        return None

    async def publish_stopped_window(
        self,
        item: TraceReportItemConfig,
        *,
        data_date: date | None = None,
    ) -> str | None:
        """Window aggregate report after a realtime stream is stopped by hand (run_id includes a ``stopped`` suffix).

        Shares the sample/aggregate/report path with the T+1 daily report; the ``stop_requested_at`` latch
        makes ``complete_run`` land on ``stopped`` + partial meta (§6.8).
        """
        zone = self._item_tz(item)
        local_now = datetime.now(UTC).astimezone(zone)
        day = data_date or local_now.date()
        run_id = streaming_stopped_run_id(item.name, now=local_now)
        await self._run_item(
            TraceReportRunSpec(
                item=item,
                data_date=day,
                tz=zone,
                run_id_override=run_id,
                mark_stop=True,
            ),
        )
        return run_id

    async def _run_item(  # noqa: PLR0912, PLR0915 — sample → idempotency → aggregate → report stage chain
        self,
        item: TraceReportItemConfig | TraceReportRunSpec,
        *,
        data_date: date | None = None,
        tz: tzinfo | None = None,
    ) -> None:
        """Emit a T-1 report for one report-config item (idempotent: skip if a same-name run exists).

        Compatible with existing ``_run_item(item, data_date=, tz=)`` calls; the stop window uses
        :class:`TraceReportRunSpec`, so parameter growth does not trip FUNC-PARAMS-001.
        """
        if isinstance(item, TraceReportRunSpec):
            spec = item
        else:
            if data_date is None or tz is None:
                raise TypeError(
                    "_run_item requires data_date and tz when item is not TraceReportRunSpec",
                )
            spec = TraceReportRunSpec(item=item, data_date=data_date, tz=tz)
        item = spec.item
        data_date = spec.data_date
        tz = spec.tz
        run_id_override = spec.run_id_override
        mark_stop = spec.mark_stop
        ymd = data_date.strftime("%Y%m%d")
        run_id = run_id_override or f"{item.name}-{ymd}"
        # 1. Idempotency: skip if a same-name run exists (failed needs a manual run delete then retry,
        #    same semantics as an external trigger's "DELETE the run and retry")
        existing = await BenchmarkRun.filter(run_id=run_id).first()
        if existing is not None:
            logger.info(
                "trace_report.run_exists run_id=%s status=%s (skip)",
                run_id,
                existing.status,
            )
            return

        # 2. Resolve rubric / aggregation: RemoteConfig ``rubric_class`` alias first
        #    (take the Rubric class directly; benchmark_name may be empty); otherwise fall back to
        #    benchmark component metadata. Both paths read metadata only and trigger no execution.
        resolved = self._resolve_rubric_aggregation(item)
        if resolved is None:
            return
        rubric, aggregation, rubric_ref, service_name = resolved
        from benchmark.domain.eval_config import (  # noqa: PLC0415
            sample_evaluator_names_for_report,
        )

        registration_id = (item.agent_registration_id or "").strip() or None
        stream_instance_id = (item.stream_instance_id or "").strip() or None
        evaluator_names = sample_evaluator_names_for_report(
            rubric_refs=[c.evaluator_ref for c in rubric],
            registration_id=registration_id,
            stream_instance_id=stream_instance_id,
        )

        # 3. T-1 data window [T-1 00:00, T 00:00) (item timezone)
        window_start = datetime(data_date.year, data_date.month, data_date.day, tzinfo=tz)
        window_end = window_start + timedelta(days=1)

        started_at = time.perf_counter()

        # 4. Sample (no data → do not create a run — tomorrow's slot naturally retries that day's new T-1)
        # evaluator_name is now stored as registrationId; also accept the rubric stem for old rows.
        trace_ids = await self._benchmark_service.sample_trace_pipeline_records(
            TraceRecordSampleQuery(
                date_start=window_start,
                date_end=window_end,
                evaluator_names=evaluator_names,
                source=item.source,
                agent_name=item.agent_name,
                project_id=item.project_id,
                sample_size=item.sample_size,
                judge_content_version=item.judge_content_version,
            ),
        )
        if not trace_ids:
            logger.warning(
                "trace_report.no_data run_id=%s source=%s agent_name=%s "
                "project_id=%s judge_cv=%s registration=%s",
                run_id,
                item.source,
                item.agent_name,
                item.project_id,
                item.judge_content_version,
                item.agent_registration_id,
            )
            return

        # 5. Create the placeholder run:
        #    - dataset_name = page-registered dataset name (item.name with ``__reg__`` stripped)
        #    - benchmark_name = registrationId (if present; else the logical component name; lookups /
        #      report/log use the same rule as the realtime-stream evaluator_name)
        #    - config.trace_report_namespace = item.name (same prefix as run_id,
        #      so the trend groups by day; the benchmark_name column no longer carries the namespace)
        #    - config.benchmark_component = logical component name (display / soft match)
        #    - config does not write "benchmark_services"; rubric_ref is the report-side fallback
        from benchmark.domain.eval_config import stored_benchmark_run_name  # noqa: PLC0415

        page_dataset_name = registered_dataset_name_from_report_name(item.name)
        logical_agent_name = (item.benchmark_name or "").strip() or page_dataset_name
        run_config: dict[str, Any] = {
            "trace_daily_report": True,
            "source": item.source,
            # Overall quality is a 0~1 score; report KPI / loss waterfall read this full mark (other businesses default to 2.0)
            "score_full_mark": 1.0,
            TRACE_REPORT_NAMESPACE_CONFIG_KEY: item.name,
            # Stream-source marker (batch = non-realtime registration set; realtime = realtime registration set); for debugging, not a display key
            "trace_pipeline_source": item.source,
        }
        if mark_stop:
            run_config["stop_requested_at"] = datetime.now(UTC).isoformat()
            run_config["stop_reason"] = STOP_REASON
            run_config["stop_kind"] = STOP_KIND_MANUAL
        if rubric_ref is not None:
            run_config["rubric_ref"] = rubric_ref
        if item.benchmark_name:
            run_config["benchmark_component"] = item.benchmark_name
        if item.rule_link_url:
            run_config["rule_link_url"] = item.rule_link_url
        if item.report_html:
            run_config["report_html"] = item.report_html
        if item.agent_registration_id:
            run_config["agent_registration_id"] = item.agent_registration_id
        if item.stream_instance_id:
            run_config["stream_instance_id"] = item.stream_instance_id
        if item.judge_content_version:
            run_config["judge_content_version"] = item.judge_content_version
        tz_name = getattr(tz, "key", None) or "Asia/Shanghai"
        run_config["trace_daily_sample"] = trace_daily_sample_snapshot(
            TraceDailySampleRequest(
                source=item.source,
                timezone_name=str(tz_name),
                pass_threshold=float(aggregation.pass_threshold),
                rubric=rubric,
                agent_name=item.agent_name,
                project_id=item.project_id,
                sample_size=item.sample_size,
                registration_id=registration_id,
            )
        )
        created = await self._benchmark_service.create_placeholder_run(
            run_id,
            BenchmarkRunSpec(
                benchmark_name=stored_benchmark_run_name(
                    logical_name=logical_agent_name,
                    registration_id=registration_id,
                ),
                service_name=service_name,
                dataset_name=page_dataset_name,
                project_id=item.project_id,
                total_samples=len(trace_ids),
                total_evaluators=len(rubric),
                config_json=run_config,
            ),
        )
        if not created:
            logger.info("trace_report.placeholder_exists run_id=%s (skip)", run_id)
            return

        try:
            # 6. Aggregate: per-criterion stats (reuse the kafka aggregator — it only consumes trace_ids,
            #    independent of source) + attribution distribution
            await self._benchmark_service.update_run_status(
                run_id,
                BenchmarkStatus.RUNNING,
                started_at=datetime.now(UTC),
                total_samples=len(trace_ids),
            )
            entries = await self._benchmark_service.load_kafka_criterion_entries(
                trace_ids=trace_ids,
                rubric=rubric,
                pass_threshold=aggregation.pass_threshold,
                registration_id=registration_id,
                stream_instance_id=stream_instance_id,
            )
            final_score, _per_criterion, metrics = await self._aggregation.aggregate(
                aggregation,
                rubric,
                entries,
            )
            attr_by_cid = await self._benchmark_service.aggregate_trace_daily_root_cause(
                trace_ids=trace_ids,
                rubric=rubric,
                registration_id=registration_id,
                stream_instance_id=stream_instance_id,
            )
            for cid, attr in attr_by_cid.items():
                metrics.attribution_metrics[cid] = AttributionStats(
                    field_distributions=attr.get("field_distributions", {}),
                    value_labels=attr.get("value_labels", {}),
                    total_attributed=attr.get("total_attributed", 0),
                )
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            metrics.total_duration_ms = duration_ms

            # [stop-run window only] with no scored samples, do not emit a fake 0-score report (aligned with batch completed_cases=0).
            # A normal T+1 daily report (mark_stop=False) keeps the old semantics: emit if there is a sample.
            scored_samples = sum(int(getattr(e, "sample_count", 0) or 0) for e in entries)
            skip_stop_report = bool(mark_stop) and scored_samples <= 0

            # 7. Emit the report (shared report-generation flow)
            report_url: str | None = None
            if item.generate_report and not skip_stop_report:
                await self._benchmark_service.update_run_status(
                    run_id,
                    BenchmarkStatus.GENERATING_REPORT,
                    final_score=final_score,
                    metrics_json=metrics.model_dump(),
                )
                report_url = await self._generate_report(
                    _TraceReportJob(
                        run_id=run_id,
                        rubric=rubric,
                        metrics=metrics,
                        final_score=final_score,
                        config=run_config,
                        trace_ids=trace_ids,
                        pass_threshold=aggregation.pass_threshold,
                    ),
                )

            # 8. Complete
            report_meta: dict[str, Any] | None = None
            if report_url:
                report_meta = {
                    "report_url": report_url,
                    "generated_at": datetime.now(UTC).isoformat(),
                    # Freeze the report sample so the case-list API does not drift by resampling "the latest trace"
                    "sampled_trace_ids": list(trace_ids),
                }
            elif skip_stop_report:
                report_meta = partial_report_meta(
                    completed_cases=0,
                    stop_requested_at=str(run_config.get("stop_requested_at") or "") or None,
                )
            await self._benchmark_service.complete_run(
                CompleteRunWrite(
                    run_id=run_id,
                    final_score=final_score,
                    metrics_json=metrics.model_dump(),
                    report_meta_json=report_meta,
                    duration_ms=duration_ms,
                ),
            )
            logger.info(
                "trace_report.run_complete run_id=%s score=%.4f records=%d report_url=%s",
                run_id,
                final_score,
                len(trace_ids),
                report_url,
            )
        except Exception as exc:
            logger.exception("trace_report.run_failed run_id=%s", run_id)
            await self._benchmark_service.fail_run(run_id=run_id, reason=str(exc))

    async def _generate_report(
        self,
        job: _TraceReportJob,
    ) -> str | None:
        """Shared report-generation flow; case detail is rebuilt from real trace records.

        This path does not write synthetic detail rows to ``benchmark_run_items``; case_results are
        loaded via ``load_trace_daily_case_results`` (daily-report adapter: langfuse_url /
        triage grade / overall-quality fused / root_cause), isolated from the pipeline
        benchmark's ``load_kafka_case_results``.
        """
        trace_ids = job.trace_ids
        rubric = job.rubric
        pass_threshold = job.pass_threshold

        async def _load_cases() -> list[dict[str, Any]]:
            reg = str(job.config.get("agent_registration_id") or "").strip() or None
            stream = str(job.config.get("stream_instance_id") or "").strip() or None
            cases = await self._benchmark_service.load_trace_daily_case_results(
                trace_ids=trace_ids,
                rubric=rubric,
                pass_threshold=pass_threshold,
                registration_id=reg,
                stream_instance_id=stream,
            )
            # The eval-dimension column uses extra.weights; the same stats are persisted for the next day's radar comparison.
            section = build_weight_dimension_section(cases)
            if section is not None:
                job.metrics.criterion_metrics = {
                    cid: CriterionStats.model_validate(payload)
                    for cid, payload in section.criterion_metrics.items()
                }
            return cases

        return await generate_benchmark_report(
            ReportJobInput(
                benchmark_service=self._benchmark_service,
                run_id=job.run_id,
                metrics=job.metrics,
                final_score=job.final_score,
                config=job.config,
                rubric=job.rubric,
                case_results_loader=_load_cases,
                log_prefix="trace_report",
            ),
            webdev=self._webdev,
            pages_publisher=self._pages_publisher,
        )

    def _resolve_rubric_aggregation(
        self,
        item: TraceReportItemConfig,
    ) -> tuple[list[Criterion], AggregationConfig, str | None, str] | None:
        """Resolve rubric / aggregation / rubric_ref / service_name (metadata only).

        Priority: RemoteConfig ``rubric_class`` alias (``RUBRIC_CLASS_REFS`` allowlist;
        the explicit alias ``component_rubric`` is equivalent to unset) > ``benchmark_name`` component
        metadata. Returns ``(rubric, aggregation, rubric_ref, service_name)``;
        ``rubric_ref=None`` means use component metadata (no Rubric-class snapshot).
        On resolve failure, log error and return ``None`` (the caller skips and does not create a run).
        """
        if item.rubric_class:
            try:
                spec = resolve_rubric_class(item.rubric_class)
            except RubricClassResolutionError as exc:
                logger.error(
                    "trace_report.rubric_class_invalid name=%s alias=%s err=%s",
                    item.name,
                    item.rubric_class,
                    exc,
                )
                return None
            if spec.class_ref is not None:
                rubric_cls = load_plugin_class(spec.class_ref, what="rubric_class")
                if not (isinstance(rubric_cls, type) and issubclass(rubric_cls, BaseRubric)):
                    logger.error(
                        "trace_report.rubric_class_not_base_rubric name=%s ref=%s",
                        item.name,
                        spec.class_ref,
                    )
                    return None
                # rubric_class mode has no component: service_name takes the agent filter name,
                # falling back to the report namespace (a display field on the report spec)
                return (
                    rubric_cls.criteria(),
                    rubric_cls.default_aggregation(),
                    spec.class_ref,
                    item.agent_name or item.name,
                )
        meta = self._lookup_benchmark_metadata(item.benchmark_name)
        if meta is None:
            logger.error(
                "trace_report.benchmark_not_found name=%s",
                item.benchmark_name,
            )
            return None
        return meta.rubric, meta.aggregation, None, meta.service_name

    @staticmethod
    def _lookup_benchmark_metadata(
        benchmark_name: str,
    ) -> "BenchmarkComponentMetadata | None":
        """Look up benchmark component metadata by registered name (do not instantiate).

        The component base ``BaseAgent.__init__`` requires ``name``, so a no-arg instantiation raises
        ``TypeError``; the daily report only needs rubric / aggregation metadata, and
        ``get_metadata()`` is a classmethod, so look it up on the class directly.
        Skip undecorated classes so one bad class does not take down the whole schedule round.
        """
        for cls in BENCHMARK_SERVICES_CLASSES:
            try:
                meta = cls.get_metadata()
            except RuntimeError:
                continue
            if meta.name == benchmark_name:
                return meta
        return None


__all__ = ["TraceDailyReportScheduler", "TraceReportSchedulerDeps"]
