"""FastAPI entry point for agent-eval.

Owns the Tortoise ORM lifecycle, mounts TraceRecordService on
``app.state`` for downstream business modules, and wires the agentgraph eval runtime and HTTP routes.
"""

import asyncio
import dataclasses
import logging
import os
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
import uvicorn
from benchmark.libs.agenthost.app.providers import ProviderRegistry
from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.agenthost.tools.component_metadata import _build_tool_component_provider
from benchmark.libs.models.database import close_database, register_database
from fastapi import FastAPI
from redis.asyncio import Redis
from tortoise import connections as _tortoise_connections
from tortoise.context import get_current_context, set_global_context

from benchmark.routes.router import api_router
from benchmark.reporting.cluster import (
    registry as cluster_report_registry,  # noqa: F401 - agent registration
)
from benchmark.settings import Settings
from benchmark.settings import settings as default_settings
from benchmark.settings.setting import configure_trace_environment
from benchmark.settings.triage_loader import load_triage_configs
from benchmark.records.database import create_database_config
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import (
    BenchmarkRun,
    BenchmarkRunItem,
)
from benchmark.evaluators import registry as evaluator_registry
from benchmark.libs.jobqueue.staging_buffer import StagingBuffer
from benchmark.http.middleware.error_middleware import (
    ErrorHandlingMiddleware,
    RequestContextMiddleware,
)
from benchmark.http.middleware.trace_context import trace_context_middleware
from benchmark.ingest.mq.runtime.lifecycle import (
    start_eval_mq_consumer,
    stop_eval_mq_consumer,
)
from benchmark.paths import app_config_path, default_conf_dir
from benchmark.services import TraceRecordService
from benchmark.runs import (
    registry as benchmark_registry,
)
from benchmark.runs.common.external_client_factory import (
    ExternalClientFactory,
    ExternalClientFactoryConfig,
)
from benchmark.runs.common.config_helpers import (
    load_aggregation_from_config,
    load_rubric_from_config,
)
from benchmark.runs.common.dimension_stats import uses_external_report_assembly
from benchmark.runs.config import load_benchmark_config
from benchmark.runs.run_service import (
    BenchmarkService,
    CompleteRunWrite,
    ExpandDimensionMetricsSpec,
)
from benchmark.runs.runner import BenchmarkRunner, BenchmarkRunnerDeps, ReportGenerationSpec
from benchmark.runs.schema import ExternalJudgeItemResult
from benchmark.streaming.runner import StreamingBenchmarkRunner
from benchmark.reporting.framework.aggregation_calculator import AggregationCalculator
from benchmark.reporting.framework.report_storage import (
    build_data_storage_client,
    build_report_storage_client,
)
from benchmark.reporting.framework.webdev_client import WebdevReportClient
from benchmark.queue import EvalQueueRegistry
from benchmark.queue.queue_factory import (
    BENCHMARK_CASES_L2,
    apply_slots_to_settings,
    load_queue_prefix_from_yaml,
    load_slots_from_yaml,
)
from benchmark.queue.queue_prefix_policy import (
    STANDARD_DEPLOY_PROFILES,
    QueuePrefixPolicyError,
    QueuePrefixResolveInput,
    RuntimeLaneDetectInput,
    assert_committed_standard_yaml_prefixes,
    resolve_queue_prefix,
    uses_ambiguous_standard_profile_fallback,
)
from benchmark.workers.attribution_worker import create_attribution_wait_worker
from benchmark.workers.benchmark_worker import (
    BenchmarkWorkerDeps,
    create_benchmark_worker,
)
from benchmark.workers.code_attribution_worker import create_code_attribution_worker
from benchmark.workers.daily_scheduler import (
    DailyExternalBenchmarkScheduler,
    DailySchedulerOptions,
)
from benchmark.workers.external_runner import (
    ExternalBenchmarkRunner,
    ExternalReportPublish,
    ExternalRunnerDeps,
)
from benchmark.workers.external_worker import create_external_wait_worker
from benchmark.workers.finalize_trigger import (
    register_finalize_callback,
    try_finalize_external,
)
from benchmark.workers.odps_export_scheduler import OdpsExportScheduler
from benchmark.workers.running_healer import (
    HealerCallbacks,
    RunningHealer,
    recover_stale_with_queue_guard,
)
from benchmark.workers.scheduler_hub import SchedulerHub, SchedulerHubConfig
from benchmark.workers.trace_report_scheduler import (
    TraceDailyReportScheduler,
    TraceReportSchedulerDeps,
)
from benchmark.workers.webui_task_scheduler import (
    WebuiTaskScheduler,
    WebuiTaskSchedulerDeps,
)
from benchmark.services.evaluation import EvaluatorService, build_app_runtime
from benchmark.labeling.case_review_hook import LsCaseReviewAfterRunHook
from benchmark.labeling.client import LabelStudioAuth, LabelStudioClient
from benchmark.labeling.dispatcher import LabelStudioDispatcher
from benchmark.labeling.project_resolver import ProjectResolver, ProjectResolverConfig
from benchmark.labeling.webhook_service import LabelStudioWebhookService
from benchmark.reporting.publish import ReportSiteReportPublisher, build_pages_publisher
from benchmark.trace.tools import TraceToolProvider
from benchmark.attribution.service.service import TriageAssembly, TriageAttributionService
from benchmark.ingest.sidecar.lifecycle import stop_sidecar_ingest
from benchmark.attribution import (
    registry as triage_attribution_registry,
)
from benchmark.attribution.classifiers_registry import triage_classifiers
from benchmark.utils.error_handler import install_error_handlers
from benchmark.utils.logging_setup import ensure_application_logging

logger = get_logger("agent_eval.main")

#: app/main.py → parents[1] == service root (services/agent-eval).
_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# LoggingMetricsProvider / observability middleware logs "metric emitted" at INFO on every call,
# which under high throughput washes out business logs. Raise those loggers to WARNING,
# keep error-level metric logs, and drop the INFO noise.
for _metric_logger in ("message_queue", "agentgraph.observability.middleware"):
    logging.getLogger(_metric_logger).setLevel(logging.WARNING)

# The vipserver_client SDK emits a lot of INFO on host refresh and similar paths (vipserver_client.host_reactor
# and child loggers), which drowns business logs. Raise it to WARNING and keep only exception/failure signals.
logging.getLogger("vipserver_client").setLevel(logging.WARNING)


@dataclasses.dataclass
class _LifespanState:
    """Shared component bag for lifespan assembly stages (orchestration in ``_build_lifespan``).

    Stage functions (``_ls_*``) only read and write declared fields; shutdown closes non-None components in reverse dependency order.
    """

    app: FastAPI
    settings: Settings
    db_config: dict[str, Any]
    trace_record_service: TraceRecordService
    runtime: Any = None
    triage_service: TriageAttributionService | None = None
    ls_client: LabelStudioClient | None = None
    evaluator_service: EvaluatorService | None = None
    benchmark_cfg: Any = None
    benchmark_service: BenchmarkService | None = None
    benchmark_runner: BenchmarkRunner | None = None
    aggregation_calculator: AggregationCalculator | None = None
    webdev_client: WebdevReportClient | None = None
    pages_publisher: ReportSiteReportPublisher | None = None
    scheduler_hub: SchedulerHub | None = None
    daily_scheduler: DailyExternalBenchmarkScheduler | None = None
    trace_report_scheduler: TraceDailyReportScheduler | None = None
    webui_task_scheduler: WebuiTaskScheduler | None = None
    odps_export_scheduler: OdpsExportScheduler | None = None
    mq_service: Any = None
    sidecar_ingest_service: Any = None
    streaming_runner: StreamingBenchmarkRunner | None = None
    eval_queue_manager: EvalQueueRegistry | None = None
    eval_workers: list[Any] = dataclasses.field(default_factory=list)
    staging_buffer: StagingBuffer | None = None
    running_healer: RunningHealer | None = None
    external_judge_client: Any = None
    attribution_provider_registry: Any = None


def create_app(*, settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI app.

    In the lifespan, initialize Tortoise ORM (register_database from libs/data_models),
    instantiate TraceRecordService and mount it on ``app.state``, and wire the
    agentgraph eval runtime (evaluators) and HTTP routes.
    """
    # Strip VIP root FileHandler; unify all logs to stdout → application.log.
    ensure_application_logging(reason="create_app")

    app_settings = settings or default_settings
    configure_trace_environment(app_settings)

    db_config = create_database_config(
        db_url=app_settings.database_url,
        environment=app_settings.environment,
    )

    trace_record_service = TraceRecordService()
    lifespan = _build_lifespan(app_settings, db_config, trace_record_service)

    # Security line (ALIBABA-2026-46066077 swagger information leak): docs endpoints are open by default only on daily;
    # pre/prod are always closed (404); AGENT_EVAL_DOCS_ENABLED may override explicitly.
    docs_enabled = app_settings.api_docs_enabled
    app = FastAPI(
        title="agent_eval",
        description="Agent quality signal triage and bad case attribution service",
        version="0.1.0",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.trace_record_service = trace_record_service
    # Middleware: last-added is outermost, so RequestContextMiddleware wraps
    # ErrorHandlingMiddleware — the 500 envelope still gets the X-Request-Id header.
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestContextMiddleware)
    # Bind libs `init_request_context(workflow_name=...)` per HTTP request so
    # LangfuseMiddleware forwards it to the trace root `name` (see middleware docstring).
    app.middleware("http")(trace_context_middleware)
    install_error_handlers(app)
    app.include_router(api_router)
    # Label Studio webhook router (returns 503 if LS not configured)
    from benchmark.routes.label_studio import router as _ls_router  # noqa: PLC0415

    app.include_router(_ls_router)
    return app


def _build_lifespan(
    app_settings: Settings,
    db_config: dict[str, Any],
    trace_record_service: TraceRecordService,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the lifespan: the closure holds the three dependencies resolved by create_app.

    The assembly body is ``lifespan``; large stages will be extracted into module-level ``_ls_*`` functions over time.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup log of eager-import side effects: the registry manifest and @tool_component
        # registration depend on the import (ProviderRegistry.reset() has wiped registrations before; see the re-pin
        # logic below). Log the manifest size at startup so it can be reconciled.
        logger.info(
            "component manifests loaded: evaluators=%d triage_attribution=%d "
            "trace_tool=%s amap_trace_tool=%s",
            len(evaluator_registry.__all__),
            len(triage_attribution_registry.__all__),
            TraceToolProvider.__name__,
            "none",
        )
        # Initialise Tortoise ORM + auto-register observers via
        # register_database() from libs/data_models.
        logger.info(
            "Initialising Tortoise ORM env=%s",
            app_settings.environment,
        )
        await register_database(app, db_config)
        from benchmark.domain.runtime_config_store import start_runtime_config_refresh  # noqa: PLC0415

        await start_runtime_config_refresh()

        # Register as the global fallback so an HTTP request task can see the Tortoise context built by the lifespan task.
        db_ctx = get_current_context()
        if db_ctx is not None:
            set_global_context(db_ctx)

        state = _LifespanState(
            app=app,
            settings=app_settings,
            db_config=db_config,
            trace_record_service=trace_record_service,
        )
        await _ls_check_benchmark_schema()

        # Inject settings before bootstrap so TraceToolProvider.from_config can reuse them.
        # Pin the tool metadata adapter again: an import side effect may have been cleared by ProviderRegistry.reset();
        # without the re-pin, @tool_component(trace) coerce fails → LookupError.
        ProviderRegistry.register_metadata_adapter("tool", _build_tool_component_provider)
        runtime = build_app_runtime(app_config_path(app_settings.env), _SERVICE_ROOT)
        runtime.settings = app_settings  # type: ignore[attr-defined]
        await runtime.bootstrap()
        # yaml observability.log_level may have seeded LOG_LEVEL; refresh namespaces.
        ensure_application_logging(reason="after_runtime_bootstrap")
        app.state.runtime = runtime
        state.runtime = runtime

        # Build TriageAttributionService (the agent fetches traces itself via ToolProvider).
        # Judge-context hooks are owned inside the service (set/clear_triage_judge_context);
        # the assembly layer only passes TriageAssembly (aligned with main's current signature).
        triage_config, notification_config = load_triage_configs(app_config_path(app_settings.env))
        state.triage_service = TriageAttributionService.with_defaults(
            assembly=TriageAssembly(
                agent_manager=runtime.agent_manager,
                classifiers=triage_classifiers(),
                triage_config=triage_config,
                notification_config=notification_config,
                trace_record_service=trace_record_service,
            ),
        )

        await _ls_init_label_studio(state)
        app.state.triage_service = state.triage_service

        # Singleton owned by the lifespan; routes reach it via the
        # get_evaluator_service dependency (Depends), never request.app.state.
        state.evaluator_service = EvaluatorService(
            runtime=runtime,
            trace_record_service=trace_record_service,
            triage_service=state.triage_service,
        )
        app.state.evaluator_service = state.evaluator_service

        # ── Benchmark infrastructure assembly ─────────────────────────────
        _ls_init_benchmark_services(state)

        # Materialize the unified agent view: sync the assembly result into runtime_config_records
        # (kind=agent_view); the catalog API reads the DB directly. Failure is log-only and does not block startup.
        from benchmark.routes.benchmark import sync_agent_view_catalog  # noqa: PLC0415

        await sync_agent_view_catalog()

        # Defer stale recover until after BullMQ init (D16: do not false-kill when wait/running is non-empty)

        # ── External benchmark trigger assembly (§5.1) ──────────────────
        await _ls_init_external_trigger(state)

        await _ls_init_eval_mq(state)

        # ── Benchmark streaming: sidecar MetaQ + optional Kafka ─────────────
        # Kafka is controlled by BENCHMARK_STREAMING_ENABLED, default false.
        # (The sidecar always attempts to start; its own switch is decided by the sidecar config.)
        await _ls_init_benchmark_streaming(state)

        # ── Run scanner: cross-run worker pool ────────────────────────────
        # Periodically scans for non-terminal runs with pending tasks and
        # spawns workers for runs not already being processed by this node.
        # Shares the global worker semaphore with API-spawned workers.
        # → Replaced by BullMQ: Workers are driven by BullMQ queues; RunScanner is no longer needed.

        # ── BullMQ eval-task dispatch startup ───────────────────────────────
        # 1-7. YAML config / Manager / Staging / finalize callback registration / Workers
        #     / stale recover / running healer / queue_manager injection (including degrade).
        await _ls_init_bullmq_stack(state)
        # hub start must come after QM injection (or degrade); a separate call keeps bullmq init from blowing up.
        try:
            await _ls_start_hub_schedulers(state)
        except Exception:
            logger.exception(
                "hub schedulers start failed; continuing without daily/trace/webui triggers",
            )

        try:
            yield
        finally:
            # Stop the daily scheduler first (no new external benchmarks),
            # then healer / BullMQ Workers (no new jobs consumed), then background consume and watch,
            # then runtime / DB (reverse-order detail is in _ls_shutdown).
            await _ls_shutdown(state)

    return lifespan


async def _ls_check_benchmark_schema() -> None:
    """Startup schema fail-fast: a missing column only warns (route layer 503 is the fallback) and does not block startup.

    If benchmark_runs lacks service_name / benchmark_run_items lacks attribution_job_id,
    every BenchmarkRunItem query raises UndefinedColumnError — run the migration script first.
    """
    # Fail fast if benchmark_runs is missing / still on business_agent_name,
    # or benchmark_run_items is missing attribution_job_id (async attribution
    # callback matching). Without attribution_job_id ALL BenchmarkRunItem
    # queries fail with UndefinedColumnError — apply migrate_attribution_job_id.sql.
    try:
        _conn = _tortoise_connections.get("default")
        await _conn.execute_query(
            "SELECT service_name FROM benchmark_runs WHERE false",
        )
        await _conn.execute_query(
            "SELECT attribution_job_id FROM benchmark_run_items WHERE false",
        )
        # A missing streaming-task mirror table only warns (mirror is fail-soft and does not block consumption).
        try:
            await _conn.execute_query(
                "SELECT task_id FROM benchmark_kafka_tasks WHERE false",
            )
        except Exception as _mirror_exc:
            logger.warning(
                "benchmark_kafka_tasks schema check failed: %s — apply "
                "services/agent-eval/sql/schema.sql (section 20). "
                "Streaming task mirror persistence will be skipped until fixed.",
                _mirror_exc,
            )
    except Exception as _schema_exc:
        _msg = str(_schema_exc).lower()
        if "benchmark_runs" in _msg or "service_name" in _msg:
            logger.error(
                "benchmark schema check failed: %s — apply "
                "services/agent-eval/sql/schema.sql and/or "
                "ALTER TABLE benchmark_runs RENAME COLUMN "
                "business_agent_name TO service_name. "
                "POST /v1/benchmarks/*/run will return 503 until fixed.",
                _schema_exc,
            )
        elif "attribution_job_id" in _msg or "benchmark_run_items" in _msg:
            logger.error(
                "benchmark_run_items schema check failed: %s — apply "
                "services/agent-eval/scripts/migrate_attribution_job_id.sql "
                "(ALTER TABLE benchmark_run_items ADD COLUMN "
                "IF NOT EXISTS attribution_job_id VARCHAR(256)). "
                "Judge/attribution callbacks will fail with "
                "OperationalError until fixed.",
                _schema_exc,
            )
        else:
            logger.warning(
                "benchmark schema check skipped/unexpected: %s",
                _schema_exc,
            )


async def _ls_init_label_studio(state: _LifespanState) -> None:
    """Wire the Label Studio integration from config; degrade to disabled when credentials are incomplete or init fails."""
    settings = state.settings
    ls_client = None
    _ls_auth_mode = settings.label_studio_auth_mode
    _ls_creds_ok = (_ls_auth_mode == "token" and bool(settings.label_studio_api_key)) or (
        _ls_auth_mode == "compat"
        and bool(settings.label_studio_email)
        and bool(settings.label_studio_password)
    )
    if settings.label_studio_enabled and not _ls_creds_ok:
        logger.warning(
            "label_studio enabled but credentials incomplete (auth_mode=%s) — disabled",
            _ls_auth_mode,
        )
    if settings.label_studio_enabled and _ls_creds_ok:
        try:
            ls_client = LabelStudioClient(
                base_url=settings.label_studio_base_url,
                auth=LabelStudioAuth(
                    api_key=settings.label_studio_api_key,
                    mode=settings.label_studio_auth_mode,
                    email=settings.label_studio_email,
                    password=settings.label_studio_password,
                ),
            )
            project_resolver = ProjectResolver(
                client=ls_client,
                config=ProjectResolverConfig(
                    callback_url=settings.label_studio_callback_url,
                    webhook_secret=settings.label_studio_webhook_secret,
                    default_label_config=str(
                        _SERVICE_ROOT / settings.label_studio_default_label_config
                    ),
                    label_config_map={
                        k: str(_SERVICE_ROOT / v)
                        for k, v in settings.label_studio_label_config_map.items()
                    },
                ),
            )
            count = await project_resolver.warm()
            logger.info("ls_cache warmed: %d projects", count)

            ls_dispatcher = LabelStudioDispatcher(
                client=ls_client,
                project_resolver=project_resolver,
                trace_record_service=state.trace_record_service,
            )
            state.triage_service.set_after(
                LsCaseReviewAfterRunHook(
                    dispatcher=ls_dispatcher,
                    inner=state.triage_service.after_hook,
                )
            )
            state.app.state.ls_webhook_service = LabelStudioWebhookService(
                project_resolver=project_resolver,
                trace_record_service=state.trace_record_service,
            )
            logger.info("label_studio integration enabled")
            if not settings.label_studio_webhook_secret:
                logger.warning(
                    "label_studio_webhook_secret not configured — "
                    "webhook endpoint is unauthenticated"
                )
        except Exception:
            logger.warning("label_studio init failed — disabled", exc_info=True)
            if ls_client is not None:
                ls_client.close()
            ls_client = None
    state.ls_client = ls_client


def _ls_init_benchmark_services(state: _LifespanState) -> None:
    """Wire benchmark infrastructure (service / runner / aggregation / webdev client)."""
    benchmark_cfg = load_benchmark_config(app_config_path(state.settings.env))
    state.benchmark_cfg = benchmark_cfg
    state.benchmark_service = BenchmarkService()
    state.aggregation_calculator = AggregationCalculator()
    state.webdev_client = WebdevReportClient(
        webdev_agent_url=benchmark_cfg.webdev_agent_url,
        timeout=benchmark_cfg.webdev_report_timeout_s,
        storage_client=build_report_storage_client(),
    )
    # Aone Pages republisher: None when RemoteConfig is disabled / config is missing (graceful degrade)
    state.pages_publisher = build_pages_publisher()
    state.benchmark_runner = BenchmarkRunner(
        BenchmarkRunnerDeps(
            evaluator_service=state.evaluator_service,
            benchmark_service=state.benchmark_service,
            aggregation_calculator=state.aggregation_calculator,
            webdev_client=state.webdev_client,
            triage_service=state.triage_service,
            evaluate_timeout_s=benchmark_cfg.evaluate_timeout_s,
            pages_publisher=state.pages_publisher,
        ),
    )
    app = state.app
    app.state.benchmark_config = benchmark_cfg
    app.state.benchmark_service = state.benchmark_service
    app.state.benchmark_runner = state.benchmark_runner
    app.state.benchmark_background_tasks: set[asyncio.Task] = set()


def _scheduler_redis() -> "Redis | None":
    """Shared Redis for scheduled-task leader election (exists only after the BullMQ stack inits; None if not ready)."""
    from benchmark.libs.queue_runtime import get_redis  # noqa: PLC0415

    return get_redis()


async def _ls_init_external_trigger(state: _LifespanState) -> None:
    """Wire the external benchmark runner + daily scheduler (degrade on failure; do not block startup)."""
    benchmark_cfg = state.benchmark_cfg
    try:
        _external_storage_client = build_data_storage_client()
        _callback_url_base = benchmark_cfg.external_callback_url_base

        # Aone client factory (§6.3): the constructor takes the process-level
        # base_url / api_key / default pipeline id once (the read path is unchanged),
        # and at runtime caches and reuses a client per full per-run target URL.
        _external_judge_base = benchmark_cfg.external_judge_base_url
        # Attribution base URL: explicit if set, else fallback to judge's
        _external_attr_base = benchmark_cfg.external_attribution_base_url or _external_judge_base
        _external_api_key = benchmark_cfg.external_api_key
        _external_judge_pipeline_id = benchmark_cfg.external_rubric_pipeline_id
        # Attribution pipeline: dedicated ID if set, else fallback to judge's
        _external_attr_pipeline_id = (
            benchmark_cfg.external_attribution_pipeline_id or _external_judge_pipeline_id
        )
        _external_client_factory = None
        _external_judge_client = None
        _external_attribution_client = None
        if _external_judge_base and _external_api_key and _external_judge_pipeline_id:
            _external_client_factory = ExternalClientFactory(
                ExternalClientFactoryConfig(
                    judge_base_url=_external_judge_base,
                    api_key=_external_api_key,
                    default_judge_pipeline_id=_external_judge_pipeline_id,
                    attribution_base_url=_external_attr_base,
                    default_attribution_pipeline_id=_external_attr_pipeline_id,
                ),
            )
            # Client for the default URL (equivalent to today's singleton; a per-run target is
            # built dynamically by the factory from the URL)
            _external_judge_client = _external_client_factory.get_judge_client()
            if _external_attr_base:
                _external_attribution_client = _external_client_factory.get_attribution_client()
        elif _external_judge_base:
            logger.warning(
                "aone_base_url set but api_key/pipeline_id missing; skipping Aone clients",
            )

        external_benchmark_runner = ExternalBenchmarkRunner(
            benchmark_service=state.benchmark_service,
            storage_client=_external_storage_client,
            deps=ExternalRunnerDeps(
                aggregation_calculator=state.aggregation_calculator,
                webdev_client=state.webdev_client,
                external_judge_client=_external_judge_client,
                external_attribution_client=_external_attribution_client,
                external_client_factory=_external_client_factory,
                evaluator_service=state.evaluator_service,
                triage_service=state.triage_service,
                pages_publisher=state.pages_publisher,
                callback_url_base=_callback_url_base,
            ),
        )
        state.app.state.external_benchmark_runner = external_benchmark_runner
        await _ls_wire_hub_schedulers(state, external_benchmark_runner)
    except Exception:
        logger.exception(
            "external benchmark scheduler init failed; continuing without daily trigger",
        )


def _bind_scheduler_hub_leader_lock(state: _LifespanState) -> None:
    """Bind the SchedulerHub leader lock to the current BullMQ ``eval_queue_prefix`` lane.

    Call before ``hub.start`` / the first lock acquire: prod / pre / isolated prefixes stay independent,
    so they do not share ``agent-eval:scheduler:leader`` and race one lock across environments.
    """
    hub = state.scheduler_hub
    if hub is None:
        return
    prefix = str(state.settings.eval_queue_prefix or "").strip()
    if not prefix:
        logger.warning(
            "scheduler.leader_lock_bind_skipped reason=empty_queue_prefix lock_key=%s",
            hub.leader_lock_key,
        )
        return
    hub.bind_queue_prefix(prefix)


async def _ls_wire_hub_schedulers(
    state: _LifespanState,
    external_benchmark_runner: ExternalBenchmarkRunner,
) -> None:
    """Build the shared SchedulerHub and wire daily / trace / webui (**without** ``start``).

    The process holds one hub: if several hubs share the default leader-lock key, a hub started later
    in the same process can never become leader, and jobs stay silent all day.

    After BullMQ resolves ``queue_prefix``, the leader lock is bound by
    :func:`_bind_scheduler_hub_leader_lock` as ``{prefix}:scheduler:leader``.

    ``start`` must come after BullMQ ``set_queue_manager`` (see
    :func:`_ls_start_hub_schedulers`): if startup compensation runs before the queue is injected, the
    search_eval family falls into ``LocalEvalService``'s default Langfuse lookup.
    """
    # lock_key is provisional for now; bind it after ``_ls_init_bullmq_stack`` resolves the prefix
    hub = SchedulerHub(
        SchedulerHubConfig(
            redis_getter=_scheduler_redis,
        )
    )
    state.scheduler_hub = hub

    state.daily_scheduler = DailyExternalBenchmarkScheduler(
        runner=external_benchmark_runner,
        options=DailySchedulerOptions(
            benchmark_registry=benchmark_registry,
            hub=hub,
        ),
    )

    # Trace daily report (T+1): only aggregate trace_pipeline_records; do not fetch OSS / do not judge
    state.trace_report_scheduler = TraceDailyReportScheduler(
        TraceReportSchedulerDeps(
            benchmark_service=state.benchmark_service,
            aggregation_calculator=state.aggregation_calculator,
            webdev=state.webdev_client,
            pages_publisher=state.pages_publisher,
            hub=hub,
        ),
    )

    if state.benchmark_service is None or state.runtime is None:
        return
    state.webui_task_scheduler = WebuiTaskScheduler(
        WebuiTaskSchedulerDeps(
            hub=hub,
            benchmark_service=state.benchmark_service,
            runtime=state.runtime,
            external_runner=external_benchmark_runner,
        ),
    )


async def _ls_start_hub_schedulers(state: _LifespanState) -> None:
    """Start daily / trace / webui schedules (must be after ``set_queue_manager``).

    Plan A: the schedule execution side all lives in agent-eval (WebUI only writes the DB). ``start``
    only after BullMQ is ready (or explicitly degraded), so ``startup_compensation`` / WebUI
    catch_up does not race queue injection; WebUI sync fills today's missed fires at the end of each round.
    """
    _bind_scheduler_hub_leader_lock(state)
    enabled = state.settings.scheduled_tasks_enabled
    daily = state.daily_scheduler
    if daily is not None:
        if not enabled:
            logger.warning(
                "daily_scheduler.disabled by AGENT_EVAL_SCHEDULED_TASKS_ENABLED "
                "(default false; set true to run scheduled tasks)",
            )
        elif state.settings.external_daily_benchmark_disabled:
            logger.warning(
                "daily_scheduler.disabled by EXTERNAL_DAILY_BENCHMARK_DISABLED; "
                "ignoring RemoteConfig external_daily.enabled",
            )
        else:
            await daily.start()

    trace = state.trace_report_scheduler
    if trace is not None:
        if not enabled:
            logger.warning(
                "trace_report_scheduler.disabled by AGENT_EVAL_SCHEDULED_TASKS_ENABLED "
                "(default false; set true to run scheduled tasks)",
            )
        elif state.settings.trace_daily_report_disabled:
            logger.warning(
                "trace_report_scheduler.disabled by TRACE_DAILY_REPORT_DISABLED; "
                "ignoring RemoteConfig trace_daily_report.enabled",
            )
        else:
            await trace.start()

    webui = state.webui_task_scheduler
    if webui is None:
        return
    if not enabled:
        logger.warning(
            "webui_task_scheduler.disabled by AGENT_EVAL_SCHEDULED_TASKS_ENABLED "
            "(default false; set true to run scheduled tasks)",
        )
        return
    await webui.start()


async def _ls_init_eval_mq(state: _LifespanState) -> None:
    """Wire realtime message consumption. The sidecar is started by the streaming runner."""
    settings = state.settings
    app = state.app

    # Eval-message MetaQ consumption is assembled in mq_consumer.runtime.lifecycle, keeping main small.
    # Default None: shutdown's stop_eval_mq_consumer(mq_service) always has a binding (None→no-op),
    # so even if this assignment is later moved into a try or an early exception, it will not raise UnboundLocalError.
    state.mq_service = None
    state.mq_service = await start_eval_mq_consumer(
        evaluator_service=state.evaluator_service,
        settings=settings,
    )
    # The sidecar is started by the streaming runner. Set None first: if start() fails and returns early,
    # shutdown can still call stop_sidecar_ingest without an AttributeError.
    state.sidecar_ingest_service = None
    if state.mq_service is not None:
        app.state.mq_consumer_service = state.mq_service
    # MQ / RemoteConfig may reset logging; re-bind app namespaces (do not clear root).
    ensure_application_logging(reason="after_mq_consumer")

    logger.info(
        "Service switches resolved: benchmark_streaming=%s label_studio=%s source=env_var",
        settings.benchmark_streaming_enabled,
        settings.label_studio_enabled,
    )


async def _ls_init_benchmark_streaming(state: _LifespanState) -> None:
    """Wire the benchmark runner: the sidecar always attempts to start; Kafka follows the env var.

    ``BENCHMARK_STREAMING_ENABLED=false`` (default) does not create a Kafka consumer;
    when ``true``, the runner attaches the reconciler and consumes per the declared config.
    The sidecar's own switch is still decided by the sidecar config. Startup failure does not block HTTP.
    """
    settings = state.settings
    app = state.app
    kafka_enabled = settings.benchmark_streaming_enabled
    runner = StreamingBenchmarkRunner(
        evaluator_service=state.evaluator_service,
        settings=settings,
        kafka_enabled=kafka_enabled,
    )
    state.sidecar_ingest_service = None
    try:
        await runner.start()
    except Exception:
        logger.exception("Benchmark streaming failed to start; continuing without it")
        try:
            await runner.stop()
        except Exception:
            logger.exception("Benchmark runner stop after failed start failed")
        return
    state.streaming_runner = runner
    app.state.streaming_runner = runner
    state.sidecar_ingest_service = runner.sidecar_service
    if runner.sidecar_service is not None:
        app.state.sidecar_ingest_service = runner.sidecar_service
    # The MetaQ PushConsumer connects only when the sidecar subscribes, and may attach the VIP FileHandler again;
    # reclaim after runner.start(), or consumption-processing logs go only to client.log.
    ensure_application_logging(reason="after_sidecar_ingest")
    sidecar = runner.sidecar_service
    sidecar_consumers = int(getattr(sidecar, "active_consumer_count", 0) or 0) if sidecar else 0
    logger.info(
        "Benchmark runner started: sidecar=%s kafka=%s consumers=%s",
        sidecar is not None,
        kafka_enabled,
        sidecar_consumers,
    )


async def _ls_finalize_benchmark(state: _LifespanState, run_id: str) -> None:
    """Sync benchmark CAS winner: aggregate metrics + generate the report → completed.

    Distributed finalize on the BullMQ path (aligned with BenchmarkRunner
    ``_try_generate_distributed_report``). ``try_finalize_benchmark``
    only completes RUNNING → AGGREGATING (including the stuck-recovery branch that lost the CAS but
    observed aggregating); the first thing this function does is the Phase 2 CAS
    (AGGREGATING → GENERATING_REPORT) — the same transition as external ``_finalize_run``, so when
    several triggers (judge callback / processor finally / healer) race, only one executor enters
    the later aggregate+report, and the same run does not generate the report twice or complete_run twice
    (ieval_external-20260914-q4 once had both paths generate a report concurrently).

    OSS/external style (including search_eval sync judge), if ``external_benchmark_runner`` is mounted,
    delegates the report phase to ``publish_external_report``, the same path as regenerate.
    """
    benchmark_service = state.benchmark_service
    benchmark_runner = state.benchmark_runner
    run = await benchmark_service.get_run(run_id)
    if run is None:
        logger.warning(
            "benchmark.finalize_run_missing run_id=%s",
            run_id,
        )
        return
    # Phase 2 CAS: only the executor that wins AGGREGATING → GENERATING_REPORT
    # may continue. The external callback path ``_finalize_run`` holds the same lock.
    won = await BenchmarkRun.filter(
        run_id=run_id,
        status=BenchmarkStatus.AGGREGATING.value,
    ).update(status=BenchmarkStatus.GENERATING_REPORT.value)
    if won == 0:
        logger.info(
            "benchmark.finalize_cas_lost run_id=%s — 已被其他触发源推进，跳过",
            run_id,
        )
        return
    config = run.config_json if isinstance(run.config_json, dict) else {}
    rubric = load_rubric_from_config(config)
    aggregation = load_aggregation_from_config(config)
    generate_report = config.get("generate_report", True)
    benchmark_name = (
        config.get("benchmark_services", "") or getattr(run, "benchmark_name", "") or ""
    )

    # OSS sync-judge (search_eval and similar): shares the report path with regenerate / async-external
    ext_runner = getattr(state.app.state, "external_benchmark_runner", None)
    if ext_runner is not None and uses_external_report_assembly(
        config,
        str(benchmark_name) if benchmark_name else getattr(run, "benchmark_name", None),
    ):
        logger.info(
            "benchmark.finalize_via_external_report run_id=%s benchmark=%s",
            run_id,
            benchmark_name,
        )
        await ext_runner.publish_external_report(
            ExternalReportPublish(
                run_id=run_id,
                config=config,
                rubric=rubric,
                aggregation=aggregation,
            ),
        )
        return

    # 1. SQL-aggregate case scores → job metrics
    sql_entries = await benchmark_service.load_criterion_entries_from_pipeline(
        run_id=run_id,
        rubric=rubric,
        pass_threshold=aggregation.pass_threshold,
    )
    # Restore react_rounds from persisted data (the worker writes metrics_json as progress advances)
    react = benchmark_runner._extract_persisted_react_rounds(
        run.metrics_json if run.metrics_json else None,
    )
    if react and sql_entries:
        sql_entries[0] = dataclasses.replace(
            sql_entries[0],
            react_rounds_json=react,
        )
    final_score, _per_criterion, metrics = await state.aggregation_calculator.aggregate(
        aggregation,
        rubric,
        sql_entries,
    )
    # Dimension expansion (no-op when ieval has no registration; search_eval falls back to the external path above)
    await benchmark_service.expand_dimension_metrics(
        ExpandDimensionMetricsSpec(
            run_id=run_id,
            benchmark_name=getattr(run, "benchmark_name", None) or benchmark_name or None,
            metrics=metrics,
            config=config,
        ),
    )
    await benchmark_runner._attach_attribution_to_metrics(
        metrics,
        run_id,
        benchmark_name=getattr(run, "benchmark_name", None) or benchmark_name or None,
    )

    # fused_avg override (fused mode has fused rows; per_eval returns None)
    fused_avg = await benchmark_service.load_fused_score_average(
        run_id,
    )
    if fused_avg is not None:
        final_score = fused_avg
        metrics.aggregation_method = "fused_avg"

    # duration_ms (prefer run.started_at, consistent across nodes)
    duration_ms = benchmark_runner._run_duration_ms(run, 0.0)
    metrics.total_duration_ms = duration_ms

    # 2. Report generation
    report_url: str | None = None
    if generate_report and benchmark_name:
        await benchmark_service.update_run_status(
            run_id,
            BenchmarkStatus.GENERATING_REPORT,
            final_score=final_score,
            metrics_json=metrics.model_dump(),
        )
        try:
            benchmark_services = state.runtime.agent_manager.get(
                benchmark_name,
            )
            report_url = await benchmark_runner._generate_report(
                ReportGenerationSpec(
                    benchmark_services=benchmark_services,
                    run_id=run_id,
                    metrics=metrics,
                    final_score=final_score,
                ),
            )
        except (KeyError, LookupError):
            logger.warning(
                "benchmark.finalize_component_not_found run_id=%s name=%s",
                run_id,
                benchmark_name,
            )
        except Exception:
            logger.exception(
                "benchmark.finalize_report_failed run_id=%s — completing without report",
                run_id,
            )

    # 3. Complete
    await benchmark_service.complete_run(
        CompleteRunWrite(
            run_id=run_id,
            final_score=final_score,
            metrics_json=metrics.model_dump(),
            report_meta_json=(
                {
                    "report_url": report_url,
                    "generated_at": datetime.now(UTC).isoformat(),
                }
                if report_url
                else None
            ),
            duration_ms=duration_ms,
        ),
    )


async def _ls_finalize_external(state: _LifespanState, run_id: str) -> None:
    runner = getattr(state.app.state, "external_benchmark_runner", None)
    if runner is None:
        raise RuntimeError(
            "external_benchmark_runner not initialized; cannot finalize external run",
        )
    await runner._finalize_run(run_id)


def _ls_register_finalize_callbacks(state: _LifespanState) -> None:
    """Register the benchmark-batch / external finalize callbacks (fired by the BullMQ CAS winner).

    - external → ExternalBenchmarkRunner._finalize_run (aggregate + report)
    - batch benchmark → BenchmarkRunner distributed report finish
    """
    register_finalize_callback("benchmark", partial(_ls_finalize_benchmark, state))
    register_finalize_callback("external", partial(_ls_finalize_external, state))


async def _ls_start_eval_workers(state: _LifespanState) -> None:
    """Start the benchmark-batch / external-wait / attribution-wait Workers."""
    qm = state.eval_queue_manager
    eval_workers = state.eval_workers
    settings = state.settings

    # When the external runner is wired, take its public factory (§6.3); None if unwired / unconfigured
    _ext_runner = getattr(state.app.state, "external_benchmark_runner", None)
    _external_client_factory = _ext_runner.external_client_factory if _ext_runner is not None else None

    # Worker queue split (multi-instance isolation §12.7): profile=all consumes the default
    # and L2 queues in one process (single-process compatible); default consumes only the default queue; l2 consumes only the L2
    # isolation queue (a separate process, for physical fault/resource isolation).
    profile = (settings.worker_profile or "all").strip().lower()
    if profile not in ("all", "default", "l2"):
        logger.warning("worker_profile invalid=%s fallback=all", profile)
        profile = "all"
    _bm_deps = BenchmarkWorkerDeps(
        run_service=state.benchmark_service,
        evaluator_service=state.evaluator_service,
        triage_service=state.triage_service,
        settings=settings,
        # Unified execution layer (§8.3): per-run target resolution for async judge
        external_client_factory=_external_client_factory,
    )
    if profile in ("all", "default"):
        eval_workers.append(create_benchmark_worker(qm, _bm_deps))
    if profile in ("all", "l2"):
        eval_workers.append(create_benchmark_worker(qm, _bm_deps, queue_name=BENCHMARK_CASES_L2))
    logger.info("benchmark_workers.started profile=%s", profile)

    # external-wait worker (needs the Aone judge client)
    # §8.3 queue-merge compatibility window: keep draining in-flight legacy jobs; new runs always use
    # the unified benchmark-cases queue (this worker idles for one version cycle, then goes away)
    _external_judge_client = (
        getattr(_ext_runner, "_external_judge_client", None) if _ext_runner is not None else None
    )
    state.external_judge_client = _external_judge_client
    _ext_worker = create_external_wait_worker(
        queue_manager=qm,
        run_service=state.benchmark_service,
        external_judge_client=_external_judge_client,
        settings=settings,
    )
    eval_workers.append(_ext_worker)

    # attribution-wait worker (needs the attribution provider registry)
    _external_attr_client = (
        getattr(_ext_runner, "_external_attribution_client", None) if _ext_runner is not None else None
    )
    # Factory-aware registry (§6.4): take the client from the per-run target URL;
    # if the factory is missing, fall back to today's singleton registry (behavior unchanged)
    _attr_provider_registry = None
    state.attribution_provider_registry = _attr_provider_registry
    _attr_worker = create_attribution_wait_worker(
        queue_manager=qm,
        run_service=state.benchmark_service,
        attribution_provider_registry=_attr_provider_registry,
        settings=settings,
    )
    eval_workers.append(_attr_worker)

    _code_attr_worker = create_code_attribution_worker(
        queue_manager=qm,
        run_service=state.benchmark_service,
        settings=settings,
    )
    eval_workers.append(_code_attr_worker)

    logger.info(
        "BullMQ workers started: dataset + benchmark + external-wait "
        "+ attribution-wait + code-attribution (%d workers)",
        len(eval_workers),
    )


#: D16 startup-recovery multi-node dedupe lock TTL (seconds) — covers the rolling-release batch gap (CD-AONE-001 ≥300s)
_STARTUP_RECOVER_LOCK_TTL_S = 300


async def _ls_recover_stale_runs(state: _LifespanState) -> None:
    """D16 stale recover: refresh heartbeats for in-flight runs first, then judge stale (with a queue guard).

    On multi-node startup/rolling release a short Redis lock (``SET NX EX``) makes only one node run it:
    recover is based on global PG state, so any one node running it once is enough and a skipped node misses nothing;
    the lock is not released early, and the TTL expires on its own to cover the rolling-release window. A lock-acquire
    error fails open (run anyway) — a one-shot fallback recovery would rather write twice (idempotent) than miss one.
    """
    qm = state.eval_queue_manager
    benchmark_service = state.benchmark_service

    lock_key = f"{qm.prefix}:healer:startup_recover"
    try:
        lock_redis = aioredis.from_url(qm.redis_url, decode_responses=True)
        try:
            acquired = await lock_redis.set(
                lock_key,
                f"{socket.gethostname()}:{os.getpid()}",
                nx=True,
                ex=_STARTUP_RECOVER_LOCK_TTL_S,
            )
        finally:
            await lock_redis.aclose()
    except Exception:
        logger.warning(
            "healer.startup_recover_lock_failed (fail-open, proceeding)",
            exc_info=True,
        )
        acquired = True
    if not acquired:
        logger.info(
            "healer.startup_recover_skipped lock_key=%s (another node recovering)",
            lock_key,
        )
        return

    async def _guard_heartbeat(run_id: str) -> None:
        await benchmark_service.heartbeat(run_id)

    try:
        recovered = await recover_stale_with_queue_guard(
            queue_manager=qm,
            recover_fn=benchmark_service.recover_stale_runs,
            heartbeat_fn=_guard_heartbeat,
            stale_timeout_s=state.benchmark_cfg.stale_heartbeat_timeout_s,
        )
        if recovered > 0:
            logger.info(
                "benchmark.recovered %d stale runs on startup (D16)",
                recovered,
            )
    except Exception:
        logger.warning("benchmark.recover_stale_runs failed", exc_info=True)


async def _ls_healer_external_aggregating_finalize(
    state: _LifespanState,
    run_id: str,
) -> None:
    """Recover an external run stuck in RUNNING / AGGREGATING / GENERATING_REPORT.

    - ``running`` → ``try_finalize_external`` (Phase 1 CAS)
    - ``aggregating`` → ``_finalize_run`` (Phase 2 CAS)
    - ``generating_report`` → reset → aggregating → ``_finalize_run``

    Case: after the Nth judge callback increments ``judge_received_count``,
    an intermediate step such as ``_release_and_promote_next_judge`` raises, so
    ``try_finalize_external`` never runs → the run sticks on ``running``.
    Or the process crashes during aggregate/report after winning the Phase 2 CAS → the run sticks on
    ``generating_report``. The healer calls this method when it finds one.
    """
    if "external" not in run_id:
        return
    benchmark_service = state.benchmark_service
    run = await benchmark_service.get_run(run_id)
    if run is None or run.status not in (
        "running",
        "aggregating",
        "generating_report",
    ):
        return
    runner = getattr(state.app.state, "external_benchmark_runner", None)
    if runner is None:
        return
    if run.status == "running":
        logger.info(
            "healer.external_running_recovery run_id=%s (judge done but CAS not fired)",
            run_id,
        )
        await try_finalize_external(run_id, benchmark_service)
    elif run.status == "generating_report":
        # Phase 2 CAS already won (aggregating→generating_report),
        # but aggregate/report crashed. reset → aggregating, then run Phase 2 again.
        n = await BenchmarkRun.filter(
            run_id=run_id,
            status=BenchmarkStatus.GENERATING_REPORT.value,
        ).update(status=BenchmarkStatus.AGGREGATING.value)
        if n == 0:
            return  # Concurrent race: another node already advanced it
        logger.info(
            "healer.external_generating_report_recovery "
            "run_id=%s (reset generating_report→aggregating)",
            run_id,
        )
        await runner._finalize_run(run_id)
    else:
        logger.info(
            "healer.external_aggregating_recovery run_id=%s",
            run_id,
        )
        await runner._finalize_run(run_id)


async def _ls_healer_benchmark_aggregating_finalize(
    state: _LifespanState,
    run_id: str,
) -> None:
    """Recover a benchmark run stuck in AGGREGATING / GENERATING_REPORT.

    - ``aggregating`` → ``_ls_finalize_benchmark`` (its internal Phase 2 CAS
      picks the single executor)
    - ``generating_report`` → conditionally reset back to ``aggregating`` first (aligned with
      the external recovery path; reset affecting 0 rows means another node is already handling it,
      so return), then let the CAS in ``_ls_finalize_benchmark`` decide
    """
    run = await state.benchmark_service.get_run(run_id)
    if run is None or run.status not in ("aggregating", "generating_report"):
        return
    logger.info(
        "healer.benchmark_%s_recovery run_id=%s",
        run.status,
        run_id,
    )
    if run.status == "generating_report":
        n = await BenchmarkRun.filter(
            run_id=run_id,
            status=BenchmarkStatus.GENERATING_REPORT.value,
        ).update(status=BenchmarkStatus.AGGREGATING.value)
        if n == 0:
            return  # Concurrent race: another node already advanced it
        logger.info(
            "healer.benchmark_generating_report_reset run_id=%s "
            "(reset generating_report→aggregating)",
            run_id,
        )
    await _ls_finalize_benchmark(state, run_id)


async def _ls_start_running_healer(state: _LifespanState) -> None:
    """Start the running healer (crash recovery / TTL / heartbeat / aggregating recovery)."""
    if state.settings.eval_queue_healer_disabled:
        logger.warning(
            "running_healer.disabled by EVAL_QUEUE_HEALER_DISABLED; "
            "crash recovery / TTL / heartbeat will NOT run",
        )
        return
    benchmark_service = state.benchmark_service

    async def _healer_benchmark_mark_failed(
        d: dict[str, Any],
        *,
        retry_count: int = 0,
        reason: str = "healer",
        **_: object,
    ) -> None:
        # Idempotency guard: failed→failed must not be counted twice
        updated = (
            await BenchmarkRunItem.filter(
                run_id=d["run_id"],
                item_id=d["item_id"],
            )
            .exclude(status__in=["success", "failed"])
            .update(
                status="failed",
                score=0.0,
                error_message=str(reason)[:2000],
            )
        )
        if updated > 0:
            await benchmark_service.increment_progress(
                d["run_id"],
                failed_delta=updated,
            )

    async def _healer_external_mark_failed(
        d: dict[str, Any],
        *,
        retry_count: int = 0,
        reason: str = "healer",
        **_: object,
    ) -> None:
        await benchmark_service.cas_advance_external_judge_item(
            result=ExternalJudgeItemResult(
                run_id=d["run_id"],
                item_id=d["item_id"],
                failed=True,
                score=0.0,
                judge_json={"error": str(reason), "reason": str(reason)},
                error_message=str(reason)[:2000],
            ),
        )

    _ext_runner = getattr(state.app.state, "external_benchmark_runner", None)
    running_healer = RunningHealer(
        queue_manager=state.eval_queue_manager,
        settings=state.settings,
        callbacks=HealerCallbacks(
            benchmark_heartbeat=benchmark_service.heartbeat,
            benchmark_mark_failed=_healer_benchmark_mark_failed,
            external_mark_failed=_healer_external_mark_failed,
            external_aggregating_finalize=partial(
                _ls_healer_external_aggregating_finalize,
                state,
            ),
            benchmark_aggregating_finalize=partial(
                _ls_healer_benchmark_aggregating_finalize,
                state,
            ),
            external_judge_client=state.external_judge_client,
            attribution_provider_registry=state.attribution_provider_registry,
            benchmark_run_service=benchmark_service,
            external_callback_handler=(
                _ext_runner.handle_callback if _ext_runner is not None else None
            ),
        ),
    )
    await running_healer.start()
    state.running_healer = running_healer
    state.app.state.running_healer = running_healer


async def _ls_close_eval_workers(workers: list[Any]) -> None:
    """Close BullMQ Workers one by one (one failure does not block the rest)."""
    for w in workers:
        try:
            await w.close()
        except Exception:
            logger.warning("BullMQ worker close failed", exc_info=True)


async def _ls_bullmq_init_failed_fallback(
    state: _LifespanState,
    exc: Exception | None = None,
) -> None:
    """Degrade when BullMQ init fails: tear down half-built Workers and fall back to stale recover with no queue guard.

    After degrade, ``app.state.eval_queue_manager`` is None: the health probe reports
    ``degraded`` from that, and trigger-style entries (manual-runs and similar) 503 — so a request is not
    accepted with no worker to consume it and a run frozen until the stale recoverer harvests a "202 false success".
    """
    logger.exception(
        "BullMQ eval_queue init failed; continuing without BullMQ workers",
    )
    # Surface the degrade reason on app.state (health / the accept guard carries it to the caller).
    state.app.state.eval_queue_degraded_reason = (
        f"{type(exc).__name__}: {exc}" if exc is not None else "unknown"
    )
    # On a half-init, tear down Workers already started so a half-built state with no healer does not keep consuming
    await _ls_close_eval_workers(state.eval_workers)
    state.eval_workers.clear()
    if state.eval_queue_manager is not None:
        try:
            await state.eval_queue_manager.close()
        except Exception:
            logger.warning(
                "BullMQ queue_manager close after init failure failed",
                exc_info=True,
            )
    state.eval_queue_manager = None
    state.app.state.eval_queue_manager = None
    # close() before setting None: otherwise the underlying Redis connection leaks (_ls_shutdown
    # closes staging_buffer only when it is non-None, and will not run after it is cleared).
    if state.staging_buffer is not None:
        try:
            await state.staging_buffer.close()
        except Exception:
            logger.warning("staging_buffer close after init failure failed", exc_info=True)
        state.staging_buffer = None
        state.app.state.staging_buffer = None
    # When BullMQ is unavailable, fall back to stale recover with no queue guard
    try:
        recovered = await state.benchmark_service.recover_stale_runs(
            stale_timeout_s=state.benchmark_cfg.stale_heartbeat_timeout_s,
        )
        if recovered > 0:
            logger.info(
                "benchmark.recovered %d stale runs (no BullMQ fallback)",
                recovered,
            )
    except Exception:
        logger.warning("benchmark.recover_stale fallback failed", exc_info=True)


async def _ls_init_bullmq_stack(state: _LifespanState) -> None:
    """Assemble the full BullMQ eval-queue stack (config / manager / workers / healer + failure degrade)."""
    settings = state.settings
    app = state.app
    # 1. Load the queue prefix + slot config from YAML (mutate settings in place; a prefix-policy failure fails startup)
    # Pin only the current profile's bare stored prefix; if the image mounts only one app.<env>.yaml,
    # missing other standard files degrade to a warning, so a deployment that could start is not taken down.
    _yaml_profiles = [settings.env] if settings.env in STANDARD_DEPLOY_PROFILES else []
    try:
        assert_committed_standard_yaml_prefixes(
            default_conf_dir(),
            profiles=_yaml_profiles,
            missing="warn",
            warn=lambda msg: logger.warning(
                "BullMQ queue_prefix policy: %s",
                msg,
            ),
        )
    except QueuePrefixPolicyError:
        logger.exception("BullMQ queue_prefix policy: committed YAML violated")
        raise

    _env_prefix_before_yaml = (settings.eval_queue_prefix or "").strip()
    _yaml_prefix = load_queue_prefix_from_yaml(
        app_config_path(settings.env),
    )
    _lane_input = RuntimeLaneDetectInput(
        env_sign=settings.env_sign,
        sigma_app_stage=settings.sigma_app_stage,
        deploy_profile=settings.env,
        deploy_kind=settings.eval_queue_deploy_kind,
    )
    if uses_ambiguous_standard_profile_fallback(_lane_input):
        # model_fields_set: set from an env var/constructor argument means explicit; a pure default daily is not in the set
        _explicit_env = "env" in settings.model_fields_set
        logger.warning(
            "BullMQ queue_prefix: 无 envSign/SIGMA_APP_STAGE，按 profile=%r "
            "强制标准前缀（AGENT_EVAL_ENV %s）。本地/CI 请设 "
            "AGENT_EVAL_DEPLOY_KIND=local 并指定非保留 EVAL_QUEUE_PREFIX，"
            "避免写入共享 daily/pre/prod Redis。",
            settings.env,
            "已显式设置" if _explicit_env else "未显式设置，正在使用默认值 daily",
        )
    try:
        _resolved_prefix = resolve_queue_prefix(
            QueuePrefixResolveInput(
                deploy_profile=settings.env,
                yaml_prefix=_yaml_prefix,
                env_prefix=_env_prefix_before_yaml,
                env_sign=settings.env_sign,
                sigma_app_stage=settings.sigma_app_stage,
                deploy_kind=settings.eval_queue_deploy_kind,
            )
        )
        if _env_prefix_before_yaml and _env_prefix_before_yaml != _resolved_prefix:
            logger.warning(
                "BullMQ queue_prefix overridden by policy: "
                "requested=%r resolved=%r envSign=%r sigma=%r profile=%s",
                _env_prefix_before_yaml,
                _resolved_prefix,
                settings.env_sign,
                settings.sigma_app_stage,
                settings.env,
            )
        settings.eval_queue_prefix = _resolved_prefix
        # Bind the hub leader lock as soon as the prefix is persisted (must be before ODPS / daily / webui start)
        _bind_scheduler_hub_leader_lock(state)
    except QueuePrefixPolicyError:
        logger.exception(
            "BullMQ queue_prefix policy rejected profile=%s kind=%s "
            "envSign=%r sigma=%r yaml=%r env_prefix=%r",
            settings.env,
            settings.eval_queue_deploy_kind,
            settings.env_sign,
            settings.sigma_app_stage,
            _yaml_prefix,
            _env_prefix_before_yaml,
        )
        raise

    _yaml_slots = load_slots_from_yaml(
        app_config_path(settings.env),
    )
    if _yaml_slots:
        apply_slots_to_settings(settings, _yaml_slots)

    try:
        # 2. Create EvalQueueRegistry (6 Queues)
        qm = EvalQueueRegistry(settings)
        state.eval_queue_manager = qm
        app.state.eval_queue_manager = qm
        logger.info(
            "BullMQ eval_queue config — prefix=%s redis_url=%s "
            "max_attempts=%d sync_retry_ms=%d external_retry_ms=%d "
            "lockDuration=%dms sync_concurrency=%d external_concurrency=%d "
            "sync_slots=%d external_slots=%d max_probes=%d",
            qm.prefix,
            "(set)" if qm.redis_url else "(none)",
            settings.eval_queue_max_attempts,
            settings.sync_retry_max_duration_ms,
            settings.external_retry_max_duration_ms,
            settings.eval_queue_lock_duration_ms,
            settings.eval_queue_worker_concurrency,
            settings.external_worker_concurrency,
            settings.sync_slots_per_run_per_node,
            settings.external_slots_per_run_per_node,
            settings.max_slot_probes_per_run,
        )

        # 2b. StagingBuffer — Redis-backed staging for stream API items
        _staging_redis = aioredis.from_url(
            qm.redis_url,
            decode_responses=True,
        )
        try:
            state.staging_buffer = StagingBuffer(_staging_redis)
        except Exception:
            # Constructor-failure fallback: close the bare Redis handle so the connection does not leak
            await _staging_redis.aclose()
            raise
        app.state.staging_buffer = state.staging_buffer

        # 3. Register finalize callbacks
        _ls_register_finalize_callbacks(state)

        # 4. Start Workers
        await _ls_start_eval_workers(state)

        # 5. D16 stale recover: refresh heartbeats for in-flight runs first, then judge stale
        await _ls_recover_stale_runs(state)

        # 6. Start the running healer (crash recovery / TTL / heartbeat / aggregating recovery)
        await _ls_start_running_healer(state)

        # 7. Inject queue_manager → external runner (BullMQ enqueue + callback promote)
        _ext_runner = getattr(app.state, "external_benchmark_runner", None)
        if _ext_runner is not None and hasattr(_ext_runner, "set_queue_manager"):
            _ext_runner.set_queue_manager(qm)

        # 8. ODPS sampled-export scheduler (depends on queue_runtime Redis; multi-host SET NX dedupes)
        await _ls_start_odps_export_scheduler(state)
    except Exception as _bullmq_exc:
        await _ls_bullmq_init_failed_fallback(state, exc=_bullmq_exc)


async def _ls_start_odps_export_scheduler(state: _LifespanState) -> None:
    """Start OdpsExportScheduler (call only after BullMQ/Redis is ready).

    Do not start when ``AGENT_EVAL_SCHEDULED_TASKS_ENABLED`` is false (the default);
    force it off when ``ODPS_EXPORT_DISABLED=true``; otherwise the odps_job rows in DB
    ``runtime_config_records`` decide. The Redis lock makes only one host run it.
    """
    if not state.settings.scheduled_tasks_enabled:
        logger.warning(
            "odps_export_scheduler.disabled by AGENT_EVAL_SCHEDULED_TASKS_ENABLED "
            "(default false; set true to run scheduled tasks)",
        )
        return
    if state.settings.odps_export_disabled:
        logger.warning(
            "odps_export_scheduler.disabled by ODPS_EXPORT_DISABLED; "
            "ignoring stored odps export jobs",
        )
        return
    try:
        from benchmark.domain.runtime_config_store import (  # noqa: PLC0415
            refresh_odps_export_snapshot,
        )

        _bind_scheduler_hub_leader_lock(state)
        if state.scheduler_hub is None:
            # If wire did not run, build the hub as a fallback; the prefix was already resolved by BullMQ into settings
            state.scheduler_hub = SchedulerHub(
                SchedulerHubConfig(
                    redis_getter=_scheduler_redis,
                    prefix=str(state.settings.eval_queue_prefix or "").strip(),
                )
            )
        hub = state.scheduler_hub
        scheduler = OdpsExportScheduler(
            config_refresher=refresh_odps_export_snapshot,
            hub=hub,
        )
        state.odps_export_scheduler = scheduler
        await scheduler.start()
    except Exception:
        logger.exception(
            "odps_export_scheduler.init_failed; continuing without ODPS export schedule",
        )
        state.odps_export_scheduler = None


async def _ls_stop_schedulers(state: _LifespanState) -> None:
    """Stop ODPS / webui / external-daily / trace schedules, then close the shared SchedulerHub."""
    if state.odps_export_scheduler is not None:
        try:
            await state.odps_export_scheduler.stop()
        except Exception:
            logger.warning("odps_export_scheduler stop failed", exc_info=True)
    if state.webui_task_scheduler is not None:
        try:
            await state.webui_task_scheduler.stop()
        except Exception:
            logger.warning("webui_task_scheduler stop failed", exc_info=True)
    if state.daily_scheduler is not None:
        await state.daily_scheduler.stop()
    if state.trace_report_scheduler is not None:
        try:
            await state.trace_report_scheduler.stop()
        except Exception:
            logger.warning("trace_report_scheduler stop failed", exc_info=True)
    if state.scheduler_hub is not None:
        try:
            await state.scheduler_hub.shutdown()
        except Exception:
            logger.warning("scheduler_hub shutdown failed", exc_info=True)


async def _ls_shutdown(state: _LifespanState) -> None:
    """Shut down in reverse dependency order: schedulers → healer → workers → queue → MQ → …."""
    from benchmark.domain.runtime_config_store import stop_runtime_config_refresh  # noqa: PLC0415

    await stop_runtime_config_refresh()
    await _ls_stop_schedulers(state)
    if state.running_healer is not None:
        try:
            await state.running_healer.stop()
        except Exception:
            logger.warning("running_healer stop failed", exc_info=True)
    # Stop BullMQ Workers
    await _ls_close_eval_workers(state.eval_workers)
    if state.eval_queue_manager is not None:
        await state.eval_queue_manager.close()
    # Close StagingBuffer Redis connection (prevent __del__ on closed loop)
    if state.staging_buffer is not None:
        try:
            await state.staging_buffer.close()
        except Exception:
            logger.warning("staging_buffer close failed", exc_info=True)
    await stop_eval_mq_consumer(state.mq_service)
    # The sidecar is closed by runner.stop. If the runner was never attached, stop it alone so it does not leak.
    if state.streaming_runner is not None:
        logger.info("Stopping benchmark runner (sidecar ingest)...")
        await state.streaming_runner.stop(timeout=60.0)
        logger.info("Benchmark runner stopped")
    else:
        await stop_sidecar_ingest(state.sidecar_ingest_service)
    if state.ls_client is not None:
        state.ls_client.close()
    if state.runtime is not None:
        await state.runtime.shutdown()
    await close_database()
    logger.info("Tortoise ORM connections closed")


if __name__ == "__main__":
    uvicorn.run(
        "benchmark.service.main:create_app",
        factory=True,
        host=default_settings.host,
        port=default_settings.port,
    )
