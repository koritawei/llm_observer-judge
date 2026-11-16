from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.service import main as main_module
from benchmark.settings import Settings
from benchmark.settings import setting as setting_module


def test_create_app_sets_trace_environment_from_agent_eval_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_settings = Settings(env="pre")
    environment: dict[str, str] = {}
    monkeypatch.setattr(setting_module.os, "environ", environment)

    main_module.create_app(settings=app_settings)

    assert environment["TRACE_TRACING_ENVIRONMENT"] == "pre"
    assert environment["TRACE_TIMEOUT"] == "30"


def test_create_app_preserves_explicit_trace_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_settings = Settings(env="daily")
    environment: dict[str, str] = {"TRACE_TIMEOUT": "60"}
    monkeypatch.setattr(setting_module.os, "environ", environment)

    main_module.create_app(settings=app_settings)

    assert environment["TRACE_TIMEOUT"] == "60"


def _external_trigger_state(settings: Settings) -> MagicMock:
    state = MagicMock()
    state.settings = settings
    state.benchmark_cfg = SimpleNamespace(
        external_callback_url_base="",
        external_judge_base_url="",
        external_attribution_base_url="",
        external_api_key="",
        external_rubric_pipeline_id="",
        external_attribution_pipeline_id="",
    )
    state.app.state = SimpleNamespace()
    state.benchmark_service = MagicMock()
    state.aggregation_calculator = MagicMock()
    state.webdev_client = MagicMock()
    state.pages_publisher = MagicMock()
    state.evaluator_service = MagicMock()
    state.triage_service = MagicMock()
    # Skip wiring WebuiTaskScheduler; the unit test watches only the daily.start ordering
    state.runtime = None
    state.daily_scheduler = None
    state.trace_report_scheduler = None
    state.webui_task_scheduler = None
    state.scheduler_hub = None
    return state


def _patch_external_trigger_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncMock:
    """Stub storage / runner / hub schedule construction, and return a daily.start spy."""
    start = AsyncMock()
    scheduler = MagicMock()
    scheduler.start = start
    monkeypatch.setattr(main_module, "build_data_storage_client", lambda: MagicMock())
    monkeypatch.setattr(main_module, "ExternalBenchmarkRunner", MagicMock())
    monkeypatch.setattr(main_module, "SchedulerHub", MagicMock())
    monkeypatch.setattr(
        main_module,
        "DailyExternalBenchmarkScheduler",
        MagicMock(return_value=scheduler),
    )
    monkeypatch.setattr(
        main_module,
        "TraceDailyReportScheduler",
        MagicMock(return_value=MagicMock(start=AsyncMock())),
    )
    return start


@pytest.mark.asyncio
async def test_daily_scheduler_not_started_when_scheduled_tasks_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_EVAL_SCHEDULED_TASKS_ENABLED", raising=False)
    start = _patch_external_trigger_deps(monkeypatch)
    state = _external_trigger_state(Settings())
    await main_module._ls_init_external_trigger(state)
    start.assert_not_awaited()
    await main_module._ls_start_hub_schedulers(state)
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_wire_hub_schedulers_does_not_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wire only constructs; start must wait until queue_manager is injected and is then triggered by _ls_start_hub_schedulers."""
    monkeypatch.setenv("AGENT_EVAL_SCHEDULED_TASKS_ENABLED", "true")
    start = _patch_external_trigger_deps(monkeypatch)
    state = _external_trigger_state(Settings())
    await main_module._ls_init_external_trigger(state)
    start.assert_not_awaited()
    assert state.daily_scheduler is not None


@pytest.mark.asyncio
async def test_daily_scheduler_starts_when_scheduled_tasks_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_EVAL_SCHEDULED_TASKS_ENABLED", "true")
    start = _patch_external_trigger_deps(monkeypatch)
    state = _external_trigger_state(Settings())
    await main_module._ls_init_external_trigger(state)
    start.assert_not_awaited()
    await main_module._ls_start_hub_schedulers(state)
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_odps_scheduler_not_started_when_scheduled_tasks_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_EVAL_SCHEDULED_TASKS_ENABLED", raising=False)
    start = AsyncMock()
    monkeypatch.setattr(
        main_module,
        "OdpsExportScheduler",
        MagicMock(return_value=SimpleNamespace(start=start)),
    )
    state = MagicMock()
    state.settings = Settings()
    await main_module._ls_start_odps_export_scheduler(state)
    start.assert_not_awaited()


# ════════════════════════════════════════════════════════════════════════════
# _ls_finalize_benchmark Phase 2 CAS (dedupe across several trigger sources)
# ════════════════════════════════════════════════════════════════════════════


def _make_finalize_state(run: Any) -> MagicMock:
    state = MagicMock()
    state.benchmark_service.get_run = AsyncMock(return_value=run)
    state.app.state = SimpleNamespace(external_benchmark_runner=None)
    return state


class TestLsFinalizeBenchmarkCas:
    """_ls_finalize_benchmark must win the AGGREGATING→GENERATING_REPORT CAS before it runs.

    Live incident ieval_external-20260914-q4: the stuck-recovery branch of try_finalize_benchmark
    and the try_finalize_external callback fired concurrently. The same run took both paths and
    generated a report once each, and complete_run was written twice.
    """

    @pytest.mark.asyncio
    async def test_cas_lost_skips_finalize(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = SimpleNamespace(
            run_id="r1",
            status="aggregating",
            config_json={},
            benchmark_name="bm",
        )
        state = _make_finalize_state(run)
        qs = MagicMock()
        qs.update = AsyncMock(return_value=0)  # CAS did not win (another node already advanced)
        monkeypatch.setattr(main_module.BenchmarkRun, "filter", MagicMock(return_value=qs))

        await main_module._ls_finalize_benchmark(state, "r1")

        qs.update.assert_awaited_once()
        # After losing the CAS, do not touch aggregation/report/completion again
        state.benchmark_service.load_criterion_entries_from_pipeline.assert_not_called()
        state.benchmark_service.complete_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_cas_won_delegates_external_assembly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = SimpleNamespace(
            run_id="r1",
            status="aggregating",
            config_json={},
            benchmark_name="search_eval",
        )
        state = _make_finalize_state(run)
        ext_runner = MagicMock()
        ext_runner.publish_external_report = AsyncMock(return_value="https://oss/r.html")
        state.app.state = SimpleNamespace(external_benchmark_runner=ext_runner)
        qs = MagicMock()
        qs.update = AsyncMock(return_value=1)  # CAS won
        monkeypatch.setattr(main_module.BenchmarkRun, "filter", MagicMock(return_value=qs))
        monkeypatch.setattr(main_module, "uses_external_report_assembly", lambda *a, **k: True)

        await main_module._ls_finalize_benchmark(state, "r1")

        ext_runner.publish_external_report.assert_awaited_once()
        # complete_run is finished inside publish_external_report; this function does not repeat it
        state.benchmark_service.complete_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_missing_returns_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = _make_finalize_state(None)
        filter_mock = MagicMock()
        monkeypatch.setattr(main_module.BenchmarkRun, "filter", filter_mock)

        await main_module._ls_finalize_benchmark(state, "r-missing")

        filter_mock.assert_not_called()  # When the run does not exist, do not even start the CAS


class TestLsHealerBenchmarkAggregatingFinalize:
    """Healer recovery entry: generating_report must be reset first, then handed to the CAS to decide."""

    @pytest.mark.asyncio
    async def test_generating_report_resets_then_finalizes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = SimpleNamespace(run_id="r1", status="generating_report")
        state = _make_finalize_state(run)
        qs = MagicMock()
        qs.update = AsyncMock(return_value=1)  # reset succeeded
        monkeypatch.setattr(main_module.BenchmarkRun, "filter", MagicMock(return_value=qs))
        finalize = AsyncMock()
        monkeypatch.setattr(main_module, "_ls_finalize_benchmark", finalize)

        await main_module._ls_healer_benchmark_aggregating_finalize(state, "r1")

        qs.update.assert_awaited_once()
        finalize.assert_awaited_once_with(state, "r1")

    @pytest.mark.asyncio
    async def test_generating_report_reset_lost_skips(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = SimpleNamespace(run_id="r1", status="generating_report")
        state = _make_finalize_state(run)
        qs = MagicMock()
        qs.update = AsyncMock(return_value=0)  # A concurrent race: another node already advanced
        monkeypatch.setattr(main_module.BenchmarkRun, "filter", MagicMock(return_value=qs))
        finalize = AsyncMock()
        monkeypatch.setattr(main_module, "_ls_finalize_benchmark", finalize)

        await main_module._ls_healer_benchmark_aggregating_finalize(state, "r1")

        finalize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_aggregating_goes_straight_to_finalize(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = SimpleNamespace(run_id="r1", status="aggregating")
        state = _make_finalize_state(run)
        filter_mock = MagicMock()
        monkeypatch.setattr(main_module.BenchmarkRun, "filter", filter_mock)
        finalize = AsyncMock()
        monkeypatch.setattr(main_module, "_ls_finalize_benchmark", finalize)

        await main_module._ls_healer_benchmark_aggregating_finalize(state, "r1")

        filter_mock.assert_not_called()  # aggregating does not need a reset
        finalize.assert_awaited_once_with(state, "r1")

    @pytest.mark.asyncio
    async def test_completed_status_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = SimpleNamespace(run_id="r1", status="completed")
        state = _make_finalize_state(run)
        finalize = AsyncMock()
        monkeypatch.setattr(main_module, "_ls_finalize_benchmark", finalize)

        await main_module._ls_healer_benchmark_aggregating_finalize(state, "r1")

        finalize.assert_not_awaited()


def _lifespan_state() -> main_module._LifespanState:
    return main_module._LifespanState(
        app=MagicMock(),
        settings=Settings(),
        db_config={},
        trace_record_service=MagicMock(),
        evaluator_service=MagicMock(),
    )


@pytest.mark.asyncio
async def test_eval_mq_init_presets_sidecar_ingest_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "start_eval_mq_consumer", AsyncMock(return_value=None))
    state = _lifespan_state()
    del state.sidecar_ingest_service

    await main_module._ls_init_eval_mq(state)

    assert state.sidecar_ingest_service is None


@pytest.mark.asyncio
async def test_runner_start_failure_leaves_sidecar_none_and_stops_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = MagicMock()
    runner.start = AsyncMock(side_effect=RuntimeError("boom"))
    runner.stop = AsyncMock()
    monkeypatch.setattr(
        main_module,
        "StreamingBenchmarkRunner",
        MagicMock(return_value=runner),
    )
    state = _lifespan_state()
    del state.sidecar_ingest_service

    await main_module._ls_init_benchmark_streaming(state)

    assert state.sidecar_ingest_service is None
    assert state.streaming_runner is None
    runner.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_benchmark_streaming_reclaims_logging_after_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a sidecar MetaQ subscribe, logs must be reclaimed, so processing logs do not go into the VIP client.log."""
    runner = MagicMock()
    runner.start = AsyncMock()
    runner.sidecar_service = MagicMock(active_consumer_count=2)
    monkeypatch.setattr(
        main_module,
        "StreamingBenchmarkRunner",
        MagicMock(return_value=runner),
    )
    ensure = MagicMock(return_value=True)
    monkeypatch.setattr(main_module, "ensure_application_logging", ensure)
    state = _lifespan_state()

    await main_module._ls_init_benchmark_streaming(state)

    ensure.assert_called_with(reason="after_sidecar_ingest")
    assert state.sidecar_ingest_service is runner.sidecar_service
    assert state.streaming_runner is runner


@pytest.mark.asyncio
async def test_shutdown_stops_sidecar_when_runner_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_sidecar = AsyncMock()
    monkeypatch.setattr(main_module, "stop_sidecar_ingest", stop_sidecar)
    monkeypatch.setattr(main_module, "stop_eval_mq_consumer", AsyncMock())
    monkeypatch.setattr(main_module, "_ls_stop_schedulers", AsyncMock())
    monkeypatch.setattr(main_module, "_ls_close_eval_workers", AsyncMock())
    monkeypatch.setattr(main_module, "close_database", AsyncMock())
    state = _lifespan_state()
    state.sidecar_ingest_service = None
    state.streaming_runner = None

    await main_module._ls_shutdown(state)

    stop_sidecar.assert_awaited_once_with(None)
