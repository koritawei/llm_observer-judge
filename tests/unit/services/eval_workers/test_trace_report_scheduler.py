"""Unit tests for TraceDailyReportScheduler.

Locks:
- trigger semantics: RemoteConfig enabled / the benchmark master switch / today's fixed time has not arrived / fired prevents a repeat;
- ``_run_item``: run_id naming (``{name}-{yyyymmdd(T-1)}``), idempotent skip,
  no data means no run is created, a missing benchmark component means no run is created, the T-1 window calculation,
  run_config does not contain the ``benchmark_services`` key (trends are grouped by item.name),
  the full chain (sample → aggregate → attribute → emit the report → complete_run) and fail_run on failure.
"""

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from benchmark.domain.trace_report_config import TraceReportConfig, TraceReportItemConfig
from benchmark.reporting.framework.schema import AggregatedMetrics
from benchmark.workers import trace_report_scheduler as trs_mod
from benchmark.workers.trace_report_scheduler import (
    TraceDailyReportScheduler,
    TraceReportRunSpec,
)

#: Fixed "now": 2026-09-23 10:00 UTC = 18:00 Asia/Shanghai
_FIXED_NOW = datetime(2026, 9, 23, 10, 0, tzinfo=UTC)


def _item(**overrides: Any) -> TraceReportItemConfig:
    base: dict[str, Any] = {
        "name": "globalvoice_daily",
        "benchmark_name": "example",
        "source": "kafka",
    }
    base.update(overrides)
    return TraceReportItemConfig(**base)


def _config(*items: TraceReportItemConfig, enabled: bool = True) -> TraceReportConfig:
    return TraceReportConfig(enabled=enabled, reports=list(items))


def _make_scheduler(config: TraceReportConfig) -> TraceDailyReportScheduler:
    from benchmark.workers.trace_report_scheduler import TraceReportSchedulerDeps

    return TraceDailyReportScheduler(
        TraceReportSchedulerDeps(
            benchmark_service=MagicMock(),
            aggregation_calculator=MagicMock(),
            config=config,
        )
    )


def _patch_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix datetime.now in the scheduler module (construction calls still use the real datetime)."""
    fake_dt = MagicMock(wraps=datetime)
    fake_dt.now = lambda tz=None: _FIXED_NOW
    monkeypatch.setattr(trs_mod, "datetime", fake_dt)


def _patch_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trs_mod, "is_benchmark_disabled", lambda: False)


class _FakeComponent:
    """Minimal benchmark-component stand-in (the scheduler only calls get_metadata)."""

    rubric: list[Any] = []
    aggregation: Any = None

    @classmethod
    def get_metadata(cls) -> MagicMock:
        meta = MagicMock()
        meta.name = "example"
        meta.service_name = "demo_svc"
        meta.rubric = cls.rubric
        meta.aggregation = cls.aggregation
        return meta


def _patch_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rubric: list[Any],
    aggregation: Any,
) -> None:
    _FakeComponent.rubric = rubric
    _FakeComponent.aggregation = aggregation
    monkeypatch.setattr(trs_mod, "BENCHMARK_SERVICES_CLASSES", (_FakeComponent,))


def _criterion(cid: str = "c1", evaluator_ref: str = "e1") -> Any:
    return SimpleNamespace(id=cid, name=cid.upper(), weight=1.0, evaluator_ref=evaluator_ref)


class TestStart:
    @pytest.mark.asyncio
    async def test_start_disabled_noop(self) -> None:
        scheduler = _make_scheduler(_config(_item(), enabled=False))
        await scheduler.start()
        assert scheduler._hub.started is False

    @pytest.mark.asyncio
    async def test_start_registers_tick_and_compensation_via_hub(self) -> None:
        """Both tick and startup compensation go through the hub (via the leader); a direct create_task call is forbidden."""
        scheduler = _make_scheduler(_config(_item(), enabled=True))
        await scheduler.start()
        assert scheduler._hub.started is True
        assert scheduler._hub.has_job("trace-daily-report-tick")
        assert scheduler._hub.has_job("trace-daily-report-startup-compensation")
        tick = scheduler._hub._scheduler.get_job("trace-daily-report-tick")
        assert tick is not None
        assert tick.next_run_time is not None
        await scheduler.stop()
        assert not scheduler._hub.has_job("trace-daily-report-tick")

    @pytest.mark.asyncio
    async def test_tick_emits_alive_heartbeat(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _patch_switches(monkeypatch)
        cfg = _config(_item(schedule_time="23:59"))
        monkeypatch.setattr(trs_mod, "fetch_trace_report_config", lambda: cfg)
        monkeypatch.setattr(
            trs_mod,
            "list_db_trace_report_items",
            AsyncMock(return_value=[]),
        )
        scheduler = _make_scheduler(cfg)
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]
        with caplog.at_level("WARNING", logger="agent_eval.eval_workers.trace_report_scheduler"):
            await scheduler._tick()
        assert any("tick_alive" in r.message for r in caplog.records)


class TestTriggerOnce:
    @pytest.mark.asyncio
    async def test_skip_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            trs_mod,
            "fetch_trace_report_config",
            lambda: _config(_item(), enabled=False),
        )
        scheduler = _make_scheduler(_config(enabled=False))
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]
        await scheduler._trigger_once(reason="test")
        scheduler._run_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_benchmark_switch_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            trs_mod,
            "fetch_trace_report_config",
            lambda: _config(_item()),
        )
        monkeypatch.setattr(trs_mod, "is_benchmark_disabled", lambda: True)
        scheduler = _make_scheduler(_config(_item()))
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]
        await scheduler._trigger_once(reason="test")
        scheduler._run_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_after_slot_with_t_minus_1_date(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The fixed time has passed (18:00 Shanghai > 02:00) → trigger, and the data date = 2026-09-22."""
        _patch_now(monkeypatch)
        _patch_switches(monkeypatch)
        cfg = _config(_item(schedule_time="02:00"))
        monkeypatch.setattr(trs_mod, "fetch_trace_report_config", lambda: cfg)
        scheduler = _make_scheduler(cfg)
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]

        await scheduler._trigger_once(reason="scheduled")

        scheduler._run_item.assert_awaited_once()
        kwargs = scheduler._run_item.await_args.kwargs
        assert kwargs["data_date"] == date(2026, 9, 22)

    @pytest.mark.asyncio
    async def test_skip_before_slot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The fixed time has not arrived (18:00 Shanghai < 23:59) → do not trigger."""
        _patch_now(monkeypatch)
        _patch_switches(monkeypatch)
        cfg = _config(_item(schedule_time="23:59"))
        monkeypatch.setattr(trs_mod, "fetch_trace_report_config", lambda: cfg)
        scheduler = _make_scheduler(cfg)
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]

        await scheduler._trigger_once(reason="scheduled")

        scheduler._run_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_report_items_merged_with_remote_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The two sources are merged: a kind=trace_report entry registered on the streaming side and the RemoteConfig reports

        both trigger; on the same name the DB entry overrides RemoteConfig (a repeated page submit keeps the latest).
        """
        _patch_now(monkeypatch)
        _patch_switches(monkeypatch)
        remote_config_item = _item(name="dup_name", benchmark_name="old", schedule_time="02:00")
        cfg = _config(remote_config_item)
        monkeypatch.setattr(trs_mod, "fetch_trace_report_config", lambda: cfg)
        db_item = _item(
            name="dup_name",
            benchmark_name="amap_agent_benchmark",
            source="sidecar",
            schedule_time="02:00",
        )
        extra_item = _item(
            name="rt_new",
            benchmark_name="amap_agent_benchmark",
            schedule_time="02:00",
        )

        async def _fake_db_items() -> list[TraceReportItemConfig]:
            return [db_item, extra_item]

        monkeypatch.setattr(trs_mod, "list_db_trace_report_items", _fake_db_items)
        scheduler = _make_scheduler(cfg)
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]

        await scheduler._trigger_once(reason="scheduled")

        items = {call.args[0].name: call.args[0] for call in scheduler._run_item.await_args_list}
        # Same-name override: benchmark_name uses the DB entry's amap_agent_benchmark
        assert set(items) == {"dup_name", "rt_new"}
        assert items["dup_name"].benchmark_name == "amap_agent_benchmark"
        assert items["rt_new"].source == "kafka"

    @pytest.mark.asyncio
    async def test_remote_config_custom_overrides_auto_default_db_item(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A user customization overrides the default registration: when the DB item's origin=auto_default, the same-name RemoteConfig item wins."""
        _patch_now(monkeypatch)
        _patch_switches(monkeypatch)
        remote_config_item = _item(
            name="dup_name",
            benchmark_name="custom_benchmark",
            rubric_class="amap_agent_kafka_rubric",
            schedule_time="02:00",
        )
        cfg = _config(remote_config_item)
        monkeypatch.setattr(trs_mod, "fetch_trace_report_config", lambda: cfg)
        db_item = _item(
            name="dup_name",
            benchmark_name="amap_agent_benchmark",
            schedule_time="02:00",
            origin="auto_default",
        )

        async def _fake_db_items() -> list[TraceReportItemConfig]:
            return [db_item]

        monkeypatch.setattr(trs_mod, "list_db_trace_report_items", _fake_db_items)
        scheduler = _make_scheduler(cfg)
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]

        await scheduler._trigger_once(reason="scheduled")

        items = {call.args[0].name: call.args[0] for call in scheduler._run_item.await_args_list}
        assert items["dup_name"].benchmark_name == "custom_benchmark"
        assert items["dup_name"].rubric_class == "amap_agent_kafka_rubric"

    @pytest.mark.asyncio
    async def test_db_items_failure_degrades_to_remote_config_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A DB read exception degrades to RemoteConfig-only, and existing report scheduling is unaffected."""
        _patch_now(monkeypatch)
        _patch_switches(monkeypatch)
        cfg = _config(_item(schedule_time="02:00"))
        monkeypatch.setattr(trs_mod, "fetch_trace_report_config", lambda: cfg)

        async def _boom() -> list[TraceReportItemConfig]:
            raise RuntimeError("db down")

        monkeypatch.setattr(trs_mod, "list_db_trace_report_items", _boom)
        scheduler = _make_scheduler(cfg)
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]

        await scheduler._trigger_once(reason="scheduled")

        scheduler._run_item.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fired_set_prevents_duplicate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_now(monkeypatch)
        _patch_switches(monkeypatch)
        cfg = _config(_item(schedule_time="02:00"))
        monkeypatch.setattr(trs_mod, "fetch_trace_report_config", lambda: cfg)
        scheduler = _make_scheduler(cfg)
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]

        await scheduler._trigger_once(reason="scheduled")
        await scheduler._trigger_once(reason="scheduled")

        scheduler._run_item.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_item_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_now(monkeypatch)
        _patch_switches(monkeypatch)
        cfg = _config(_item(name="", schedule_time="02:00"))
        monkeypatch.setattr(trs_mod, "fetch_trace_report_config", lambda: cfg)
        scheduler = _make_scheduler(cfg)
        scheduler._run_item = AsyncMock()  # type: ignore[method-assign]

        await scheduler._trigger_once(reason="scheduled")

        scheduler._run_item.assert_not_called()


class TestMergedReportItems:
    """The two sources are merged: RemoteConfig reports[] ∪ the DB auto-registered items (regression for a production missed report).

    Production incident: when RemoteConfig ``reports=[]`` the old implementation looked only at RemoteConfig, and a DB-registered item
    (default fixed time 09:00) could never enter the trigger set. After the hub migration, waking is cron's job,
    and the merge result must still be complete — otherwise ``_trigger_once`` likewise misses the DB item.
    """

    @pytest.mark.asyncio
    async def test_merged_includes_db_item_when_remote_config_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = TraceReportConfig(
            enabled=True,
            schedule_time="02:00",
            schedule_timezone="Asia/Shanghai",
            reports=[],
        )

        async def _fake_db_items() -> list[TraceReportItemConfig]:
            return [_item(name="db_daily", schedule_time="09:00")]

        monkeypatch.setattr(trs_mod, "list_db_trace_report_items", _fake_db_items)
        scheduler = _make_scheduler(cfg)

        merged = await scheduler._merged_report_items()

        assert [it.name for it in merged] == ["db_daily"]
        assert merged[0].schedule_time == "09:00"

    @pytest.mark.asyncio
    async def test_merged_unions_remote_config_and_db(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = _config(_item(name="remote_config_daily", schedule_time="02:00"))

        async def _fake_db_items() -> list[TraceReportItemConfig]:
            return [_item(name="db_daily", schedule_time="20:00")]

        monkeypatch.setattr(trs_mod, "list_db_trace_report_items", _fake_db_items)
        scheduler = _make_scheduler(cfg)

        merged = await scheduler._merged_report_items()

        names = {it.name for it in merged}
        assert names == {"remote_config_daily", "db_daily"}

    @pytest.mark.asyncio
    async def test_merged_empty_when_no_items(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = TraceReportConfig(
            enabled=True,
            schedule_time="02:00",
            schedule_timezone="Asia/Shanghai",
            reports=[],
        )

        async def _no_db_items() -> list[TraceReportItemConfig]:
            return []

        monkeypatch.setattr(trs_mod, "list_db_trace_report_items", _no_db_items)
        scheduler = _make_scheduler(cfg)

        merged = await scheduler._merged_report_items()

        assert merged == []


class TestRunItem:
    @pytest.fixture()
    def shanghai(self) -> ZoneInfo:
        return ZoneInfo("Asia/Shanghai")

    @pytest.mark.asyncio
    async def test_existing_run_skipped(self, shanghai: ZoneInfo) -> None:
        scheduler = _make_scheduler(_config(_item()))
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                trs_mod.BenchmarkRun,
                "filter",
                MagicMock(
                    return_value=SimpleNamespace(
                        first=AsyncMock(return_value=SimpleNamespace(status="completed")),
                    ),
                ),
            )
            await scheduler._run_item(_item(), data_date=date(2026, 9, 22), tz=shanghai)
            scheduler._benchmark_service.sample_trace_pipeline_records.assert_not_called()  # type: ignore[union-attr]
        finally:
            monkeypatch.undo()

    @pytest.mark.asyncio
    async def test_no_data_skips_placeholder(self, shanghai: ZoneInfo) -> None:
        scheduler = _make_scheduler(_config(_item()))
        scheduler._benchmark_service.sample_trace_pipeline_records = AsyncMock(  # type: ignore[method-assign]
            return_value=[],
        )
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                trs_mod.BenchmarkRun,
                "filter",
                MagicMock(
                    return_value=SimpleNamespace(first=AsyncMock(return_value=None)),
                ),
            )
            _patch_registry(
                monkeypatch,
                rubric=[_criterion()],
                aggregation=SimpleNamespace(pass_threshold=0.6),
            )
            await scheduler._run_item(_item(), data_date=date(2026, 9, 22), tz=shanghai)
            scheduler._benchmark_service.create_placeholder_run.assert_not_called()  # type: ignore[union-attr]
        finally:
            monkeypatch.undo()

    def test_resolve_metadata_without_instantiating_component(self) -> None:
        """Regression: the component base-class constructor requires name, and daily-report metadata parsing must not instantiate the component class.

        In production ``cls()`` with no arguments once raised TypeError, and the whole report item failed silently
        before sampling (only an item_failed log). The stand-in requires name at construction and records the instantiation.
        """

        class _NamedComponent:
            instantiated = False

            def __init__(self, name: str) -> None:
                _NamedComponent.instantiated = True
                self.name = name

            @classmethod
            def get_metadata(cls) -> SimpleNamespace:
                return SimpleNamespace(
                    name="amap_agent_benchmark",
                    service_name="amap_agent",
                    rubric=[_criterion(evaluator_ref="amap_agent")],
                    aggregation=SimpleNamespace(pass_threshold=0.5),
                )

        scheduler = _make_scheduler(_config(_item()))
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(trs_mod, "BENCHMARK_SERVICES_CLASSES", (_NamedComponent,))
            resolved = scheduler._resolve_rubric_aggregation(
                _item(benchmark_name="amap_agent_benchmark"),
            )
        finally:
            monkeypatch.undo()
        assert resolved is not None
        assert resolved[3] == "amap_agent"
        assert resolved[0][0].evaluator_ref == "amap_agent"
        assert _NamedComponent.instantiated is False

    @pytest.mark.asyncio
    async def test_benchmark_not_found_skips(self, shanghai: ZoneInfo) -> None:
        scheduler = _make_scheduler(_config(_item()))
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                trs_mod.BenchmarkRun,
                "filter",
                MagicMock(
                    return_value=SimpleNamespace(first=AsyncMock(return_value=None)),
                ),
            )
            monkeypatch.setattr(trs_mod, "BENCHMARK_SERVICES_CLASSES", ())
            await scheduler._run_item(_item(), data_date=date(2026, 9, 22), tz=shanghai)
            scheduler._benchmark_service.sample_trace_pipeline_records.assert_not_called()  # type: ignore[union-attr]
        finally:
            monkeypatch.undo()

    @pytest.mark.asyncio
    async def test_full_pipeline(  # noqa: PLR0915 — the full-flow assertions are gathered in one place
        self,
        shanghai: ZoneInfo,
    ) -> None:
        scheduler = _make_scheduler(_config(_item()))
        service = scheduler._benchmark_service
        service.sample_trace_pipeline_records = AsyncMock(return_value=["t1", "t2"])
        service.load_kafka_criterion_entries = AsyncMock(return_value=["entry"])
        service.aggregate_trace_daily_root_cause = AsyncMock(
            return_value={
                "fused": {
                    "field_distributions": {"root_cause": {"tool": 2}},
                    "value_labels": {},
                    "total_attributed": 2,
                },
            },
        )
        service.update_run_status = AsyncMock()
        service.complete_run = AsyncMock()
        service.fail_run = AsyncMock()
        service.create_placeholder_run = AsyncMock(return_value=True)

        metrics = AggregatedMetrics(
            aggregation_method="weighted_avg",
            total_samples=2,
            total_duration_ms=0,
        )
        scheduler._aggregation.aggregate = AsyncMock(  # type: ignore[method-assign]
            return_value=(1.5, {}, metrics),
        )

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                trs_mod.BenchmarkRun,
                "filter",
                MagicMock(
                    return_value=SimpleNamespace(first=AsyncMock(return_value=None)),
                ),
            )
            rubric = [_criterion("c1", "e1"), _criterion("c2", "e2")]
            _patch_registry(
                monkeypatch,
                rubric=rubric,
                aggregation=SimpleNamespace(pass_threshold=0.6),
            )
            generate_report = AsyncMock(return_value="https://oss.example.com/r.html")
            monkeypatch.setattr(trs_mod, "generate_benchmark_report", generate_report)

            await scheduler._run_item(
                _item(project_id="p1", agent_name="amap_chat"),
                data_date=date(2026, 9, 22),
                tz=shanghai,
            )
        finally:
            monkeypatch.undo()

        # The T-1 window + filter-condition assertions (the sample arguments)
        sample_query = service.sample_trace_pipeline_records.await_args.args[0]
        assert sample_query.date_start == datetime(2026, 9, 22, tzinfo=shanghai)
        assert sample_query.date_end == datetime(2026, 9, 23, tzinfo=shanghai)
        assert sample_query.evaluator_names == ["e1", "e2"]
        assert sample_query.source == "kafka"
        assert sample_query.project_id == "p1"
        assert sample_query.agent_name == "amap_chat"

        # Placeholder run: run_id still uses the report-item name; the column benchmark_name = the internal agent;
        # dataset_name = the page dataset name; the namespace goes into config.trace_report_namespace
        run_id, spec = service.create_placeholder_run.await_args.args
        assert run_id == "globalvoice_daily-20260922"
        assert spec.benchmark_name == "example"
        assert spec.service_name == "demo_svc"
        assert spec.dataset_name == "globalvoice_daily"
        assert spec.total_samples == 2
        assert "benchmark_services" not in spec.config_json
        assert spec.config_json["trace_daily_report"] is True
        assert spec.config_json["score_full_mark"] == 1.0
        assert spec.config_json["trace_report_namespace"] == "globalvoice_daily"
        assert spec.config_json["trace_pipeline_source"] == "kafka"
        assert spec.config_json["benchmark_component"] == "example"
        sample = spec.config_json["trace_daily_sample"]
        assert sample["source"] == "kafka"
        assert sample["agent_name"] == "amap_chat"
        assert sample["project_id"] == "p1"
        assert sample["timezone"] == "Asia/Shanghai"
        assert sample["pass_threshold"] == 0.6
        assert sample["evaluator_names"] == ["e1", "e2"]

        # The attribution distribution enters metrics, and report emission finishes + complete_run
        assert "fused" in metrics.attribution_metrics
        generate_report.assert_awaited_once()
        job = generate_report.await_args.args[0]
        assert job.run_id == run_id
        assert job.rubric == rubric
        service.complete_run.assert_awaited_once()
        complete_write = service.complete_run.await_args.args[0]
        assert complete_write.run_id == run_id
        assert complete_write.final_score == 1.5
        assert complete_write.report_meta_json["report_url"] == ("https://oss.example.com/r.html")
        service.fail_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_mark_stop_skips_report_when_no_scored_samples(
        self,
        shanghai: ZoneInfo,
    ) -> None:
        """[stop-run only] No scored samples: complete_run writes skipped_no_cases, and the report is not emitted."""
        scheduler = _make_scheduler(_config(_item(generate_report=True)))
        service = scheduler._benchmark_service
        service.sample_trace_pipeline_records = AsyncMock(return_value=["t1"])
        # sample_count=0 → no valid score inside the window
        service.load_kafka_criterion_entries = AsyncMock(
            return_value=[SimpleNamespace(sample_count=0, criterion_id="c1")],
        )
        service.aggregate_trace_daily_root_cause = AsyncMock(return_value={})
        service.update_run_status = AsyncMock()
        service.complete_run = AsyncMock()
        service.fail_run = AsyncMock()
        service.create_placeholder_run = AsyncMock(return_value=True)
        metrics = AggregatedMetrics(
            aggregation_method="weighted_avg",
            total_samples=0,
            total_duration_ms=0,
        )
        scheduler._aggregation.aggregate = AsyncMock(  # type: ignore[method-assign]
            return_value=(0.0, {}, metrics),
        )

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                trs_mod.BenchmarkRun,
                "filter",
                MagicMock(
                    return_value=SimpleNamespace(first=AsyncMock(return_value=None)),
                ),
            )
            _patch_registry(
                monkeypatch,
                rubric=[_criterion()],
                aggregation=SimpleNamespace(pass_threshold=0.6),
            )
            generate_report = AsyncMock(return_value="https://oss.example.com/r.html")
            monkeypatch.setattr(trs_mod, "generate_benchmark_report", generate_report)

            await scheduler._run_item(
                TraceReportRunSpec(
                    item=_item(generate_report=True),
                    data_date=date(2026, 9, 22),
                    tz=shanghai,
                    run_id_override="globalvoice_daily-20260922-stopped",
                    mark_stop=True,
                ),
            )
        finally:
            monkeypatch.undo()

        generate_report.assert_not_awaited()
        service.complete_run.assert_awaited_once()
        write = service.complete_run.await_args.args[0]
        assert write.report_meta_json["report_status"] == "skipped_no_cases"
        assert write.report_meta_json["partial"] is True
        assert write.report_meta_json["completed_cases"] == 0
        # The placeholder run still writes the stop latch
        _run_id, spec = service.create_placeholder_run.await_args.args
        assert spec.config_json["stop_requested_at"]

    @pytest.mark.asyncio
    async def test_sample_uses_registration_id_as_evaluator_name(
        self,
        shanghai: ZoneInfo,
    ) -> None:
        """When a registrationId is present, sampling prefers stem@reg (and also accepts a bare reg / an old stem-name row)."""
        scheduler = _make_scheduler(
            _config(_item(agent_registration_id="example--biz--default--v56")),
        )
        service = scheduler._benchmark_service
        service.sample_trace_pipeline_records = AsyncMock(return_value=["t1"])
        service.load_kafka_criterion_entries = AsyncMock(return_value=[])
        service.aggregate_trace_daily_root_cause = AsyncMock(return_value={})
        service.update_run_status = AsyncMock()
        service.complete_run = AsyncMock()
        service.fail_run = AsyncMock()
        service.create_placeholder_run = AsyncMock(return_value=True)
        scheduler._aggregation.aggregate = AsyncMock(  # type: ignore[method-assign]
            return_value=(
                0.0,
                {},
                AggregatedMetrics(
                    aggregation_method="weighted_avg",
                    total_samples=1,
                    total_duration_ms=0,
                ),
            ),
        )
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                trs_mod.BenchmarkRun,
                "filter",
                MagicMock(
                    return_value=SimpleNamespace(first=AsyncMock(return_value=None)),
                ),
            )
            rubric = [_criterion("c1", "example")]
            _patch_registry(
                monkeypatch,
                rubric=rubric,
                aggregation=SimpleNamespace(pass_threshold=0.5),
            )
            monkeypatch.setattr(
                trs_mod,
                "generate_benchmark_report",
                AsyncMock(return_value=None),
            )
            await scheduler._run_item(
                _item(agent_registration_id="example--biz--default--v56"),
                data_date=date(2026, 9, 22),
                tz=shanghai,
            )
        finally:
            monkeypatch.undo()

        sample_query = service.sample_trace_pipeline_records.await_args.args[0]
        assert sample_query.evaluator_names[0] == "example@example--biz--default--v56"
        assert "example--biz--default--v56" in sample_query.evaluator_names
        assert "example" in sample_query.evaluator_names
        entries_kw = service.load_kafka_criterion_entries.await_args.kwargs
        assert entries_kw["registration_id"] == "example--biz--default--v56"
        run_id, spec = service.create_placeholder_run.await_args.args
        sample = spec.config_json["trace_daily_sample"]
        assert sample["evaluator_names"][0] == "example@example--biz--default--v56"
        assert sample["agent_registration_id"] == "example--biz--default--v56"
        # The column benchmark_name stores stem@reg; the logical name goes into config.benchmark_component
        assert spec.benchmark_name == "example@example--biz--default--v56"
        assert spec.config_json.get("benchmark_component") == "example"

    @pytest.mark.asyncio
    async def test_rubric_class_mode_bypasses_component(self, shanghai: ZoneInfo) -> None:
        """The rubric_class alias wins: rubric/aggregation are taken straight from the Rubric class,
        the benchmark component is not looked up, and the run_config snapshot writes rubric_ref."""
        scheduler = _make_scheduler(_config(_item()))
        service = scheduler._benchmark_service
        service.sample_trace_pipeline_records = AsyncMock(return_value=["t1"])
        service.create_placeholder_run = AsyncMock(return_value=True)
        service.update_run_status = AsyncMock()
        service.load_kafka_criterion_entries = AsyncMock(return_value=["entry"])
        service.aggregate_trace_daily_root_cause = AsyncMock(return_value={})
        service.complete_run = AsyncMock()

        metrics = AggregatedMetrics(
            aggregation_method="weighted_avg",
            total_samples=1,
            total_duration_ms=0,
        )
        scheduler._aggregation.aggregate = AsyncMock(  # type: ignore[method-assign]
            return_value=(0.8, {}, metrics),
        )

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                trs_mod.BenchmarkRun,
                "filter",
                MagicMock(
                    return_value=SimpleNamespace(first=AsyncMock(return_value=None)),
                ),
            )
            # The registry is emptied: if it mistakenly took the component path it would return benchmark_not_found immediately
            monkeypatch.setattr(trs_mod, "BENCHMARK_SERVICES_CLASSES", ())
            monkeypatch.setattr(
                trs_mod,
                "generate_benchmark_report",
                AsyncMock(return_value=None),
            )

            await scheduler._run_item(
                _item(
                    benchmark_name="",  # It may be empty in rubric_class mode
                    rubric_class="amap_agent_kafka_rubric",
                    agent_name="amap-agent",
                ),
                data_date=date(2026, 9, 22),
                tz=shanghai,
            )
        finally:
            monkeypatch.undo()

        # The sampled evaluator_names come from the evaluator_ref of the Rubric class criteria
        sample_query = service.sample_trace_pipeline_records.await_args.args[0]
        assert sample_query.evaluator_names == ["amap_agent"]

        # Placeholder run: service_name takes agent_name; the config snapshot writes rubric_ref;
        # when benchmark_name is empty, fall back to the page dataset name, and the namespace still goes into config
        run_id, spec = service.create_placeholder_run.await_args.args
        assert run_id == "globalvoice_daily-20260922"
        assert spec.benchmark_name == "globalvoice_daily"
        assert spec.dataset_name == "globalvoice_daily"
        assert spec.service_name == "amap-agent"
        assert spec.config_json["rubric_ref"] == (
            "business.runs.amap_agent.rubric:AmapAgentKafkaRubric"
        )
        assert "benchmark_component" not in spec.config_json
        assert spec.config_json["trace_report_namespace"] == "globalvoice_daily"

        # The aggregation threshold comes from the Rubric class (pass_threshold=0.5)
        entries_call = service.load_kafka_criterion_entries.await_args.kwargs
        assert entries_call["pass_threshold"] == 0.5
        assert [c.id for c in entries_call["rubric"]] == ["amap_agent_overall"]
        service.complete_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rubric_class_invalid_alias_skips(self, shanghai: ZoneInfo) -> None:
        """An illegal rubric_class alias: log an error and skip, and do not create a run."""
        scheduler = _make_scheduler(_config(_item()))
        service = scheduler._benchmark_service
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                trs_mod.BenchmarkRun,
                "filter",
                MagicMock(
                    return_value=SimpleNamespace(first=AsyncMock(return_value=None)),
                ),
            )
            await scheduler._run_item(
                _item(rubric_class="no_such_rubric"),
                data_date=date(2026, 9, 22),
                tz=shanghai,
            )
        finally:
            monkeypatch.undo()
        service.sample_trace_pipeline_records.assert_not_called()  # type: ignore[union-attr]
        service.create_placeholder_run.assert_not_called()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_failure_marks_run_failed(self, shanghai: ZoneInfo) -> None:
        scheduler = _make_scheduler(_config(_item()))
        service = scheduler._benchmark_service
        service.sample_trace_pipeline_records = AsyncMock(return_value=["t1"])
        service.create_placeholder_run = AsyncMock(return_value=True)
        service.update_run_status = AsyncMock()
        service.load_kafka_criterion_entries = AsyncMock(side_effect=RuntimeError("db down"))
        service.fail_run = AsyncMock()

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(
                trs_mod.BenchmarkRun,
                "filter",
                MagicMock(
                    return_value=SimpleNamespace(first=AsyncMock(return_value=None)),
                ),
            )
            _patch_registry(
                monkeypatch,
                rubric=[_criterion()],
                aggregation=SimpleNamespace(pass_threshold=0.6),
            )
            await scheduler._run_item(_item(), data_date=date(2026, 9, 22), tz=shanghai)
        finally:
            monkeypatch.undo()

        service.fail_run.assert_awaited_once()
        assert service.fail_run.await_args.kwargs["run_id"] == "globalvoice_daily-20260922"
        assert "db down" in service.fail_run.await_args.kwargs["reason"]
