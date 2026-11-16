"""Daily-report trend: replay only trace_pipeline_records, and cache the result on benchmark_runs."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from benchmark.records.enums import BenchmarkStatus
from benchmark.runs.run_service import BenchmarkService
from benchmark.runs.trace_daily_trend import attach_trace_daily_trend

_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _run(
    run_id: str,
    *,
    status: BenchmarkStatus,
    final_score: float | None = None,
    metrics: dict | None = None,
    config: dict | None = None,
    total_samples: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=run_id,
        benchmark_name="amap_agent_daily",
        status=status,
        final_score=final_score,
        metrics_json=metrics,
        config_json=config,
        total_samples=total_samples,
        project_id="proj",
    )


def _filter(runs: dict[str, SimpleNamespace]) -> MagicMock:
    def _apply(**kwargs: object) -> MagicMock:
        run = runs.get(str(kwargs.get("run_id") or ""))
        status = kwargs.get("status")
        if run is not None and status is not None and run.status != status:
            run = None
        query = MagicMock()
        query.order_by.return_value = query
        query.first = AsyncMock(return_value=run)
        return query

    return MagicMock(side_effect=_apply)


def _sample() -> dict:
    return {
        "source": "kafka",
        "agent_name": "amap-agent",
        "project_id": "proj",
        "sample_size": None,
        "timezone": "Asia/Shanghai",
        "pass_threshold": 0.5,
        "evaluator_names": ["amap_agent"],
        "criteria": [
            {
                "id": "amap_agent_overall",
                "name": "综合质量",
                "weight": 1.0,
                "evaluator_ref": "amap_agent",
                "description": "",
            }
        ],
    }


@pytest.mark.asyncio
async def test_daily_trend_uses_traces_and_run_cache() -> None:
    """The data date is anchored on run_id; a cache hit does not scan the table; a miss scans traces and writes them back onto the run."""
    current = _run(
        "amap_agent_daily-20261012",
        status=BenchmarkStatus.GENERATING_REPORT,
        config={"trace_daily_report": True, "trace_daily_sample": _sample()},
    )
    cached = _run(
        "amap_agent_daily-20261011",
        status=BenchmarkStatus.COMPLETED,
        final_score=0.7,
        metrics={
            "daily_trend_point": {
                "fused_avg": 0.7,
                "excellent_good_rate": 50.0,
                "criterion_means": {"expression": 0.4},
                "grade_distribution": {"GOOD": 2},
            }
        },
    )
    missing = _run(
        "amap_agent_daily-20261010",
        status=BenchmarkStatus.COMPLETED,
        final_score=0.55,
        total_samples=1,
    )
    runs = {run.run_id: run for run in (current, cached, missing)}
    service = BenchmarkService()
    service.sample_trace_pipeline_records = AsyncMock(return_value=["t-1"])
    service.load_trace_daily_case_results = AsyncMock(
        return_value=[
            {
                "grade": "GOOD",
                "scores": {"fused": {"score": 0.6, "eval_failed": False}},
                "weight_dimensions": [
                    {
                        "id": "expression",
                        "name": "表达质量",
                        "weight": 0.5,
                        "score": 3.0,
                        "full_score": 5.0,
                    }
                ],
            }
        ],
    )
    service._patch_metrics = AsyncMock()
    spec = {
        "trace_daily_report": True,
        "benchmark_name": "amap_agent_daily",
        "run_id": "amap_agent_daily-20261012",
        "final_score": 0.8,
        "case_results": [{"grade": "EXCELLENT"}],
        "total_samples": 4,
        "total_failed": 1,
        "criterion_metrics": {"expression": {"mean": 0.9}},
    }
    with patch(
        "benchmark.runs.trace_daily_trend.rh.BenchmarkRun.filter",
        _filter(runs),
    ):
        await attach_trace_daily_trend(
            service,
            spec,
            benchmark_name="amap_agent_daily",
            days=3,
        )

    assert spec["trend_7d"]["dates"] == ["10-10", "10-11", "10-12"]
    assert spec["trend_7d"]["fused_avg"] == [0.55, 0.7, 0.8]
    assert spec["trend_7d"]["excellent_good_rate"] == [100.0, 50.0, 100.0]
    assert spec["trend_7d"]["eval_success_rate"][2] == 75.0
    assert spec["yesterday_criterion_means"] == {"expression": 0.4}
    assert spec["yesterday_grade_distribution"] == {"GOOD": 2}
    service.sample_trace_pipeline_records.assert_awaited_once()
    query = service.sample_trace_pipeline_records.await_args.args[0]
    assert query.date_start == datetime(2026, 10, 10, tzinfo=_SHANGHAI)
    assert query.date_end == datetime(2026, 10, 11, tzinfo=_SHANGHAI)
    assert query.agent_name == "amap-agent"
    assert query.source == "kafka"
    patched = [call.args[0] for call in service._patch_metrics.await_args_list]
    assert patched == ["amap_agent_daily-20261010", "amap_agent_daily-20261012"]
    stored = service._patch_metrics.await_args_list[0].args[1]["daily_trend_point"]
    assert stored["fused_avg"] == 0.55
    assert stored["trace_scanned"] is True
    assert stored["criterion_means"]["expression"] == 3.0


@pytest.mark.asyncio
async def test_daily_trend_amap_agent_benchmark_joins_legacy_daily() -> None:
    """When the new amap_agent_benchmark namespace is missing a day, fall back to amap_agent_daily-{ymd}."""
    report_ns = (
        "amap-agent-sampling-2026-10-30-1793370626586__reg__"
        "ba-82a6e3f2-724c-4e82-8c4d-111935b81bc7__ts__ds-rt-1793370626586"
    )
    current_id = f"{report_ns}-20261101"
    current = _run(
        current_id,
        status=BenchmarkStatus.GENERATING_REPORT,
        config={
            "trace_daily_report": True,
            "trace_daily_sample": _sample(),
            "benchmark_component": "amap_agent_benchmark",
            "trace_report_namespace": report_ns,
        },
    )
    current.benchmark_name = "amap_agent_benchmark@ba-82a6e3f2-724c-4e82-8c4d-111935b81bc7"
    legacy = _run(
        "amap_agent_daily-20261031",
        status=BenchmarkStatus.COMPLETED,
        final_score=0.66,
        metrics={
            "daily_trend_point": {
                "fused_avg": 0.66,
                "excellent_good_rate": 40.0,
                "criterion_means": {"expression": 0.5},
                "grade_distribution": {"ACCEPTABLE": 1},
            }
        },
    )
    runs = {current.run_id: current, legacy.run_id: legacy}
    service = BenchmarkService()
    service.sample_trace_pipeline_records = AsyncMock(return_value=[])
    service.load_trace_daily_case_results = AsyncMock(return_value=[])
    service._patch_metrics = AsyncMock()
    spec = {
        "trace_daily_report": True,
        "benchmark_name": "amap_agent_benchmark",
        "run_id": current_id,
        "final_score": 0.81,
        "case_results": [{"grade": "GOOD"}],
        "total_samples": 2,
        "total_failed": 0,
    }
    with patch(
        "benchmark.runs.trace_daily_trend.rh.BenchmarkRun.filter",
        _filter(runs),
    ):
        await attach_trace_daily_trend(
            service,
            spec,
            benchmark_name=report_ns,
            days=2,
        )
    assert spec["trend_7d"]["dates"] == ["10-31", "11-01"]
    assert spec["trend_7d"]["fused_avg"] == [0.66, 0.81]
    assert spec["yesterday_criterion_means"] == {"expression": 0.5}
    service.sample_trace_pipeline_records.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_trend_non_amap_does_not_join_legacy_daily() -> None:
    """Anything other than amap_agent_benchmark must not fall back to amap_agent_daily."""
    report_ns = "other-dataset__reg__ba-x__ts__ds-rt-1"
    current_id = f"{report_ns}-20261101"
    current = _run(
        current_id,
        status=BenchmarkStatus.GENERATING_REPORT,
        config={"trace_daily_report": True, "trace_daily_sample": _sample()},
    )
    legacy = _run(
        "amap_agent_daily-20261031",
        status=BenchmarkStatus.COMPLETED,
        final_score=0.66,
        metrics={"daily_trend_point": {"fused_avg": 0.66}},
    )
    runs = {current.run_id: current, legacy.run_id: legacy}
    service = BenchmarkService()
    service.sample_trace_pipeline_records = AsyncMock(return_value=[])
    service.load_trace_daily_case_results = AsyncMock(return_value=[])
    service._patch_metrics = AsyncMock()
    spec = {
        "trace_daily_report": True,
        "benchmark_name": "other_benchmark",
        "run_id": current_id,
        "final_score": 0.9,
        "case_results": [{"grade": "EXCELLENT"}],
        "total_samples": 1,
        "total_failed": 0,
    }
    with patch(
        "benchmark.runs.trace_daily_trend.rh.BenchmarkRun.filter",
        _filter(runs),
    ):
        await attach_trace_daily_trend(
            service,
            spec,
            benchmark_name=report_ns,
            days=2,
        )
    assert spec["trend_7d"]["fused_avg"][0] is None
    assert spec["trend_7d"]["fused_avg"][1] == 0.9


@pytest.mark.asyncio
async def test_attach_context_keeps_other_benchmarks_on_items() -> None:
    """A non-daily report still goes through load_daily_trend, and does not enter trace replay."""
    service = BenchmarkService()
    spec = {"benchmark_name": "ieval_external", "final_score": 1.5, "case_results": []}
    with (
        patch.object(service, "load_daily_trend", AsyncMock(return_value=None)) as item_trend,
        patch.object(service, "load_yesterday_criterion_means", AsyncMock(return_value={})),
        patch.object(service, "load_yesterday_grade_distribution", AsyncMock(return_value={})),
        patch(
            "benchmark.runs.run_metrics.attach_trace_daily_trend",
            AsyncMock(),
        ) as daily,
    ):
        await service.attach_daily_report_context(spec)
    item_trend.assert_awaited_once()
    daily.assert_not_called()


@pytest.mark.asyncio
async def test_attach_context_daily_skips_item_trend() -> None:
    """The daily-report branch does not call the trend query based on benchmark_run_items."""
    service = BenchmarkService()
    spec = {
        "trace_daily_report": True,
        "benchmark_name": "amap_agent_daily",
        "run_id": "amap_agent_daily-20261012",
        "final_score": 0.8,
    }
    with (
        patch.object(service, "load_daily_trend", AsyncMock()) as item_trend,
        patch(
            "benchmark.runs.run_metrics.attach_trace_daily_trend",
            AsyncMock(),
        ) as daily,
    ):
        await service.attach_daily_report_context(spec)
    item_trend.assert_not_called()
    daily.assert_awaited_once()
