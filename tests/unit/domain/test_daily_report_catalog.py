"""Today's report catalog: pure logic plus assembly (no real DB)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from benchmark.domain.daily_report_catalog import (
    CatalogBuildOptions,
    batch_catalog_key,
    batch_run_rank,
    build_daily_report_catalog,
    build_report_snapshot,
    is_date_ended_run_id,
    is_realtime_daily_run,
    is_streaming_stopped_run_id,
    pages_report_url,
    rubric_health_status,
    shanghai_day_window,
)

_SH = timezone(timedelta(hours=8))


def test_shanghai_day_window_bounds() -> None:
    now = datetime(2026, 11, 3, 15, 30, tzinfo=_SH)
    window = shanghai_day_window(now)
    assert window.ymd == "20261103"
    assert window.start == datetime(2026, 11, 3, 0, 0, tzinfo=_SH)
    assert window.end == datetime(2026, 11, 4, 0, 0, tzinfo=_SH)


def test_pages_report_url_prefers_report_site() -> None:
    assert (
        pages_report_url(
            {
                "report_site_url": "https://agent-report.io/a.html",
                "report_url": "https://oss.example/b.html",
            }
        )
        == "https://agent-report.io/a.html"
    )
    assert pages_report_url({"report_url": "https://oss.example/b.html"}) is None
    assert pages_report_url(None) is None


def test_batch_catalog_key_strips_external_suffix() -> None:
    assert batch_catalog_key("ieval_external") == "ieval"
    assert batch_catalog_key("ieval_external@ba-1") == "ieval"
    assert batch_catalog_key("search_eval") == "search_eval"
    assert batch_catalog_key("search_eval_v2") == "search_eval_v2"
    assert batch_catalog_key("") == ""


def test_batch_run_rank_prefers_exact_daily_id() -> None:
    assert batch_run_rank("ieval_external-20261103", "ieval_external", "20261103") == 0
    assert batch_run_rank("ieval_external-20261102", "ieval_external", "20261103") == 1
    assert batch_run_rank("ieval_external-20261103-m1", "ieval_external", "20261103") == 2
    assert batch_run_rank("other-run", "ieval_external", "20261103") == 2


def test_rubric_health_status_thresholds() -> None:
    assert rubric_health_status(1.0) == "Healthy"
    assert rubric_health_status(0.8) == "Healthy"
    assert rubric_health_status(0.79) == "Watch"
    assert rubric_health_status(0.5) == "Watch"
    assert rubric_health_status(0.49) == "Degraded"
    assert rubric_health_status(None) is None


def test_is_realtime_daily_run() -> None:
    assert is_realtime_daily_run({"trace_daily_report": True})
    assert not is_realtime_daily_run({"trace_daily_report": False})
    assert not is_realtime_daily_run({})


def test_is_streaming_stopped_run_id() -> None:
    assert is_streaming_stopped_run_id("amap-sampling-20261103-stopped-143022")
    assert is_streaming_stopped_run_id("trace-20261103-stopped-000001")
    assert not is_streaming_stopped_run_id("amap-sampling-20261102")
    assert not is_streaming_stopped_run_id("ieval_external-20261103")
    assert not is_streaming_stopped_run_id("ieval_external-20261103-m1")


def test_is_date_ended_run_id() -> None:
    assert is_date_ended_run_id("ieval_external-20261103")
    assert is_date_ended_run_id("ieval_external-20261103", ymd="20261103")
    assert not is_date_ended_run_id("ieval_external-20261103", ymd="20261102")
    assert not is_date_ended_run_id("ieval_external-20261103-m1")
    assert not is_date_ended_run_id("ieval_external-20261103-s2")
    assert not is_date_ended_run_id("amap-20261103-stopped-143022")


def _run(
    *,
    run_id: str,
    benchmark_name: str,
    url: str,
    finished_at: datetime,
    realtime: bool = False,
    registration_id: str | None = None,
    agent_scope: str | None = None,
) -> SimpleNamespace:
    config: dict = {}
    if realtime:
        config["trace_daily_report"] = True
        if registration_id:
            config["agent_registration_id"] = registration_id
        if agent_scope is not None:
            config["agent_registration"] = {
                "id": registration_id or "ba-x",
                "agent_scope": agent_scope,
            }
    return SimpleNamespace(
        run_id=run_id,
        benchmark_name=benchmark_name,
        finished_at=finished_at,
        created_at=finished_at,
        updated_at=finished_at,
        report_meta_json={"report_site_url": url},
        config_json=config,
    )


@pytest.mark.asyncio
async def test_build_catalog_splits_realtime_and_batch() -> None:
    day = datetime(2026, 11, 3, 10, 0, tzinfo=_SH)
    runs = [
        _run(
            run_id="amap-agent-sampling-2026-11-02-x__reg__ba-1__ts__ds-1-20261102",
            benchmark_name="amap_agent@ba-1",
            url="https://agent-report.io/rt-amap.html",
            finished_at=day,
            realtime=True,
            registration_id="ba-1",
            agent_scope="amap-agent",
        ),
        _run(
            run_id="amap-agent-sampling-2026-11-02-y__reg__ba-2__ts__ds-2-20261102",
            benchmark_name="amap_agent@ba-2",
            url="https://agent-report.io/rt-amap-2.html",
            finished_at=day + timedelta(minutes=5),
            realtime=True,
            registration_id="ba-2",
            agent_scope="amap-agent",
        ),
        _run(
            run_id="vehicle-rt-20261102",
            benchmark_name="vehicle_agent@ba-v",
            url="https://agent-report.io/rt-vehicle.html",
            finished_at=day,
            realtime=True,
            registration_id="ba-v",
            agent_scope="vehicle-agent",
        ),
        # A manually stopped window: even with Pages it does not enter the catalog (distinct from T+1)
        _run(
            run_id="amap-agent-sampling-20261103-stopped-143022",
            benchmark_name="amap_agent@ba-1",
            url="https://agent-report.io/rt-stopped.html",
            finished_at=day + timedelta(minutes=10),
            realtime=True,
            registration_id="ba-1",
            agent_scope="amap-agent",
        ),
        _run(
            run_id="ieval_external-20261103",
            benchmark_name="ieval_external",
            url="https://agent-report.io/ieval.html",
            finished_at=day,
        ),
        _run(
            run_id="ieval_external-20261103-m1",
            benchmark_name="ieval_external",
            url="https://agent-report.io/ieval-manual.html",
            finished_at=day + timedelta(hours=1),
        ),
        _run(
            run_id="search_eval-20261103",
            benchmark_name="search_eval",
            url="https://agent-report.io/search.html",
            finished_at=day,
        ),
        _run(
            run_id="search_eval_v2-20261103",
            benchmark_name="search_eval_v2",
            url="https://agent-report.io/search-v2.html",
            finished_at=day,
        ),
        # No Pages URL → ignore
        SimpleNamespace(
            run_id="search_eval-20261103-m2",
            benchmark_name="search_eval",
            finished_at=day,
            created_at=day,
            updated_at=day,
            report_meta_json={"report_url": "https://oss.example/only.html"},
            config_json={},
        ),
    ]
    catalog = await build_daily_report_catalog(
        CatalogBuildOptions(
            now=day,
            runs=runs,
            active_scopes={"amap-agent", "vehicle-agent"},
        )
    )
    # When one scope has several rows, keep only the one with the newer finished_at, and no longer return a URL list
    amap = catalog.realtime["amap-agent"]
    assert amap.url == "https://agent-report.io/rt-amap-2.html"
    assert amap.run_id.endswith("ds-2-20261102")
    assert catalog.realtime["vehicle-agent"].url == "https://agent-report.io/rt-vehicle.html"
    # A stopped window must not be mixed into realtime
    assert "rt-stopped" not in str(catalog.model_dump())
    # Batch includes only runs whose id ends with today's date; -m1 does not enter the catalog
    assert {key: link.url for key, link in catalog.batch.items()} == {
        "ieval": "https://agent-report.io/ieval.html",
        "search_eval": "https://agent-report.io/search.html",
        "search_eval_v2": "https://agent-report.io/search-v2.html",
    }


@pytest.mark.asyncio
async def test_build_catalog_excludes_stopped_and_non_date_batch() -> None:
    day = datetime(2026, 11, 3, 11, 0, tzinfo=_SH)
    runs = [
        _run(
            run_id="rt-ns-20261103-stopped-090011",
            benchmark_name="amap_agent",
            url="https://agent-report.io/stopped-only.html",
            finished_at=day,
            realtime=True,
            agent_scope="amap-agent",
        ),
        _run(
            run_id="search_eval-20261103-s1",
            benchmark_name="search_eval",
            url="https://agent-report.io/manual-s.html",
            finished_at=day,
        ),
    ]
    catalog = await build_daily_report_catalog(
        CatalogBuildOptions(now=day, runs=runs, active_scopes={"amap-agent"})
    )
    assert catalog.model_dump() == {"realtime": {}, "batch": {}}


@pytest.mark.asyncio
async def test_build_catalog_batch_accepts_prior_day_run_id() -> None:
    """When updated_at is inside today's window, the run_id date may be yesterday or n days ago."""
    day = datetime(2026, 11, 11, 11, 39, 45, 593784, tzinfo=_SH)
    runs = [
        _run(
            run_id="ieval_external-20261110",
            benchmark_name="ieval_external",
            url="https://agent-report.io/ieval-prior.html",
            finished_at=day,
        ),
        _run(
            run_id="ieval_external-20261111",
            benchmark_name="ieval_external",
            url="https://agent-report.io/ieval-today.html",
            finished_at=day + timedelta(minutes=1),
        ),
        _run(
            run_id="search_eval-20261109",
            benchmark_name="search_eval",
            url="https://agent-report.io/search-old.html",
            finished_at=day,
        ),
    ]
    catalog = await build_daily_report_catalog(
        CatalogBuildOptions(now=day, runs=runs, active_scopes=set())
    )
    # For the same key, prefer today's exact daily run
    assert catalog.batch["ieval"].url == "https://agent-report.io/ieval-today.html"
    assert catalog.batch["ieval"].run_id == "ieval_external-20261111"
    assert catalog.batch["search_eval"].run_id == "search_eval-20261109"


@pytest.mark.asyncio
async def test_build_catalog_filters_inactive_realtime_scope() -> None:
    day = datetime(2026, 11, 3, 9, 0, tzinfo=_SH)
    runs = [
        _run(
            run_id="rt-1",
            benchmark_name="amap_agent",
            url="https://agent-report.io/a.html",
            finished_at=day,
            realtime=True,
            agent_scope="amap-agent",
        ),
        _run(
            run_id="rt-2",
            benchmark_name="vehicle_agent",
            url="https://agent-report.io/v.html",
            finished_at=day,
            realtime=True,
            agent_scope="vehicle-agent",
        ),
    ]
    catalog = await build_daily_report_catalog(
        CatalogBuildOptions(
            now=day,
            runs=runs,
            active_scopes={"amap-agent"},
        )
    )
    assert list(catalog.realtime.keys()) == ["amap-agent"]
    assert catalog.realtime["amap-agent"].url == "https://agent-report.io/a.html"
    assert catalog.batch == {}


@pytest.mark.asyncio
async def test_build_catalog_omits_missing_batch() -> None:
    day = datetime(2026, 11, 3, 9, 0, tzinfo=_SH)
    catalog = await build_daily_report_catalog(
        CatalogBuildOptions(now=day, runs=[], active_scopes=set())
    )
    assert catalog.model_dump() == {"realtime": {}, "batch": {}}


def _rubric_metric(*, mean: float, pass_rate: float, samples: int = 202) -> dict:
    return {
        "evaluator_ref": "amap_agent",
        "mean": mean,
        "median": mean,
        "p95": mean,
        "min": mean,
        "max": mean,
        "std": 0.0,
        "pass_rate": pass_rate,
        "sample_count": samples,
        "success_count": samples,
        "failed_count": 0,
    }


def test_snapshot_joins_all_rubrics_including_zero_weight() -> None:
    run = SimpleNamespace(
        final_score=1.72,
        service_name="amap_agent",
        total_samples=202,
        evaluated_items=200,
        failed_items=2,
        metrics_json={
            "total_samples": 202,
            "criterion_metrics": {
                "fused": _rubric_metric(mean=1.5, pass_rate=0.9, samples=202),
                "D0": _rubric_metric(mean=2.0, pass_rate=1.0),
                "D1": _rubric_metric(mean=1.2, pass_rate=0.6),
            },
        },
        config_json={
            "rubric": [
                {
                    "id": "D0",
                    "name": "安全与合规",
                    "description": "门控",
                    "weight": 0.0,
                    "evaluator_ref": "amap_agent",
                },
                {
                    "id": "D1",
                    "name": "意图",
                    "description": "",
                    "weight": 1.0,
                    "evaluator_ref": "amap_agent",
                },
            ]
        },
    )
    link = build_report_snapshot(run, url="https://agent-report.io/a.html", run_id="run-1")
    payload = link.model_dump()
    overall = payload["overall"]
    assert isinstance(overall, dict)
    assert overall == {
        "current": 1.72,
        "success": 200,
        "failed": 2,
        "samples": 202,
        "evaluator": "amap_agent",
    }
    assert overall["success"] + overall["failed"] == overall["samples"]
    rubrics = payload["rubrics"]
    assert isinstance(rubrics, list)
    assert [item["id"] for item in rubrics] == ["D0", "D1"]
    safety = rubrics[0]
    assert safety["weight"] == 0.0
    assert safety["current"] == 2.0
    assert safety["status"] == "Healthy"
    assert safety["success"] == 202
    assert rubrics[1]["status"] == "Watch"


def test_snapshot_current_falls_back_to_fused_mean() -> None:
    run = SimpleNamespace(
        final_score=None,
        service_name=None,
        total_samples=None,
        evaluated_items=4,
        failed_items=0,
        metrics_json={
            "criterion_metrics": {
                "fused": _rubric_metric(mean=1.25, pass_rate=0.4, samples=4),
            }
        },
        config_json={"rubric": []},
    )
    link = build_report_snapshot(run, url="https://agent-report.io/a.html", run_id="run-2")
    assert link.overall.current == 1.25
    assert link.overall.model_dump() == {
        "current": 1.25,
        "success": 4,
        "failed": 0,
        "samples": 4,
        "evaluator": "amap_agent",
    }
    assert link.rubrics == []


def test_snapshot_overall_caps_success_to_sample_cap() -> None:
    """success=evaluated_items and does not exceed total_samples; samples=success+failed."""
    run = SimpleNamespace(
        final_score=0.9,
        service_name="amap_agent",
        total_samples=100,
        evaluated_items=150,
        failed_items=10,
        metrics_json={"criterion_metrics": {}},
        config_json={"rubric": []},
    )
    link = build_report_snapshot(run, url="https://agent-report.io/a.html", run_id="run-3")
    assert link.overall.success == 100
    assert link.overall.failed == 10
    assert link.overall.samples == 110
    assert link.overall.success + link.overall.failed == link.overall.samples
