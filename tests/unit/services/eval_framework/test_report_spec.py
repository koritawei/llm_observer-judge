"""Unit tests for the unified report_spec assembly core."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.reporting.framework.report_spec import (
    PerformanceMetricsContext,
    ReportJobInput,
    ReportSpecFields,
    assemble_report_spec,
    attach_performance_metrics,
    build_report_spec_dict,
    generate_benchmark_report,
    prefer_db_metrics,
    resolve_rule_link_url,
)
from benchmark.reporting.framework.schema import (
    AggregatedMetrics,
    Criterion,
    CriterionStats,
)


def _metrics(*, mean: float = 1.5, failed_count: int = 3) -> AggregatedMetrics:
    return AggregatedMetrics(
        criterion_metrics={
            "D1": CriterionStats(
                criterion_id="D1",
                evaluator_ref="x",
                sample_count=10,
                success_count=7,
                failed_count=failed_count,
                mean=mean,
                median=mean,
                p95=mean,
                min=0.0,
                max=2.0,
                std=0.1,
                pass_rate=0.7,
            ),
        },
        aggregation_method="fused_avg",
        total_samples=10,
        total_duration_ms=1234,
    )


def test_resolve_rule_link_url_prefers_snapshot() -> None:
    assert resolve_rule_link_url({"rule_link_url": "https://a.com/r"}) == "https://a.com/r"
    assert resolve_rule_link_url({}) is None


def test_build_report_spec_dict_strips_failed_and_sets_totals() -> None:
    run = SimpleNamespace(
        benchmark_name="plain_bm",
        service_name="svc",
        dataset_name="d1",
    )
    spec = build_report_spec_dict(
        ReportSpecFields(
            run=run,
            run_id="run-1",
            config={"benchmark_services": "plain_bm", "rule_link_url": "https://rule"},
            metrics=_metrics(),
            final_score=1.6,
            rubric=[Criterion(id="D1", name="维1", weight=1.0, evaluator_ref="x")],
            total_failed=2,
        ),
    )
    assert spec["run_id"] == "run-1"
    assert spec["rule_link_url"] == "https://rule"
    assert spec["total_samples"] == 10
    assert spec["total_failed"] == 2
    assert spec["total_success"] == 8
    assert "failed_count" not in spec["criterion_metrics"]["D1"]
    assert spec["total_duration_ms"] == 1234
    assert spec["criteria"][0]["id"] == "D1"
    assert "report_parser_class" not in spec


def test_build_report_spec_falls_back_to_holistic_when_no_dim_metrics() -> None:
    """stat_dimensions is registered but sub_scores were not collected → criteria fall back to the primary dimension, so the report is not blank."""
    from benchmark.runs.common.dimension_stats import (
        DimensionStatSpec,
        clear_stat_dimensions,
        register_stat_dimensions,
    )

    clear_stat_dimensions()
    register_stat_dimensions(
        "example_like_bm",
        (
            DimensionStatSpec(key="准确度", name="准确度", full_score=10.0),
            DimensionStatSpec(key="相关性", name="相关性", full_score=10.0),
        ),
    )
    try:
        spec = build_report_spec_dict(
            ReportSpecFields(
                run=SimpleNamespace(
                    benchmark_name="example_like_bm",
                    service_name="example",
                    dataset_name="d1",
                ),
                run_id="run-fallback",
                config={"benchmark_services": "example_like_bm"},
                # criterion_metrics has only holistic, with no sub-dimension key
                metrics=_metrics(),
                final_score=5.0,
                rubric=[
                    Criterion(id="example", name="综合质量", weight=1.0, evaluator_ref="example")
                ],
                total_failed=0,
            ),
        )
        assert [c["id"] for c in spec["criteria"]] == ["example"]
    finally:
        clear_stat_dimensions()


def test_build_report_spec_dict_copies_page_report_binding() -> None:
    spec = build_report_spec_dict(
        ReportSpecFields(
            run=None,
            run_id="run-2",
            config={
                "benchmark_services": "plain_bm",
                "report_parser_class": (
                    "benchmark.reporting.framework.unified_report."
                    "report_parser:SearchEvalReportParser"
                ),
                "rubric_class": "business.runs.search_eval.rubric:SearchEvalRubric",
                "rubric_doc_url": "  ",
                "report_html": "amap_agent_daily.html",
            },
            metrics=_metrics(),
            final_score=1.0,
            rubric=[Criterion(id="D1", name="维1", weight=1.0, evaluator_ref="x")],
            total_failed=0,
        ),
    )
    assert spec["report_parser_class"].endswith("SearchEvalReportParser")
    assert spec["rubric_class"].endswith("SearchEvalRubric")
    assert spec["rubric_doc_url"] == ""
    assert spec["report_html"] == "amap_agent_daily.html"


def test_prefer_db_metrics_overrides_memory() -> None:
    memory = _metrics(mean=0.1)
    run = SimpleNamespace(metrics_json=_metrics(mean=1.9).model_dump(), final_score=1.9)
    metrics, score = prefer_db_metrics(run, memory, 0.1)
    assert metrics.criterion_metrics["D1"].mean == 1.9
    assert score == 1.9


def test_prefer_db_metrics_raises_when_db_missing() -> None:
    """No in-memory fallback: when the DB has no metrics_json / final_score, raise directly."""
    memory = _metrics(mean=0.1)
    with pytest.raises(RuntimeError, match="run not found"):
        prefer_db_metrics(None, memory, 0.1)
    with pytest.raises(RuntimeError, match="metrics_json missing"):
        prefer_db_metrics(SimpleNamespace(metrics_json=None, final_score=1.0), memory, 0.1)
    with pytest.raises(RuntimeError, match="metrics_json missing"):
        prefer_db_metrics(
            SimpleNamespace(metrics_json={"finalize_source": "partial"}, final_score=1.0),
            memory,
            0.1,
        )
    with pytest.raises(RuntimeError, match="final_score missing"):
        prefer_db_metrics(
            SimpleNamespace(metrics_json=_metrics().model_dump(), final_score=None),
            memory,
            0.1,
        )


@pytest.mark.asyncio
async def test_assemble_merges_component_extra_keys_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmark.reporting.framework.report_spec.count_distinct_failed_items",
        AsyncMock(return_value=1),
    )
    service = MagicMock()
    service.get_run = AsyncMock(
        return_value=SimpleNamespace(
            config_json={
                "rubric": [
                    {
                        "id": "D1",
                        "name": "维1",
                        "weight": 1.0,
                        "evaluator_ref": "x",
                    },
                ],
            },
            benchmark_name="bm",
            service_name="svc",
            dataset_name="ds",
            metrics_json=_metrics().model_dump(),
            final_score=1.2,
        ),
    )
    service.load_case_results = AsyncMock(return_value=[{"case_id": "c1"}])
    service.attach_daily_report_context = AsyncMock()
    service.merge_performance_metrics = AsyncMock()

    component = MagicMock()
    component._build_report_spec.return_value = {
        "benchmark_name": "should_not_override",
        "custom_note": "biz-extra",
    }

    spec = await assemble_report_spec(
        ReportJobInput(
            benchmark_service=service,
            run_id="run-1",
            metrics=_metrics(),
            final_score=1.2,
            component_services=component,
            log_prefix="test",
        ),
    )
    assert spec["benchmark_name"] == "bm"
    assert spec["custom_note"] == "biz-extra"
    assert spec["case_results"] == [{"case_id": "c1"}]
    assert spec["total_failed"] == 1
    service.attach_daily_report_context.assert_awaited_once()
    # A case with no latency / token data → do not write performance_metrics, and do not persist it
    assert "performance_metrics" not in spec
    service.merge_performance_metrics.assert_not_awaited()


def _perf_case(*, ttft_ms: str, elapsed_seconds: str, total_tokens: str) -> dict[str, Any]:
    return {
        "case_id": "c1",
        "output": {"ttft_ms": ttft_ms, "elapsed_seconds": elapsed_seconds},
        "scores": {
            "fused": {
                "score": 1.5,
                "judge": {"raw_json": {"odps": {"total_tokens": total_tokens}}},
            },
        },
    }


@pytest.mark.asyncio
async def test_attach_performance_metrics_writes_spec_and_persists() -> None:
    service = MagicMock()
    service.merge_performance_metrics = AsyncMock()
    spec: dict[str, Any] = {"react_rounds_summary": {"2": 4}}

    await attach_performance_metrics(
        spec,
        PerformanceMetricsContext(
            benchmark_service=service,
            run_id="run-1",
            case_results=[
                _perf_case(ttft_ms="620", elapsed_seconds="2.8", total_tokens="1540"),
            ],
        ),
    )

    perf = spec["performance_metrics"]
    assert perf["latency"]["p50_ms"] == 2800.0
    assert perf["ttft"]["p50_ms"] == 620.0
    assert perf["tokens"]["avg_total"] == 1540.0
    assert perf["react"]["avg_rounds"] == 2.0
    service.merge_performance_metrics.assert_awaited_once_with(
        "run-1",
        performance_metrics=perf,
    )


@pytest.mark.asyncio
async def test_attach_performance_metrics_survives_persist_failure() -> None:
    """A persist failure only degrades: spec still carries the metrics, and report emission is not interrupted."""
    service = MagicMock()
    service.merge_performance_metrics = AsyncMock(side_effect=RuntimeError("db down"))
    spec: dict[str, Any] = {}

    await attach_performance_metrics(
        spec,
        PerformanceMetricsContext(
            benchmark_service=service,
            run_id="run-1",
            case_results=[
                _perf_case(ttft_ms="620", elapsed_seconds="2.8", total_tokens="1540"),
            ],
        ),
    )

    assert spec["performance_metrics"]["ttft"]["p50_ms"] == 620.0


@pytest.mark.asyncio
async def test_attach_performance_metrics_noop_without_data() -> None:
    service = MagicMock()
    service.merge_performance_metrics = AsyncMock()
    spec: dict[str, Any] = {}

    await attach_performance_metrics(
        spec,
        PerformanceMetricsContext(
            benchmark_service=service,
            run_id="run-1",
            case_results=[{"case_id": "c1", "output": {"response_text": "a"}}],
        ),
    )

    assert spec == {}
    service.merge_performance_metrics.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_benchmark_report_dispatches_and_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmark.reporting.framework.report_spec.count_distinct_failed_items",
        AsyncMock(return_value=0),
    )
    captured: dict[str, Any] = {}

    async def _dispatch(*, spec: dict[str, Any], webdev: Any) -> tuple[str, str | None]:
        captured["spec"] = spec
        return "<html>ok</html>", "https://oss/r.html"

    monkeypatch.setattr(
        "benchmark.reporting.framework.report_spec.dispatch_report_generation",
        _dispatch,
    )

    service = MagicMock()
    service.get_run = AsyncMock(
        return_value=SimpleNamespace(
            config_json={"rule_link_url": "https://rule"},
            benchmark_name="bm",
            service_name="",
            dataset_name="",
            metrics_json=_metrics().model_dump(),
            final_score=1.5,
        ),
    )
    service.load_case_results = AsyncMock(return_value=[])
    service.attach_daily_report_context = AsyncMock()
    service.merge_performance_metrics = AsyncMock()
    service.merge_report_meta = AsyncMock()

    pages = MagicMock()
    pages.publish = AsyncMock(return_value="https://pages/r")

    url = await generate_benchmark_report(
        ReportJobInput(
            benchmark_service=service,
            run_id="run-1",
            metrics=_metrics(mean=0.2),
            final_score=0.2,
            log_prefix="test",
        ),
        webdev=MagicMock(),
        pages_publisher=pages,
    )
    assert url == "https://oss/r.html"
    assert captured["spec"]["final_score"] == 1.5  # The DB is the source of truth
    assert captured["spec"]["rule_link_url"] == "https://rule"
    pages.publish.assert_awaited_once()
    service.merge_report_meta.assert_awaited_once()
