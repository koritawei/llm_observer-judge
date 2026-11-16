"""Unit tests for performance and efficiency metric aggregation (latency / TTFT / Token / ReAct + trend points)."""

from typing import Any

from benchmark.reporting.framework.performance_metrics import (
    PERF_TREND_KEYS,
    performance_trend_point,
    summarize_performance_metrics,
)


def _case(
    *,
    ttft_ms: Any = None,
    elapsed_seconds: Any = None,
    odps: dict[str, Any] | None = None,
    odps_under_raw_json: bool = True,
) -> dict[str, Any]:
    output: dict[str, Any] = {"response_text": "answer"}
    if ttft_ms is not None:
        output["ttft_ms"] = ttft_ms
    if elapsed_seconds is not None:
        output["elapsed_seconds"] = elapsed_seconds
    judge: dict[str, Any] = {"criterion_scores": {"D1": {"score": 1.5}}}
    if odps is not None:
        if odps_under_raw_json:
            judge["raw_json"] = {"odps": odps}
        else:
            judge["odps"] = odps
    return {
        "case_id": "c1",
        "output": output,
        "scores": {"fused": {"score": 1.5, "eval_failed": False, "judge": judge}},
    }


class TestSummarizePerformanceMetrics:
    """``summarize_performance_metrics``: the definition, string parsing, and empty data."""

    def test_latency_and_ttft_percentiles(self) -> None:
        cases = [
            _case(ttft_ms=500, elapsed_seconds=1.0),
            _case(ttft_ms=700, elapsed_seconds=3.0),
            _case(ttft_ms=900, elapsed_seconds=5.0),
        ]

        metrics = summarize_performance_metrics(cases)

        # elapsed_seconds is converted from seconds → milliseconds
        assert metrics["latency"]["p50_ms"] == 3000.0
        assert metrics["latency"]["avg_ms"] == 3000.0
        assert metrics["latency"]["sample_count"] == 3
        assert metrics["ttft"]["p50_ms"] == 700.0
        assert metrics["ttft"]["sample_count"] == 3

    def test_csv_string_numbers_are_parsed(self) -> None:
        cases = [_case(ttft_ms="620", elapsed_seconds="2.80")]

        metrics = summarize_performance_metrics(cases)

        assert metrics["latency"]["p50_ms"] == 2800.0
        assert metrics["ttft"]["p50_ms"] == 620.0

    def test_tokens_from_odps_raw_json(self) -> None:
        odps = {
            "total_input_tokens": "1200",
            "total_output_tokens": "340",
            "total_tokens": "1540",
        }
        cases = [_case(odps=odps), _case(odps=odps)]

        tokens = summarize_performance_metrics(cases)["tokens"]

        assert tokens["total_tokens"] == 3080.0
        assert tokens["avg_total"] == 1540.0
        assert tokens["avg_input"] == 1200.0
        assert tokens["avg_output"] == 340.0
        assert tokens["sample_count"] == 2

    def test_tokens_from_top_level_odps(self) -> None:
        """external callback: ``odps`` also appears at the top level of judge_json."""
        cases = [_case(odps={"total_tokens": 900}, odps_under_raw_json=False)]

        assert summarize_performance_metrics(cases)["tokens"]["avg_total"] == 900.0

    def test_tokens_fallback_to_input_plus_output(self) -> None:
        cases = [_case(odps={"total_input_tokens": 800, "total_output_tokens": 200})]

        assert summarize_performance_metrics(cases)["tokens"]["avg_total"] == 1000.0

    def test_react_rounds_weighted_average(self) -> None:
        metrics = summarize_performance_metrics(
            [_case(ttft_ms=1)],
            react_rounds_summary={1: 2, 3: 2},
        )

        assert metrics["react"]["avg_rounds"] == 2.0
        assert metrics["react"]["sample_count"] == 4
        assert metrics["react"]["distribution"] == {"1": 2, "3": 2}

    def test_missing_sections_are_omitted(self) -> None:
        """When there is only latency data, do not emit the ttft / tokens / react sections."""
        metrics = summarize_performance_metrics([_case(elapsed_seconds=2.0)])

        assert set(metrics) == {"latency"}

    def test_no_data_returns_empty(self) -> None:
        assert summarize_performance_metrics(None) == {}
        assert summarize_performance_metrics([]) == {}
        assert summarize_performance_metrics([_case()]) == {}

    def test_invalid_values_are_skipped(self) -> None:
        cases = [
            _case(ttft_ms="", elapsed_seconds="N/A", odps={"total_tokens": "oops"}),
            _case(ttft_ms=True, elapsed_seconds=None),
        ]

        assert summarize_performance_metrics(cases) == {}


class TestPerformanceTrendPoint:
    """``performance_trend_point``: trend-key projection and missing-value handling."""

    def test_projects_all_trend_keys(self) -> None:
        metrics = summarize_performance_metrics(
            [_case(ttft_ms=620, elapsed_seconds=2.8, odps={"total_tokens": 1540})],
            react_rounds_summary={2: 1},
        )

        point = performance_trend_point(metrics)

        assert point == {
            "latency_p50_ms": 2800.0,
            "ttft_p50_ms": 620.0,
            "token_avg": 1540.0,
            "react_avg_rounds": 2.0,
        }
        assert set(point) == set(PERF_TREND_KEYS)

    def test_missing_sections_produce_no_keys(self) -> None:
        assert performance_trend_point(None) == {}
        assert performance_trend_point({}) == {}
        assert performance_trend_point({"latency": {"p50_ms": 1200.0}}) == {
            "latency_p50_ms": 1200.0,
        }
