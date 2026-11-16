"""A manually triggered request can carry the report parser class configured on the page."""

from benchmark.runs.schema import ExternalBenchmarkRequest


def test_manual_request_keeps_report_parser_class() -> None:
    payload = ExternalBenchmarkRequest(
        benchmark_name="gpt-judge-pro",
        report_parser_class=(
            "benchmark.reporting.framework.unified_report.report_parser:IEvalReportParser"
        ),
        report_html="amap_agent_daily.html",
    )
    assert payload.report_parser_class.endswith("IEvalReportParser")
    assert payload.report_html == "amap_agent_daily.html"
    assert payload.run_id_suffix is None


def test_manual_request_keeps_parallel_evaluators_and_attribution_report() -> None:
    payload = ExternalBenchmarkRequest(
        benchmark_name="gpt-judge-pro",
        judges=[
            {
                "execution": "sync",
                "target": "business.runs.search_eval.local_judge:SearchEvalLocalJudge",
                "provider": "local",
                "class_ref": "business.runs.search_eval.local_judge:SearchEvalLocalJudge",
            },
            {
                "execution": "async",
                "target": "https://aone.example/judge",
                "provider": "external",
                "class_ref": "benchmark.runs.common.execution_classes:RemoteJudge",
            },
        ],
        attributions=[
            {
                "execution": "sync",
                "target": "benchmark.runs.common.execution_classes:LocalAttribution",
                "provider": "local",
                "class_ref": "benchmark.runs.common.execution_classes:LocalAttribution",
            }
        ],
        attribution_report_parser_class=(
            "benchmark.reporting.framework.unified_report.report_parser:DefaultReportParser"
        ),
        attribution_report_html="benchmark_report.html",
    )
    assert payload.judge is not None
    assert payload.judge.class_ref is not None
    assert payload.judge.class_ref.endswith("SearchEvalLocalJudge")
    assert payload.judges is not None
    assert len(payload.judges) == 2
    assert payload.attribution is not None
    assert payload.attribution_report_html == "benchmark_report.html"
