"""Private HTML presentation adapter tests for Cluster Report."""

import json
import re
import subprocess

import pytest
from benchmark.reporting.cluster.render.html import render_html
from benchmark.reporting.cluster.render.presentation import (
    build_base_payload,
    inject_private_fragment,
    render_private_fragment,
)
from benchmark.reporting.cluster.schema import AttributionView
from benchmark.reporting.cluster.stats.attribution import build_attribution_stats
from benchmark.reporting.framework.unified_report.template_loader import load_unified_template_sync

from tests.unit.cluster_report.test_html_payload import _record, _report


def test_vui_presentation_keeps_metrics_out_of_standard_payload() -> None:
    report = _report()
    report.evaluator_name = "vui_agent"
    report.cases = [_record(score=1.0)]

    payload = build_base_payload(report)
    fragment = render_private_fragment(report)

    assert "vui_metrics" not in payload.report_data_json()
    assert payload.criteria[0]["name"].startswith("D0 ·")
    assert "build_vui_metrics" not in fragment
    assert "var metrics =" in fragment
    assert 'id = "vui-cluster-metrics"' in fragment
    assert "低分问题聚类" in fragment
    assert "DOMContentLoaded" in fragment


def test_presentation_uses_data_window_for_report_date() -> None:
    report = _report()
    report.data_window = {
        "start": "2026-10-01T00:00:00+08:00",
        "end": "2026-10-02T00:00:00+08:00",
    }

    payload = build_base_payload(report)

    assert payload.report_date == "数据范围：2026-10-01 00:00 ～ 2026-10-02 00:00"


def test_private_payload_links_original_business_trace() -> None:
    report = _report()
    record = _record(score=1.0)
    record.trace_id = "raw-primary-id"
    record.project_id = "project-vui"
    report.cases = [record]

    payload = build_base_payload(report)

    assert payload.case_results[0]["trace_url"] == (
        "https://agentfuse-v4.amap-inc.com/project/project-vui/traces/raw-primary-id"
        "?traceId=raw-primary-id"
    )


def test_vui_presentation_decodes_output_and_translates_root_cause() -> None:
    report = _report()
    report.evaluator_name = "vui_agent"
    record = _record(query="", score=0.0)
    record.attribution = AttributionView(root_cause_code="rejection_evidence_missing")
    record.trace_meta["input_snapshot"] = json.dumps(
        {"request_body": {"user_query": "帮我导航去机场"}}, ensure_ascii=True
    )
    record.trace_meta["output"] = json.dumps(
        {"component": {"text": "找到机场，您想看路线还是导航呢？"}}, ensure_ascii=True
    )
    report.cases = [record]

    case = build_base_payload(report).case_results[0]

    assert case["display_input"] == "帮我导航去机场"
    assert case["display_output"] == "找到机场，您想看路线还是导航呢？"
    assert case["root_cause_display"] == "输入与识别｜拒识证据缺失"
    assert case["attribution"]["root_cause"] == "输入与识别｜拒识证据缺失"


def test_vehicle_presentation_renders_private_diagnostics() -> None:
    report = _report()
    report.evaluator_name = "vehicle_agent"
    report.llm_summary = {
        "clusters": [{"title": "路线工具缺失", "affected_count": 2, "summary": "缺少路线工具调用"}]
    }

    fragment = render_private_fragment(report)

    assert "Vehicle 运行诊断" in fragment
    assert "路线工具缺失" in fragment
    assert "vui_metrics" not in fragment
    assert "VUI 运行观测" not in render_html(report)


def test_injector_uses_case_anchor_or_appends_to_content() -> None:
    fragment = '<section class="vui-cluster-fragment">fragment</section>'

    anchored, found = inject_private_fragment(
        '<main><div id="case-section"></div></main>', fragment
    )
    fallback, missing = inject_private_fragment("<main>base</main>", fragment)

    assert found is True
    assert anchored.index("fragment") < anchored.index('id="case-section"')
    assert missing is False
    assert fallback.endswith("</main>")
    assert "fragment" in fallback


def _template_js(payload: object, expression: str) -> object:
    """Execute the shared template's actual score/reason functions in Node."""
    template = load_unified_template_sync()
    names = (
        "isVetoDim",
        "fullScore",
        "scoreColor",
        "escHtml",
        "caseDisplayId",
        "pickJudgeReason",
        "buildBasicInfoHtml",
        "buildDimScoresHtml",
    )
    functions = []
    for name in names:
        match = re.search(
            rf"^function {name}\(.*?(?=^function |^window.pickJudgeReason)",
            template,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None, name
        functions.append(match.group())
    script = (
        "var window={REPORT:{}};var REPORT={};"
        + "\n".join(functions)
        + "var data="
        + json.dumps(payload, ensure_ascii=True)
        + ";"
        + "REPORT.criteria=data.criteria||[];"
        + "console.log(JSON.stringify("
        + expression
        + "));"
    )
    result = subprocess.run(
        ["node", "-e", script], check=True, capture_output=True, text=True, timeout=10
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("evaluator", ["vehicle_agent", "vui_agent"])
@pytest.mark.parametrize("score,percent", [(1.0, 50.0), (2.0, 100.0)])
def test_d0_uses_two_point_full_score_in_shared_template(
    evaluator: str, score: float, percent: float
) -> None:
    report = _report()
    report.evaluator_name = evaluator
    record = _record(score=score)
    record.dimension_results = [{"dimension": "D0", "score_hint": score, "status": "pass"}]
    report.cases = [record]
    payload = build_base_payload(report)
    criterion = next(item for item in payload.criteria if item["id"] == "D0")
    assert (
        _template_js(
            {"criterion": criterion, "mean": payload.criterion_metrics["D0"]["mean"]},
            "data.mean/fullScore(data.criterion)*100",
        )
        == percent
    )


def test_completion_counts_valid_judge_results_including_low_scores() -> None:
    report = _report()
    report.cases = [_record(score=score) for score in (2.0, 1.0, 0.0, None)]
    report.stats.effectiveness = {"total": 4, "score_distribution": {"2": 1, "1": 1, "0": 1}}
    payload = build_base_payload(report)
    assert (payload.total_samples, payload.total_success, payload.total_failed) == (4, 3, 1)
    assert payload.grade_distribution["FAIL"] == 1


def test_attribution_top_ten_preserves_full_total_and_other_bucket() -> None:
    report = _report()
    report.cases = [
        _record(attribution=AttributionView(status="completed", root_cause_code=f"cause_{i}"))
        for i in range(11)
    ]
    report.stats.attribution = build_attribution_stats(report.cases)
    assert len(report.stats.attribution["root_cause_top"]) == 10
    metrics = build_base_payload(report).attribution_metrics["fused"]
    assert metrics["total_attributed"] == 11
    counts = metrics["field_distributions"]["根因（Top10，其余合并为其他）"]
    assert counts["其他"] == 1
    assert sum(counts.values()) == 11
    assert round(counts["cause_0"] / metrics["total_attributed"] * 100, 1) == 9.1


def test_judge_and_dimension_reasons_reach_shared_template() -> None:
    report = _report()
    record = _record(score=0.0, reason="导航缺少CMD")
    record.dimension_results = [
        {"dimension": "D1", "score_hint": 0.0, "status": "fail", "reason": "未执行导航工具"}
    ]
    report.cases = [record]
    payload = build_base_payload(report)
    case = payload.case_results[0]
    assert _template_js(case, "pickJudgeReason(data)") == "导航缺少CMD"
    assert case["scores"]["D1"]["reason"] == "未执行导航工具"
    html = _template_js(
        {"criteria": payload.criteria, "case": case},
        "buildBasicInfoHtml(data.case)+buildDimScoresHtml(data.case)",
    )
    assert "评分理由" in html
    assert "导航缺少CMD" in html
    assert "未执行导航工具" in html
