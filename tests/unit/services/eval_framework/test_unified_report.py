"""Unit tests for the unified daily report's ReportPayload / filler / plugin / flag."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from benchmark.reporting.framework.unified_report import template_loader
from benchmark.reporting.framework.unified_report.filler import fill_report_template
from benchmark.reporting.framework.unified_report.payload import (
    CaseColumnSpec,
    ReportPayload,
    derive_report_date,
)
from benchmark.reporting.framework.unified_report.pipeline import (
    _template_name_from,
    generate_unified_report_from_spec,
    unified_report_enabled,
)
from benchmark.reporting.framework.unified_report.plugin import (
    clear_report_payload_plugins,
    register_report_payload_plugin,
    resolve_report_payload_plugin,
)
from benchmark.reporting.framework.unified_report.plugins.default import DefaultReportPayloadPlugin
from benchmark.reporting.framework.unified_report.template_loader import (
    clear_template_cache,
    load_unified_template,
    resolve_report_template_name,
)


def test_derive_report_date_from_explicit_and_run_id() -> None:
    assert derive_report_date({"report_date": "2026-09-16"}) == "2026-09-16"
    assert derive_report_date({"run_id": "ieval_external-20260916"}) == "2026-09-16"
    assert derive_report_date({"run_id": "bm-20260916-retry"}) == "2026-09-16"
    assert derive_report_date({"run_id": "no-date-here"}) is None


def test_report_payload_roundtrip_and_split() -> None:
    payload = ReportPayload(
        run_id="bm-20260916",
        benchmark_name="search_eval",
        final_score=1.5,
        total_samples=10,
        total_success=9,
        total_failed=1,
        case_results=[{"case_id": "c1", "grade": "POOR"}],
        case_columns=[CaseColumnSpec(id="tag", header="业务标签", path="extra.tag")],
        fused_attr_field="扣分维度",
    )
    report = payload.report_data_json()
    assert "case_results" not in report
    assert report["fused_attr_field"] == "扣分维度"
    assert report["report_date"] == "2026-09-16"
    assert payload.cases_json()[0]["case_id"] == "c1"
    assert payload.case_columns_json()[0]["header"] == "业务标签"


def test_fill_report_template_injects_three_tags() -> None:
    html = (
        "<html><body>"
        '<script type="application/json" id="report-data">{}</script>'
        '<script type="application/json" id="cases-data">[]</script>'
        '<script type="application/json" id="case-columns-data">[]</script>'
        '<span class="rule-link-slot">评分规则</span>'
        "</body></html>"
    )
    payload = ReportPayload(
        run_id="r1",
        benchmark_name="bm",
        rule_link_url="https://example.com/rules",
        case_results=[{"case_id": "a"}],
        case_columns=[CaseColumnSpec(id="x", header="X", path="x")],
    )
    out = fill_report_template(html, payload)
    assert '"run_id": "r1"' in out or '"run_id":"r1"' in out
    assert '"case_id": "a"' in out or '"case_id":"a"' in out
    assert "case-columns-data" in out
    assert '"header": "X"' in out or '"header":"X"' in out
    assert 'href="https://example.com/rules"' in out
    assert "评分规则 ↗" in out
    assert "rule-link-slot" not in out


def test_fill_injects_missing_tags_before_body_close() -> None:
    html = "<html><body><div>ok</div></body></html>"
    out = fill_report_template(html, ReportPayload(run_id="r2"))
    assert 'id="report-data"' in out
    assert 'id="cases-data"' in out
    assert 'id="case-columns-data"' in out


@pytest.mark.asyncio
async def test_load_unified_template_contains_slots() -> None:
    clear_template_cache()
    html = await load_unified_template()
    # Skeleton slots aligned with the production search_eval-20260916 page
    for slot in (
        "qhub-banner",
        "insight-section",
        "kpi-grid",
        "grades-section",
        "trend-section",
        "dim-table",
        "radar-chart-box",
        "waterfall-delta-section",
        "heatmap-section",
        "attribution-section",
        "case-section",
        "detail-modal",
        "report-data",
        "cases-data",
        "case-columns-data",
        "rule-link-slot",
    ):
        assert slot in html, f"missing slot {slot}"
    assert "meta-date" in html
    assert "resolveReportDate" in html
    assert "new Date(" not in html
    assert 'data-slot="report_date"' in html
    assert "onQueryClick" in html
    assert "toggleTraceDetail" in html
    assert "buildBasicInfoHtml" in html
    assert "pickJudgeReason" in html
    assert "judge_reason_preferred_keys" in html
    assert "['example','fused','综合质量']" not in html
    assert "评分理由" in html
    assert "buildDimScoresHtml" in html
    assert "Array.isArray(c.dim_scores)" in html
    assert "unitIntervalColor" in html
    assert "escHtml(String(sc.raw_score))" in html
    assert "buildAttributionHtml" in html
    assert "sanitizeUserIdFields" in html
    assert "extractCaseQuery" in html
    assert "extractCaseResult" in html
    assert "renderMarkdown" in html
    assert "trace-detail-row" in html
    assert "query-toggle" in html
    assert "<th>query</th>" in html
    assert "<th>用例ID</th>" not in html
    assert "综合得分" in html
    assert "各维度评分" in html
    assert "用例归因" in html
    assert "escHtml(String(c.input))" not in html
    assert "escHtml(String(c.output))" not in html
    assert "loss/start" in html
    assert "ac-fill" in html
    assert "query-tip-float" in html
    assert "rule_link_url" in html or "rule-link" in html
    assert "Langfuse" in html
    # Data slots must be empty; no external link / no caller UX patch (the pipeline injects those later)
    assert 'id="report-data">{}</script>' in html
    assert 'id="cases-data">[]</script>' in html
    assert 'id="case-columns-data">[]</script>' in html
    assert "http://" not in html and "https://" not in html
    assert "report-ux-patch" not in html
    assert "query-ux-patch" not in html
    assert "dingtalk.com" not in html
    assert "webdev-agent-watermark" not in html


def test_resolve_plugin_defaults() -> None:
    clear_report_payload_plugins()
    p = resolve_report_payload_plugin(None)
    assert isinstance(p, DefaultReportPayloadPlugin)
    assert p.fused_attr_field() == "primary_issue_category"

    ieval = resolve_report_payload_plugin("ieval_external")
    assert ieval.fused_attr_field() == "primary_issue_category"

    search = resolve_report_payload_plugin("search_eval")
    assert search.fused_attr_field() == "primary_issue_category"

    example = resolve_report_payload_plugin("example_benchmark")
    assert isinstance(example, DefaultReportPayloadPlugin)
    assert example.fused_attr_field() == "primary_issue_category"
    assert example.judge_reason_preferred_keys()[0] == "example"
    assert p.judge_reason_preferred_keys() == ("fused", "综合质量")


def test_template_name_from_resolves_plugin_without_nameerror() -> None:
    """Regression: ``_template_name_from`` must be able to resolve the plugin, and a NameError must not fail report emission."""
    clear_report_payload_plugins()
    assert _template_name_from({"report_html": "custom.html"}) == "custom.html"
    # The search_eval plugin's template_name may be None (it uses the generic template); what matters is that it does not raise NameError
    assert _template_name_from({"benchmark_name": "search_eval"}) == getattr(
        resolve_report_payload_plugin("search_eval"), "template_name", None
    )


def test_register_plugin_override() -> None:
    clear_report_payload_plugins()

    class _Custom(DefaultReportPayloadPlugin):
        name = "custom"

        def fused_attr_field(self) -> str:
            return "custom_field"

    register_report_payload_plugin("my_bm", _Custom())
    assert resolve_report_payload_plugin("my_bm").fused_attr_field() == "custom_field"
    clear_report_payload_plugins()


def test_unified_report_enabled_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_EVAL_UNIFIED_REPORT", raising=False)
    monkeypatch.delenv("AGENT_EVAL_FORCE_WEBDEV_REPORT", raising=False)
    assert unified_report_enabled() is True

    monkeypatch.setenv("AGENT_EVAL_FORCE_WEBDEV_REPORT", "1")
    assert unified_report_enabled() is False

    monkeypatch.delenv("AGENT_EVAL_FORCE_WEBDEV_REPORT", raising=False)
    monkeypatch.setenv("AGENT_EVAL_UNIFIED_REPORT", "0")
    assert unified_report_enabled() is False

    monkeypatch.setenv("AGENT_EVAL_UNIFIED_REPORT", "1")
    assert unified_report_enabled() is True


@pytest.mark.asyncio
async def test_generate_unified_from_spec_fills_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_template_cache()
    clear_report_payload_plugins()
    monkeypatch.setattr(
        "benchmark.reporting.framework.unified_report.pipeline._apply_static_postprocess",
        lambda html, _spec: html,
    )
    spec: dict[str, Any] = {
        "run_id": "search_eval-20260916",
        "benchmark_name": "search_eval",
        "final_score": 1.2,
        "total_samples": 2,
        "total_success": 2,
        "total_failed": 0,
        "criteria": [{"id": "d1", "name": "维度1", "weight": 1}],
        "case_results": [
            {
                "case_id": "c1",
                "trace_id": "abc123xyz",
                "display_input": "去西湖",
                "grade": "POOR",
                "scores": {"fused": {"score": 0.5}},
            },
        ],
        "attribution_metrics": {
            "fused": {
                "field_distributions": {"扣分维度": {"intent(意图)": 1}},
                "total_attributed": 1,
            },
        },
    }
    html = await generate_unified_report_from_spec(spec)
    assert "search_eval-20260916" in html
    assert "去西湖" in html or "case_id" in html
    assert 'id="report-data"' in html
    assert DefaultReportPayloadPlugin().fused_attr_field() == "primary_issue_category"


@pytest.mark.asyncio
async def test_webdev_generate_report_uses_unified_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.reporting.framework.webdev_client import WebdevReportClient

    monkeypatch.delenv("AGENT_EVAL_FORCE_WEBDEV_REPORT", raising=False)
    monkeypatch.delenv("AGENT_EVAL_UNIFIED_REPORT", raising=False)
    client = WebdevReportClient(webdev_agent_url="http://unused")
    client._refine_insight_from_html = AsyncMock(side_effect=lambda html, _s: html)  # type: ignore[method-assign]
    with patch(
        "benchmark.reporting.framework.unified_report.pipeline.generate_unified_report_from_spec",
        AsyncMock(return_value="<html><body>unified</body></html>"),
    ) as gen:
        html, url = await client.generate_report({"run_id": "r1", "benchmark_name": "bm"})
    gen.assert_awaited_once()
    assert html == "<html><body>unified</body></html>"
    assert url is None


@pytest.mark.asyncio
async def test_webdev_falls_back_when_template_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.reporting.framework.unified_report.template_loader import UnifiedTemplateError
    from benchmark.reporting.framework.webdev_client import WebdevReportClient

    monkeypatch.delenv("AGENT_EVAL_FORCE_WEBDEV_REPORT", raising=False)
    monkeypatch.delenv("AGENT_EVAL_UNIFIED_REPORT", raising=False)
    client = WebdevReportClient(webdev_agent_url="http://unused")
    with (
        patch(
            "benchmark.reporting.framework.unified_report.pipeline.generate_unified_report_from_spec",
            AsyncMock(side_effect=UnifiedTemplateError("missing")),
        ),
        patch.object(
            client,
            "_generate_html_once",
            AsyncMock(return_value="<html><body>llm-fallback</body></html>"),
        ),
        patch.object(client, "_refine_insight_from_html", AsyncMock(side_effect=lambda h, _s: h)),
        patch.object(client, "_attach_grade_distribution"),
        patch.object(client, "_attach_today_criterion_means"),
        patch.object(client, "_attach_root_cause_display"),
        patch.object(client, "_attach_report_insights"),
        patch.object(client, "_attach_enhance_modules"),
        patch.object(client, "_validate_case_required_fields", return_value=[]),
        patch.object(client, "_build_template_prompt", return_value="prompt"),
    ):
        html, _url = await client.generate_report({"run_id": "r1", "case_results": []})
    assert "llm-fallback" in html


@pytest.mark.asyncio
async def test_webdev_generate_report_skips_unified_when_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.reporting.framework.webdev_client import WebdevReportClient

    monkeypatch.setenv("AGENT_EVAL_FORCE_WEBDEV_REPORT", "1")
    client = WebdevReportClient(webdev_agent_url="http://unused")
    with (
        patch(
            "benchmark.reporting.framework.unified_report.pipeline.generate_unified_report_from_spec",
            AsyncMock(return_value="<html>u</html>"),
        ) as gen,
        patch.object(
            client,
            "_generate_html_once",
            AsyncMock(return_value="<html><body>llm</body></html>"),
        ),
        patch.object(client, "_refine_insight_from_html", AsyncMock(side_effect=lambda h, _s: h)),
        patch.object(client, "_attach_grade_distribution"),
        patch.object(client, "_attach_today_criterion_means"),
        patch.object(client, "_attach_root_cause_display"),
        patch.object(client, "_attach_report_insights"),
        patch.object(client, "_attach_enhance_modules"),
        patch.object(client, "_validate_case_required_fields", return_value=[]),
        patch.object(client, "_build_template_prompt", return_value="prompt"),
    ):
        html, _url = await client.generate_report({"run_id": "r1", "case_results": []})
    gen.assert_not_awaited()
    assert "llm" in html


def test_default_plugin_type() -> None:
    assert isinstance(DefaultReportPayloadPlugin(), DefaultReportPayloadPlugin)


def test_resolve_report_template_name_defaults() -> None:
    assert resolve_report_template_name(None) == "benchmark_report.html"
    assert resolve_report_template_name("  ") == "benchmark_report.html"
    assert resolve_report_template_name("../etc/passwd.html") == "benchmark_report.html"
    assert resolve_report_template_name("not-html.txt") == "benchmark_report.html"


@pytest.mark.asyncio
async def test_load_custom_report_template(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "amap_agent_daily.html"
    custom.write_text(
        '<html><script id="report-data" type="application/json">{}</script>'
        '<script id="cases-data" type="application/json">[]</script>'
        "<body>custom-daily</body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(template_loader, "_TEMPLATES_DIR", tmp_path)
    clear_template_cache()
    html = await load_unified_template("amap_agent_daily.html")
    assert "custom-daily" in html
    clear_template_cache()


@pytest.mark.asyncio
async def test_missing_custom_template_falls_back(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "benchmark_report.html"
    default.write_text(
        '<html><script id="report-data"></script><script id="cases-data"></script>default</html>',
        encoding="utf-8",
    )
    monkeypatch.setattr(template_loader, "_TEMPLATES_DIR", tmp_path)
    clear_template_cache()
    html = await load_unified_template("missing.html")
    assert "default" in html
    clear_template_cache()


def test_resolve_report_template_stat_error_falls_back(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When stat raises OSError, fall back to the default template and do not bubble the exception to the caller."""
    monkeypatch.setattr(template_loader, "_TEMPLATES_DIR", tmp_path)
    original = Path.is_file

    def _is_file(self: Path) -> bool:
        if self.name == "broken.html":
            raise OSError(116, "Stale file handle")
        return original(self)

    monkeypatch.setattr(Path, "is_file", _is_file)
    assert resolve_report_template_name("broken.html") == "benchmark_report.html"


@pytest.mark.asyncio
async def test_missing_spec_selects_template_after_assemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When spec is empty, assemble it first, and take the template name from the assembled result rather than the empty job.config."""
    from benchmark.reporting.framework.report_spec import ReportJobInput
    from benchmark.reporting.framework.schema import AggregatedMetrics
    from benchmark.reporting.framework.unified_report import pipeline as pipe

    job = ReportJobInput(
        benchmark_service=object(),  # type: ignore[arg-type]
        run_id="r1",
        metrics=AggregatedMetrics(
            aggregation_method="weighted_avg",
            total_samples=0,
            total_duration_ms=0,
        ),
        final_score=0.0,
    )
    assembled = {
        "report_html": "amap_agent_daily.html",
        "benchmark_name": "amap_agent_daily",
        "run_id": "r1",
        "case_results": [],
    }

    async def _assemble(received: ReportJobInput) -> dict[str, Any]:
        assert received is job
        return assembled

    loaded: dict[str, str | None] = {}

    async def _load(name: str | None) -> str:
        loaded["name"] = name
        return "TEMPLATE"

    async def _plugin(
        *,
        spec: dict[str, Any],
        job: ReportJobInput,
    ) -> tuple[ReportPayload, dict[str, Any]]:
        assert spec is assembled
        assert job.run_id == "r1"
        return ReportPayload(run_id="r1", benchmark_name="amap_agent_daily"), spec

    monkeypatch.setattr(pipe, "assemble_report_spec", _assemble)
    monkeypatch.setattr(pipe, "load_unified_template", _load)
    monkeypatch.setattr(pipe, "_payload_via_plugin", _plugin)
    monkeypatch.setattr(pipe, "fill_report_template", lambda template, _payload: template)
    monkeypatch.setattr(pipe, "_apply_static_postprocess", lambda html, _spec: html)

    html = await pipe.generate_unified_report_html(job)
    assert loaded["name"] == "amap_agent_daily.html"
    assert html == "TEMPLATE"
