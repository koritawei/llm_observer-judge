"""Unit tests for WebdevReportClient — A2A protocol construction and HTML post-processing."""

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from benchmark.reporting.framework import webdev_client
from benchmark.reporting.framework.webdev_client import (
    _QUERY_UX_PATCH,
    _REPORT_UX_PATCH,
    _TRACE_LINK_PATCH,
    WebdevReportClient,
    _InsightChatRequest,
)


def _client(
    storage: object = None,
    insight_llm_complete: Callable[[str], Awaitable[str]] | None = None,
) -> WebdevReportClient:
    return WebdevReportClient(
        webdev_agent_url="http://webdev-agent.test/",
        storage_client=storage,  # type: ignore[arg-type]
        insight_llm_complete=insight_llm_complete,
    )


@pytest.fixture(autouse=True)
def _disable_insight_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module does not hit the Amap gateway by default, so a local AMAP_LLM_API_KEY does not trigger a real network request."""
    monkeypatch.setattr(
        WebdevReportClient,
        "_insight_llm_credentials",
        staticmethod(lambda: None),
    )


@pytest.fixture(autouse=True)
def _force_webdev_report_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most cases in this module mock AsyncClient, and must take the LLM skill path rather than the unified static template."""
    monkeypatch.setenv("AGENT_EVAL_FORCE_WEBDEV_REPORT", "1")


# ════════════════════════════════════════════════════════════════════════════
# _extract_html — unpack the A2A response
# ════════════════════════════════════════════════════════════════════════════


def test_extract_html_walks_artifacts_for_first_text_part() -> None:
    resp = {
        "result": {
            "artifacts": [
                {"parts": [{"kind": "data", "data": {}}]},
                {"parts": [{"kind": "text", "text": "<html>ok</html>"}]},
            ],
        },
    }
    assert WebdevReportClient._extract_html(resp) == "<html>ok</html>"


def test_extract_html_empty_when_no_text_part() -> None:
    assert WebdevReportClient._extract_html({"result": {"artifacts": []}}) == ""
    assert WebdevReportClient._extract_html({}) == ""


# ════════════════════════════════════════════════════════════════════════════
# _ensure_charset
# ════════════════════════════════════════════════════════════════════════════


def test_ensure_charset_noop_when_present() -> None:
    html = '<html><head><meta charset="utf-8"></head></html>'
    assert WebdevReportClient._ensure_charset(html) == html


def test_ensure_charset_injected_after_head() -> None:
    html = "<html><head><title>t</title></head></html>"
    out = WebdevReportClient._ensure_charset(html)
    assert out.startswith('<html><head><meta charset="UTF-8">')


def test_ensure_charset_falls_back_to_html_tag_then_prefix() -> None:
    assert WebdevReportClient._ensure_charset("<html><body>x</body></html>").startswith(
        '<html><meta charset="UTF-8">',
    )
    assert WebdevReportClient._ensure_charset("plain").startswith('<meta charset="UTF-8">plain')


# ════════════════════════════════════════════════════════════════════════════
# _validate_html_completeness
# ════════════════════════════════════════════════════════════════════════════


def test_validate_completeness_full_document_passes() -> None:
    html = "<html><head><style></style></head><body>x</body></html>"
    assert WebdevReportClient._validate_html_completeness(html) == []


def test_validate_completeness_allows_document_without_style() -> None:
    html = "<html><head></head><body>x</body></html>"
    assert WebdevReportClient._validate_html_completeness(html) == []


def test_validate_completeness_flags_truncated_document() -> None:
    """CSS only, with no body/html close, means the LLM output was truncated, and uploading to OSS is forbidden."""
    missing = WebdevReportClient._validate_html_completeness("<html><head><style></style>")
    assert "<body" in missing
    assert "</body>" in missing
    assert "</html>" in missing


# ════════════════════════════════════════════════════════════════════════════
# _normalize_case_io / _fix_case_data
# ════════════════════════════════════════════════════════════════════════════


def test_normalize_case_io_converts_non_string_fields() -> None:
    cases = [
        {
            "input": {"q": "你好"},
            "output": ["a", "b"],
            "expected": None,
            "display_input": "已是字符串",
            "extra_int": 3,
        },
    ]
    out = WebdevReportClient._normalize_case_io(cases)
    assert json.loads(out[0]["input"]) == {"q": "你好"}
    assert json.loads(out[0]["output"]) == ["a", "b"]
    assert out[0]["expected"] is None
    assert out[0]["display_input"] == "已是字符串"
    # A field outside the allowlist is not touched
    assert out[0]["extra_int"] == 3


def test_fix_case_data_replaces_llm_abbreviated_content() -> None:
    """cases-data that the LLM truncated must be overwritten unconditionally with the raw data from PG."""
    spec = {"case_results": [{"input": "完整输入", "output": "完整输出"}]}
    html = (
        '<html><body><script type="application/json" id="cases-data">'
        '[{"input": "完整输..."}]</script></body></html>'
    )
    out = WebdevReportClient._fix_case_data(html, spec)
    assert "完整输入" in out
    assert "完整输..." not in out


def test_fix_case_data_injects_fallback_tag_before_body_close() -> None:
    spec = {"case_results": [{"input": "q"}]}
    out = WebdevReportClient._fix_case_data("<html><body></body></html>", spec)
    assert 'id="cases-data"' in out
    assert out.index('id="cases-data"') < out.index("</body>")


def test_fix_case_data_noop_without_case_results() -> None:
    html = "<html><body>x</body></html>"
    assert WebdevReportClient._fix_case_data(html, {}) == html
    assert WebdevReportClient._fix_case_data(html, {"case_results": []}) == html


def test_fix_case_data_escapes_script_close_sequence() -> None:
    """When the content contains `</` it must be escaped, or the <script> tag is closed early."""
    spec = {"case_results": [{"output": "</script>evil"}]}
    out = WebdevReportClient._fix_case_data("<html><body></body></html>", spec)
    assert "<\\/script>" in out


# ════════════════════════════════════════════════════════════════════════════
# _inject_report_data / _attach_grade_distribution
# ════════════════════════════════════════════════════════════════════════════


def test_inject_report_data_excludes_case_results() -> None:
    spec = {
        "run_id": "r1",
        "final_score": 0.9,
        "case_results": [{"input": "bulk"}],
    }
    html = '<html><body><script type="application/json" id="report-data">{}</script></body></html>'
    out = WebdevReportClient._inject_report_data(html, spec)
    payload = out.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    data = json.loads(payload)
    assert data["run_id"] == "r1"
    assert "case_results" not in data


def test_inject_report_data_fallback_when_tag_missing() -> None:
    out = WebdevReportClient._inject_report_data(
        "<html><body></body></html>",
        {"run_id": "r1"},
    )
    assert 'id="report-data"' in out


def test_attach_grade_distribution_counts_grades() -> None:
    spec: dict[str, Any] = {
        "case_results": [
            {"grade": "EXCELLENT"},
            {"grade": "GOOD"},
            {"grade": "GOOD"},
            {"grade": ""},  # An empty string is not counted
            {"output": "无 grade 的 case 不计"},
            "非 dict 行不炸",
        ],
    }
    WebdevReportClient._attach_grade_distribution(spec)
    assert spec["grade_distribution"] == {"EXCELLENT": 1, "GOOD": 2}


def test_attach_grade_distribution_respects_existing_value() -> None:
    spec: dict[str, Any] = {
        "grade_distribution": {"POOR": 5},
        "case_results": [{"grade": "GOOD"}],
    }
    WebdevReportClient._attach_grade_distribution(spec)
    assert spec["grade_distribution"] == {"POOR": 5}


def test_attach_grade_distribution_noop_without_grades() -> None:
    spec: dict[str, Any] = {"case_results": [{"output": "x"}]}
    WebdevReportClient._attach_grade_distribution(spec)
    assert "grade_distribution" not in spec


# ════════════════════════════════════════════════════════════════════════════
# _attach_today_criterion_means (the flat "today" data source for the radar chart)
# ════════════════════════════════════════════════════════════════════════════


def test_attach_today_criterion_means_flattens_mean() -> None:
    """A nested criterion_metrics stats object → a flat {cid: mean} (the same shape as yesterday)."""
    spec: dict[str, Any] = {
        "criterion_metrics": {
            "intent": {"mean": 1.8, "median": 2, "sample_count": 5},
            "safety": {"mean": 1, "p95": 1},
        },
    }
    WebdevReportClient._attach_today_criterion_means(spec)
    assert spec["today_criterion_means"] == {"intent": 1.8, "safety": 1.0}
    # The value must be a plain numeric float (the frontend divides it directly)
    assert all(isinstance(v, float) for v in spec["today_criterion_means"].values())


def test_attach_today_criterion_means_skips_invalid() -> None:
    """A non-dict stats value / a missing mean / a bool mean is always skipped."""
    spec: dict[str, Any] = {
        "criterion_metrics": {
            "ok": {"mean": 1.5},
            "bad_str": {"mean": "1.5"},
            "bad_bool": {"mean": True},
            "no_mean": {"median": 1},
            "not_dict": "broken",
        },
    }
    WebdevReportClient._attach_today_criterion_means(spec)
    assert spec["today_criterion_means"] == {"ok": 1.5}


def test_attach_today_criterion_means_noop_when_empty() -> None:
    """When criterion_metrics is missing or entirely invalid, do not inject the key."""
    spec: dict[str, Any] = {}
    WebdevReportClient._attach_today_criterion_means(spec)
    assert "today_criterion_means" not in spec
    spec2: dict[str, Any] = {"criterion_metrics": {"bad": {"mean": None}}}
    WebdevReportClient._attach_today_criterion_means(spec2)
    assert "today_criterion_means" not in spec2


def test_attach_today_criterion_means_idempotent() -> None:
    """When spec already has a non-empty today_criterion_means, do not overwrite it (a caller pre-assembly wins)."""
    spec: dict[str, Any] = {
        "today_criterion_means": {"intent": 2.0},
        "criterion_metrics": {"intent": {"mean": 1.8}},
    }
    WebdevReportClient._attach_today_criterion_means(spec)
    assert spec["today_criterion_means"] == {"intent": 2.0}


# ════════════════════════════════════════════════════════════════════════════
# _attach_report_insights / _generate_insight_text
# ════════════════════════════════════════════════════════════════════════════


def test_attach_report_insights_computes_score_loss() -> None:
    """When attribution / root-cause display is present → inject insight_text; rank by total lost-score contribution, using the Chinese root-cause name."""
    spec: dict[str, Any] = {
        "final_score": 1.7,
        "total_samples": 10,
        "total_success": 10,
        "criteria": [
            {"id": "d1_intent", "name": "意图识别", "weight": 0.2},
            {"id": "d3_accuracy", "name": "信息真实性", "weight": 0.2},
        ],
        "case_results": [
            {
                "root_cause_display": "规划/策略错误",
                "attribution": {"primary_issue_category": "strategy_error"},
                "scores": {
                    "fused": {"score": 1.5},
                    "d1_intent": {"score": 1.6},
                    "d3_accuracy": {"score": 1.8},
                },
            },
            {
                "root_cause_display": "规划/策略错误",
                "attribution": {"primary_issue_category": "strategy_error"},
                "scores": {
                    "fused": {"score": 1.3},
                    "d1_intent": {"score": 1.4},
                    "d3_accuracy": {"score": 1.9},
                },
            },
            {
                "root_cause_display": "生成/事实性错误",
                "attribution": {"primary_issue_category": "factuality_error"},
                "scores": {
                    "fused": {"score": 1.0},
                    "d1_intent": {"score": 1.8},
                    "d3_accuracy": {"score": 0.5},
                },
            },
        ],
    }
    WebdevReportClient._attach_report_insights(spec)

    assert "score_loss_attribution" not in spec
    assert "score_loss_matrix" not in spec
    assert "insight_text" in spec
    assert isinstance(spec["insight_text"], str)
    assert "规划/策略错误" in spec["insight_text"]
    assert "生成/事实性错误" in spec["insight_text"]
    assert len(spec["insight_text"]) <= 200
    # The English raw category must no longer dominate the copy
    assert "tool_unavailable" not in spec["insight_text"]


def test_attach_report_insights_ranks_by_total_not_mean() -> None:
    """A large volume of medium loss should outweigh a few samples with a high mean (so tool is not misreported as Top)."""
    spec: dict[str, Any] = {
        "final_score": 1.5,
        "total_samples": 100,
        "total_success": 99,
        "case_results": [
            # 53 strategy cases: each loses 1.0 → total 53
            *[
                {
                    "root_cause_display": "规划/策略错误",
                    "scores": {"fused": {"score": 1.0}},
                }
                for _ in range(53)
            ],
            # 2 tool cases: each loses 1.8 → total 3.6 (a higher mean but a small total)
            {
                "root_cause_display": "工具/不可用",
                "scores": {"fused": {"score": 0.2}},
            },
            {
                "root_cause_display": "工具/执行错误",
                "scores": {"fused": {"score": 0.2}},
            },
        ],
    }
    WebdevReportClient._attach_report_insights(spec)
    text = spec["insight_text"]
    assert "规划/策略错误" in text
    # Top2 should be strategy plus one of the two tools, with strategy first
    assert text.index("规划/策略错误") < text.index("工具")


def test_attach_report_insights_always_refreshes() -> None:
    """Recompute insight_text from the current cases every time, overwriting the old copy."""
    spec: dict[str, Any] = {
        "insight_text": "已有洞察-过期",
        "final_score": 1.4,
        "total_samples": 100,
        "total_success": 99,
        "case_results": [
            {
                "root_cause_display": "规划/策略错误",
                "scores": {"fused": {"score": 1.0}},
            },
            {
                "root_cause_display": "生成/事实性错误",
                "scores": {"fused": {"score": 1.2}},
            },
        ],
    }
    WebdevReportClient._attach_report_insights(spec)
    text = spec["insight_text"]
    assert "已有洞察-过期" not in text
    assert "规划/策略错误" in text
    assert "生成/事实性错误" in text


def test_attach_report_insights_no_attribution() -> None:
    """A case with no attribution data → do not inject insight_text (the two retired fields are never injected either)."""
    spec: dict[str, Any] = {
        "final_score": 1.8,
        "total_samples": 5,
        "total_success": 5,
        "criteria": [{"id": "d1", "name": "意图", "weight": 0.2}],
        "case_results": [
            {"scores": {"fused": {"score": 1.9}}},
            {"scores": {"fused": {"score": 1.7}}},
        ],
    }
    WebdevReportClient._attach_report_insights(spec)
    assert "score_loss_attribution" not in spec
    assert "score_loss_matrix" not in spec
    assert "insight_text" not in spec


def test_attach_report_insights_no_case_results() -> None:
    """No case_results → do not inject insight_text."""
    spec: dict[str, Any] = {"final_score": 1.8}
    WebdevReportClient._attach_report_insights(spec)
    assert "insight_text" not in spec


# ════════════════════════════════════════════════════════════════════════════
# _attach_root_cause_display (precompute the display text for the "root cause" column)
# ════════════════════════════════════════════════════════════════════════════


def test_root_cause_display_prefers_attribution_root_cause() -> None:
    """Attribution is present and contains the primary-cause field root_cause → use the primary-cause text (highest priority)."""
    spec: dict[str, Any] = {
        "criteria": [],
        "case_results": [
            {
                "attribution": {
                    "root_cause": "工具超时导致回答退化",
                    "primary_issue_layer": "tool",
                    "primary_issue_category": "tool/tool_timeout",
                },
                "scores": {"fused": {"score": 1.0}},
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "工具超时导致回答退化"


def test_root_cause_display_ieval_layer_category_chinese() -> None:
    """ieval: primary_issue_layer/category (value_labels + a deduped prefix) → Chinese display text.

    Aligned with the "composite attribution" stats table: value_labels maps a raw category value to
    the ``{layer}/{label}`` shape (for example strategy_error → planning/strategy_error),
    joins it with primary_issue_layer=planning, dedupes the prefix, then converts it to Chinese
    (planning/strategy_error → the Chinese label for planning/strategy error).
    """
    spec: dict[str, Any] = {
        "benchmark_name": "ieval_external",
        "criteria": [],
        "attribution_metrics": {
            "fused": {
                "value_labels": {
                    "primary_issue_category": {
                        "strategy_error": "planning/strategy_error",
                    },
                },
            },
        },
        "case_results": [
            {
                "attribution": {
                    "primary_issue_layer": "planning",
                    "primary_issue_category": "strategy_error",
                },
                "scores": {"fused": {"score": 1.0}},
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "规划/策略错误"


def test_root_cause_display_other_uses_root_cause_only() -> None:
    """Other businesses: use only the primary-cause root_cause text, not layer/category.

    The clarified rule: other businesses "use the primary cause inside the default attribution" — when root_cause is present
    show the primary-cause text, and when it is absent fall back to dimensions short of full score (do not read layer/category).
    """
    spec: dict[str, Any] = {
        "criteria": [{"id": "D1", "name": "意图识别"}],
        "case_results": [
            {
                "attribution": {
                    "root_cause": "工具超时导致回答退化",
                    "primary_issue_layer": "tool",
                    "primary_issue_category": "tool_timeout",
                },
                "scores": {"fused": {"score": 1.0}, "D1": {"score": 1.2}},
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "工具超时导致回答退化"


def test_root_cause_display_other_no_root_cause_falls_back_to_dims() -> None:
    """Other businesses: attribution has no primary cause root_cause → fall back to the Chinese names of dimensions short of full score (do not read layer/category)."""
    spec: dict[str, Any] = {
        "criteria": [{"id": "D1", "name": "意图识别"}, {"id": "D3", "name": "信息真实性"}],
        "case_results": [
            {
                "attribution": {
                    "primary_issue_layer": "tool",
                    "primary_issue_category": "tool_timeout",
                },
                "scores": {
                    "fused": {"score": 1.0},
                    "D1": {"score": 1.2},  # Short of full score → count it
                    "D3": {"score": 2.0},  # Full score → do not count it
                },
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "意图识别"


def test_root_cause_display_ieval_dedupes_layer_category_prefix() -> None:
    """ieval: layer=planning + category label=planning/strategy_error → deduped to planning/strategy_error.

    Reproduces the repeated-prefix root-cause column ``planning/planning/strategy_error`` on the ieval_external-20260915 production report:
    value_labels maps strategy_error to planning/strategy_error,
    which is joined with primary_issue_layer=planning and then deduped.
    """
    spec: dict[str, Any] = {
        "benchmark_name": "ieval_external",
        "criteria": [],
        "attribution_metrics": {
            "fused": {
                "value_labels": {
                    "primary_issue_category": {
                        "strategy_error": "planning/strategy_error",
                        "intent_error": "planning/intent_error",
                    },
                },
            },
        },
        "case_results": [
            {
                "attribution": {
                    "primary_issue_layer": "planning",
                    "primary_issue_category": "strategy_error",
                },
                "scores": {"fused": {"score": 1.0}},
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "规划/策略错误"


def test_root_cause_display_none_layer_falls_back_to_dims() -> None:
    """primary_issue_layer=none counts as no valid attribution → fall back to the Chinese names of dimensions short of full score."""
    spec: dict[str, Any] = {
        "benchmark_name": "ieval_external",
        "criteria": [{"id": "D1", "name": "意图识别"}, {"id": "D3", "name": "信息真实性"}],
        "case_results": [
            {
                "attribution": {"primary_issue_layer": "none"},
                "scores": {
                    "fused": {"score": 1.0},
                    "D1": {"score": 1.2},  # Short of full score → count it
                    "D3": {"score": 2.0},  # Full score → do not count it
                },
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "意图识别"


def test_root_cause_display_ieval_attribution_in_fused_scores() -> None:
    """ieval: no case-level attribution → fall back to scores.fused.attribution, and convert layer/category to Chinese."""
    spec: dict[str, Any] = {
        "benchmark_name": "ieval_external",
        "criteria": [],
        "case_results": [
            {
                "scores": {
                    "fused": {
                        "score": 1.0,
                        "attribution": {
                            "primary_issue_layer": "generation",
                            "primary_issue_category": "factuality_error",
                        },
                    },
                },
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "生成/事实性错误"


def test_root_cause_display_ieval_underfull_dimension_names() -> None:
    """ieval with no attribution → the Chinese names of scores dimensions short of full score (D0 full score 1, the rest full score 2)."""
    spec: dict[str, Any] = {
        "criteria": [
            {"id": "D0", "name": "安全与合规"},
            {"id": "D1", "name": "意图识别"},
            {"id": "D3", "name": "信息真实性"},
        ],
        "case_results": [
            {
                "scores": {
                    "fused": {"score": 1.5},
                    "D0": {"score": 1},  # Full score 1 → no points lost, do not count it
                    "D1": {"score": 1.2},  # Full score 2 → short of full score
                    "D3": {"score": 2.0},  # Full score → do not count it
                },
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "意图识别"


def test_root_cause_display_ieval_skips_root_cause() -> None:
    """When an ieval business has attribution, do not use the raw root-cause text; convert primary_issue_layer/category to Chinese directly.

    On an ieval report (benchmark_name starts with ieval, for example ieval_external) the "root cause" column
    does not render the long root_cause text — when attribution data exists it shows primary_issue_layer /
    primary_issue_category (deduped prefix + converted to Chinese, aligned with the composite-attribution stats table),
    and when there is no attribution data it shows the Chinese names of dimensions short of full score.
    """
    spec: dict[str, Any] = {
        "benchmark_name": "ieval_external",
        "criteria": [],
        "case_results": [
            {
                "attribution": {
                    "root_cause": "规划阶段未建立约束检查，产生连锁问题……",
                    "primary_issue_layer": "planning",
                    "primary_issue_category": "strategy_error",
                },
                "scores": {"fused": {"score": 1.0}},
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "规划/策略错误"


def test_root_cause_display_ieval_root_cause_only_falls_back_to_dims() -> None:
    """ieval attribution contains only root_cause (no layer/category) → skip the root cause, and fall back to the Chinese names of dimensions short of full score."""
    spec: dict[str, Any] = {
        "benchmark_name": "ieval_external",
        "criteria": [
            {"id": "D1", "name": "意图识别"},
            {"id": "D3", "name": "信息真实性"},
        ],
        "case_results": [
            {
                "attribution": {"root_cause": "长文本根因……"},
                "scores": {
                    "fused": {"score": 1.0},
                    "D1": {"score": 1.2},  # Short of full score → count it
                    "D3": {"score": 2.0},  # Full score → do not count it
                },
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "意图识别"


def test_root_cause_display_ieval_ignores_sub_dimension_sub_scores() -> None:
    """ieval sub_scores are sub-dimensions (such as D6a/D6b, which are not in criteria) → prefer scores[cid].

    Reproduces the ieval_external-20260915 production-report incident where the root-cause column showed the raw keys ``D6a/D6b``
    instead of the Chinese names of dimensions short of full score: the case contains both scores (D0-D6 criteria) and
    sub_scores (D6a/D6b sub-dimensions), so scores[cid] should be preferred, on the same scale as the "per-dimension scores" column,
    and a non-criteria key from sub_scores is not read.
    """
    spec: dict[str, Any] = {
        "benchmark_name": "ieval_external",
        "criteria": [
            {"id": "D0", "name": "安全与合规"},
            {"id": "D6", "name": "信息充分性与决策帮助度"},
        ],
        "case_results": [
            {
                "scores": {
                    "fused": {"score": 1.5},
                    "D0": {"score": 1.0},  # Full score 1 → do not count it
                    "D6": {"score": 0.0},  # Short of full score → count it
                },
                "sub_scores": {"D6a": {"score": 1.0}, "D6b": {"score": 0.5}},
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "信息充分性与决策帮助度"


def test_root_cause_display_search_eval_sub_scores() -> None:
    """search_eval with no attribution → dimensions in sub_scores that are short of full score, joined with " / ", in criteria order."""
    spec: dict[str, Any] = {
        "benchmark_name": "search_eval",
        "criteria": [
            {"id": "intent_understanding", "name": "意图理解"},
            {"id": "relevance", "name": "内容切题"},
            {"id": "factual_accuracy", "name": "信息准确性"},
        ],
        "case_results": [
            {
                "scores": {"fused": {"score": 1.5}},
                "sub_scores": {
                    "relevance": 1.0,  # The dict order is scrambled — output must follow criteria order
                    "intent_understanding": 2.0,  # Full score → do not count it
                    "factual_accuracy": 1.5,
                },
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "内容切题/信息准确性"


def test_root_cause_display_search_eval_ignores_attribution() -> None:
    """search_eval: even when attribution is present, use only the sub_scores dimensions that are short of full score (the clarified rule)."""
    spec: dict[str, Any] = {
        "benchmark_name": "search_eval",
        "criteria": [
            {"id": "intent_understanding", "name": "意图理解"},
            {"id": "relevance", "name": "内容切题"},
        ],
        "case_results": [
            {
                "attribution": {
                    "root_cause": "应被忽略的主因",
                    "primary_issue_layer": "planning",
                    "primary_issue_category": "strategy_error",
                },
                "scores": {"fused": {"score": 1.5}},
                "sub_scores": {
                    "intent_understanding": 2.0,  # Full score → do not count it
                    "relevance": 1.0,  # Short of full score → count it
                },
            },
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert spec["case_results"][0]["root_cause_display"] == "内容切题"


def test_root_cause_display_full_score_unset_and_idempotent() -> None:
    """All at full score → do not set it; a case that already has root_cause_display is not overwritten (idempotent)."""
    spec: dict[str, Any] = {
        "criteria": [{"id": "D1", "name": "意图识别"}],
        "case_results": [
            {"scores": {"fused": {"score": 2.0}, "D1": {"score": 2.0}}},
            {"root_cause_display": "已有值", "scores": {"D1": {"score": 0.5}}},
        ],
    }
    WebdevReportClient._attach_root_cause_display(spec)
    assert "root_cause_display" not in spec["case_results"][0]
    assert spec["case_results"][1]["root_cause_display"] == "已有值"


def test_build_template_prompt_root_cause_display_column() -> None:
    """The root-cause column must render the caller-precomputed root_cause_display directly (JS must not derive it)."""
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "case.root_cause_display" in prompt
    assert "禁止 JS 从 attribution/scores" in prompt
    assert "只允许这 7 类" in prompt
    assert "禁止生成「用例ID」「输入」「操作」列" in prompt
    # The output column (after query): display_output truncated to 40 characters, allowed to wrap, plus the full text on hover
    assert "output：display_output" in prompt
    assert "removeColByName(table,'output')" not in _REPORT_UX_PATCH
    assert "removeColByName(table,'操作')" in _REPORT_UX_PATCH
    assert "removeColByName(table,'用例ID')" in _REPORT_UX_PATCH
    assert "window.filteredCases" in _REPORT_UX_PATCH
    assert "filtered[pageOffset+i]" in _REPORT_UX_PATCH


# ════════════════════════════════════════════════════════════════════════════
# _attach_enhance_modules (waterfall / heatmap / period-over-period — only new fields are added)
# ════════════════════════════════════════════════════════════════════════════


def test_attach_criterion_deltas_from_today_yesterday() -> None:
    """today/yesterday_criterion_means are both present → criterion_deltas contains delta."""
    spec: dict[str, Any] = {
        "criteria": [{"id": "D1", "name": "意图识别"}, {"id": "D2", "name": "约束"}],
        "today_criterion_means": {"D1": 1.9, "D2": 1.5},
        "yesterday_criterion_means": {"D1": 1.8, "D2": 1.6},
        "case_results": [],
    }
    WebdevReportClient._attach_enhance_modules(spec)
    deltas = {d["id"]: d for d in spec["criterion_deltas"]}
    assert deltas["D1"]["delta"] == round(1.9 - 1.8, 4)
    assert deltas["D2"]["name"] == "约束"
    # Idempotent
    WebdevReportClient._attach_enhance_modules(spec)
    assert len(spec["criterion_deltas"]) == 2


def test_attach_score_loss_waterfall_uses_root_cause_display() -> None:
    """The waterfall apportions the composite-score drop by root_cause_display weight (the bars sum to about Δ, and a single bar is ≤2)."""
    spec: dict[str, Any] = {
        "final_score": 1.54,
        "criteria": [],
        "case_results": [
            {
                "root_cause_display": "规划/策略错误",
                "scores": {"fused": {"score": 1.0}},  # loss 1.0
            },
            {
                "root_cause_display": "规划/策略错误",
                "scores": {"fused": {"score": 0.5}},  # loss 1.5 → raw 2.5
            },
            {
                "root_cause_display": "生成/事实性错误",
                "scores": {"fused": {"score": 1.5}},  # loss 0.5
            },
            # No root cause → count it as "unattributed", so the lost score is not apportioned onto an existing root cause
            {"scores": {"fused": {"score": 0.2}}, "grade": "FAIL"},
        ],
    }
    WebdevReportClient._attach_enhance_modules(spec)
    wf = spec["score_loss_waterfall"]
    assert wf["start"] == 2.0
    assert wf["end"] == 1.54
    assert wf["delta"] == round(2.0 - 1.54, 4)
    assert wf["steps"][0]["label"] == "规划/策略错误"
    by_label = {step["label"]: step for step in wf["steps"]}
    # Δ=0.46; strategy raw=2.5 / total_raw=4.8 (including 1.8 unattributed)
    assert by_label["规划/策略错误"]["loss"] == round(0.46 * (2.5 / 4.8), 3)
    assert by_label["未归因"]["raw_loss"] == 1.8
    assert by_label["未归因"]["count"] == 1
    assert wf["steps"][0]["count"] == 2
    assert all(s["loss"] <= 2.0 for s in wf["steps"])
    assert abs(sum(s["loss"] for s in wf["steps"]) - wf["delta"]) < 0.01


def test_attach_score_loss_waterfall_start_uses_score_full_mark() -> None:
    """When the trace daily report's full score is 1.0, the waterfall start and delta share the same scale."""
    spec: dict[str, Any] = {
        "final_score": 0.6,
        "score_full_mark": 1.0,
        "criteria": [],
        "case_results": [
            {
                "root_cause_display": "规划/策略错误",
                "scores": {"fused": {"score": 0.4}},
            },
        ],
    }
    WebdevReportClient._attach_enhance_modules(spec)
    wf = spec["score_loss_waterfall"]
    assert wf["start"] == 1.0
    assert wf["end"] == 0.6
    assert wf["delta"] == round(1.0 - 0.6, 4)
    assert wf["steps"][0]["loss"] <= wf["start"]


def test_attach_root_cause_dim_matrix_accepts_string_full_mark() -> None:
    """When score_full_mark is a string, share _score_full_mark with the waterfall, and do not fall back to 2.0."""
    spec: dict[str, Any] = {
        "score_full_mark": "1.0",
        "criteria": [{"id": "D1", "name": "意图识别"}],
        "case_results": [
            {
                "root_cause_display": "规划/策略错误",
                "scores": {"D1": {"score": 0.4}},
            },
        ],
    }
    WebdevReportClient._attach_enhance_modules(spec)
    cell = spec["root_cause_dim_matrix"]["cells"]["规划/策略错误"]["D1"]
    assert cell == round(1.0 - 0.4, 3)


def test_attach_report_insights_clamps_overfull_fused_score() -> None:
    """When the full score is 1.0 and fused=1.5, the lost score is 0, and a negative loss is not counted into the share."""
    spec: dict[str, Any] = {
        "final_score": 0.5,
        "score_full_mark": 1.0,
        "total_samples": 4,
        "total_success": 4,
        "case_results": [
            {
                "root_cause_display": "规划/策略错误",
                "scores": {"fused": {"score": 1.5}},
            },
            {
                "root_cause_display": "生成/事实性错误",
                "scores": {"fused": {"score": 0.0}},
            },
        ],
    }
    WebdevReportClient._attach_report_insights(spec)
    text = spec["insight_text"]
    assert "生成/事实性错误占失分 100%" in text
    assert "规划/策略错误占 0%" in text


def test_attach_root_cause_dim_matrix_accumulates_loss() -> None:
    """The heatmap is the average loss by root cause × dimension (including full-score cases; a few zeros are not exaggerated)."""
    spec: dict[str, Any] = {
        "criteria": [
            {"id": "D0", "name": "安全与合规"},
            {"id": "D1", "name": "意图识别"},
        ],
        "case_results": [
            {
                "root_cause_display": "规划/策略错误",
                "scores": {
                    "fused": {"score": 1.0},
                    "D0": {"score": 1.0},  # Full score → loss 0, and it still counts in the denominator
                    "D1": {"score": 1.0},  # Loss of 1
                },
            },
            {
                "root_cause_display": "规划/策略错误",
                "scores": {
                    "fused": {"score": 0.8},
                    "D0": {"score": 0.5},  # Loss of 0.5
                    "D1": {"score": 1.5},  # Loss of 0.5
                },
            },
            {
                # No root cause but a valid dimension score: bucket it as "unattributed", and a loss of 0 still displays 0.000
                "scores": {
                    "fused": {"score": 2.0},
                    "D0": {"score": 1.0},
                    "D1": {"score": 2.0},
                },
            },
        ],
    }
    WebdevReportClient._attach_enhance_modules(spec)
    mx = spec["root_cause_dim_matrix"]
    assert mx["rows"] == ["规划/策略错误", "未归因"]
    assert mx["unit"] == "mean_loss"
    # D1 mean = (1.0+0.5)/2 = 0.75；D0 mean = (0+0.5)/2 = 0.25
    assert mx["cells"]["规划/策略错误"]["D1"] == 0.75
    assert mx["cells"]["规划/策略错误"]["D0"] == 0.25
    assert mx["cells"]["未归因"]["D0"] == 0.0
    assert mx["cells"]["未归因"]["D1"] == 0.0
    assert all(v <= 2.0 for row in mx["cells"].values() for v in row.values())
    assert mx["cells"]["规划/策略错误"]["D0"] <= 1.0


def test_attach_root_cause_dim_matrix_does_not_inflate_sparse_zero() -> None:
    """When only 1 of 3 cases has D6=0, the cell should be ~0.67 rather than 2.0."""
    spec: dict[str, Any] = {
        "criteria": [{"id": "D6", "name": "信息充分性"}],
        "case_results": [
            {
                "root_cause_display": "工具/执行错误",
                "scores": {"fused": {"score": 1.0}, "D6": {"score": 2.0}},
            },
            {
                "root_cause_display": "工具/执行错误",
                "scores": {"fused": {"score": 1.0}, "D6": {"score": 2.0}},
            },
            {
                "root_cause_display": "工具/执行错误",
                "scores": {"fused": {"score": 0.5}, "D6": {"score": 0.0}},
            },
        ],
    }
    WebdevReportClient._attach_enhance_modules(spec)
    cell = spec["root_cause_dim_matrix"]["cells"]["工具/执行错误"]["D6"]
    assert cell == round((0 + 0 + 2) / 3, 3)
    assert cell < 2.0


def test_inject_report_enhance_idempotent() -> None:
    """The enhance-patch injection is idempotent; a native module that already exists is not built again."""
    html = "<html><body><div class='container'></div></body></html>"
    out = WebdevReportClient._inject_report_enhance(html)
    assert 'id="report-enhance-patch"' in out
    assert "score_loss_waterfall" in out
    assert "qhub-banner" in out
    assert "function hasNativeSection" in out
    assert "'waterfall-delta-section','分数损失与维度环比'" in out
    assert "'heatmap-section','根因 × 维度失分矩阵'" in out
    assert "buildBlock(true,!hasMatrix)" in out
    # When a native waterfall section exists, replace it wholesale (the template's .wf-* bar heights collapsing is unreliable)
    assert "removeChild(nativeSec)" in out
    assert "#waterfall-delta-section .card" not in out
    assert "#heatmap-section .heatmap-wrap" not in out
    assert "#waterfall-delta-section{display:grid" in out
    assert "#waterfall-delta-section>.wd-grid{grid-column:1/-1" in out
    # Waterfall bar height shares an axis with the full score (value/start); normalizing by max(step.loss) is forbidden
    assert "loss/start*100" in out
    assert "maxLoss" not in out
    assert "start.toFixed(3)" in out
    assert "loss.toFixed(3)" in out
    assert "hasValue&&Number.isFinite(v)?v.toFixed(3):'—'" in out
    assert "计算方法：总落差" in out
    assert "计算方法：维度环比" in out
    assert "计算方法：单元格" in out
    assert "var isFlat=Math.abs(delta)<=0.01" in out
    assert "绝对变化 ≤ 0.010 按持平显示" in out
    out2 = WebdevReportClient._inject_report_enhance(out)
    assert out2.count('id="report-enhance-patch"') == 1


def test_report_enhance_waterfall_scales_against_start_not_max_loss() -> None:
    """The tallest root-cause bar must not be normalized to 100% by max(loss) (the incident where it lined up with the full score)."""
    patch = webdev_client._REPORT_ENHANCE_PATCH
    assert "loss/start*100" in patch
    assert "s.loss/maxLoss" not in patch
    assert "if(s.loss>maxLoss) maxLoss=s.loss" not in patch
    # The composite-score bar shares the axis: end/start, and a hardcoded /2 is forbidden
    assert "end/start*100" in patch
    assert "(Number(wf.end||0)/2)*100" not in patch


def test_report_ux_patch_normalizes_unknown_and_aligns_grade() -> None:
    """Composite-attribution filter: an empty category normalizes to unknown; clicking a category switches to all statuses."""
    assert "function normCat" in _REPORT_UX_PATCH
    assert "normCat(caseAttr(c).primary_issue_category)" in _REPORT_UX_PATCH
    assert "origFRC.__ruxAlign" in _REPORT_UX_PATCH
    assert "g.value=''" in _REPORT_UX_PATCH
    # The "all statuses" value drifts with the template ('' or 'ALL'): the filter condition must exclude 'ALL',
    # otherwise gv='ALL' is truthy → c.grade!==gv is always true → case detail is always 0 rows
    # (the ieval_external-20260916-a1 production incident); when switching to all statuses, setting value to empty misses,
    # so fall back to selectedIndex=0, and the dropdown is not left blank.
    assert "gv!=='ALL'" in _REPORT_UX_PATCH
    assert "g.selectedIndex=0" in _REPORT_UX_PATCH


def test_report_ux_patch_case_id_trace_and_output_column() -> None:  # noqa: PLR0915
    """Drop the case-id column + an output column after query + Trace fill (search_eval-20260921)."""
    # Case detail no longer shows a "case id" column; the detail entry moves to the query dropdown.
    assert "removeColByName(table,'用例ID')" in _REPORT_UX_PATCH
    assert "th.textContent='用例ID'" not in _REPORT_UX_PATCH
    # Only query can be clicked to expand detail: remove the template's row-level onclick
    assert "row.removeAttribute('onclick')" in _REPORT_UX_PATCH
    # Locating the table must not look only for #data-table (ieval_external-20260916: class=case-tbl
    # / tbody#case-tbody with a missing id → query hover-click and the Trace column stay '-')
    assert "function findCaseTable" in _REPORT_UX_PATCH
    assert "getElementById('case-tbody')" in _REPORT_UX_PATCH
    assert "table.case-tbl" in _REPORT_UX_PATCH
    assert "var table=findCaseTable()" in _REPORT_UX_PATCH
    assert "t==='query'||t==='用例ID'" in _REPORT_UX_PATCH
    # The trace-link column is filled by reshapeCaseColumns with a real <a> from c.trace_url
    assert "colIndex(table,'Trace链接')" in _REPORT_UX_PATCH
    assert "ta.className='trace-link'" in _REPORT_UX_PATCH
    # The modal header can still inject Langfuse (the openDetail path is kept; the data source is asserted on the next line)
    assert "function injectTraceLink" in _REPORT_UX_PATCH
    assert "rux-trace-link" in _REPORT_UX_PATCH
    assert "__ruxActiveTraceUrl" in _REPORT_UX_PATCH
    assert "__ruxTrace" in _REPORT_UX_PATCH
    # The query dropdown click path (_QUERY_UX_PATCH.ensureTraceToggle) writes trace_url in sync,
    # so the openDetail wrapper (_REPORT_UX_PATCH.injectTraceLink) can read the data
    # (otherwise !url returns early, and the Langfuse external link in the modal header is never injected)
    assert "window.__ruxActiveTraceUrl=String(c.trace_url||'').trim()" in _QUERY_UX_PATCH
    # Opening it a second time must update href; an early-return must not leave a stale Langfuse link
    assert "existing.href=url" in _REPORT_UX_PATCH
    assert "if(modal.querySelector('.rux-trace-link')) return;" not in _REPORT_UX_PATCH
    assert "window.open(url,'_blank'" not in _REPORT_UX_PATCH
    # The modal header also accepts the template's .dm-header (not only .detail-modal-header)
    assert ".dm-header,.dm-title" in _REPORT_UX_PATCH
    # interactive after DCL has passed must run immediately, and must not wait for DOMContentLoaded again
    assert "readyState==='loading'" in _REPORT_UX_PATCH
    assert "readyState!=='complete'" not in _REPORT_UX_PATCH
    # The wrapper chain tolerates errors with try
    assert "try{ origRender.apply(this,arguments); }catch(e){" in _QUERY_UX_PATCH
    assert "try{ origRT.apply(this,arguments); }catch(e){" in _REPORT_UX_PATCH
    assert "try{ if(typeof renderPagination==='function')" in _REPORT_UX_PATCH
    assert ".hover-overlay" in _QUERY_UX_PATCH
    assert ".cell-truncate" in _QUERY_UX_PATCH
    assert "if(!full) full=fullTextOf(clipEl)" in _QUERY_UX_PATCH
    assert "__ruxRowGuard" in _REPORT_UX_PATCH
    assert "a.trace-link,a.rux-trace-link" in _REPORT_UX_PATCH
    # Composite attribution: accept both .attr-title (search) and .dist-section-title
    assert ".attr-title,.dist-section-title" in _REPORT_UX_PATCH
    assert "function isAttrFieldTitle" in _REPORT_UX_PATCH
    # The search template's .dist-bar-fill is missing a color, so fall back
    assert ".dist-bar-fill,.attr-bar-fill{background:#4f46e5}" in _REPORT_UX_PATCH
    # output column: display_output wins, truncated to 40 characters, and wraps onto multiple lines
    assert "function outputTextOf" in _REPORT_UX_PATCH
    assert "c.display_output" in _REPORT_UX_PATCH
    assert "clipText(ofull,40)" in _REPORT_UX_PATCH
    assert ".output-clip" in _REPORT_UX_PATCH
    assert "word-break:break-word" in _REPORT_UX_PATCH
    # The output column has a fixed width of 10em (about 10 CJK characters per line): under auto layout it is not squeezed by a nowrap column
    assert ".output-clip{display:inline-block;width:10em" in _REPORT_UX_PATCH
    prompt = WebdevReportClient._build_template_prompt(
        {"benchmark_name": "b", "criteria": [], "case_results": []}
    )
    assert "列宽固定 10em" in prompt
    # The hover-layer delegated selector also covers output (extended on the query-ux side)
    assert ".output-clip[data-full]" in _QUERY_UX_PATCH
    assert "function findCaseTable" in _QUERY_UX_PATCH
    assert "var table=findCaseTable()" in _QUERY_UX_PATCH
    assert "t==='query'||t==='用例ID'" in _QUERY_UX_PATCH
    # A data-full already on the template td must be removed, so hover-tip and query-tip-float do not both apply
    # Two hover layers stacked (search_eval-20260916)
    assert "ocell.removeAttribute('data-full')" in _REPORT_UX_PATCH
    assert "td.removeAttribute('data-full')" in _QUERY_UX_PATCH


def test_report_ux_patch_compact_dim_section_and_single_canvas() -> None:
    """The evaluation-dimension card is compacted (table on the left + radar centered + legend) + 3ch between dimension columns."""
    assert "function compactDimSection" in _REPORT_UX_PATCH
    assert "function installRadarRenderer" in _REPORT_UX_PATCH
    assert "function ensureRadarLegend" in _REPORT_UX_PATCH
    assert "function compactDimTable" in _REPORT_UX_PATCH
    assert "gridTemplateColumns='1fr'" in _REPORT_UX_PATCH
    assert "维度雷达（今日 vs 昨日）" in _REPORT_UX_PATCH
    assert "padding-right:3ch" in _REPORT_UX_PATCH
    assert "rux-dim-name" in _REPORT_UX_PATCH
    assert "querySelectorAll('canvas')" in _REPORT_UX_PATCH
    # Legend: today is a solid line / yesterday is a dashed line, and a missing legend is built
    assert "addItem('today','今日'" in _REPORT_UX_PATCH
    assert "addItem('yesterday','昨日'" in _REPORT_UX_PATCH
    assert "flex='1 1 0'" in _REPORT_UX_PATCH


def test_trace_link_patch_matches_truncated_case_id_via_data_full() -> None:
    """After the case id is truncated, row matching for the trace-link column reads the full traceID from data-full."""
    assert "querySelector('[data-full]')" in _TRACE_LINK_PATCH


def test_report_ux_patch_fixes_trend_chart_js_guard() -> None:
    """search_eval-20260916: a leftover Chart.js guard on the native 2D trend chart → the whole section is blank."""
    # Patch side: detect the leftover guard, inject a stub only when there is no new Chart, and redraw once
    assert "function fixTrendChartGuard" in _REPORT_UX_PATCH
    assert "typeof Chart==='undefined'" in _REPORT_UX_PATCH
    assert "new Chart" in _REPORT_UX_PATCH
    assert "__ruxStub" in _REPORT_UX_PATCH
    assert "window.renderTrend()" in _REPORT_UX_PATCH
    # Prompt side: explicitly forbid a chart-library existence check
    prompt = WebdevReportClient._build_template_prompt(
        {"benchmark_name": "b", "criteria": [], "case_results": [], "trend_7d": {}}
    )
    assert "禁止任何图表库" in prompt
    assert "存在性检查" in prompt


def test_report_ux_patch_search_eval_deducted_dim_filter() -> None:
    """search_eval "deduction dimensions" click: match dimensions short of full score, and do not force primary_issue_category."""
    assert "function fusedAttrField" in _REPORT_UX_PATCH
    assert "fds['扣分维度']" in _REPORT_UX_PATCH
    assert "function matchDeductedDim" in _REPORT_UX_PATCH
    assert "field==='扣分维度'" in _REPORT_UX_PATCH
    assert "function readDimScore" in _REPORT_UX_PATCH
    # When the template lacks filterByRootCause, add a minimal implementation so a click is not a no-op
    assert "typeof window.filterByRootCause!=='function'" in _REPORT_UX_PATCH


def test_report_ux_patch_grade_filter_id_drift_defensive() -> None:
    """search_eval-20260916: the status dropdown id drifts (grade-filter); normalize defensively and do not break the chain.

    The template writes the status dropdown as id="grade-filter" (not case-grade-filter) → the old
    setupDefaultFilter returned early, and the wrappers for applyFilters / filterByRootCause / renderTable
    all failed (composite-attribution click showed an empty table, case-id column shaping was lost, and output-clip did not take effect).
    Fix: a defensive normalized id lookup + a missing sel does not block the wrappers.
    """
    # id normalization: case-grade-filter / grade-filter / a fallback scan of the FAIL+POOR options
    assert "function findGradeFilter" in _REPORT_UX_PATCH
    assert "document.getElementById('grade-filter')" in _REPORT_UX_PATCH
    assert "document.getElementById('case-grade-filter')" in _REPORT_UX_PATCH
    assert "document.querySelectorAll('select')" in _REPORT_UX_PATCH
    # The search box is normalized the same way (case-search / .search-box / input[type=search])
    assert "function findCaseSearch" in _REPORT_UX_PATCH
    assert "document.querySelector('.search-box')" in _REPORT_UX_PATCH
    # A missing sel only skips inserting the option, and must not return early (the wrappers must always be attached)
    assert "var sel=findGradeFilter();" in _REPORT_UX_PATCH
    assert "if(sel&&!sel.getAttribute('data-rux')){" in _REPORT_UX_PATCH
    # Inside the wrappers, lookups always go through the normalized search
    assert "var g=findGradeFilter();" in _REPORT_UX_PATCH
    assert "var searchEl=findCaseSearch();" in _REPORT_UX_PATCH
    # Truncated-prefix matching for search (the capability does not regress after it covers the query-ux search wrapper)
    assert "indexOf(qp)===0" in _REPORT_UX_PATCH


# ════════════════════════════════════════════════════════════════════════════
# _validate_case_required_fields (check that Case detail's required-value columns are present)
# ════════════════════════════════════════════════════════════════════════════


def test_validate_case_required_fields_all_present() -> None:
    """query / case status / composite score / per-dimension scores / trace link are all present → no problem."""
    spec: dict[str, Any] = {
        "criteria": [{"id": "D1", "name": "意图识别"}],
        "case_results": [
            {
                "case_id": "c1",
                "display_input": "附近火锅店",
                "grade": "GOOD",
                "trace_url": "https://langfuse.example/t/1",
                "scores": {"fused": {"score": 1.8}, "D1": {"score": 2.0}},
            },
        ],
    }
    assert WebdevReportClient._validate_case_required_fields(spec) == []


def test_validate_case_required_fields_missing() -> None:
    """All five columns missing → report each column (including the case_id location)."""
    spec: dict[str, Any] = {
        "criteria": [{"id": "D1", "name": "意图识别"}],
        "case_results": [{"case_id": "c1", "scores": {}}],
    }
    problems = WebdevReportClient._validate_case_required_fields(spec)
    assert len(problems) == 1
    text = problems[0]
    assert "c1" in text
    for col in ("query", "case状态", "综合分", "各维度得分", "Trace链接"):
        assert col in text


def test_validate_case_required_fields_partial_dim_missing() -> None:
    """Some dimensions lack score → the per-dimension-scores column reports the missing dimension ids."""
    spec: dict[str, Any] = {
        "criteria": [{"id": "D1", "name": "意图识别"}, {"id": "D3", "name": "真实性"}],
        "case_results": [
            {
                "case_id": "c2",
                "display_input": "q",
                "grade": "POOR",
                "trace_url": "https://langfuse.example/t/2",
                "scores": {"fused": {"score": 1.0}, "D1": {"score": 1.2}},
            },
        ],
    }
    problems = WebdevReportClient._validate_case_required_fields(spec)
    assert len(problems) == 1
    assert "各维度得分(D3)" in problems[0]


def test_validate_case_required_fields_empty_cases() -> None:
    """No case_results → no problem (the Case detail section is not rendered in the first place)."""
    assert WebdevReportClient._validate_case_required_fields({"case_results": []}) == []
    assert WebdevReportClient._validate_case_required_fields({}) == []


def test_trace_link_patch_falls_back_to_detail_button_onclick() -> None:
    """When there is no case-id column, extract case_id from the "details" button's onclick as the match anchor."""
    patch = webdev_client._TRACE_LINK_PATCH
    assert "caseIdFromRow" in patch
    assert 'onclick*="openDetail"' in patch


def test_attach_report_insights_skips_non_dict_rows() -> None:
    """A non-dict row / a row missing attribution / a row missing the fused score is skipped."""
    spec: dict[str, Any] = {
        "final_score": 1.5,
        "total_samples": 10,
        "total_success": 9,
        "criteria": [{"id": "d1", "name": "意图", "weight": 0.2}],
        "case_results": [
            "not-a-dict",
            {"attribution": None},
            {"attribution": {"primary_issue_category": ""}},
            {"attribution": {"primary_issue_category": "tool_error"}},  # no scores
            {
                "attribution": {"primary_issue_category": "tool_error"},
                "scores": {"fused": {"score": 1.2}, "d1": {"score": 1.0}},
            },
        ],
    }
    WebdevReportClient._attach_report_insights(spec)
    assert "score_loss_attribution" not in spec
    assert "insight_text" in spec  # The last row's valid attribution case drives insight generation


def test_generate_insight_text_high_completion_low_score() -> None:
    """A high completion rate but a low score → the insight includes a gap diagnosis, each root cause's share, and the weakest dimension; ≤200 characters."""
    spec: dict[str, Any] = {
        "final_score": 1.56,  # 78%
        "total_samples": 200,
        "total_success": 199,  # 99.5%
        "criteria": [
            {"id": "D2", "name": "需求约束满足度"},
            {"id": "D1", "name": "意图识别与任务拆解"},
        ],
        "criterion_metrics": {
            "D2": {"mean": 1.31},
            "D1": {"mean": 1.70},
        },
    }
    score_loss = {"规划/策略错误": 48.0, "生成/事实性错误": 26.0, "工具/执行错误": 26.0}
    result = WebdevReportClient._generate_insight_text(spec, score_loss)
    assert result is not None
    assert "99.5%" in result
    assert "78%" in result
    assert "规划/策略错误" in result
    assert "生成/事实性错误" in result
    assert "占失分" in result
    assert "需求约束满足度" in result
    assert "规划质量与事实校验" in result
    assert "流程中断" in result
    assert len(result) <= 200


def test_generate_insight_text_normalizes_by_score_full_mark() -> None:
    """When the daily report's full score is 1.0 and the composite score is full, do not misreport the composite score as only 50% by dividing by 2."""
    spec: dict[str, Any] = {
        "final_score": 1.0,
        "score_full_mark": 1.0,
        "total_samples": 100,
        "total_success": 99,
        "criteria": [{"id": "D1", "name": "意图识别"}],
        "criterion_metrics": {"D1": {"mean": 1.0}},
    }
    score_loss = {"规划/策略错误": 1.0, "生成/事实性错误": 1.0}
    result = WebdevReportClient._generate_insight_text(spec, score_loss)
    assert result is not None
    assert "综合分仅 50%" not in result
    assert len(result) <= 200


def test_generate_insight_text_low_completion() -> None:
    """A low completion rate → the insight includes a prompt that the flow is not connected end to end."""
    spec: dict[str, Any] = {
        "final_score": 1.0,
        "total_samples": 100,
        "total_success": 70,
    }
    score_loss = {"工具/执行错误": 50.0}
    result = WebdevReportClient._generate_insight_text(spec, score_loss)
    assert result is not None
    assert "流程贯通" in result
    assert len(result) <= 200


def test_generate_insight_text_tool_advice_when_top_is_tool() -> None:
    """Top1 is a tool-class root cause → suggest investigating the tool chain."""
    spec: dict[str, Any] = {
        "final_score": 1.4,
        "total_samples": 100,
        "total_success": 98,
    }
    score_loss = {"工具/不可用": 40.0, "工具/执行错误": 30.0}
    result = WebdevReportClient._generate_insight_text(spec, score_loss)
    assert result is not None
    assert "工具调用链路" in result


def test_generate_insight_text_no_score_loss() -> None:
    """No deduction data → return None."""
    result = WebdevReportClient._generate_insight_text({}, {})
    assert result is None


def test_clip_insight_truncates_at_sentence() -> None:
    """When it exceeds 200 characters, truncate at a semicolon or period, and do not cut a word in the middle."""
    text = "甲" * 80 + "；" + "乙" * 80 + "；" + "丙" * 80
    out = WebdevReportClient._clip_insight(text, max_chars=200)
    assert len(out) <= 200
    assert out.endswith("；")
    assert "丙" not in out


def test_strip_html_for_insight_keeps_cases_drops_patches() -> None:
    html = (
        "<html><body>"
        '<script type="application/json" id="report-data">'
        '{"final_score":1.56}</script>'
        '<script type="application/json" id="cases-data">'
        '[{"id":"very-long-case"}]</script>'
        '<style id="query-ux-patch">.x{color:red}</style>'
        '<script id="query-ux-patch-js">var x=1;</script>'
        "<p>visible</p></body></html>"
    )
    stripped = WebdevReportClient._strip_html_for_insight(html)
    assert "very-long-case" in stripped
    assert "final_score" in stripped
    assert "query-ux-patch" not in stripped
    assert "visible" in stripped


def test_build_insight_prompt_includes_case_details() -> None:
    spec = {
        "final_score": 1.7,
        "total_samples": 2,
        "total_success": 2,
        "case_results": [
            {
                "case_id": "c-001",
                "display_input": "帮我导航去西湖",
                "display_output": "已为您规划路线",
                "grade": "POOR",
                "root_cause_display": "规划/策略错误",
                "scores": {"fused": {"score": 1.1}},
            }
        ],
    }
    prompt = WebdevReportClient._build_insight_prompt("<html></html>", "规则洞察", spec)
    assert "帮我导航去西湖" in prompt
    assert "规划/策略错误" in prompt
    assert "今日摘要" in prompt
    assert "top_problems" in prompt
    assert "规则洞察" in prompt


def test_sanitize_insight_metric_drops_placeholders() -> None:
    """The model writing "insufficient data" is a cannot-analyze placeholder, and it must be cleared and not shown."""
    assert WebdevReportClient._sanitize_insight_metric("数据不足") == ""
    assert WebdevReportClient._sanitize_insight_metric("无法计算") == ""
    assert WebdevReportClient._sanitize_insight_metric("-0.36") == "-0.36"
    problems = WebdevReportClient._coerce_top_problems(
        [
            {
                "name": "意图理解",
                "score_drag": "数据不足",
                "share_pct": "52.3%",
                "affected_cases": 23,
            },
        ],
    )
    assert problems[0]["score_drag"] == ""
    assert problems[0]["share_pct"] == "52.3%"


def test_normalize_insight_structured_parses_summary_and_top() -> None:
    raw = """```json
{
  "summary": {
    "conclusion": "综合得分 1.703（+0.05），任务满足率回升至 95.5%。",
    "bullets": ["「信息真实性」+0.08", "POOR+FAIL 共 9 例"],
    "action": "关注项：排查 strategy_error"
  },
  "top_problems": [
    {
      "rank": 1,
      "name": "planning/strategy_error · 策略拆解错误",
      "trend": "连续 5 天占比 >35%",
      "affected_cases": 33,
      "score_drag": "-0.36",
      "share_pct": "38.4%",
      "impacted_dims": "需求约束",
      "chain": ["输入", "拆解错误", "偏离意图"],
      "typical_case_id": "TC-0042",
      "typical_case_desc": "自驾路线约束优先级错误"
    }
  ],
  "insight_text": "综合得分 1.703（+0.05），任务满足率回升至 95.5%。"
}
```"""
    parsed = WebdevReportClient._normalize_insight_structured(raw)
    assert parsed is not None
    assert "1.703" in parsed["insight_text"]
    assert parsed["insight_summary"]["action"].startswith("关注项")
    assert len(parsed["top_problems"]) == 1
    assert parsed["top_problems"][0]["affected_cases"] == 33


def test_strip_html_for_insight_truncates_over_max() -> None:
    html = "<html><body>" + ("字" * 80) + "</body></html>"
    stripped = WebdevReportClient._strip_html_for_insight(html, max_chars=40)
    assert stripped.endswith("<!-- truncated -->")
    assert len(stripped) < len(html)


def test_normalize_insight_output_strips_fence_and_json() -> None:
    raw = '```json\n{"insight": "完成率 **99.5%** 但综合分仅 78%。"}\n```'
    out = WebdevReportClient._normalize_insight_output(raw)
    assert out is not None
    assert "99.5%" in out
    assert "```" not in out


def test_normalize_insight_output_rejects_short() -> None:
    assert WebdevReportClient._normalize_insight_output("太短") is None
    assert WebdevReportClient._normalize_insight_output(None) is None


def test_replace_insight_in_html_updates_json_and_banner() -> None:
    html = (
        "<html><body>"
        '<div class="insight-banner">💡 旧洞察</div>'
        '<script type="application/json" id="report-data">'
        '{"insight_text":"旧洞察","final_score":1.56}</script>'
        "</body></html>"
    )
    new = "任务完成率 **99.5%** 但综合分仅 78%，规划与事实性合计贡献 74% 失分。"
    out = WebdevReportClient._replace_insight_in_html(html, new)
    assert "旧洞察" not in out
    assert "<strong>99.5%</strong>" in out
    m = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>',
        out,
        re.DOTALL,
    )
    assert m is not None
    data = json.loads(m.group(1))
    assert data["insight_text"] == new
    assert data["final_score"] == 1.56


@pytest.mark.asyncio
async def test_refine_insight_from_html_uses_injected_complete() -> None:
    async def complete(prompt: str) -> str:
        assert "very-long-case" in prompt  # Case detail must be sent to the insight model
        assert "今日摘要" in prompt
        return (
            '{"summary":{"conclusion":"完成率 **99.5%** 但综合分仅 78%，规划质量需优先提升。",'
            '"bullets":["规划/策略错误是主因"],"action":"优先排查多约束拆解"},'
            '"top_problems":[{"rank":1,"name":"planning/strategy_error",'
            '"trend":"高占比","affected_cases":3,"score_drag":"-0.3",'
            '"share_pct":"40%","impacted_dims":"需求约束",'
            '"chain":["输入","拆解错","偏意图"],'
            '"typical_case_id":"very-long-case","typical_case_desc":"示例"}],'
            '"insight_text":"完成率 **99.5%** 但综合分仅 78%，规划质量需优先提升。"}'
        )

    html = (
        "<html><body>"
        '<div class="insight-banner">💡 规则洞察</div>'
        '<script type="application/json" id="report-data">'
        '{"insight_text":"规则洞察"}</script>'
        '<script type="application/json" id="cases-data">'
        '[{"id":"very-long-case"}]</script>'
        "</body></html>"
    )
    spec: dict[str, Any] = {
        "insight_text": "规则洞察",
        "run_id": "r1",
        "case_results": [
            {
                "id": "very-long-case",
                "display_input": "去西湖",
                "grade": "POOR",
                "scores": {"fused": {"score": 1.0}},
            }
        ],
    }
    client = _client(insight_llm_complete=complete)
    out = await client._refine_insight_from_html(html, spec)
    assert spec["insight_text"].startswith("完成率")
    assert spec["insight_summary"]["action"].startswith("优先排查")
    assert len(spec["top_problems"]) == 1
    assert "<strong>99.5%</strong>" in out
    assert "规则洞察" not in out
    assert '[{"id":"very-long-case"}]' in out
    assert out.count('id="cases-data"') == 1
    data = json.loads(
        re.search(
            r'<script type="application/json" id="report-data">(.*?)</script>',
            out,
            re.DOTALL,
        ).group(1)
    )
    assert data["top_problems"][0]["name"] == "planning/strategy_error"


@pytest.mark.asyncio
async def test_refine_insight_from_html_keeps_fallback_on_empty() -> None:
    async def complete(_prompt: str) -> str:
        return "短"

    html = (
        '<html><body><div class="insight-banner">💡 规则洞察</div>'
        '<script type="application/json" id="report-data">'
        '{"insight_text":"规则洞察"}</script></body></html>'
    )
    spec: dict[str, Any] = {"insight_text": "规则洞察"}
    out = await _client(insight_llm_complete=complete)._refine_insight_from_html(
        html,
        spec,
    )
    assert spec["insight_text"] == "规则洞察"
    assert "规则洞察" in out


@pytest.mark.asyncio
async def test_refine_insight_from_html_keeps_case_section_and_payload() -> None:
    """When case_results is present, the CASE detail section and cases-data must be kept unchanged after the insight is rewritten."""

    async def complete(_prompt: str) -> str:
        return (
            '{"insight_text":"完成率 **99.5%** 但综合分仅 78%，规划质量需优先提升。",'
            '"summary":{"conclusion":"完成率 **99.5%** 但综合分仅 78%。",'
            '"bullets":["规划优先"],"action":"排查策略"},"top_problems":[]}'
        )

    html = (
        "<html><body>"
        '<div class="insight-banner">💡 规则洞察</div>'
        "<section>效果指标</section><section>评测维度</section>"
        "<section>CASE 明细</section>"
        '<script type="application/json" id="report-data">'
        '{"insight_text":"规则洞察","final_score":1.56,"total_samples":1}</script>'
        '<script type="application/json" id="cases-data">'
        '[{"case_id":"abc123xyz","display_input":"去西湖"}]</script>'
        "</body></html>"
    )
    spec: dict[str, Any] = {
        "insight_text": "规则洞察",
        "case_results": [{"case_id": "abc123xyz", "display_input": "去西湖"}],
    }
    out = await _client(insight_llm_complete=complete)._refine_insight_from_html(
        html,
        spec,
    )
    assert "<strong>99.5%</strong>" in out
    assert "CASE 明细" in out
    assert '[{"case_id":"abc123xyz","display_input":"去西湖"}]' in out
    assert spec.get("insight_summary", {}).get("action") == "排查策略"


@pytest.mark.asyncio
async def test_refine_insight_from_html_skips_without_banner_or_text() -> None:
    html = "<html><body><section>效果指标</section></body></html>"
    spec: dict[str, Any] = {"run_id": "r1"}
    out = await _client()._refine_insight_from_html(html, spec)
    assert out == html


@pytest.mark.asyncio
async def test_post_insight_chat_reads_openai_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "  洞察正文来自网关。  "}}]}

    class _Http:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout
            self.posts: list[dict[str, Any]] = []

        async def __aenter__(self) -> "_Http":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            json: dict[str, Any],
            headers: dict[str, str],
        ) -> _Resp:
            self.posts.append({"url": url, "json": json, "headers": headers})
            return _Resp()

    created: list[_Http] = []

    def _factory(timeout: float) -> _Http:
        client = _Http(timeout)
        created.append(client)
        return client

    monkeypatch.setattr("benchmark.utils.http_transport.AsyncClient", _factory)
    text = await WebdevReportClient._post_insight_chat(
        _InsightChatRequest(
            base_url="https://ai-llm-gateway.amap.com/v1",
            api_key="test-key",
            model="gpt-5.6-sol",
            prompt="hello",
            timeout_s=12.0,
        ),
    )
    assert text == "洞察正文来自网关。"
    assert created[0].timeout == 12.0
    sent = created[0].posts[0]
    assert sent["url"] == "https://ai-llm-gateway.amap.com/v1/chat/completions"
    assert sent["json"]["model"] == "gpt-5.6-sol"
    assert sent["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_generate_report_invokes_insight_refine_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call the insight rewrite after HTML validation passes and before upload."""
    seen: list[str] = []

    async def fake_refine(
        self: WebdevReportClient,
        html: str,
        spec: dict[str, Any],
    ) -> str:
        seen.append(html)
        return html.replace("</body>", "<!--insight-refined--></body>")

    monkeypatch.setattr(WebdevReportClient, "_refine_insight_from_html", fake_refine)
    fake = _FakeHTTPClient(_a2a_response(_FULL_HTML))
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    html, url = await _client().generate_report(
        {"run_id": "r1", "final_score": 0.9, "total_samples": 1}
    )
    assert url is None
    assert seen
    assert "<!--insight-refined-->" in html
    assert len(fake.posts) == 1


# ════════════════════════════════════════════════════════════════════════════
# _find_matching_brace / _fix_window_function_hoisting / _fix_iife_global_scope
# ════════════════════════════════════════════════════════════════════════════


def test_find_matching_brace_nested_and_string_aware() -> None:
    """Nested braces plus a '}' inside a string do not affect pairing."""
    text = "{ a: { b: 1 }, s: '}' }"
    close = WebdevReportClient._find_matching_brace(text, 0)
    assert close == len(text) - 1
    assert text[close] == "}"


def test_find_matching_brace_escaped_quote_inside_string() -> None:
    """An escaped quote inside a string does not close the string early, and a later '}' is not misjudged."""
    text = '{s:"\\"}",x:1}'
    close = WebdevReportClient._find_matching_brace(text, 0)
    assert close == len(text) - 1


def test_find_matching_brace_unmatched_returns_none() -> None:
    assert WebdevReportClient._find_matching_brace("{ { }", 0) is None


def test_fix_window_function_hoisting_converts_assignment() -> None:
    html = (
        "<html><body><script>(function(){"
        "window.render=function x(){return 1;};"
        "})();</script></body></html>"
    )
    # window.render=function(...) { — both anonymous and named function assignments are captured as window.X
    html = html.replace(
        "window.render=function x(){return 1;};",
        "window.render = function(){ return 1; };",
    )
    out = WebdevReportClient._fix_window_function_hoisting(html)
    assert "function render()" in out
    assert "window.render=render;" in out
    assert "window.render = function" not in out


def test_fix_window_function_hoisting_skips_json_script_tags() -> None:
    html = (
        '<html><body><script type="application/json" id="report-data">'
        '{"code": "window.x = function(){ return 1; };"}'
        "</script></body></html>"
    )
    assert WebdevReportClient._fix_window_function_hoisting(html) == html


def test_fix_iife_global_scope_exposes_handler_referenced_functions() -> None:
    html = (
        '<html><body><button onclick="openDetail(1)">x</button>'
        "<script>(function(){"
        "function openDetail(id){ return id; }"
        "})();</script></body></html>"
    )
    out = WebdevReportClient._fix_iife_global_scope(html)
    assert "window.openDetail=openDetail;" in out
    # The assignment is inserted before the IIFE closes
    assert out.index("window.openDetail=openDetail;") < out.index("})();")


def test_fix_iife_global_scope_noop_without_inline_handlers() -> None:
    html = "<html><body><script>(function(){function f(){}})();</script></body></html>"
    assert WebdevReportClient._fix_iife_global_scope(html) == html


def test_fix_iife_global_scope_noop_when_function_not_defined() -> None:
    """A function referenced by an inline handler is not defined in the document → do not blindly add a window assignment."""
    html = (
        '<html><body><button onclick="extLib.run()">x</button>'
        "<script>(function(){ var a = 1; })();</script></body></html>"
    )
    assert WebdevReportClient._fix_iife_global_scope(html) == html


def test_fix_render_table_row_bind_rewrites_outer_loop_index() -> None:
    """The rows[idx].addEventListener IIFE argument i → ri (ieval_external-20260916)."""
    html = (
        "<script>tbody.innerHTML=h;\n"
        "var rows=tbody.querySelectorAll('tr');\n"
        "for(var ri=0;ri<rows.length;ri++){\n"
        "  (function(idx){\n"
        "    rows[idx].addEventListener('click',function(e){\n"
        "      if(e.target.closest('.hover-overlay'))return;\n"
        "      var cc=page[idx];\n"
        "      if(cc)openDetail(cc.case_id);\n"
        "    });\n"
        "  })(i);\n"
        "}\n</script>"
    )
    out = WebdevReportClient._fix_render_table_row_bind(html)
    assert "})(ri);" in out
    assert "})(i);" not in out
    assert WebdevReportClient._fix_render_table_row_bind(out) == out


def test_fix_render_table_row_bind_noop_without_pattern() -> None:
    html = "<script>function renderTable(){ tbody.innerHTML=h; }</script>"
    assert WebdevReportClient._fix_render_table_row_bind(html) == html


def test_inject_query_ux_inserts_before_body_close_and_idempotent() -> None:
    html = "<html><body><table></table></body></html>"
    out = WebdevReportClient._inject_query_ux(html)
    assert 'id="query-ux-patch"' in out
    assert out.index('id="query-ux-patch"') < out.index("</body>")
    # Idempotent: what is already injected is not inserted again
    assert WebdevReportClient._inject_query_ux(out) == out


def test_query_ux_patch_fulltext_holders_cover_hover_naming() -> None:
    """The fullTextOf full-text container selector must also cover the .query-hover name variant.

    search_eval-20260909 regression: the LLM template named the full-text hover layer
    .query-hover / .detail-query-hover (rather than .query-overlay), and when the holders
    selector did not cover it the fallback kids textContent concatenated "the visible truncated text + the hidden full text"
    into a doubled query (the visible text repeated twice).
    """
    # A JS selector may be split across lines (line-width limit); restore it to a complete string before matching
    patch = re.sub(r"'\s*\+\s*\n\s*'", "", webdev_client._QUERY_UX_PATCH)
    m = re.search(r"querySelectorAll\('([^']*\.query-overlay[^']*)'\)", patch)
    assert m, "fullTextOf holders 选择器未找到"
    selectors = m.group(1)
    assert ".query-hover" in selectors
    assert ".detail-query-hover" in selectors


def test_query_ux_patch_query_click_dropdown_and_hides_user_id() -> None:
    """Clicking query expands a rich detail dropdown; the detail modal drops the user id / the full input and output."""
    patch = webdev_client._QUERY_UX_PATCH
    assert "query-clip query-toggle" in patch
    assert "function ensureTraceToggle" in patch
    assert "trace-detail-row" in patch
    assert "基本信息" in patch
    assert "各维度评分" in patch
    assert "用例归因" in patch
    assert "综合得分" in patch
    assert "用户id" in patch
    assert "lab!=='输入'&&lab!=='输出'" in patch
    assert "document.__queryTraceBind" in patch
    assert "buildBasicInfoHtml" in patch or "__richDetail" in patch


# ════════════════════════════════════════════════════════════════════════════
# _inject_dim_scores_layout
# ════════════════════════════════════════════════════════════════════════════


def test_inject_dim_scores_layout_inserts_before_body_close_and_idempotent() -> None:
    html = "<html><body><table></table></body></html>"
    out = WebdevReportClient._inject_dim_scores_layout(html)
    assert 'id="dim-scores-layout-patch"' in out
    assert out.index('id="dim-scores-layout-patch"') < out.index("</body>")
    # Idempotent: what is already injected is not inserted again
    assert WebdevReportClient._inject_dim_scores_layout(out) == out


def test_inject_dim_scores_layout_appends_without_body_close() -> None:
    html = "<html><body><table></table>"
    out = WebdevReportClient._inject_dim_scores_layout(html)
    assert 'id="dim-scores-layout-patch"' in out


# ════════════════════════════════════════════════════════════════════════════
# _inject_trace_link_patch
# ════════════════════════════════════════════════════════════════════════════


def test_inject_trace_link_patch_inserts_before_body_close_and_idempotent() -> None:
    html = "<html><body><table></table></body></html>"
    out = WebdevReportClient._inject_trace_link_patch(html)
    assert 'id="trace-link-patch"' in out
    assert out.index('id="trace-link-patch"') < out.index("</body>")
    # Idempotent: what is already injected is not inserted again
    assert WebdevReportClient._inject_trace_link_patch(out) == out


def test_inject_trace_link_patch_appends_without_body_close() -> None:
    html = "<html><body><table></table>"
    out = WebdevReportClient._inject_trace_link_patch(html)
    assert 'id="trace-link-patch"' in out


def test_trace_link_patch_builds_links_from_cases_data() -> None:
    """The patch injects a real <a> row by row from cases-data (the contract that fixes the q3 rejection incident).

    The q3 incident: the prompt asked the LLM to build a Langfuse external link with the DOM API and kept discussing
    the platform red line, which hit webdev's semantic external-link red line and rejected the whole ticket. After the fix the LLM only produces
    a plain-text slot column, and the real link is injected by this patch — it must match the case by the text of the "case id" column
    (trying all three of trace_id / display_case_id / case_id), build the link with the DOM API
    (target=_blank + rel=noopener), and wrap renderTable so paging is covered.
    """
    patch = webdev_client._TRACE_LINK_PATCH
    for kw in (
        "cases-data",
        "trace_id",
        "display_case_id",
        "case_id",
        "trace_url",
        "createElement('a')",
        "setAttribute('href',url)",
        "setAttribute('target','_blank')",
        "setAttribute('rel','noopener')",
        "Trace ↗",
        "renderTable",
        "用例ID",
        "Trace链接",
        "a.trace-link",
        "window.filteredCases",
        "filtered[pageOffset+r]",
        "function findCaseTable",
        "getElementById('case-tbody')",
        "table.case-tbl",
        "var table=findCaseTable()",
        "slice(0,3)===m[1]",
    ):
        assert kw in patch, kw


def test_build_template_prompt_case_table_must_have_data_table_id() -> None:
    """The prompt must force the Case detail table to carry id=data-table (ieval_external-20260916)."""
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert 'id="data-table"' in prompt
    assert 'class="case-tbl"' in prompt


def test_build_template_prompt_dim_scores_vertical_layout() -> None:
    """The prompt must require the "per-dimension scores" column to give each dimension its own line (stacked vertically)."""
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "每个 criterion 独占一行" in prompt
    assert "纵向堆叠" in prompt


def test_build_template_prompt_radar_full_score_label() -> None:
    """The prompt must require radar-chart axis labels to carry a full-score suffix, and must not draw percentage tick labels."""
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "加满分后缀" in prompt
    assert "不画任何刻度文字" in prompt


def test_build_template_prompt_radar_legend_toggle_no_full_score_text() -> None:
    """The radar must let the legend be clicked to filter series, and the chart must not contain the words "full score"."""
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "可点选图例" in prompt
    assert "只 stroke 当前选中系列" in prompt
    assert "setLineDash([8,4])" in prompt
    assert "禁止**在图上绘制" in prompt
    assert "「满分」字样" in prompt
    # Old: a "full score" annotation beside the chart must no longer be required
    assert "旁侧 10px 「满分」字样" not in prompt
    assert "#16a34a（radius 5）+ 旁侧" not in prompt
    assert "维度列按内容收窄" in prompt
    assert "padding-right:3ch" in prompt


def test_build_template_prompt_radar_per_dim_normalize() -> None:
    """The prompt must require normalization by each dimension's full score: the full-score polyline reaches the outermost ring (plan A).

    This replaces the old absolute-score positioning: v = mean / full_score, and mean/2 is forbidden because it pins D0 at half the radius.
    """
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "按维满分归一化" in prompt
    assert "v = clamp(mean / full_score, 0, 1)" in prompt
    assert "折线顶点到达最外圈" in prompt
    assert "禁止再用 mean/2" in prompt
    assert "mean/full_score≥0.95" in prompt
    # The old absolute-score convention must not remain
    assert "绝对分定位，禁止归一化" not in prompt
    assert "折线顶点只到 1 分位置" not in prompt
    assert "满分帽" not in prompt


def test_build_template_prompt_radar_flat_means_source() -> None:
    """Radar data-source contract: today and yesterday are both a flat {cid: number} map, and reading criterion_metrics is forbidden.

    search_eval-20260914-q4 production incident: the LLM treated criterion_metrics[id] (a nested
    stats object) as a number and divided it directly → NaN → a blank radar with no console error. The prompt must
    point the radar data source at the two isomorphic flat maps today_criterion_means / yesterday_criterion_means.
    """
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "today_criterion_means[id]" in prompt
    assert "yesterday_criterion_means[id]" in prompt
    assert "扁平 map" in prompt
    assert "雷达图「今日」数据集唯一数据源" in prompt
    # The old contract (taking the radar value from the nested criterion_metrics) must not remain
    assert "今日值 = criterion_metrics[id].mean" not in prompt


def test_build_template_prompt_conditional_section_flags() -> None:
    """The prompt carries conditional-section has_* flags; enhanced sections use the new field names, and the old field names are retired."""
    spec = {
        "benchmark_name": "b",
        "criteria": [],
        "case_results": [],
        "grade_distribution": {"EXCELLENT": 1},
        "insight_text": "洞察",
        "score_loss_waterfall": {"start": 2.0, "end": 1.5, "steps": []},
        "criterion_deltas": [
            {"id": "d1", "name": "意图", "today": 1.0, "yesterday": 0.9, "delta": 0.1}
        ],
        "root_cause_dim_matrix": {"rows": [], "cols": [], "cells": {}},
    }
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "has_grade_distribution: True" in prompt
    assert "has_insight_text: True" in prompt
    assert "has_score_loss_waterfall: True" in prompt
    assert "has_criterion_deltas: True" in prompt
    assert "has_root_cause_dim_matrix: True" in prompt
    assert "has_* 标志是条件分节的唯一判断依据" in prompt
    # A retired old field appears only in the "do not render" note, and must not be a has_* render flag
    assert "has_score_loss:" not in prompt
    assert "禁止**渲染废止字段 score_loss_attribution" in prompt

    empty_prompt = WebdevReportClient._build_template_prompt(
        {"benchmark_name": "b", "criteria": [], "case_results": []}
    )
    assert "has_insight_text: False" in empty_prompt
    assert "has_score_loss_waterfall: False" in empty_prompt


def test_build_template_prompt_no_delta_rows() -> None:
    """The prompt must forbid the KPI card from rendering period-over-period information (a delta row / a significance label).

    2026-09 product decision: the report page no longer shows a KPI-card "vs yesterday" comparison.
    (The dimension comparison is a separate section, criterion_deltas, and is unrelated to the KPI card.)
    """
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "KPI 卡禁止渲染环比信息" in prompt
    assert "不渲染任何 delta/环比行" in prompt
    # The old rendering instruction (the delta-row calculation rule) must not remain
    assert "效果指标 delta 行" not in prompt
    assert "序列最后两点" not in prompt


def test_build_template_prompt_enhance_sections() -> None:
    """Enhanced sections are written into the prompt; a retired field name appears only as a do-not-render note."""
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "score_loss_waterfall" in prompt
    assert "root_cause_dim_matrix" in prompt
    assert "criterion_deltas" in prompt
    assert "分数损失与维度环比" in prompt
    assert "桌面端同一行左右两列且等宽" in prompt
    assert "增强分析计算说明（必须紧贴对应栏底部渲染）" in prompt
    assert "含“未归因”" in prompt
    assert "今日同维度有效样本均分−昨日同 benchmark" in prompt
    assert "（维度满分−得分）均值" in prompt
    # Do not list an old field as a report-data key that should be rendered
    assert "- score_loss_attribution:" not in prompt
    assert "- score_loss_matrix:" not in prompt


def test_build_template_prompt_radar_tooltip_inside_box() -> None:
    """The prompt must require the tooltip to be written inside radar-box (the insertBefore mount constraint).

    ieval_external-20260915: the LLM wrote radar-tooltip as a sibling of radar-box,
    and JS box.insertBefore(canvas, tooltip) threw NotFoundError → a blank radar chart.
    """
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "tooltip div 必须写在 radar-box" in prompt
    assert "insertBefore(canvas, tooltip)" in prompt


def test_build_template_prompt_init_steps_try_catch() -> None:
    """The prompt must require every init step to be isolated in its own try/catch (one step failing does not abort the rest)."""
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "每步调用必须用独立 try/catch 包裹" in prompt


# ════════════════════════════════════════════════════════════════════════════
# _fix_chart_tooltip_nesting / _isolate_init_steps
# ════════════════════════════════════════════════════════════════════════════


def test_fix_chart_tooltip_nesting_moves_sibling_tooltip_into_box() -> None:
    """Move a sibling tooltip into the box — fixes insertBefore failing to mount a node that is not a child."""
    html = (
        '<div class="card"><div class="chart-box" id="radar-box"></div>'
        '<div class="radar-legend" id="radar-legend"></div>'
        '<div class="chart-tooltip" id="radar-tooltip"></div></div>'
    )
    out = WebdevReportClient._fix_chart_tooltip_nesting(html)
    box_open_end = out.index(">", out.index('id="radar-box"')) + 1
    box_close = out.index("</div>", box_open_end)
    tip_pos = out.index('id="radar-tooltip"')
    assert box_open_end <= tip_pos < box_close
    # The legend stays where it is
    assert 'id="radar-legend"' in out
    # Idempotent
    assert WebdevReportClient._fix_chart_tooltip_nesting(out) == out


def test_fix_chart_tooltip_nesting_leaves_nested_tooltip_untouched() -> None:
    """The tooltip is already inside the box (the normal waterfall/trend structure) → leave it."""
    html = (
        '<div class="chart-box" id="trend-box">'
        '<div class="chart-tooltip" id="trend-tooltip"></div></div>'
    )
    assert WebdevReportClient._fix_chart_tooltip_nesting(html) == html


def test_fix_chart_tooltip_nesting_ignores_box_without_tooltip() -> None:
    """There is no same-prefix tooltip after the box → leave it."""
    html = '<div class="chart-box" id="waterfall-box"></div>'
    assert WebdevReportClient._fix_chart_tooltip_nesting(html) == html


def test_fix_chart_tooltip_nesting_matches_root_prefix_variant() -> None:
    """The radar-chart-box ↔ radar-tooltip name variant (search_eval-20260915).

    When the box id is radar-chart-box but the tooltip id is radar-tooltip (one word-root shorter),
    the old strict {name}-tooltip pairing fails; once pairing is relaxed it should be moved into the box correctly.
    """
    html = (
        '<div class="radar-box" id="radar-box">'
        '<div class="chart-box" id="radar-chart-box"></div>'
        '<div class="chart-tooltip" id="radar-tooltip"></div>'
        "</div>"
    )
    out = WebdevReportClient._fix_chart_tooltip_nesting(html)
    box_open_end = out.index(">", out.index('id="radar-chart-box"')) + 1
    box_close = out.index("</div>", box_open_end)
    tip_pos = out.index('id="radar-tooltip"')
    assert box_open_end <= tip_pos < box_close
    # Idempotent
    assert WebdevReportClient._fix_chart_tooltip_nesting(out) == out


# ════════════════════════════════════════════════════════════════════════════
# _inject_safe_insert_before_patch
# ════════════════════════════════════════════════════════════════════════════


def test_inject_safe_insert_before_patch_injected_and_idempotent() -> None:
    """The insertBefore fallback patch is injected before </body>, and is idempotent.

    ieval_external-20260915 / search_eval-20260915: the LLM wrote the tooltip as
    a sibling, or called insertBefore after createElement before the node was in the DOM, so mounting the canvas failed and
    the radar was blank. Wrapping insertBefore at runtime and degrading to append is the general fallback.
    """
    html = "<html><body><div></div></body></html>"
    out = WebdevReportClient._inject_safe_insert_before_patch(html)
    assert 'id="safe-insert-before-patch"' in out
    assert "Element.prototype.insertBefore" in out
    assert "refNode.parentNode!==this" in out
    # Idempotent
    assert WebdevReportClient._inject_safe_insert_before_patch(out) == out


def test_inject_safe_insert_before_patch_appends_when_no_body_close() -> None:
    """When there is no </body>, append at the end."""
    html = "<html><div></div>"
    out = WebdevReportClient._inject_safe_insert_before_patch(html)
    assert 'id="safe-insert-before-patch"' in out


def test_build_template_prompt_insert_before_ref_must_be_child() -> None:
    """The prompt must state that the insertBefore reference node is a child of the mount container itself."""
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "参照节点必须是该挂载容器自身的子元素" in prompt


def test_build_template_prompt_radar_series_closed_loop() -> None:
    """The prompt must require the radar data-series polygon to close (the last vertex connects back to the start).

    search_eval-20260915: after lineTo per dimension, the LLM's drawSeries was missing closePath,
    so the radar's last edge had a gap and did not close (the data was fine, there was no error, and it was a purely visual gap).
    """
    spec = {"benchmark_name": "b", "criteria": [], "case_results": []}
    prompt = WebdevReportClient._build_template_prompt(spec)
    assert "数据系列多边形必须闭环" in prompt
    assert "closePath" in prompt


def test_isolate_init_steps_wraps_top_level_calls() -> None:
    """Wrap each top-level render call in init with try/catch — one step throwing does not abort the later sections."""
    html = (
        "<script>"
        "function init(){\n"
        "renderHeader();\n"
        "renderRadar();\n"
        "renderAttribution();\n"
        "}\n"
        "init();"
        "</script>"
    )
    out = WebdevReportClient._isolate_init_steps(html)
    assert "try{renderRadar();}catch(e){console.error(" in out
    assert "try{renderAttribution();}catch(e){console.error(" in out
    # The init() call itself is not wrapped
    assert "try{init();}" not in out
    # Idempotent
    assert WebdevReportClient._isolate_init_steps(out) == out


def test_isolate_init_steps_no_init_noop() -> None:
    html = "<script>function main(){renderRadar();}</script>"
    assert WebdevReportClient._isolate_init_steps(html) == html


# ════════════════════════════════════════════════════════════════════════════
# _fix_undefined_function_calls
# ════════════════════════════════════════════════════════════════════════════


def test_fix_undefined_calls_aliases_unique_suffix_candidate() -> None:
    """The ieval_external-20260908-q1 case: the call is renderXxx but the definition is drawXxx."""
    html = (
        "<html><body><script>"
        "function drawTrendChart(){return 1;}"
        "function init(){ renderTrendChart(); renderTable(); }"
        "function renderTable(){}"
        "init();"
        "</script></body></html>"
    )
    out = WebdevReportClient._fix_undefined_function_calls(html)
    # Inject an alias for renderTrendChart → drawTrendChart
    assert "function renderTrendChart(){try{return drawTrendChart.apply(this,arguments);}" in out
    # The alias is inserted at the top of the script block that contains the call (it takes effect before the init() call site)
    assert out.index("function renderTrendChart()") < out.index("init();")
    # An already defined renderTable does not get a stub injected
    assert "renderTable is not defined" not in out
    # Idempotent
    assert WebdevReportClient._fix_undefined_function_calls(out) == out


def test_fix_undefined_calls_stubs_when_no_candidate() -> None:
    html = "<html><body><script>function init(){ fooBar(); } init();</script></body></html>"
    out = WebdevReportClient._fix_undefined_function_calls(html)
    assert (
        "function fooBar(){console.error('[agent-eval] fooBar is not defined — stubbed');}" in out
    )
    assert WebdevReportClient._fix_undefined_function_calls(out) == out


def test_fix_undefined_calls_stubs_when_multiple_candidates() -> None:
    """When there are several candidates with the same suffix (drawX + showX), do not guess an alias; degrade to a stub so they are not mismatched."""
    html = (
        "<html><body><script>"
        "function drawChart(){} function showChart(){}"
        "function init(){ renderChart(); } init();"
        "</script></body></html>"
    )
    out = WebdevReportClient._fix_undefined_function_calls(html)
    assert "renderChart is not defined — stubbed" in out
    assert "alias renderChart->" not in out


def test_fix_undefined_calls_ignores_builtins_methods_and_typeof_guards() -> None:
    html = (
        "<html><body><script>"
        "function init(){"
        "setTimeout(function(){},1);"  # Built-in
        "obj.renderChart();"  # Method call
        "var s = String(1);"  # Built-in constructor
        "if (typeof optionalRender === 'function') optionalRender();"  # typeof guard
        "}"
        "init();"
        "</script></body></html>"
    )
    assert WebdevReportClient._fix_undefined_function_calls(html) == html


def test_fix_undefined_calls_skips_json_tags_and_no_scripts() -> None:
    html = (
        '<html><body><script type="application/json" id="cases-data">'
        '[{"a": "fooBar(1)"}]</script></body></html>'
    )
    assert WebdevReportClient._fix_undefined_function_calls(html) == html
    assert WebdevReportClient._fix_undefined_function_calls("plain text") == "plain text"


def test_fix_undefined_calls_var_arrow_defs_count_as_defined() -> None:
    """A const/let/var arrow function / function expression also counts as defined."""
    html = (
        "<html><body><script>"
        "const renderChart = () => {};"
        "let drawTable = function(){};"
        "var buildRadar = async () => {};"
        "function init(){ renderChart(); drawTable(); buildRadar(); }"
        "init();"
        "</script></body></html>"
    )
    assert WebdevReportClient._fix_undefined_function_calls(html) == html


# ════════════════════════════════════════════════════════════════════════════
# _unique_suffix_alias — prefix-contains matching (search_eval-20260910 regression:
# init calls renderTrend() but the definition is drawTrendChart(), and the exact suffix Trend≠TrendChart
# so the trend section was hidden by the stub)
# ════════════════════════════════════════════════════════════════════════════


def test_unique_suffix_alias_exact_match() -> None:
    assert (
        webdev_client._unique_suffix_alias("renderTrendChart", {"drawTrendChart", "renderKPI"})
        == "drawTrendChart"
    )


def test_unique_suffix_alias_caller_shorter_prefix_match() -> None:
    """The call-name suffix is a prefix of the definition-name suffix: renderTrend→drawTrendChart."""
    assert (
        webdev_client._unique_suffix_alias("renderTrend", {"drawTrendChart", "renderKPI"})
        == "drawTrendChart"
    )


def test_unique_suffix_alias_caller_longer_prefix_match() -> None:
    """The definition-name suffix is a prefix of the call-name suffix: renderTrendChart→drawTrend."""
    assert (
        webdev_client._unique_suffix_alias("renderTrendChart", {"drawTrend", "renderKPI"})
        == "drawTrend"
    )


def test_unique_suffix_alias_exact_preferred_over_prefix() -> None:
    """When an exact candidate exists, use it directly, and do not enter prefix-contains matching."""
    pool = {"drawTrendChart", "drawTrendChartExtended"}
    assert webdev_client._unique_suffix_alias("renderTrendChart", pool) == "drawTrendChart"


def test_unique_suffix_alias_ambiguous_prefix_returns_none() -> None:
    """Prefix-contains yields several candidates (drawTrendChart / updateTrendLegend) → refuse to bind."""
    pool = {"drawTrendChart", "updateTrendLegend", "renderKPI"}
    assert webdev_client._unique_suffix_alias("renderTrend", pool) is None


def test_unique_suffix_alias_no_match_returns_none() -> None:
    assert webdev_client._unique_suffix_alias("renderTrend", {"drawRadarChart"}) is None
    # The call name has no verb prefix (camelCase boundary) → the suffix cannot be stripped
    assert webdev_client._unique_suffix_alias("trend", {"drawTrendChart"}) is None


def test_fix_undefined_calls_aliases_prefix_matched_candidate() -> None:
    """End to end for the search_eval-20260910 case: a renderTrend() call should alias to drawTrendChart."""
    html = (
        "<html><body><script>"
        "function drawTrendChart(){return 1;}"
        "function init(){ renderTrend(); }"
        "init();"
        "</script></body></html>"
    )
    out = WebdevReportClient._fix_undefined_function_calls(html)
    assert "function renderTrend(){try{return drawTrendChart.apply(this,arguments);}" in out
    assert "renderTrend is not defined" not in out
    # The alias is inserted so it takes effect before the init() call site; idempotent
    assert out.index("function renderTrend()") < out.index("init();")
    assert WebdevReportClient._fix_undefined_function_calls(out) == out


def test_fix_undefined_calls_prefix_ambiguity_still_stubs() -> None:
    """When prefix-contains is ambiguous (drawTrendChart + showTrendPanel), still degrade to a stub, and do not guess."""
    html = (
        "<html><body><script>"
        "function drawTrendChart(){} function showTrendPanel(){}"
        "function init(){ renderTrend(); } init();"
        "</script></body></html>"
    )
    out = WebdevReportClient._fix_undefined_function_calls(html)
    assert "renderTrend is not defined — stubbed" in out
    assert "alias renderTrend->" not in out


# ════════════════════════════════════════════════════════════════════════════
# _upload_to_storage
# ════════════════════════════════════════════════════════════════════════════


class _FakeStorage:
    def __init__(
        self, uploaded: object = True, signed: object = "https://cdn.example.com/signed"
    ) -> None:
        self._uploaded = uploaded
        self._signed = signed
        self.upload_calls: list[tuple[str, bytes, str]] = []

    async def upload(self, key: str, data: bytes, *, content_type: str) -> object:
        self.upload_calls.append((key, data, content_type))
        return self._uploaded

    async def sign_url(self, key: str, *, expires_seconds: int) -> object:
        return self._signed


async def test_upload_to_storage_none_when_no_client() -> None:
    assert await _client(storage=None)._upload_to_storage({}, "<html></html>") is None


async def test_upload_to_storage_prefers_signed_url() -> None:
    storage = _FakeStorage()
    url = await _client(storage=storage)._upload_to_storage(
        {"run_id": "run-1"},
        "<html></html>",
    )
    assert url == "https://cdn.example.com/signed"
    key, data, content_type = storage.upload_calls[0]
    assert key == "run-1/report.html"
    assert content_type.startswith("text/html")


async def test_upload_to_storage_falls_back_to_upload_url_when_sign_fails() -> None:
    storage = _FakeStorage(uploaded="https://cdn.example.com/raw", signed=None)
    url = await _client(storage=storage)._upload_to_storage({"run_id": "r"}, "x")
    assert url == "https://cdn.example.com/raw"


async def test_upload_to_storage_none_when_upload_fails() -> None:
    storage = _FakeStorage(uploaded=False)
    assert await _client(storage=storage)._upload_to_storage({"run_id": "r"}, "x") is None


async def test_upload_to_storage_sanitizes_run_id_in_object_key() -> None:
    """A path separator inside run_id is replaced — an extra path level cannot be injected."""
    storage = _FakeStorage()
    await _client(storage=storage)._upload_to_storage({"run_id": "../../etc/passwd"}, "x")
    key = storage.upload_calls[0][0]
    # The key may contain only one '/' — the separator between run_id and report.html
    assert key.endswith("/report.html")
    assert key.count("/") == 1


# ════════════════════════════════════════════════════════════════════════════
# generate_report — protocol construction + error paths (AsyncClient mocks out the network)
# ════════════════════════════════════════════════════════════════════════════


class _FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.status_code = 200
        self.content = json.dumps(data).encode()

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


class _FakeHTTPClient:
    """An AsyncClient stand-in that captures the payload (async context manager)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.posts: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_FakeHTTPClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> _FakeResponse:
        self.posts.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self._data)


def _a2a_response(html: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "result": {"artifacts": [{"parts": [{"kind": "text", "text": html}]}]},
    }


_FULL_HTML = (
    '<html><head><style>body{}</style></head><body><div class="container">'
    "<header>天级跟踪日报</header>"
    "<section>效果指标</section><section>评测维度</section>"
    "<div>report</div></body></html>"
)


async def test_generate_report_builds_jsonrpc_message_send_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeHTTPClient(_a2a_response(_FULL_HTML))
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    client = _client()
    html, url = await client.generate_report(
        {"run_id": "r1", "final_score": 0.9, "total_samples": 1}
    )

    assert url is None  # storage is not configured
    assert html  # Return the post-processed HTML
    sent = fake.posts[0]
    assert sent["url"] == "http://webdev-agent.test/"
    payload = sent["json"]
    assert payload["jsonrpc"] == "2.0"
    assert payload["method"] == "message/send"
    parts = payload["params"]["message"]["parts"]
    assert parts[0]["kind"] == "text"
    assert "benchmark-report" in parts[0]["text"]


async def test_generate_report_returns_empty_on_empty_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeHTTPClient(_a2a_response(""))
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    assert await _client().generate_report({"run_id": "r1"}) == ("", None)


async def test_generate_report_rejects_non_html_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The LLM returns an error message instead of HTML → reject it, and do not upload."""
    fake = _FakeHTTPClient(_a2a_response("抱歉，我无法生成该报告"))
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    assert await _client().generate_report({"run_id": "r1"}) == ("", None)


async def test_generate_report_rejects_truncated_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated document missing </body>/</html> → reject it (uploading it to OSS would be a blank screen)."""
    fake = _FakeHTTPClient(_a2a_response("<html><head><style>body{}"))
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    assert await _client().generate_report({"run_id": "r1"}) == ("", None)


async def test_generate_report_storage_failure_does_not_lose_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upload raises → report_url=None, but the HTML is still returned."""

    class _ExplodingStorage:
        async def upload(self, *args: object, **kwargs: object) -> object:
            raise ConnectionError("oss down")

    fake = _FakeHTTPClient(_a2a_response(_FULL_HTML))
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    html, url = await _client(storage=_ExplodingStorage()).generate_report(
        {"run_id": "r1", "final_score": 0.9, "total_samples": 1}
    )
    assert html
    assert url is None


# ════════════════════════════════════════════════════════════════════════════
# _validate_js_scripts — syntax fallback for the LLM's render script (bracket balance)
# ════════════════════════════════════════════════════════════════════════════


class TestValidateJsScripts:
    """Use a real JS parser to catch a SyntaxError at parse time."""

    def test_balanced_scripts_pass(self) -> None:
        html = (
            "<html><body><script>function init(){var a={b:1};init()}"
            'function x(){console.log("a(b)")}</script></body></html>'
        )
        assert WebdevReportClient._validate_js_scripts(html) == []

    def test_extra_closing_brace_flagged(self) -> None:
        """Reproduce the production defect: one extra } after init()."""
        html = "<html><body><script>function init(){var a=1;}init()}</script></body></html>"
        problems = WebdevReportClient._validate_js_scripts(html)
        assert len(problems) == 1
        assert "SyntaxError" in problems[0]

    def test_unclosed_brace_flagged(self) -> None:
        html = "<html><body><script>function x(){var a=1;</script></body></html>"
        problems = WebdevReportClient._validate_js_scripts(html)
        assert len(problems) == 1
        assert "SyntaxError" in problems[0]

    def test_invalid_declaration_flagged(self) -> None:
        html = "<html><body><script>const = 1;</script></body></html>"
        problems = WebdevReportClient._validate_js_scripts(html)
        assert len(problems) == 1
        assert "SyntaxError" in problems[0]

    def test_unterminated_string_flagged(self) -> None:
        html = "<html><body><script>var x = 'abc;</script></body></html>"
        problems = WebdevReportClient._validate_js_scripts(html)
        assert len(problems) == 1
        assert "SyntaxError" in problems[0]

    def test_unterminated_comment_flagged(self) -> None:
        html = "<html><body><script>var x=1; /* never closes</script></body></html>"
        problems = WebdevReportClient._validate_js_scripts(html)
        assert len(problems) == 1
        assert "SyntaxError" in problems[0]

    def test_string_comment_template_brackets_ignored(self) -> None:
        html = (
            '<html><body><script>var s="}{";var c=1;// {\nvar t=`((`;/* } */</script></body></html>'
        )
        assert WebdevReportClient._validate_js_scripts(html) == []

    def test_regex_literal_brackets_ignored(self) -> None:
        """Quotes / brackets / character classes inside a regex literal are not misjudged (the production line 38 case), and division is not misjudged."""
        html = (
            "<html><body><script>var h='';h+='<td>'+caseId.replace(/'/g,\"\\\\'\")+'</td>';"
            "var r=/[{}]/g;var s=a/b;return /x{2}/.test(s)</script></body></html>"
        )
        assert WebdevReportClient._validate_js_scripts(html) == []

    def test_json_data_tags_skipped(self) -> None:
        """The report-data/cases-data data tags do not participate in the JS check (brackets inside the JSON are guaranteed separately by json.loads)."""
        html = (
            '<html><body><script type="application/json" id="report-data">{"a":{}</script>'
            '<script type="application/json" id="cases-data">[}</script>'
            "<script>var a=1;</script></body></html>"
        )
        assert WebdevReportClient._validate_js_scripts(html) == []


# ════════════════════════════════════════════════════════════════════════════
# generate_report — retry the whole thing when the LLM output is unusable
# ════════════════════════════════════════════════════════════════════════════


class _SeqFakeHTTPClient:
    """An AsyncClient stand-in that returns a different response per call (for the retry-chain test)."""

    def __init__(self, datas: list[dict[str, Any]]) -> None:
        self._datas = datas
        self.posts: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_SeqFakeHTTPClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> _FakeResponse:
        self.posts.append({"url": url, "json": json, "headers": headers})
        idx = min(len(self.posts) - 1, len(self._datas) - 1)
        return _FakeResponse(self._datas[idx])


#: Structurally complete, but the main render JS is syntactically broken (reproduces search_eval-20260911: an extra } after init())
_BAD_JS_HTML = (
    '<html><head><style>body{}</style></head><body><div class="container">report</div>'
    "<script>function init(){var a=1;}init()}</script></body></html>"
)


async def test_generate_report_retries_once_on_js_syntax_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first JS is syntactically broken → regenerate; the second is fine → return success."""
    fake = _SeqFakeHTTPClient([_a2a_response(_BAD_JS_HTML), _a2a_response(_FULL_HTML)])
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    html, url = await _client().generate_report(
        {"run_id": "r1", "final_score": 0.9, "total_samples": 1}
    )
    assert len(fake.posts) == 2
    assert "report" in html
    assert url is None


async def test_generate_report_gives_up_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both generations have broken JS → reject and do not upload, returning ("", None)."""
    fake = _SeqFakeHTTPClient([_a2a_response(_BAD_JS_HTML), _a2a_response(_BAD_JS_HTML)])
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    assert await _client().generate_report({"run_id": "r1"}) == ("", None)
    assert len(fake.posts) == 2


async def test_generate_report_retries_on_truncated_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated document also triggers regeneration (the unified "LLM output is unusable" retry semantics)."""
    fake = _SeqFakeHTTPClient(
        [_a2a_response("<html><head><style>body{}"), _a2a_response(_FULL_HTML)]
    )
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    html, _url = await _client().generate_report(
        {"run_id": "r1", "final_score": 0.9, "total_samples": 1}
    )
    assert len(fake.posts) == 2
    assert html


# ════════════════════════════════════════════════════════════════════════════
# _validate_data_sections — data-section completeness + visibility (the last gate before upload)
# ════════════════════════════════════════════════════════════════════════════


def _data_report_html(style: str = "body{}") -> str:
    """A complete data-report HTML (every section title + data tag, for the validation tests)."""
    return (
        f'<html><head><style>{style}</style></head><body><div class="container">'
        "<header>天级跟踪日报</header>"
        "<section>效果指标</section><section>档位分布</section>"
        "<section>近 7 天趋势</section><section>评测维度</section>"
        "<section>不合格归因</section><section>CASE 明细</section>"
        '<script type="application/json" id="report-data">'
        '{"final_score": 1.7, "total_samples": 2}</script>'
        '<script type="application/json" id="cases-data">'
        '[{"case_id": "a"}, {"case_id": "b"}]</script>'
        '<style id="query-ux-patch">.query-clip{}</style>'
        '<style id="report-enhance-patch">'
        "#waterfall-delta-section{display:grid}</style>"
        '<script id="report-ux-patch-js">'
        "function findCaseTable(){}"
        "window.filteredCases; filtered[pageOffset+i];"
        "removeColByName(table,'用例ID');"
        "'<th>分类</th><th>分布</th><th>个数</th><th>百分比</th>'"
        "</script>"
        '<script id="query-ux-patch-js">'
        "基本信息; ensureTraceToggle;"
        "</script>"
        '<script id="trace-link-patch">'
        "window.filteredCases; filtered[pageOffset+r]</script>"
        "</div></body></html>"
    )


_FULL_DATA_SPEC: dict[str, Any] = {
    "run_id": "r1",
    "final_score": 1.7,
    "total_samples": 2,
    "case_results": [{"case_id": "a"}, {"case_id": "b"}],
    "grade_distribution": {"EXCELLENT": 1},
    "trend_7d": {"dates": ["2026-09-14"], "final_score": [1.7]},
    "attribution_metrics": {"total_attributed": 1},
}


def test_validate_data_sections_passes_on_complete_report() -> None:
    """The data tags are complete and every section is rendered → pass."""
    assert WebdevReportClient._validate_data_sections(_data_report_html(), _FULL_DATA_SPEC) == []


def test_validate_data_sections_flags_missing_total_samples() -> None:
    """report-data is missing total_samples → reject."""
    html = _data_report_html().replace(
        '"final_score": 1.7, "total_samples": 2', '"final_score": 1.7'
    )
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("total_samples" in p for p in problems)


def test_validate_data_sections_flags_cases_length_mismatch() -> None:
    """The cases-data length does not match spec.case_results → reject (the data is missing rows)."""
    html = _data_report_html().replace('[{"case_id": "a"}, {"case_id": "b"}]', '[{"case_id": "a"}]')
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("cases-data length" in p for p in problems)


def test_validate_data_sections_flags_missing_trend_section() -> None:
    """spec has trend_7d but the HTML has no "last 7 days trend" title → reject (the LLM dropped the section)."""
    html = _data_report_html().replace("<section>近 7 天趋势</section>", "")
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("近 7 天趋势" in p for p in problems)


def test_validate_data_sections_allows_js_section_title_with_badge() -> None:
    """A badge child after the "failing attribution" title inside a JS template should count as rendered.

    ieval_external-20260915: the LLM wrote
    ``<div class="sec">failing attribution <span class="badge"…></span></div>``,
    and the old regex required the title to close immediately → a false rejection → regeneration then hit the external-link red line again.
    """
    html = (
        '<html><head><style id="report-enhance-patch">'
        "#waterfall-delta-section{display:grid}</style></head><body>"
        '<div class="container"><header>天级跟踪日报</header>'
        "<section>效果指标</section><section>档位分布</section>"
        "<section>近 7 天趋势</section><section>评测维度</section>"
        "<section>CASE 明细</section>"
        "<script>"
        'var html=\'<div class="sec">不合格归因 '
        '<span class="badge" id="attr-total-badge"></span></div>\';'
        "</script>"
        '<script type="application/json" id="report-data">'
        '{"final_score": 1.7, "total_samples": 2}</script>'
        '<script type="application/json" id="cases-data">'
        '[{"case_id": "a"}, {"case_id": "b"}]</script>'
        '<style id="query-ux-patch">.query-clip{}</style>'
        '<script id="report-ux-patch-js">'
        "function findCaseTable(){}"
        "window.filteredCases; filtered[pageOffset+i];"
        "removeColByName(table,'用例ID');"
        "'<th>分类</th><th>分布</th><th>个数</th><th>百分比</th>'"
        "</script>"
        '<script id="query-ux-patch-js">'
        "基本信息; ensureTraceToggle;"
        "</script>"
        '<script id="trace-link-patch">'
        "window.filteredCases; filtered[pageOffset+r]</script>"
        "</div></body></html>"
    )
    assert WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC) == []


def test_reject_unusable_html_flags_external_link_refusal() -> None:
    """A webdev external-link red-line rejection message takes the dedicated log and is rejected."""
    refusal = "无法完成该请求：页面中不允许嵌入用户指定的外部链接。如需跳转，请使用高德端内能力（POI 详情、路线规划、导航等）。"
    assert _client()._reject_unusable_html(refusal, "req-1") is True


def test_build_template_prompt_avoids_clickable_link_instruction() -> None:
    """The prompt must no longer say "clickable link", so the external-link red line is less likely to fire by mistake."""
    prompt = WebdevReportClient._build_template_prompt(
        {"benchmark_name": "b", "criteria": [], "case_results": []}
    )
    assert "可点击链接" not in prompt
    # Case detail no longer opens a separate "case id" column; the case id is written in the query dropdown's basic info
    assert "可点击文本" not in prompt
    assert "禁止生成「用例ID」" in prompt
    assert "rule-link-slot" in prompt
    assert "trace-link-slot" in prompt
    assert "平台红线" in prompt


def test_validate_data_sections_ignores_titles_inside_scripts() -> None:
    """A title that appears only inside a JS string is not the same as a real section having been rendered."""
    html = (
        '<html><head><style id="report-enhance-patch">'
        "#waterfall-delta-section{display:grid}</style></head><body>"
        '<script type="application/json" id="report-data">'
        '{"final_score": 1.7, "total_samples": 2}</script>'
        '<script type="application/json" id="cases-data">[{"case_id": "a"}, {"case_id": "b"}]</script>'
        '<script id="report-ux-patch-js">'
        "window.filteredCases; filtered[pageOffset+i];"
        "'<th>分类</th><th>分布</th><th>个数</th><th>百分比</th>';"
        "const labels=['效果指标','评测维度','档位分布','近 7 天趋势','不合格归因','CASE 明细'];"
        "</script>"
        '<script id="trace-link-patch">window.filteredCases; filtered[pageOffset+r]</script>'
        "</body></html>"
    )
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("效果指标" in p for p in problems)
    assert any("CASE 明细" in p for p in problems)


def test_validate_data_sections_flags_hidden_data_section() -> None:
    """Production reproduction (ieval_external-20260914): a stylesheet display:none hits a data
    section (JS style.display='' clearing an inline style cannot override a stylesheet rule) → reject."""
    html = _data_report_html(style="#sec-cases{display:none}.tip{display:none}").replace(
        "<section>CASE 明细</section>", '<section id="sec-cases">CASE 明细</section>'
    )
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("data section hidden" in p and "sec-cases" in p for p in problems)


def test_validate_data_sections_flags_class_and_visibility_hidden_section() -> None:
    html = _data_report_html(style=".hidden-sec{visibility:hidden}").replace(
        "<section>CASE 明细</section>",
        '<section class="hidden-sec">CASE 明细</section>',
    )
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("hidden-sec" in p for p in problems)


def test_validate_data_sections_does_not_treat_hidden_child_as_hidden_section() -> None:
    """#section .tooltip hides a child element, which is not the same as hiding the whole section."""
    html = _data_report_html(style="#sec-cases .tooltip{display:none}").replace(
        "<section>CASE 明细</section>",
        '<section id="sec-cases">CASE 明细<span class="tooltip">tip</span></section>',
    )
    assert WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC) == []


def test_validate_data_sections_ignores_hidden_non_section() -> None:
    """display:none hits a non-data section (a tooltip and similar) → do not false-report it."""
    html = _data_report_html(style=".chart-tip{display:none}")
    assert WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC) == []


def test_validate_data_sections_allows_missing_conditional_without_data() -> None:
    """When spec has no trend_7d/attribution_metrics/grade_distribution, the corresponding section may be omitted."""
    spec = {
        k: v
        for k, v in _FULL_DATA_SPEC.items()
        if k not in ("trend_7d", "attribution_metrics", "grade_distribution")
    }
    html = (
        _data_report_html()
        .replace("<section>档位分布</section>", "")
        .replace("<section>近 7 天趋势</section>", "")
        .replace("<section>不合格归因</section>", "")
    )
    assert WebdevReportClient._validate_data_sections(html, spec) == []


def test_validate_data_sections_allows_empty_cases_without_tag() -> None:
    """When case_results is empty, the cases-data tag may be omitted (_fix_case_data does not inject it)."""
    spec = {**_FULL_DATA_SPEC, "case_results": []}
    html = (
        _data_report_html()
        .replace(
            '<script type="application/json" id="cases-data">'
            '[{"case_id": "a"}, {"case_id": "b"}]</script>',
            "",
        )
        .replace("<section>CASE 明细</section>", "")
    )
    assert WebdevReportClient._validate_data_sections(html, spec) == []


def test_validate_data_sections_tolerates_spaced_section_titles() -> None:
    """Whitespace the LLM is free to put in a title ("root cause × dimension" / "last 7 days trend") is not a false rejection.

    ieval_external-20260914-q4: the validator used to match the spaceless string "root cause×dimension",
    so a faithfully rendered title with spaces, "root cause × dimension loss matrix", was misjudged as a dropped section and triggered regeneration.
    """
    spec = {**_FULL_DATA_SPEC, "score_loss_matrix": {"rc": {"d1": 0.5}}}
    html = _data_report_html().replace(
        "<section>近 7 天趋势</section>",
        "<section>近7天趋势</section><section>根因 × 维度 失分矩阵</section>",
    )
    assert WebdevReportClient._validate_data_sections(html, spec) == []


def test_validate_data_sections_ignores_score_loss_fields() -> None:
    """When spec contains the retired field score_loss_matrix, the corresponding section is no longer required."""
    spec = {**_FULL_DATA_SPEC, "score_loss_matrix": {"rc": {"d1": 0.5}}}
    assert WebdevReportClient._validate_data_sections(_data_report_html(), spec) == []


def test_validate_data_sections_flags_trace_patch_without_row_fallback() -> None:
    """When trace data exists, the patch must be able to backfill from filteredCases even with no action column."""
    spec = {
        **_FULL_DATA_SPEC,
        "case_results": [
            {
                "case_id": "a",
                "trace_id": "tid-a",
                "trace_url": "https://example.test/tid-a",
            },
            {
                "case_id": "b",
                "trace_id": "tid-b",
                "trace_url": "https://example.test/tid-b",
            },
        ],
    }
    html = _data_report_html().replace("filtered[pageOffset+i]", "caseIdFromRow(row)")
    problems = WebdevReportClient._validate_data_sections(html, spec)
    assert any("report-ux-patch-js lacks filteredCases row fallback" in p for p in problems)


def test_validate_data_sections_flags_missing_query_ux_patch() -> None:
    """When there are cases, query-ux-patch must be injected (hover for query/output/case id)."""
    html = _data_report_html().replace('<style id="query-ux-patch">.query-clip{}</style>', "")
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("missing query-ux-patch" in p for p in problems)


def test_validate_data_sections_flags_missing_case_id_removal() -> None:
    html = _data_report_html().replace("removeColByName(table,'用例ID');", "")
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("lacks 用例ID column removal" in p for p in problems)


def test_validate_data_sections_flags_missing_find_case_table() -> None:
    html = _data_report_html().replace("function findCaseTable(){}", "")
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("lacks findCaseTable fallback" in p for p in problems)


def test_validate_data_sections_flags_residual_row_bind_outer_i() -> None:
    """A failed rewrite that leaves rows[idx].addEventListener…})(i) → reject."""
    bad_bind = (
        "tbody.innerHTML=h;\n"
        "var rows=tbody.querySelectorAll('tr');\n"
        "for(var ri=0;ri<rows.length;ri++){\n"
        "  (function(idx){\n"
        "    rows[idx].addEventListener('click',function(e){\n"
        "      if(e.target.closest('.hover-overlay'))return;\n"
        "      var cc=page[idx];\n"
        "      if(cc)openDetail(cc.case_id);\n"
        "    });\n"
        "  })(i);\n"
        "}\n"
    )
    html = _data_report_html() + f"<script>{bad_bind}</script>"
    problems = WebdevReportClient._validate_data_sections(html, _FULL_DATA_SPEC)
    assert any("outer loop index i" in p for p in problems)


def test_validate_data_sections_flags_missing_two_column_layout_patch() -> None:
    spec = {
        **_FULL_DATA_SPEC,
        "score_loss_waterfall": {"steps": [{"label": "x", "loss": 0.1}]},
        "criterion_deltas": [{"id": "d1", "name": "D1", "today": 1.0, "yesterday": 0.9}],
    }
    html = _data_report_html().replace("#waterfall-delta-section{display:grid}", "")
    problems = WebdevReportClient._validate_data_sections(html, spec)
    assert any("two-column grid" in p for p in problems)


async def test_generate_report_rejects_hidden_data_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the LLM returns a report that hides a data section → reject it and trigger regeneration."""
    bad_html = _data_report_html(style="#sec-cases{display:none}").replace(
        "<section>CASE 明细</section>", '<section id="sec-cases">CASE 明细</section>'
    )
    fake = _SeqFakeHTTPClient([_a2a_response(bad_html), _a2a_response(bad_html)])
    monkeypatch.setattr(
        "benchmark.reporting.framework.webdev_client.AsyncClient",
        lambda timeout: fake,
    )
    assert await _client().generate_report(_FULL_DATA_SPEC) == ("", None)
    assert len(fake.posts) == 2  # Both attempts are rejected
