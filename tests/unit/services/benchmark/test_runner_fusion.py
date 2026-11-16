"""Tests for PR4: ScoreFuser, AggregationCalculator, WebdevReportClient, BenchmarkRunner helpers."""

import json
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.evaluators.base import EvaluatorResult
from benchmark.runs.run_helpers import percentile_cont
from benchmark.reporting.framework import webdev_client as webdev_client_mod
from benchmark.reporting.framework.aggregation_calculator import (
    AggregationCalculator,
    EvaluatorResultEntry,
)
from benchmark.reporting.framework.fusion import EvalFailure, ScoreFuser, is_failure
from benchmark.reporting.framework.schema import (
    AggregationConfig,
    Criterion,
)
from benchmark.reporting.framework.webdev_client import WebdevReportClient
from benchmark.attribution.service.service import TriageOutcome
from benchmark.utils.http_transport import ReadTimeout

# ════════════════════════════════════════════════════════════════════════════
# Test fixtures
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rubric() -> list[Criterion]:
    return [
        Criterion(
            id="accuracy",
            name="Accuracy",
            description="Factual accuracy",
            weight=0.6,
            evaluator_ref="exact_match",
        ),
        Criterion(
            id="helpfulness",
            name="Helpfulness",
            description="Response quality",
            weight=0.4,
            evaluator_ref="llm_judge",
        ),
    ]


@pytest.fixture
def agg_weighted() -> AggregationConfig:
    return AggregationConfig(method="weighted_avg", pass_threshold=0.8)


@pytest.fixture
def agg_all_pass() -> AggregationConfig:
    return AggregationConfig(method="all_pass", pass_threshold=0.8)


@pytest.fixture
def agg_majority() -> AggregationConfig:
    return AggregationConfig(method="majority_vote", pass_threshold=0.8)


def _result(score: float) -> EvaluatorResult:
    return EvaluatorResult(score=score, reason="test")


# ════════════════════════════════════════════════════════════════════════════
# ScoreFuser tests
# ════════════════════════════════════════════════════════════════════════════


class TestEvalFailure:
    def test_repr(self):
        err = ValueError("timeout")
        f = EvalFailure(err)
        assert "ValueError" in repr(f)
        assert "timeout" in repr(f)

    def test_is_failure(self):
        f = EvalFailure(RuntimeError("x"))
        assert is_failure(f) is True
        assert is_failure(_result(0.5)) is False
        assert is_failure(None) is False


class TestScoreFuserWeightedAvg:
    def test_basic_weighted(self, rubric, agg_weighted):
        scores = {
            "accuracy": _result(0.9),
            "helpfulness": _result(0.7),
        }
        result = ScoreFuser.fuse(scores, rubric, agg_weighted)
        expected = 0.9 * 0.6 + 0.7 * 0.4
        assert pytest.approx(result) == expected

    def test_failure_counts_as_zero(self, rubric, agg_weighted):
        scores = {
            "accuracy": EvalFailure(ValueError("timeout")),
            "helpfulness": _result(1.0),
        }
        result = ScoreFuser.fuse(scores, rubric, agg_weighted)
        # 0.0 * 0.6 + 1.0 * 0.4 = 0.4
        assert pytest.approx(result) == 0.4

    def test_missing_criterion_counts_as_zero(self, rubric, agg_weighted):
        scores = {"accuracy": _result(1.0)}
        result = ScoreFuser.fuse(scores, rubric, agg_weighted)
        # 1.0 * 0.6 + 0.0 * 0.4 = 0.6
        assert pytest.approx(result) == 0.6

    def test_zero_total_weight(self, agg_weighted):
        rubric = [
            Criterion(id="a", name="A", description="", weight=0, evaluator_ref="e"),
        ]
        scores = {"a": _result(1.0)}
        result = ScoreFuser.fuse(scores, rubric, agg_weighted)
        assert result == 0.0


class TestScoreFuserAllPass:
    def test_all_pass(self, rubric, agg_all_pass):
        scores = {
            "accuracy": _result(0.9),
            "helpfulness": _result(0.85),
        }
        assert ScoreFuser.fuse(scores, rubric, agg_all_pass) == 1.0

    def test_one_fails(self, rubric, agg_all_pass):
        scores = {
            "accuracy": _result(0.9),
            "helpfulness": _result(0.7),
        }
        assert ScoreFuser.fuse(scores, rubric, agg_all_pass) == 0.0

    def test_eval_failure_means_fail(self, rubric, agg_all_pass):
        scores = {
            "accuracy": EvalFailure(RuntimeError("x")),
            "helpfulness": _result(1.0),
        }
        assert ScoreFuser.fuse(scores, rubric, agg_all_pass) == 0.0


class TestScoreFuserMajorityVote:
    def test_majority_pass(self, rubric, agg_majority):
        scores = {
            "accuracy": _result(0.9),
            "helpfulness": _result(0.85),
        }
        # 2/2 > 1 → pass
        assert ScoreFuser.fuse(scores, rubric, agg_majority) == 1.0

    def test_majority_fail(self, rubric, agg_majority):
        scores = {
            "accuracy": _result(0.5),
            "helpfulness": _result(0.5),
        }
        # 0/2 > 1 → false
        assert ScoreFuser.fuse(scores, rubric, agg_majority) == 0.0

    def test_tie_with_three_criteria(self, agg_majority):
        rubric = [
            Criterion(id="a", name="A", description="", weight=1, evaluator_ref="e"),
            Criterion(id="b", name="B", description="", weight=1, evaluator_ref="e"),
            Criterion(id="c", name="C", description="", weight=1, evaluator_ref="e"),
        ]
        scores = {
            "a": _result(0.9),
            "b": _result(0.5),
            "c": _result(0.5),
        }
        # 1/3 > 1.5 → false
        assert ScoreFuser.fuse(scores, rubric, agg_majority) == 0.0


class TestScoreFuserCustom:
    def test_custom_method_without_fusion_fn_ref_raises(self, rubric):
        agg = AggregationConfig(method="custom", custom_fn_ref="x:y")
        scores = {"accuracy": _result(0.5), "helpfulness": _result(0.5)}
        with pytest.raises(ValueError, match="item-level fusion requires fusion_fn_ref"):
            ScoreFuser.fuse(scores, rubric, agg)


# ════════════════════════════════════════════════════════════════════════════
# AggregationCalculator tests
# ════════════════════════════════════════════════════════════════════════════


def _entry(criterion_id: str, **overrides: Any) -> EvaluatorResultEntry:
    """Build an aggregation entry record; optional fields override defaults via kwargs.

    Supports: evaluator_ref / mean / pass_rate / sample_count /
    success_count / failed_count / attribution_stats_json /
    react_rounds_json.
    """
    fields: dict[str, Any] = {
        "evaluator_ref": "e",
        "mean": 0.8,
        "pass_rate": 0.8,
        "sample_count": 10,
        "success_count": 9,
        "failed_count": 1,
        "attribution_stats_json": None,
        "react_rounds_json": None,
    }
    fields.update(overrides)
    stats = {
        "mean": fields["mean"],
        "median": fields["mean"],
        "p95": fields["mean"],
        "min": 0.5,
        "max": 1.0,
        "std": 0.1,
        "pass_rate": fields["pass_rate"],
    }
    return EvaluatorResultEntry(
        criterion_id=criterion_id,
        evaluator_ref=fields["evaluator_ref"],
        sample_count=fields["sample_count"],
        success_count=fields["success_count"],
        failed_count=fields["failed_count"],
        criterion_stats_json=stats,
        attribution_stats_json=fields["attribution_stats_json"],
        react_rounds_json=fields["react_rounds_json"],
    )


class TestAggregationCalculator:
    @pytest.mark.asyncio
    async def test_weighted_avg(self, rubric, agg_weighted):
        entries = [
            _entry("accuracy", mean=0.9, pass_rate=0.9),
            _entry("helpfulness", mean=0.7, pass_rate=0.7),
        ]
        calc = AggregationCalculator()
        final, per_criterion, metrics = await calc.aggregate(
            agg_weighted,
            rubric,
            entries,
        )
        expected = 0.9 * 0.6 + 0.7 * 0.4
        assert pytest.approx(final) == expected
        assert "accuracy" in per_criterion
        assert "helpfulness" in per_criterion
        assert metrics.aggregation_method == "weighted_avg"
        assert metrics.total_samples == 10

    @pytest.mark.asyncio
    async def test_all_pass_pass(self, rubric, agg_all_pass):
        entries = [
            _entry("accuracy", pass_rate=0.9),
            _entry("helpfulness", pass_rate=0.85),
        ]
        calc = AggregationCalculator()
        final, _, _ = await calc.aggregate(agg_all_pass, rubric, entries)
        assert final == 1.0

    @pytest.mark.asyncio
    async def test_all_pass_fail(self, rubric, agg_all_pass):
        entries = [
            _entry("accuracy", pass_rate=0.9),
            _entry("helpfulness", pass_rate=0.7),
        ]
        calc = AggregationCalculator()
        final, _, _ = await calc.aggregate(agg_all_pass, rubric, entries)
        assert final == 0.0

    @pytest.mark.asyncio
    async def test_majority_vote(self, rubric, agg_majority):
        entries = [
            _entry("accuracy", pass_rate=0.9),
            _entry("helpfulness", pass_rate=0.85),
        ]
        calc = AggregationCalculator()
        final, _, _ = await calc.aggregate(agg_majority, rubric, entries)
        # 2/2 > 1 → pass
        assert final == 1.0

    @pytest.mark.asyncio
    async def test_missing_criterion_counts_as_fail(self, rubric, agg_all_pass):
        entries = [_entry("accuracy", pass_rate=0.9)]
        calc = AggregationCalculator()
        final, per_criterion, _ = await calc.aggregate(agg_all_pass, rubric, entries)
        # helpfulness missing → not in per_criterion → all_pass fails
        assert final == 0.0
        assert "helpfulness" not in per_criterion

    @pytest.mark.asyncio
    async def test_attribution_stats_extracted(self, rubric, agg_weighted):
        entries = [
            _entry(
                "accuracy",
                attribution_stats_json={
                    "field_distributions": {
                        "primary_issue_category": {
                            "intent_error": 6,
                            "tool_error": 4,
                        },
                    },
                    "total_attributed": 10,
                },
            ),
            _entry("helpfulness"),
        ]
        calc = AggregationCalculator()
        _, _, metrics = await calc.aggregate(agg_weighted, rubric, entries)
        assert "accuracy" in metrics.attribution_metrics
        assert (
            metrics.attribution_metrics["accuracy"].field_distributions["primary_issue_category"][
                "intent_error"
            ]
            == 6
        )
        assert metrics.attribution_metrics["accuracy"].total_attributed == 10

    @pytest.mark.asyncio
    async def test_react_rounds_extracted(self, rubric, agg_weighted):
        entries = [
            _entry(
                "accuracy",
                react_rounds_json={
                    "distribution": {"1": 5, "2": 3, "5+": 2},
                },
            ),
            _entry("helpfulness"),
        ]
        calc = AggregationCalculator()
        _, _, metrics = await calc.aggregate(agg_weighted, rubric, entries)
        assert metrics.react_rounds_summary.get(1) == 5
        assert metrics.react_rounds_summary.get(2) == 3
        assert metrics.react_rounds_summary.get(5) == 2

    @pytest.mark.asyncio
    async def test_empty_entries(self, rubric, agg_weighted):
        calc = AggregationCalculator()
        final, per_criterion, metrics = await calc.aggregate(
            agg_weighted,
            rubric,
            [],
        )
        assert final == 0.0
        assert per_criterion == {}
        assert metrics.total_samples == 0


# ════════════════════════════════════════════════════════════════════════════
# WebdevReportClient tests
# ════════════════════════════════════════════════════════════════════════════


class TestWebdevReportClient:
    @pytest.fixture(autouse=True)
    def _force_webdev_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_EVAL_FORCE_WEBDEV_REPORT", "1")

    @pytest.mark.asyncio
    async def test_generate_report_success(self):
        client = WebdevReportClient(webdev_agent_url="http://webdev:9880")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "result": {
                    "artifacts": [
                        {
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": "<!DOCTYPE html><html><head><style>body{}</style></head><body>report<section>效果指标</section><section>评测维度</section></body></html>",
                                }
                            ],
                        },
                    ],
                },
            }
        )

        with patch("benchmark.reporting.framework.webdev_client.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            html, report_url = await client.generate_report(
                {"benchmark_name": "test", "run_id": "r1", "final_score": 1.5, "total_samples": 1},
            )

        assert '<meta charset="UTF-8">' in html
        assert "<body>report" in html
        assert '"benchmark_name": "test"' in html
        assert '"run_id": "r1"' in html
        assert 'id="query-ux-patch"' in html
        assert html.index('id="query-ux-patch"') < html.index("</body>")
        assert 'id="dim-scores-layout-patch"' in html
        assert html.index('id="dim-scores-layout-patch"') < html.index("</body>")
        assert 'id="trace-link-patch"' in html
        assert html.index('id="trace-link-patch"') < html.index("</body>")
        assert report_url is None

    @pytest.mark.asyncio
    async def test_generate_report_with_storage(self):
        mock_storage = AsyncMock()
        mock_storage.upload = AsyncMock(return_value="https://cdn.example.com/r1.html")
        mock_storage.sign_url = AsyncMock(
            return_value="https://oss.example.com/r1.html?Expires=1&Signature=x",
        )

        client = WebdevReportClient(
            webdev_agent_url="http://webdev:9880",
            storage_client=mock_storage,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "result": {
                    "artifacts": [
                        {
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": "<!DOCTYPE html><html><head><style></style></head><body><section>效果指标</section><section>评测维度</section></body></html>",
                                }
                            ]
                        },
                    ],
                },
            }
        )

        with patch("benchmark.reporting.framework.webdev_client.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            _, report_url = await client.generate_report(
                {"run_id": "r1", "final_score": 1.5, "total_samples": 1},
            )

        assert report_url == "https://oss.example.com/r1.html?Expires=1&Signature=x"
        mock_storage.upload.assert_called_once()
        uploaded_key, uploaded_body = mock_storage.upload.call_args.args[:2]
        assert uploaded_key == "r1/report.html"
        uploaded_html = uploaded_body.decode("utf-8")
        assert '"run_id": "r1"' in uploaded_html
        assert 'id="query-ux-patch"' in uploaded_html
        assert 'id="dim-scores-layout-patch"' in uploaded_html
        assert 'id="trace-link-patch"' in uploaded_html
        assert mock_storage.upload.call_args.kwargs["content_type"] == ("text/html; charset=utf-8")
        mock_storage.sign_url.assert_awaited_once()
        assert mock_storage.sign_url.await_args.args[0] == "r1/report.html"
        assert mock_storage.sign_url.await_args.kwargs["expires_seconds"] == 2_592_000

    @pytest.mark.asyncio
    async def test_generate_report_storage_sign_fallback(self):
        mock_storage = AsyncMock()
        mock_storage.upload = AsyncMock(return_value="https://cdn.example.com/r1.html")
        mock_storage.sign_url = AsyncMock(return_value=None)

        client = WebdevReportClient(
            webdev_agent_url="http://webdev:9880",
            storage_client=mock_storage,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "result": {
                    "artifacts": [
                        {
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": "<!DOCTYPE html><html><head><style></style></head><body><section>效果指标</section><section>评测维度</section></body></html>",
                                }
                            ]
                        },
                    ],
                },
            }
        )

        with patch("benchmark.reporting.framework.webdev_client.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            _, report_url = await client.generate_report(
                {"run_id": "r1", "final_score": 1.5, "total_samples": 1}
            )

        assert report_url == "https://cdn.example.com/r1.html"

    @pytest.mark.asyncio
    async def test_empty_html(self):
        client = WebdevReportClient(webdev_agent_url="http://webdev:9880")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "result": {"artifacts": []},
            }
        )

        with patch("benchmark.reporting.framework.webdev_client.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            html, report_url = await client.generate_report({"run_id": "r1"})

        assert html == ""
        assert report_url is None

    def test_extract_html_from_nested(self):
        response = {
            "result": {
                "artifacts": [
                    {"parts": [{"kind": "data", "data": "ignored"}]},
                    {"parts": [{"kind": "text", "text": "found"}]},
                ],
            },
        }
        assert WebdevReportClient._extract_html(response) == "found"

    def test_extract_html_empty(self):
        assert WebdevReportClient._extract_html({}) == ""
        assert WebdevReportClient._extract_html({"result": {}}) == ""

    @pytest.mark.asyncio
    async def test_generate_report_rejects_non_html(self):
        """LLM may return error text instead of HTML; must reject and return empty."""
        client = WebdevReportClient(webdev_agent_url="http://webdev:9880")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "result": {
                    "artifacts": [
                        {"parts": [{"kind": "text", "text": "sorry, error occurred"}]},
                    ],
                },
            }
        )

        with patch("benchmark.reporting.framework.webdev_client.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            html, report_url = await client.generate_report({"run_id": "r1"})

        assert html == ""
        assert report_url is None

    def test_attach_grade_distribution_counts(self):
        """_attach_grade_distribution: count only a valid grade, and skip a non-dict row / empty value."""
        spec = {
            "case_results": [
                {"case_id": "1", "grade": "EXCELLENT"},
                {"case_id": "2", "grade": "EXCELLENT"},
                {"case_id": "3", "grade": "POOR"},
                {"case_id": "4"},
                {"case_id": "5", "grade": ""},
                "not-a-dict",
            ],
        }
        WebdevReportClient._attach_grade_distribution(spec)
        assert spec["grade_distribution"] == {"EXCELLENT": 2, "POOR": 1}

    def test_attach_grade_distribution_no_grade_keeps_absent(self):
        """No valid grade at all → do not inject an empty distribution (the whole-result stats section is omitted entirely)."""
        spec = {"case_results": [{"case_id": "1"}, {"case_id": "2"}]}
        WebdevReportClient._attach_grade_distribution(spec)
        assert "grade_distribution" not in spec

    def test_attach_grade_distribution_idempotent(self):
        """spec already has a non-empty grade_distribution → do not overwrite it."""
        spec = {
            "case_results": [{"case_id": "1", "grade": "FAIL"}],
            "grade_distribution": {"GOOD": 10},
        }
        WebdevReportClient._attach_grade_distribution(spec)
        assert spec["grade_distribution"] == {"GOOD": 10}

    @pytest.mark.asyncio
    async def test_generate_report_injects_grade_distribution(self):
        """generate_report: grade counts are injected into spec + report-data, and the prompt includes the new rendering requirements."""
        client = WebdevReportClient(webdev_agent_url="http://webdev:9880")
        spec = {
            "run_id": "r1",
            "final_score": 1.5,
            "total_samples": 2,
            "case_results": [
                {"case_id": "1", "grade": "GOOD"},
                {"case_id": "2", "grade": "FAIL"},
            ],
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "result": {
                    "artifacts": [
                        {
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": (
                                        "<html><head><style></style></head><body>"
                                        "<section>效果指标</section><section>评测维度</section>"
                                        "<section>档位分布</section><section>CASE 明细</section>"
                                        '<script type="application/json" id="cases-data">'
                                        '[{"case_id": "1"}, {"case_id": "2"}]</script>'
                                        "</body></html>"
                                    ),
                                }
                            ]
                        },
                    ],
                },
            }
        )

        with patch("benchmark.reporting.framework.webdev_client.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            html, _ = await client.generate_report(spec)

        # spec is injected with the grade distribution in place (report-data = spec minus case_results)
        assert spec["grade_distribution"] == {"GOOD": 1, "FAIL": 1}
        assert '"grade_distribution": {"GOOD": 1, "FAIL": 1}' in html
        # The prompt carries the rendering requirements for the new sections
        prompt = mock_client.post.await_args.kwargs["json"]["params"]["message"]["parts"][0]["text"]
        for keyword in ("档位分布", "case状态", "value_labels", "Trace链接"):
            assert keyword in prompt

    def test_ensure_charset_already_present(self):
        html = '<html><head><meta charset="UTF-8"><title>x</title></head></html>'
        assert WebdevReportClient._ensure_charset(html) == html

    def test_ensure_charset_injects_after_head(self):
        html = "<html><head><title>x</title></head></html>"
        result = WebdevReportClient._ensure_charset(html)
        assert '<meta charset="UTF-8">' in result
        assert result == '<html><head><meta charset="UTF-8"><title>x</title></head></html>'

    def test_ensure_charset_injects_after_html_no_head(self):
        html = "<html>content</html>"
        result = WebdevReportClient._ensure_charset(html)
        assert result == '<html><meta charset="UTF-8">content</html>'

    def test_validate_html_completeness_pass(self):
        """Complete HTML with all required tags returns empty list."""
        html = "<!DOCTYPE html><html><head><style>body{}</style></head><body>content</body></html>"
        assert WebdevReportClient._validate_html_completeness(html) == []

    def test_validate_html_completeness_fail_truncated(self):
        """Truncated HTML (only CSS, no body) returns missing tags."""
        html = "<!DOCTYPE html><html><head><style>body{color:red;"
        missing = WebdevReportClient._validate_html_completeness(html)
        # ``</style>`` has been removed from the required checklist (a legal template may have no style block)
        assert "<body" in missing
        assert "</body>" in missing
        assert "</html>" in missing

    def test_build_template_prompt_zero_external_urls(self):
        """External-link red line (production-incident regression): the prompt contains zero external URLs.

        webdev-agent's external-link-guard scans the raw LLM output, and any URL the prompt supplies
        may be copied into the output by the LLM and get the whole ticket rejected (the old alidocs /
        Chart.js CDN were both rejected for this). A future link must go through a slot + post-processing injection
        (_fill_link_slots), and must not enter the prompt.
        """
        spec = {
            "benchmark_name": "test_bm",
            "service_name": "test_svc",
            "run_id": "r1",
            "final_score": 8.5,
            "total_samples": 50,
            "criteria": [
                {"id": "acc", "name": "准确性", "weight": 0.6, "evaluator_ref": "judge1"},
            ],
            "case_results": [],
        }
        prompt = WebdevReportClient._build_template_prompt(spec)
        assert not re.findall(r"https?://\S+", prompt)
        # Charts are pure native Canvas 2D: chart libraries/CDN are forbidden, getContext is the hand-drawn entry, and
        # SVG is forbidden (the embedded viewer strips inline SVG, so the chart is blank while the data is still there)
        assert "禁止引入任何图表库" in prompt
        assert "getContext('2d')" in prompt
        assert "禁止使用 SVG" in prompt
        # The Trace link column has zero link-building instructions (the q3 rejection incident): the prompt contains no DOM link building /
        # no Langfuse-wording instruction — the real link is injected afterward by the caller via _TRACE_LINK_PATCH
        # (the same layering as _fill_link_slots). The prompt may now declare the platform red-line copy
        # (do not generate external-link elements), but it must not contain link-building instruction words.
        assert "trace-link-slot" in prompt
        for kw in ("createElement('a')", "window.open", "Langfuse"):
            assert kw not in prompt, kw

    def test_build_template_prompt_minimal(self):
        """Prompt contains structural info but not case_results data.

        Keyword assertions are grouped by topic (PLR0915: a single function is capped at 50 statements). The external-link
        red-line assertion is in ``test_build_template_prompt_zero_external_urls``.
        """
        spec = {
            "benchmark_name": "test_bm",
            "service_name": "test_svc",
            "run_id": "r1",
            "final_score": 8.5,
            "total_samples": 50,
            "criteria": [
                {"id": "acc", "name": "准确性", "weight": 0.6, "evaluator_ref": "judge1"},
            ],
            "case_results": [{"case_id": "c1", "input": "x" * 10000}],
        }
        prompt = WebdevReportClient._build_template_prompt(spec)
        # Basic structure: spec fields + the two JSON tags; case data does not enter the prompt
        for kw in ("test_bm", "test_svc", "acc", "准确性", "report-data", "cases-data"):
            assert kw in prompt, kw
        assert "x" * 10000 not in prompt
        # Column allowlist, 7 kinds: no "case id" column; the query dropdown carries basic info / dimensions / attribution
        for kw in (
            "trace_id",
            "display_case_id",
            "只允许这 7 类",
            "root_cause_display",
            "query：display_input",
            "output：display_output",
            "hover 浮层",
            "精确匹配",
            "禁止生成「用例ID」「输入」「操作」列",
            "必有值列",
            "trace_url",
            "trace-link-slot",
            "基本信息",
            "各维度评分",
            "用例归因",
        ):
            assert kw in prompt, kw
        # Filter / paging: a status dropdown + a global filter + 10 rows per page; the dimension filter has been removed
        for kw in (
            "全部状态",
            "每页 10 条",
            "禁止生成维度筛选下拉",
            "不得出现「全部维度」",
            "overflow: auto",
            "拖拽平移",
            "white-space: nowrap",
        ):
            assert kw in prompt, kw
        # Daily report, 8-section structure + the scoring-rule slot + native Canvas charts
        for kw in (
            "天级跟踪日报",
            "rule-link-slot",
            "效果指标",
            "档位分布",
            "近 7 天趋势",
            "评测维度",
            "不合格归因",
            "禁止引入任何图表库",
            "getContext('2d')",
            "devicePixelRatio",
            "trend_7d",
            "task_satisfaction_rate",
            "yesterday_criterion_means",
        ):
            assert kw in prompt, kw
        # Per-dimension full-score scale: D0/safety (veto) has a full score of 1, and the other dimensions have a full score of 2
        # (aligned with attribution_trigger_policy.DIMENSION_FULL_SCORES)
        for kw in (
            "一票否决",
            "满分按维度区分",
            "满分 1 分",
            "满分 2 分",
            "mean/1*100%",
            "mean/2*100%",
            "≥0.8",
        ):
            assert kw in prompt, kw
        # Dimension radar: drawn by hand on Canvas; normalized by each dimension's full score; the legend is clickable; the words "full score" are forbidden
        for kw in (
            "按维满分归一化",
            "v = clamp(mean / full_score, 0, 1)",
            "折线顶点到达最外圈",
            "悬停必须显示得分",
            "N 边形顶点",
            "加满分后缀",
            "可点选图例",
            "只 stroke 当前选中系列",
            "「满分」字样",
        ):
            assert kw in prompt, kw
        for kw in (
            "绝对分定位，禁止归一化",
            "v = mean / 最大满分 2",
            "满分帽",
            "旁侧 10px 「满分」字样",
        ):
            assert kw not in prompt, kw
        # Negative: deleted / forbidden content does not enter the prompt
        for kw in (
            "大小写不敏感子串",
            "pass=false",
            "word-break",
            "chart.umd.min.js",
            "alidocs",
            "整体结果统计",
            "react_rounds",
            "Trace ↗",
            "Langfuse trace 面板链接",
        ):
            assert kw not in prompt, kw
        # Canvas 2D rules (DPI scaling / setTransform / hit-testing hints / no SVG
        # notes) + per-dimension scores stacked vertically / init defense / radar normalization and other deterministic
        # rendering red lines + insights / root-cause column / mini bar chart / the platform red-line declaration and other new rules,
        # slightly longer than the SVG version
        assert len(prompt) < 19500

    def test_fill_link_slots_replaces_rule_slot(self):
        """_fill_link_slots replaces the slot with the real external link from the run config's rule_link_url."""
        html = (
            '<html><body><header><div class="meta">'
            '<span class="rule-link-slot">评分规则</span>'
            "</div></header></body></html>"
        )
        result = WebdevReportClient._fill_link_slots(
            html, "https://alidocs.dingtalk.com/i/nodes/14dA3GK8gwAG5B1mHnbDXZPG89ekBD76"
        )
        assert "rule-link-slot" not in result
        assert (
            '<a href="https://alidocs.dingtalk.com/i/nodes/14dA3GK8gwAG5B1mHnbDXZPG89ekBD76"'
            in result
        )
        assert 'target="_blank"' in result
        assert "评分规则 ↗" in result

    def test_fill_link_slots_business_url_override(self):
        """A business-custom URL: the slot is filled with the value passed in config (the business extension point)."""
        html = '<span class="rule-link-slot">评分规则</span>'
        result = WebdevReportClient._fill_link_slots(html, "https://docs.example.com/rubric-v3")
        assert '<a href="https://docs.example.com/rubric-v3"' in result

    def test_fill_link_slots_no_url_keeps_plain_slot(self):
        """config has no rule_link_url (the business did not configure it) → the slot stays plain text, and no link is rendered."""
        html = '<span class="rule-link-slot">评分规则</span>'
        assert WebdevReportClient._fill_link_slots(html) == html
        assert WebdevReportClient._fill_link_slots(html, None) == html
        assert WebdevReportClient._fill_link_slots(html, "") == html

    def test_fill_link_slots_missing_slot_is_noop(self):
        """The LLM did not generate the slot → return as-is (best-effort, and the report is not blocked)."""
        html = "<html><body><header>no slot</header></body></html>"
        assert WebdevReportClient._fill_link_slots(html, "https://docs.example.com/r") == html

    def test_fill_link_slots_tolerates_class_variant(self):
        """The slot is still hit when its class has several values / spans lines."""
        html = '<span id="x" class="badge rule-link-slot extra">\n评分规则\n</span>'
        result = WebdevReportClient._fill_link_slots(html, "https://docs.example.com/r")
        assert "rule-link-slot" not in result
        assert 'href="https://docs.example.com/r"' in result

    def test_inject_report_data_replaces_placeholder(self):
        """_inject_report_data replaces placeholder with real data."""
        html = (
            "<html><head><style></style></head><body>"
            '<script type="application/json" id="report-data">{}</script>'
            "</body></html>"
        )
        spec = {"benchmark_name": "test", "final_score": 8.5}
        result = WebdevReportClient._inject_report_data(html, spec)
        assert '"benchmark_name": "test"' in result
        assert '"final_score": 8.5' in result

    def test_inject_report_data_injects_fallback(self):
        """_inject_report_data injects tag before </body> if missing."""
        html = "<html><head><style></style></head><body>report</body></html>"
        spec = {"benchmark_name": "test"}
        result = WebdevReportClient._inject_report_data(html, spec)
        assert '<script type="application/json" id="report-data">' in result
        assert '"benchmark_name": "test"' in result
        idx_tag = result.index('<script type="application/json" id="report-data">')
        idx_body = result.index("</body>")
        assert idx_tag < idx_body

    def test_inject_report_data_excludes_case_results(self):
        """_inject_report_data does not include case_results in report-data."""
        html = (
            "<html><head><style></style></head><body>"
            '<script type="application/json" id="report-data">{}</script>'
            "</body></html>"
        )
        spec = {
            "benchmark_name": "test",
            "case_results": [{"case_id": "c1", "input": "secret"}],
        }
        result = WebdevReportClient._inject_report_data(html, spec)
        assert "secret" not in result
        assert '"benchmark_name": "test"' in result

    def test_inject_query_ux_before_body_close(self):
        """query UX patch is injected before </body> and is idempotent."""
        html = "<html><head></head><body>report</body></html>"
        result = WebdevReportClient._inject_query_ux(html)
        assert 'id="query-ux-patch"' in result
        assert "精确搜索用例ID / query" in result
        assert "query-tip-float" in result
        assert "display_input" in result
        assert result.index('id="query-ux-patch"') < result.index("</body>")
        again = WebdevReportClient._inject_query_ux(result)
        assert again.count('id="query-ux-patch"') == 1

    def test_query_ux_patch_exact_search_survives_var_naming(self):
        """The exact-search wrap must accept whatever filtered-result variable name the LLM template uses.

        Regression (ieval_external-20260826 report): the old patch assumed the variable was named
        ``filtered`` and cleared the search box before running — when the template used ``filteredCases``
        and already had exact search, both failed: the native search was disabled by the cleared value, and the compensating filter
        wrote the wrong variable name and was dropped, so the search box did nothing.
        """
        patch = webdev_client_mod._QUERY_UX_PATCH
        # Do not clear the search box: keep the input value so origApply's own search logic takes effect
        assert "box.value=''" not in patch
        # Probe several variable names: cover filtered / filteredCases / and other common names
        for name in ("filtered", "filteredCases", "filteredData"):
            assert f"'{name}'" in patch
        # Snapshot comparison probes the result variable that origApply assigns again
        assert "before[nm]!==v" in patch
        # Write the probe result back to every candidate variable, so renderTable does not read a different name
        assert "window[found[i]]=next" in patch
        # The CASES full-set fallback is kept
        assert "window.CASES||[]" in patch

    def test_query_ux_patch_dim_filter_and_fulltext_fallbacks(self):
        """Neutralize the dimension filter + extract the full text + an ellipsis-prefix fallback, three fallbacks.

        Regression (ieval_external-20260826 report):
        - the "all dimensions" dropdown is removed under the new rule — hide it and reset it rather than deleting the node,
          so template JS can still read the element safely;
        - the template query cell contains both a "truncated" element and a "full text" element, and reading
          td.textContent directly concatenates them into garbage — the hover layer shows garbage → the user
          copies the garbage into search and necessarily gets 0 results → no case can be opened for detail;
        - when the user searches truncated text ending in …/..., fall back to a display_input
          prefix match.
        """
        patch = webdev_client_mod._QUERY_UX_PATCH
        # Neutralize the dimension filter: find the "all dimensions" select, hide it, and reset it to the default option
        assert "neutralizeDimFilter" in patch
        assert "全部维度" in patch
        assert "dimSel.style.display='none'" in patch
        assert "dimSel.selectedIndex=di" in patch
        # enhanceTable runs the neutralization on every render (so a template re-render cannot restore the dropdown)
        assert "neutralizeDimFilter();" in patch
        # Full-text extraction: prefer data-full / a full-text container (.query-overlay and similar),
        # so td.textContent concatenated garbage does not pollute the hover layer
        assert "fullTextOf(td)" in patch
        assert "fullTextOf(el)" in patch
        assert ".query-overlay" in patch
        # Ellipsis-prefix fallback: when there are 0 results and the input ends in …/..., match by prefix
        assert "lastIndexOf(qp,0)===0" in patch

    def test_inject_query_ux_keeps_orig_apply_search_input(self):
        """The full patch after injection keeps the search-box value flowing through (no saved/restore hack)."""
        html = (
            "<html><head></head><body>"
            '<input class="search-box" oninput="applyFilters()">'
            "</body></html>"
        )
        result = WebdevReportClient._inject_query_ux(html)
        # The old three-step "save, clear, restore" hack has been removed
        assert "var saved=box" not in result
        assert "box.value=saved" not in result
        # wrapped carries the real input straight into origApply + idempotent exact filtering
        assert "origApply();" in result

    @pytest.mark.asyncio
    async def test_generate_report_rejects_truncated_html(self):
        """Truncated HTML (only CSS) must be rejected, not uploaded."""
        client = WebdevReportClient(webdev_agent_url="http://webdev:9880")
        truncated = "<!DOCTYPE html><html><head><style>body{color:red;}"
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "result": {
                    "artifacts": [
                        {"parts": [{"kind": "text", "text": truncated}]},
                    ],
                },
            }
        )
        with patch("benchmark.reporting.framework.webdev_client.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client
            html, report_url = await client.generate_report({"run_id": "r1"})
        assert html == ""
        assert report_url is None

    # ── _normalize_case_io tests ──────────────────────────────────────────

    def test_normalize_case_io_string_passthrough(self):
        """String values pass through unchanged."""
        cases = [{"case_id": "c1", "input": "hello", "output": "world"}]
        result = WebdevReportClient._normalize_case_io(cases)
        assert result[0]["input"] == "hello"
        assert result[0]["output"] == "world"

    def test_normalize_case_io_dict_to_json(self):
        """dict/list values are converted to pretty-printed JSON strings."""
        cases = [
            {
                "case_id": "c1",
                "input": {"query": "navigate", "lat": 39.9},
                "output": [1, 2, 3],
                "expected": {"key": "value"},
            },
        ]
        result = WebdevReportClient._normalize_case_io(cases)
        assert result[0]["input"] == json.dumps(
            {"query": "navigate", "lat": 39.9},
            ensure_ascii=False,
            indent=2,
        )
        assert result[0]["output"] == json.dumps(
            [1, 2, 3],
            ensure_ascii=False,
            indent=2,
        )
        assert result[0]["expected"] == json.dumps(
            {"key": "value"},
            ensure_ascii=False,
            indent=2,
        )

    def test_normalize_case_io_does_not_mutate_original(self):
        """The original case_results list must not be modified."""
        original_input = {"query": "test"}
        cases = [{"case_id": "c1", "input": original_input}]
        result = WebdevReportClient._normalize_case_io(cases)
        # Original dict is still a dict, not a string
        assert cases[0]["input"] is original_input
        assert isinstance(cases[0]["input"], dict)
        # Result has a string
        assert isinstance(result[0]["input"], str)

    def test_normalize_case_io_none_passthrough(self):
        """None values pass through unchanged (not converted to 'None')."""
        cases = [{"case_id": "c1", "input": None, "output": None}]
        result = WebdevReportClient._normalize_case_io(cases)
        assert result[0]["input"] is None
        assert result[0]["output"] is None

    def test_normalize_case_io_display_fields_string_passthrough(self):
        """display_input/display_output strings pass through unchanged."""
        cases = [
            {
                "case_id": "c1",
                "input": {"request_body": {"query": "test"}},
                "output": {"component": {"displayText": "reply"}},
                "display_input": "test",
                "display_output": "reply",
            }
        ]
        result = WebdevReportClient._normalize_case_io(cases)
        # display_* are already strings — pass through
        assert result[0]["display_input"] == "test"
        assert result[0]["display_output"] == "reply"
        # input/output are dicts — converted to JSON strings
        assert isinstance(result[0]["input"], str)
        assert isinstance(result[0]["output"], str)

    @pytest.mark.asyncio
    async def test_fix_case_data_normalizes_dict_io(self):
        """_fix_case_data converts dict input/output to strings in the JSON tag."""
        html = (
            "<html><head><style></style></head><body>"
            '<script type="application/json" id="cases-data">[]</script>'
            "</body></html>"
        )
        spec = {
            "case_results": [
                {
                    "case_id": "c1",
                    "input": {"query": "hi"},
                    "output": ["a", "b"],
                },
            ],
        }
        result = WebdevReportClient._fix_case_data(html, spec)
        # The dict should be serialized as a JSON string inside the tag
        assert '"\\n"' not in result  # no raw newline in JSON value
        # input should be a string value in the JSON, not an object
        # i.e. "input": "{\n  \"query\": ..." not "input": {"query": ...
        assert '"input": "{' in result or '"input":"{' in result
        assert "[object Object]" not in result

    # ── _fix_iife_global_scope tests ────────────────────────────────────

    def test_fix_iife_no_iife_returns_unchanged(self):
        """Without an IIFE, the HTML is returned unchanged."""
        html = (
            "<html><head><style></style></head><body>"
            "<button onclick=\"openDetail('c1')\">click</button>"
            "<script>function openDetail(id){return id;}</script>"
            "</body></html>"
        )
        result = WebdevReportClient._fix_iife_global_scope(html)
        assert result == html

    def test_fix_iife_no_inline_handlers_returns_unchanged(self):
        """Without inline on* attributes, the HTML is returned unchanged."""
        html = (
            "<html><head><style></style></head><body>"
            "<script>(function(){"
            "function openDetail(id){return id;}"
            "})();</script>"
            "</body></html>"
        )
        result = WebdevReportClient._fix_iife_global_scope(html)
        assert result == html

    def test_fix_iife_exposes_functions_to_window(self):
        """IIFE-scoped functions referenced in onclick are exposed to window."""
        html = (
            "<html><head><style></style></head><body>"
            "<button onclick=\"openDetail('c1')\">详情</button>"
            '<div onclick="closeDetail()">close</div>'
            "<script>(function(){"
            '"use strict";'
            "function openDetail(caseId){return caseId;}"
            "function closeDetail(){return;}"
            "function unusedFunc(){return;}"
            "})();</script>"
            "</body></html>"
        )
        result = WebdevReportClient._fix_iife_global_scope(html)
        # openDetail and closeDetail must be exposed to window
        assert "window.openDetail=openDetail" in result
        assert "window.closeDetail=closeDetail" in result
        # unusedFunc is not referenced in on* attrs → must NOT be exposed
        assert "window.unusedFunc" not in result
        # Assignments must be inserted before the IIFE closing })();
        idx_assign = result.index("window.openDetail=openDetail")
        idx_close = result.index("})();")
        assert idx_assign < idx_close

    def test_fix_iife_skips_json_script_tags(self):
        """JSON <script type="application/json"> tags are not touched."""
        html = (
            "<html><head><style></style></head><body>"
            '<script type="application/json" id="cases-data">[]</script>'
            "<button onclick=\"openDetail('c1')\">click</button>"
            "<script>(function(){"
            "function openDetail(id){return id;}"
            "})();</script>"
            "</body></html>"
        )
        result = WebdevReportClient._fix_iife_global_scope(html)
        assert "window.openDetail=openDetail" in result
        # JSON tag content must remain unchanged
        assert '<script type="application/json" id="cases-data">[]</script>' in result

    def test_fix_iife_multiple_handlers_same_function(self):
        """Same function in multiple onclick handlers is exposed only once."""
        html = (
            "<html><head><style></style></head><body>"
            '<button onclick="goToPage(1)">1</button>'
            '<button onclick="goToPage(2)">2</button>'
            '<button onclick="goToPage(3)">3</button>'
            "<script>(function(){"
            "function goToPage(n){return n;}"
            "})();</script>"
            "</body></html>"
        )
        result = WebdevReportClient._fix_iife_global_scope(html)
        # window.goToPage should appear exactly once
        assert result.count("window.goToPage=goToPage") == 1

    def test_fix_iife_oninput_and_onchange(self):
        """oninput and onchange handlers are also detected."""
        html = (
            "<html><head><style></style></head><body>"
            '<input oninput="applyFilters()"/>'
            '<select onchange="applyFilters()"></select>'
            "<script>(function(){"
            "function applyFilters(){return;}"
            "})();</script>"
            "</body></html>"
        )
        result = WebdevReportClient._fix_iife_global_scope(html)
        assert "window.applyFilters=applyFilters" in result

    def test_fix_iife_realistic_report_html(self):
        """Realistic report HTML with IIFE + multiple onclick handlers."""
        html = (
            '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">'
            "<style>body{color:red}</style></head><body>"
            '<div class="detail-overlay" onclick="closeDetail()"></div>'
            "<button onclick=\"switchTab('overview',this)\">tab</button>"
            "<span onclick=\"openDetail('case_42')\">case_42</span>"
            "<button onclick=\"openDetail('case_42')\">详情</button>"
            '<button onclick="goToPage(1)">1</button>'
            '<input oninput="applyFilters()"/>'
            '<script type="application/json" id="report-data">{}</script>'
            '<script type="application/json" id="cases-data">[]</script>'
            "<script>(function(){"
            '"use strict";'
            "function switchTab(id,btn){}"
            "function openDetail(caseId){}"
            "function closeDetail(){}"
            "function applyFilters(){}"
            "function goToPage(n){}"
            "function renderTable(){}"
            "})();</script>"
            "</body></html>"
        )
        result = WebdevReportClient._fix_iife_global_scope(html)
        # All functions referenced in on* handlers must be exposed
        for func in ["openDetail", "closeDetail", "switchTab", "applyFilters", "goToPage"]:
            assert f"window.{func}={func}" in result, f"{func} not exposed"
        # renderTable is NOT referenced in any on* handler → must NOT be exposed
        assert "window.renderTable" not in result
        # JSON script tags must be untouched
        assert '<script type="application/json" id="report-data">{}</script>' in result
        assert '<script type="application/json" id="cases-data">[]</script>' in result

    # ── _fix_window_function_hoisting tests ──

    def test_fix_window_hoisting_no_window_functions(self):
        """HTML without window.X=function(){} returns unchanged."""
        html = "<html><body><script>function foo(){return 1;}</script></body></html>"
        result = WebdevReportClient._fix_window_function_hoisting(html)
        assert result == html

    def test_fix_window_hoisting_converts_to_declaration(self):
        """window.X=function(){} is converted to function X(){} + window.X=X."""
        html = "<html><body><script>window.foo = function(){ return 1; };</script></body></html>"
        result = WebdevReportClient._fix_window_function_hoisting(html)
        assert "function foo()" in result
        assert "window.foo=foo;" in result
        # Original window.foo=function pattern should be gone
        assert "window.foo = function" not in result
        assert "window.foo=function" not in result

    def test_fix_window_hoisting_preserves_body(self):
        """Function body is preserved exactly after conversion."""
        body = "var x = 1; return x + 42;"
        html = f"<html><body><script>window.calc = function(){{{body}}};</script></body></html>"
        result = WebdevReportClient._fix_window_function_hoisting(html)
        assert body in result
        assert "function calc()" in result
        assert "window.calc=calc;" in result

    def test_fix_window_hoisting_with_params(self):
        """Functions with parameters are handled correctly."""
        html = (
            "<html><body><script>"
            "window.openDetail = function(caseId){"
            "var c = CASES.find(function(x){return x.case_id===caseId;});"
            "return c;"
            "};"
            "</script></body></html>"
        )
        result = WebdevReportClient._fix_window_function_hoisting(html)
        assert "function openDetail(caseId)" in result
        assert "window.openDetail=openDetail;" in result
        # Inner function() should NOT be extracted as a separate hoisting
        assert "function find" not in result

    def test_fix_window_hoisting_nested_braces(self):
        """Nested braces in function body are handled by brace counting."""
        html = (
            "<html><body><script>"
            "window.applyFilters = function(){"
            "if(true){var x={a:1,b:2};"
            "for(var k in x){console.log(k);}}"
            "};"
            "</script></body></html>"
        )
        result = WebdevReportClient._fix_window_function_hoisting(html)
        assert "function applyFilters()" in result
        assert "window.applyFilters=applyFilters;" in result
        # The full body should be present
        assert "if(true){var x={a:1,b:2};" in result
        assert "for(var k in x){console.log(k);}" in result

    def test_fix_window_hoisting_braces_in_strings(self):
        """Braces inside string literals don't break brace counting."""
        html = (
            "<html><body><script>"
            "window.render = function(){"
            'var s = "has {braces} inside";'
            "var t = 'also {braces}';"
            "return s + t;"
            "};"
            "</script></body></html>"
        )
        result = WebdevReportClient._fix_window_function_hoisting(html)
        assert "function render()" in result
        assert "window.render=render;" in result
        assert '"has {braces} inside"' in result

    def test_fix_window_hoisting_skips_json_script_tags(self):
        """JSON script tags are not touched."""
        html = (
            "<html><body>"
            '<script type="application/json" id="report-data">'
            '{"foo": "bar"}'
            "</script>"
            "<script>window.foo = function(){return 1;};</script>"
            "</body></html>"
        )
        result = WebdevReportClient._fix_window_function_hoisting(html)
        # JSON tag untouched
        assert '<script type="application/json" id="report-data">' in result
        assert '{"foo": "bar"}' in result
        # JS function converted
        assert "function foo()" in result
        assert "window.foo=foo;" in result

    def test_fix_window_hoisting_multiple_functions(self):
        """Multiple window.X=function patterns are all converted."""
        html = (
            "<html><body><script>"
            "window.a = function(){return 1;};"
            "window.b = function(){return 2;};"
            "window.c = function(x){return x*2;};"
            "</script></body></html>"
        )
        result = WebdevReportClient._fix_window_function_hoisting(html)
        for name in ["a", "b", "c"]:
            assert f"function {name}(" in result, f"{name} not converted"
            assert f"window.{name}={name};" in result, f"{name} not assigned to window"

    def test_fix_window_hoisting_realistic_report(self):
        """Simulates the real LLM-generated report pattern.

        The LLM defines functions as window.X = function(){} INSIDE an
        IIFE, but calls them BEFORE the assignment — causing TypeError.
        After hoisting, function declarations are available throughout.
        """
        html = (
            "<!DOCTYPE html><html><head><style>body{color:red}</style></head><body>"
            "<button onclick=\"switchTab('overview',this)\">tab</button>"
            "<span onclick=\"openDetail('case_42')\">case_42</span>"
            '<script type="application/json" id="report-data">{}</script>'
            '<script type="application/json" id="cases-data">[]</script>'
            "<script>(function(){"
            '"use strict";'
            "var REPORT = JSON.parse(document.getElementById('report-data').textContent);"
            "var CASES = JSON.parse(document.getElementById('cases-data').textContent);"
            "applyFilters();"
            "window.switchTab = function(id,btn){"
            "document.querySelectorAll('.tab-content').forEach(function(el){el.style.display='none';});"
            "};"
            "window.applyFilters = function(){"
            "currentPage=1; renderTable(); renderPagination();"
            "};"
            "window.renderTable = function(){"
            "var tbody=document.querySelector('#data-table tbody');"
            "if(!tbody) return;"
            "tbody.innerHTML=CASES.map(function(c){return '<tr>'+c.case_id+'</tr>';}).join('');"
            "};"
            "window.renderPagination = function(){};"
            "window.goToPage = function(n){currentPage=n; renderTable();};"
            "window.openDetail = function(caseId){"
            "var c=CASES.find(function(x){return x.case_id===caseId;});"
            "document.getElementById('detail-overlay').classList.add('visible');"
            "};"
            "window.closeDetail = function(){"
            "document.getElementById('detail-overlay').classList.remove('visible');"
            "};"
            "})();</script>"
            "</body></html>"
        )
        result = WebdevReportClient._fix_window_function_hoisting(html)
        # All window.X=function patterns should be converted to function declarations
        for name in [
            "switchTab",
            "applyFilters",
            "renderTable",
            "renderPagination",
            "goToPage",
            "openDetail",
            "closeDetail",
        ]:
            assert f"function {name}(" in result, f"{name} not hoisted"
            assert f"window.{name}={name};" in result, f"{name} not assigned to window"
        # No original window.X=function patterns remain
        assert "window.switchTab = function" not in result
        assert "window.applyFilters = function" not in result
        # Inner anonymous functions (forEach/find/map callbacks) are NOT extracted
        assert "function forEach" not in result
        assert "function find" not in result
        assert "function map" not in result
        # JSON tags untouched
        assert '<script type="application/json" id="report-data">{}</script>' in result
        # The critical call (applyFilters() before definition) now works
        # because function applyFilters() is hoisted
        assert "applyFilters();" in result  # original call preserved
        assert "function applyFilters(" in result  # hoisted declaration


# ════════════════════════════════════════════════════════════════════════════
# BenchmarkRunner helper tests
# ════════════════════════════════════════════════════════════════════════════


class TestComputeCriterionStats:
    def test_all_success(self):
        from benchmark.runs.runner import BenchmarkRunner

        criterion = Criterion(
            id="acc",
            name="Accuracy",
            description="",
            weight=1.0,
            evaluator_ref="exact_match",
        )
        results = [_result(0.9), _result(0.8), _result(1.0)]
        stats = BenchmarkRunner._compute_criterion_stats(
            criterion=criterion,
            results=results,
            pass_threshold=0.8,
        )
        assert stats.sample_count == 3
        assert stats.success_count == 3
        assert stats.failed_count == 0
        assert pytest.approx(stats.mean) == (0.9 + 0.8 + 1.0) / 3
        assert stats.pass_rate == 1.0
        assert stats.min == 0.8
        assert stats.max == 1.0

    def test_with_failures(self):
        from benchmark.runs.runner import BenchmarkRunner

        criterion = Criterion(
            id="acc",
            name="Accuracy",
            description="",
            weight=1.0,
            evaluator_ref="exact_match",
        )
        results = [_result(0.9), EvalFailure(RuntimeError("x")), _result(0.5)]
        stats = BenchmarkRunner._compute_criterion_stats(
            criterion=criterion,
            results=results,
            pass_threshold=0.8,
        )
        assert stats.sample_count == 3
        assert stats.success_count == 2
        assert stats.failed_count == 1
        # EvalFailure → 0.0 in scores
        assert pytest.approx(stats.mean) == (0.9 + 0.0 + 0.5) / 3
        assert stats.pass_rate == pytest.approx(1 / 3)

    def test_empty_results(self):
        from benchmark.runs.runner import BenchmarkRunner

        criterion = Criterion(
            id="acc",
            name="Accuracy",
            description="",
            weight=1.0,
            evaluator_ref="exact_match",
        )
        stats = BenchmarkRunner._compute_criterion_stats(
            criterion=criterion,
            results=[],
            pass_threshold=0.8,
        )
        assert stats.sample_count == 0
        assert stats.mean == 0.0


class TestTransposeResults:
    def test_basic(self):
        from benchmark.runs.runner import BenchmarkRunner

        eval_results = {
            "a": [_result(0.1), _result(0.2), _result(0.3)],
            "b": [_result(0.4), _result(0.5), _result(0.6)],
        }
        transposed = BenchmarkRunner._transpose_results(eval_results, 3)
        assert len(transposed) == 3
        assert transposed[0]["a"].score == 0.1
        assert transposed[0]["b"].score == 0.4
        assert transposed[2]["a"].score == 0.3
        assert transposed[2]["b"].score == 0.6


class TestSampleItems:
    def test_sample_smaller(self):
        from benchmark.runs.runner import BenchmarkRunner

        items = list(range(100))
        sampled = BenchmarkRunner._sample_items(items, 10, seed=42)
        assert len(sampled) == 10
        # Same seed → same result
        sampled2 = BenchmarkRunner._sample_items(items, 10, seed=42)
        assert sampled == sampled2

    def test_sample_larger_returns_all(self):
        from benchmark.runs.runner import BenchmarkRunner

        items = [1, 2, 3]
        sampled = BenchmarkRunner._sample_items(items, 10, seed=42)
        assert len(sampled) == 3


class TestPercentile:
    """``percentile_cont`` (the single implementation in run_helpers; shared by the runner and SQL aggregation)."""

    def test_p95(self):
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert 0.9 <= percentile_cont(data, 0.95) <= 1.0

    def test_p50_median(self):
        assert percentile_cont([1, 2, 3, 4, 5], 0.5) == 3.0

    def test_empty(self):
        assert percentile_cont([], 0.95) == 0.0

    def test_single_element(self):
        assert percentile_cont([0.5], 0.95) == 0.5


# ════════════════════════════════════════════════════════════════════════════
# _stamp_benchmark_metadata
# ════════════════════════════════════════════════════════════════════════════


class TestStampBenchmarkMetadata:
    def test_stamps_pipeline_source_and_item_id(self) -> None:
        from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY
        from benchmark.evaluators.base import EvaluatorRequest
        from benchmark.runs.runner import _stamp_benchmark_metadata

        req = EvaluatorRequest(
            input="q",
            output="a",
            metadata={"scene": "non_voice"},
        )
        out = _stamp_benchmark_metadata(req, item_id="item-9")
        assert out.metadata[PIPELINE_SOURCE_METADATA_KEY] == "benchmark"
        assert out.metadata["dataset_item_id"] == "item-9"
        assert out.metadata["scene"] == "non_voice"

    def test_overrides_existing_pipeline_source(self) -> None:
        from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY
        from benchmark.evaluators.base import EvaluatorRequest
        from benchmark.runs.runner import _stamp_benchmark_metadata

        req = EvaluatorRequest(
            input="q",
            output="a",
            metadata={
                PIPELINE_SOURCE_METADATA_KEY: "dataset",
                "dataset_item_id": "keep-me",
            },
        )
        out = _stamp_benchmark_metadata(req, item_id="other")
        assert out.metadata[PIPELINE_SOURCE_METADATA_KEY] == "benchmark"
        assert out.metadata["dataset_item_id"] == "keep-me"


# ════════════════════════════════════════════════════════════════════════════
# Attribution logic-name resolution
# ════════════════════════════════════════════════════════════════════════════


class TestAttributionLogicName:
    def test_strips_stage_prefixes(self) -> None:
        from benchmark.runs.runner import _logic_name

        assert _logic_name("amap_agent") == "amap_agent"
        assert _logic_name("judge_amap_agent") == "amap_agent"
        assert _logic_name("attribution_amap_agent") == "amap_agent"

    def test_resolve_prefers_attribution_services(self) -> None:
        from types import SimpleNamespace

        from benchmark.runs.runner import _resolve_attribution_logic_name

        meta = SimpleNamespace(
            attribution_services=["attribution_example"],
            service_name="amap_chat",
        )
        assert _resolve_attribution_logic_name(meta) == "example"

    def test_resolve_falls_back_to_evaluator_ref(self) -> None:
        from types import SimpleNamespace

        from benchmark.runs.runner import _resolve_attribution_logic_name

        meta = SimpleNamespace(
            attribution_services=[],
            service_name="amap_chat",
        )
        assert (
            _resolve_attribution_logic_name(meta, evaluator_ref="judge_amap_agent") == "amap_agent"
        )


# ════════════════════════════════════════════════════════════════════════════
# Benchmark → evaluate Langfuse report off
# ════════════════════════════════════════════════════════════════════════════


class TestBenchmarkSkipsTraceReport:
    @pytest.mark.asyncio
    async def test_evaluate_called_without_trace_report(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from benchmark.evaluators.base import EvaluatorResult
        from benchmark.runs.runner import (
            BenchmarkRunContext,
            BenchmarkRunner,
            BenchmarkRunnerDeps,
        )
        from benchmark.runs.schema import BenchmarkRequest
        from benchmark.reporting.framework.schema import Criterion, DatasetRef

        evaluate = AsyncMock(
            return_value=EvaluatorResult(score=1.0, reason="ok"),
        )
        runner = BenchmarkRunner(
            BenchmarkRunnerDeps(
                evaluator_service=SimpleNamespace(evaluate=evaluate),
                benchmark_service=MagicMock(
                    merge_criterion_metrics=AsyncMock(),
                    update_evaluator_progress=AsyncMock(),
                ),
                aggregation_calculator=MagicMock(),
                webdev_client=MagicMock(),
            ),
        )

        item = SimpleNamespace(id="i1", source_trace_id=None)
        criterion = Criterion(
            id="c1",
            name="C1",
            description="d",
            weight=1.0,
            evaluator_ref="example",
        )
        services = MagicMock()
        services.get_metadata.return_value = SimpleNamespace(
            name="bm",
            attribution_services=[],
            service_name="svc",
            aggregation=SimpleNamespace(pass_threshold=0.5),
        )
        from benchmark.evaluators.base import EvaluatorRequest

        services._build_evaluator_request.return_value = EvaluatorRequest(
            input="q",
            output="a",
            metadata={},
            trace_id="t1",
            project_id="p",
            agent_name="svc",
        )

        with patch(
            "benchmark.runs.runner.BenchmarkTraceRecordWriter",
        ) as writer_cls:
            writer_cls.return_value = MagicMock(
                upsert_score=AsyncMock(),
            )
            with patch.object(
                runner,
                "_persist_criterion_progress",
                new=AsyncMock(),
            ):
                ctx = BenchmarkRunContext(
                    benchmark_services=services,
                    run_id="run-1",
                    service_name="svc",
                    dataset_ref=DatasetRef(
                        dataset_name="ds",
                        project_id="p",
                        sample_size=1,
                        sample_seed=0,
                    ),
                    rubric=[criterion],
                    items=[item],
                    max_concurrency=1,
                    parallel_evaluators=False,
                    triage_mode="none",
                    request=BenchmarkRequest(),
                )
                await runner._run_single_evaluator(ctx, criterion)

        assert evaluate.await_count == 1
        opts = evaluate.await_args.kwargs["options"]
        assert opts.enable_trace_report is False
        assert opts.trace_client is None
        assert opts.resolve_trace is False
        assert opts.skip_triage is True


class TestFetchDatasetTimeout:
    @pytest.mark.asyncio
    async def test_read_timeout_wrapped_as_unreachable(self) -> None:
        from benchmark.runs.runner import BenchmarkRunner, BenchmarkRunnerDeps

        runner = BenchmarkRunner(
            BenchmarkRunnerDeps(
                evaluator_service=MagicMock(),
                benchmark_service=MagicMock(),
                aggregation_calculator=MagicMock(),
                webdev_client=MagicMock(),
            ),
        )
        lf = MagicMock()
        lf.get_dataset = MagicMock(side_effect=ReadTimeout("timed out"))

        with patch("benchmark.utils.trace_dataset.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Langfuse unreachable: ReadTimeout"):
                await runner._fetch_dataset(lf, "ds")


# ══════════════════════════════════════════════════════════════════════════
# _run_fused_triage attribution gate (aligned with the external judge callback)
# ══════════════════════════════════════════════════════════════════════════


class TestRunFusedTriageGate:
    """Fused triage gate: triage only a low score (< 1.4), and skip GOOD+;

    the fused-row judge_json must carry criterion_scores (aligned with the external /
    benchmark_worker fused-row format).
    """

    def _make_runner(self, triage: MagicMock) -> "object":
        from benchmark.runs.runner import BenchmarkRunner, BenchmarkRunnerDeps

        benchmark_service = MagicMock()
        # _run_fused_triage reads the run config outside the loop (the Aone push config)
        benchmark_service.get_run = AsyncMock(return_value=None)
        return BenchmarkRunner(
            BenchmarkRunnerDeps(
                evaluator_service=MagicMock(),
                benchmark_service=benchmark_service,
                aggregation_calculator=MagicMock(),
                webdev_client=MagicMock(),
                triage_service=triage,
            ),
        )

    def _make_ctx(
        self,
        rubric: list[Criterion],
        score_a: float,
        score_b: float,
    ) -> tuple[Any, dict[str, list[EvaluatorResult]]]:
        """Build (ctx, eval_results) with one item scored on both criteria."""
        from types import SimpleNamespace

        from benchmark.evaluators.base import EvaluatorRequest
        from benchmark.runs.runner import BenchmarkRunContext
        from benchmark.runs.schema import BenchmarkRequest
        from benchmark.reporting.framework.schema import DatasetRef

        services = MagicMock()
        services.get_metadata.return_value = SimpleNamespace(
            name="bm",
            attribution_services=["example"],
            service_name="svc",
        )
        services._build_evaluator_request.return_value = EvaluatorRequest(
            input="q",
            output="a",
            metadata={},
            trace_id="t1",
            project_id="p",
            agent_name="svc",
        )

        item = SimpleNamespace(id="i1", source_trace_id=None)
        ctx = BenchmarkRunContext(
            benchmark_services=services,
            run_id="run-1",
            service_name="svc",
            dataset_ref=DatasetRef(dataset_name="ds", project_id="p"),
            rubric=rubric,
            items=[item],
            max_concurrency=1,
            parallel_evaluators=False,
            triage_mode="fused",
            request=BenchmarkRequest(),
        )
        eval_results = {
            "accuracy": [_result(score_a)],
            "helpfulness": [_result(score_b)],
        }
        return ctx, eval_results

    @pytest.mark.asyncio
    async def test_low_score_triggers_triage_with_criterion_scores(
        self,
        rubric,
        agg_weighted,
    ) -> None:
        # fused = 0.9*0.6 + 0.7*0.4 = 0.82 < 1.4 → trigger attribution
        ctx, eval_results = self._make_ctx(rubric, 0.9, 0.7)
        triage = MagicMock()
        # attribution=None → do not trigger an Aone push (the push path has its own unit tests)
        triage.run = AsyncMock(return_value=TriageOutcome(status="created"))
        runner = self._make_runner(triage)

        with patch(
            "benchmark.runs.runner.BenchmarkTraceRecordWriter",
        ) as writer_cls:
            writer = MagicMock()
            writer.upsert_score = AsyncMock()
            writer_cls.return_value = writer
            await runner._run_fused_triage(ctx, eval_results, agg_weighted)

        assert triage.run.await_count == 1
        # The fused-row judge_json carries criterion_scores (aligned with the external format)
        judge_json = writer.upsert_score.await_args.args[0].judge_json
        assert judge_json["criterion_scores"]["accuracy"]["score"] == 0.9
        assert judge_json["criterion_scores"]["helpfulness"]["score"] == 0.7

    @pytest.mark.asyncio
    async def test_good_score_skips_triage(self, rubric, agg_weighted) -> None:
        # fused = 1.9*0.6 + 1.9*0.4 = 1.9 ≥ 1.4 (GOOD/EXCELLENT) → skip attribution
        ctx, eval_results = self._make_ctx(rubric, 1.9, 1.9)
        triage = MagicMock()
        triage.run = AsyncMock()
        runner = self._make_runner(triage)

        with patch(
            "benchmark.runs.runner.BenchmarkTraceRecordWriter",
        ) as writer_cls:
            writer = MagicMock()
            writer.upsert_score = AsyncMock()
            writer_cls.return_value = writer
            await runner._run_fused_triage(ctx, eval_results, agg_weighted)

        # The fused row is still written (including criterion_scores), but triage is not triggered
        assert writer.upsert_score.await_count == 1
        assert "criterion_scores" in writer.upsert_score.await_args.args[0].judge_json
        assert triage.run.await_count == 0


# ══════════════════════════════════════════════════════════════════════════
# _generate_report wires pages_publisher (Aone Pages second publish)
# ══════════════════════════════════════════════════════════════════════════


class TestGenerateReportPagesPublisher:
    """runner._generate_report: after webdev succeeds, trigger the Aone Pages publish and persist it."""

    def _make_runner(self, *, publisher=None):
        from types import SimpleNamespace

        from benchmark.runs.runner import BenchmarkRunner, BenchmarkRunnerDeps

        benchmark_service = MagicMock()
        # Report emission trusts only the DB (prefer_db_metrics): the run record must carry a complete metrics_json + final_score
        benchmark_service.get_run = AsyncMock(
            return_value=SimpleNamespace(
                config_json={},
                benchmark_name="bm",
                service_name="svc",
                dataset_name="ds",
                final_score=1.5,
                metrics_json={
                    "criterion_metrics": {
                        "c0": {
                            "criterion_id": "c0",
                            "evaluator_ref": "e0",
                            "sample_count": 1,
                            "success_count": 1,
                            "failed_count": 0,
                            "mean": 1.5,
                            "median": 1.5,
                            "p95": 1.5,
                            "min": 1.5,
                            "max": 1.5,
                            "std": 0.0,
                            "pass_rate": 1.0,
                        },
                    },
                    "aggregation_method": "weighted_avg",
                    "total_samples": 1,
                    "total_duration_ms": 0,
                },
            ),
        )
        benchmark_service.load_case_results = AsyncMock(return_value=[])
        benchmark_service.attach_daily_report_context = AsyncMock()
        benchmark_service.merge_report_meta = AsyncMock()
        benchmark_service.merge_performance_metrics = AsyncMock()

        webdev = MagicMock()
        webdev.generate_report = AsyncMock(
            return_value=("<html>report</html>", "https://oss.example.com/r.html"),
        )

        runner = BenchmarkRunner(
            BenchmarkRunnerDeps(
                evaluator_service=MagicMock(),
                benchmark_service=benchmark_service,
                aggregation_calculator=MagicMock(),
                webdev_client=webdev,
                pages_publisher=publisher,
            ),
        )
        return runner, benchmark_service

    @staticmethod
    def _make_request():
        from types import SimpleNamespace

        from benchmark.runs.runner import ReportGenerationSpec

        services = MagicMock()
        services._build_report_spec.return_value = {"benchmark_name": "bm"}
        services.get_metadata.return_value = SimpleNamespace(rubric=[])
        return ReportGenerationSpec(
            benchmark_services=services,
            run_id="run-1",
            metrics=MagicMock(),
            final_score=1.5,
        )

    @pytest.mark.asyncio
    async def test_publish_success_merges_report_site_url(self) -> None:
        publisher = MagicMock()
        publisher.publish = AsyncMock(
            return_value=("https://site.io.alibaba-inc.com/reports/2026-09/run-1/report.html"),
        )
        runner, benchmark_service = self._make_runner(publisher=publisher)

        url = await runner._generate_report(self._make_request())

        # The OSS main-path return value is unaffected
        assert url == "https://oss.example.com/r.html"
        publisher.publish.assert_awaited_once()
        assert publisher.publish.await_args.args[0] == "run-1"
        assert publisher.publish.await_args.args[1] == "<html>report</html>"
        benchmark_service.merge_report_meta.assert_awaited_once()
        call = benchmark_service.merge_report_meta.await_args
        assert call.args[0] == "run-1"
        patch_arg = call.kwargs["patch"]
        assert patch_arg["report_site_url"].endswith(
            "/reports/2026-09/run-1/report.html",
        )
        assert "report_site_published_at" in patch_arg

    @pytest.mark.asyncio
    async def test_no_publisher_skips_publish(self) -> None:
        runner, benchmark_service = self._make_runner(publisher=None)

        url = await runner._generate_report(self._make_request())

        assert url == "https://oss.example.com/r.html"
        benchmark_service.merge_report_meta.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_none_skips_merge(self) -> None:
        """Publish degrades on failure (publish → None): report_url is still returned, and nothing is persisted."""
        publisher = MagicMock()
        publisher.publish = AsyncMock(return_value=None)
        runner, benchmark_service = self._make_runner(publisher=publisher)

        url = await runner._generate_report(self._make_request())

        assert url == "https://oss.example.com/r.html"
        publisher.publish.assert_awaited_once()
        benchmark_service.merge_report_meta.assert_not_called()

    @pytest.mark.asyncio
    async def test_oss_upload_failed_skips_publish(self) -> None:
        """OSS produced nothing (report_url=None) → this node does not push to Aone Pages."""
        publisher = MagicMock()
        publisher.publish = AsyncMock()
        runner, benchmark_service = self._make_runner(publisher=publisher)
        runner._webdev.generate_report = AsyncMock(
            return_value=("<html>report</html>", None),
        )

        url = await runner._generate_report(self._make_request())

        assert url is None
        publisher.publish.assert_not_called()
        benchmark_service.merge_report_meta.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# _generate_report data has the DB as its only source of truth (in-memory data on a multi-host deploy must not go straight into the report)
# ══════════════════════════════════════════════════════════════════════════


class TestGenerateReportDbTruth:
    """The report spec's aggregate metrics follow the persisted metrics_json, not the caller's in-memory object."""

    @staticmethod
    def _db_metrics_json() -> dict:
        from benchmark.reporting.framework.schema import AggregatedMetrics, CriterionStats

        return AggregatedMetrics(
            criterion_metrics={
                "intent": CriterionStats(
                    criterion_id="intent",
                    evaluator_ref="sub_scores",
                    sample_count=9,
                    success_count=9,
                    failed_count=0,
                    mean=1.9,
                    median=2.0,
                    p95=2.0,
                    min=1.0,
                    max=2.0,
                    std=0.3,
                    pass_rate=0.9,
                ),
            },
            aggregation_method="weighted_avg",
            total_samples=9,
            total_duration_ms=777,
        ).model_dump()

    def _make_runner(self, run_record: object) -> tuple[object, MagicMock, MagicMock]:
        from benchmark.runs.runner import BenchmarkRunner, BenchmarkRunnerDeps

        benchmark_service = MagicMock()
        benchmark_service.get_run = AsyncMock(return_value=run_record)
        benchmark_service.load_case_results = AsyncMock(return_value=[])
        benchmark_service.attach_daily_report_context = AsyncMock()
        benchmark_service.merge_report_meta = AsyncMock()

        webdev = MagicMock()
        webdev.generate_report = AsyncMock(return_value=("<html>x</html>", None))

        runner = BenchmarkRunner(
            BenchmarkRunnerDeps(
                evaluator_service=MagicMock(),
                benchmark_service=benchmark_service,
                aggregation_calculator=MagicMock(),
                webdev_client=webdev,
            ),
        )
        return runner, benchmark_service, webdev

    @pytest.mark.asyncio
    async def test_metrics_rebuilt_from_db_metrics_json(self) -> None:
        """When the in-memory metrics cover only some dimensions, the report still uses the complete DB snapshot."""
        from types import SimpleNamespace

        from benchmark.runs.runner import ReportGenerationSpec

        run_record = SimpleNamespace(
            config_json={},
            benchmark_name="bm",
            metrics_json=self._db_metrics_json(),
            final_score=1.9,
        )
        runner, _bs, webdev = self._make_runner(run_record)

        services = MagicMock()
        captured: dict[str, Any] = {}

        def _capture(run: object, aggregated: object, final_score: float) -> dict:
            captured["aggregated"] = aggregated
            captured["final_score"] = final_score
            captured["run"] = run
            return {"benchmark_name": "bm"}

        services._build_report_spec.side_effect = _capture
        services.get_metadata.return_value = SimpleNamespace(rubric=[])

        # In-memory metrics have only 1 sample (this node processed only some of the items)
        memory_metrics = MagicMock()
        req = ReportGenerationSpec(
            benchmark_services=services,
            run_id="run-1",
            metrics=memory_metrics,
            final_score=0.5,
        )
        await runner._generate_report(req)

        # What _build_report_spec receives is the metrics rebuilt from the DB and the persisted final_score
        aggregated = captured["aggregated"]
        assert aggregated.criterion_metrics["intent"].mean == 1.9
        assert aggregated.total_samples == 9
        assert captured["final_score"] == 1.9
        # The run record is passed through (so _build_report_spec can read the name fields from the DB)
        assert captured["run"] is run_record
        # The in-memory object is not used
        assert captured["aggregated"] is not memory_metrics

    @pytest.mark.asyncio
    async def test_fallback_to_memory_when_metrics_json_missing(self) -> None:
        """metrics_json is missing criterion_metrics (a mid-run incremental fragment) → do not fall back to memory, and fail directly."""
        from types import SimpleNamespace

        from benchmark.runs.runner import ReportGenerationSpec

        run_record = SimpleNamespace(
            config_json={},
            benchmark_name="bm",
            metrics_json={"finalize_source": "partial"},
            final_score=None,
        )
        runner, _bs, webdev = self._make_runner(run_record)

        services = MagicMock()
        services._build_report_spec.side_effect = AssertionError(
            "must not be called — DB truth missing",
        )
        services.get_metadata.return_value = SimpleNamespace(rubric=[])

        memory_metrics = MagicMock()
        req = ReportGenerationSpec(
            benchmark_services=services,
            run_id="run-1",
            metrics=memory_metrics,
            final_score=0.5,
        )
        # No in-memory fallback: when the DB has no complete metrics_json, report emission should fail (return None)
        result = await runner._generate_report(req)
        assert result is None


class TestBuildReportSpecDbRunFields:
    """_build_report_spec prefers the name fields on the DB run record, and falls back to the registry when they are missing."""

    @staticmethod
    def _services() -> MagicMock:
        from types import SimpleNamespace

        services = MagicMock()
        services.get_metadata.return_value = SimpleNamespace(
            name="reg_bm",
            service_name="reg_svc",
            dataset=SimpleNamespace(dataset_name="reg_ds"),
            rubric=[],
        )
        return services

    @staticmethod
    def _metrics() -> object:
        from benchmark.reporting.framework.schema import AggregatedMetrics

        return AggregatedMetrics(
            aggregation_method="weighted_avg",
            total_samples=1,
            total_duration_ms=1,
        )

    def test_prefers_db_run_record(self) -> None:
        from types import SimpleNamespace

        from benchmark.reporting.framework.base import BaseBenchmarkServices

        run = SimpleNamespace(
            benchmark_name="db_bm",
            service_name="db_svc",
            dataset_name="db_ds",
        )
        spec = BaseBenchmarkServices._build_report_spec(
            self._services(),
            run,
            self._metrics(),
            1.5,
        )
        assert spec["benchmark_name"] == "db_bm"
        assert spec["service_name"] == "db_svc"
        assert spec["dataset_name"] == "db_ds"

    def test_fallback_to_registry_when_run_none(self) -> None:
        from benchmark.reporting.framework.base import BaseBenchmarkServices

        spec = BaseBenchmarkServices._build_report_spec(
            self._services(),
            None,
            self._metrics(),
            1.5,
        )
        assert spec["benchmark_name"] == "reg_bm"
        assert spec["service_name"] == "reg_svc"
        assert spec["dataset_name"] == "reg_ds"
