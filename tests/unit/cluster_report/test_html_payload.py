"""HTML and VUI payload tests migrated with their implementation."""

from benchmark.reporting.cluster.render.html import render_html
from benchmark.reporting.cluster.schema import (
    AttributionView,
    BatchCaseRecord,
    BucketPerf,
    CaseKey,
    CasePerf,
    ClusterReport,
    Coverage,
    L1Stats,
)
from benchmark.reporting.cluster.unified_payload import build_vui_metrics


def _record(  # noqa: PLR0913
    *,
    trace_id: str = "t-1",
    suite: str = "车机-Standard",
    case_id: str = "std-1",
    tag: str = "navigation.direct",
    query: str = "导航回家",
    turn_index: str = "",
    score: float | None = 1.0,
    reason: str = "缺少 CMD",
    perf: CasePerf | None = None,
    attribution: AttributionView | None = None,
    failure_modes: list[str] | None = None,
) -> BatchCaseRecord:
    return BatchCaseRecord(
        trace_id=trace_id,
        score=score,
        reason=reason,
        case_meta={"suite": suite, "case_id": case_id, "agent_mode": "standard"},
        case_key=CaseKey(
            suite=suite,
            case_id=case_id,
            tag=tag,
            query=query,
            turn_index=turn_index,
        ),
        failure_modes=[{"mode": m} for m in (failure_modes or [])],
        attribution=attribution or AttributionView(),
        perf=perf,
    )


def _perf() -> CasePerf:
    return CasePerf(
        buckets={
            "react": BucketPerf(
                calls=2,
                prompt_tokens=200,
                completion_tokens=40,
                total_tokens=240,
                latency_ms=3000.0,
                cost=0.01,
            ),
            "cgi": BucketPerf(
                calls=1,
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
                latency_ms=200.0,
            ),
        },
        e2e_ms=4200.0,
        first_text_ms=980.0,
    )


class TestHtmlRendering:
    def test_hides_notification_defect_metrics_and_checker_sections(self) -> None:
        """HTML does not yet show Aone metrics or the Checker definition for tickets that are not wired up."""

        report = _report()
        report.batch_meta["evaluator_name"] = "vui_agent"
        report.stats.performance["aone_defects"] = {
            "unique_workitems": 2,
            "related_cases": 3,
        }
        html = render_html(report)

        assert "Agent 天级跟踪日报" in html
        assert "VUI 运行观测" in html
        assert "去重 Aone 缺陷工单" not in html
        assert "关联 case 数" not in html
        assert "Checker 汇总" not in html
        assert "Checker 风险标记" not in html
        assert "CMD 标签缺失簇" in html
        assert "导航类请求未生成 CMD" in html
        assert "root_cause_top 首位 9 条" in html
        assert "Top 问题深度剖析" in html
        assert "不合格归因" in html

    def test_renders_vui_dashboard_dimensions_without_checker_sections(self) -> None:
        report = _report()
        report.cases = [
            _record(trace_id="t-1", score=1.0),
            _record(trace_id="t-2", score=2.0),
        ]
        report.cases[0].dimension_results = [
            {"dimension": "D0", "status": "pass", "score_hint": 2.0},
            {"dimension": "D4", "status": "partial", "score_hint": 1.0},
        ]
        report.cases[1].dimension_results = [
            {"dimension": "D0", "status": "not_applicable"},
            {"dimension": "D4", "status": "pass", "score_hint": 2.0},
        ]
        html = render_html(report)

        assert "D0" in html
        assert "安全与合规" in html
        assert "任务落地与执行闭环" in html
        assert "满分率" in html
        assert "分布" in html
        assert "效果指标" in html
        assert "性能 &amp; 效率" in html
        assert "VUI 运行观测" in html
        assert "D4 · 任务落地与执行闭环" in html
        assert html.index("效果指标") < html.index("VUI 运行观测") < html.index("性能 &amp; 效率")
        assert "Checker 风险标记" not in html

    def test_renders_actual_models_without_stage_buckets(self) -> None:
        report = _report()
        report.stats.performance["models"] = {
            "qwen-max": {
                "cases": 9,
                "calls": {"avg": 1.2},
                "total_tokens": {"avg": 120.0},
                "latency_ms": {"avg": 800.0},
            }
        }

        html = render_html(report)

        assert "模型" in html
        assert "qwen-max" in html
        assert 'id = "vui-cluster-metrics"' in html
        assert "调用阶段" not in html
        assert ">react<" not in html

    def test_escapes_case_content(self) -> None:
        report = _report()
        report.cases = [_record(query="<script>alert(1)</script>", score=0.0)]

        html = render_html(report)

        assert "<script>alert(1)</script>" not in html

    def test_paginates_failed_cases(self) -> None:
        report = _report()
        report.cases = [
            _record(trace_id=f"t-{index}", case_id="", query=f"问题 {index}", score=0.0)
            for index in range(21)
        ]

        html = render_html(report)

        assert 'id="case-section"' in html
        assert 'id="data-table"' in html
        assert "applyFilters" in html
        assert "实时请求" in html

    def test_renders_daily_style_enhancement_sections(self) -> None:
        report = _report()
        report.stats.enhancements = {
            "score_loss_waterfall": {
                "start": 2.0,
                "end": 1.0,
                "delta": 1.0,
                "steps": [{"label": "CMD 缺失", "loss": 1.0, "count": 2}],
            },
            "root_cause_dim_matrix": {
                "rows": ["CMD 缺失"],
                "cols": [{"id": "D4", "name": "任务落地与执行闭环"}],
                "cells": {"CMD 缺失": {"D4": 1.0}},
                "unit": "mean_loss",
            },
        }

        html = render_html(report)

        assert "分数损失瀑布" in html
        assert "CMD 缺失" in html
        assert "根因 × 维度失分矩阵" in html
        assert "insight" in html
        assert (
            html.index("分数损失瀑布")
            < html.index("VUI 运行观测")
            < html.index('id="case-section"')
        )


class TestVuiMetrics:
    def test_builds_private_vui_metrics(self) -> None:
        report = _report()
        report.cases = [_record(score=1.0)]
        report.stats.performance["route_modes"] = {
            "modes": {"lite_sdm": {"count": 1, "share": 1.0}}
        }

        metrics = build_vui_metrics(report)

        assert metrics["routes"]["modes"]["lite_sdm"]["count"] == 1
        assert metrics["clusters"][0]["name"] == "CMD 标签缺失簇"
        assert metrics["low_score_total"] == 1

    def test_low_score_clusters_do_not_fall_back_to_attribution(self) -> None:
        report = _report()
        report.llm_summary = {}
        report.stats.attribution = {"root_cause_top": [{"code": "rejection_misjudge", "count": 1}]}
        report.cases = [_record(trace_id="trace-low", score=0.0, reason="CMD 缺失")]

        metrics = build_vui_metrics(report)

        assert metrics["low_score_total"] == 1
        assert metrics["clusters"] == []

    def test_low_score_cluster_counts_only_unique_original_judge_low_scores(self) -> None:
        report = _report()
        low = _record(trace_id="trace-low", score=0.0)
        low.attribution = AttributionView(corrected_triage_level="good")
        partial = _record(trace_id="trace-partial", score=1.0)
        high = _record(trace_id="trace-high", score=2.0)
        report.cases = [low, partial, high]
        report.llm_summary = {
            "clusters": [
                {
                    "title": "D4 扣分",
                    "affected_count": 20,
                    "related_cases": [
                        "trace-low",
                        "trace-low",
                        "trace-partial",
                        "trace-high",
                        "missing",
                    ],
                }
            ]
        }

        metrics = build_vui_metrics(report)
        cluster = metrics["clusters"][0]

        assert metrics["low_score_total"] == 2
        assert cluster["count"] == 2
        assert [case["id"] for case in cluster["cases"]] == ["std-1", "std-1"]
        assert [case["score"] for case in cluster["cases"]] == [0.0, 1.0]


def _report() -> ClusterReport:
    return ClusterReport(
        batch_tag="run-20260729-153000",
        batch_meta={"model": "qwen-max", "agent_mode": "standard", "planned": 100},
        coverage=Coverage(
            expected_total=100,
            judged=95,
            should_attribute=20,
            attributed=18,
            pending=2,
            perf_collected=93,
            perf_note="2 条 Langfuse 拉取超时",
        ),
        stats=L1Stats(
            effectiveness={
                "total": 95,
                "scored": 95,
                "avg_score": 1.4,
                "pass_rate": 0.6,
                "partial_rate": 0.2,
                "fail_rate": 0.2,
                "score_distribution": {"0": 19, "1": 19, "2": 57},
                "score_views": {
                    "judge_score": {
                        "scored": 95,
                        "avg_score": 1.3,
                        "distribution": {"0": 24, "1": 18, "2": 53},
                    },
                    "attribution_score": {
                        "scored": 5,
                        "avg_score": 2.0,
                        "distribution": {"0": 0, "1": 0, "2": 5},
                    },
                    "effective_score": {
                        "scored": 95,
                        "avg_score": 1.4,
                        "distribution": {"0": 19, "1": 19, "2": 57},
                    },
                },
                "hard_rule_hit_rate": 0.12,
                "dimension_fail_rate": {"D4": {"judged": 80, "failed": 24, "fail_rate": 0.3}},
                "failure_modes": {"cmd_tag_missing": 9},
                "checker_fail_hits": {"cmd_checker": 9},
            },
            attribution={
                "should_attribute": 20,
                "attributed": 18,
                "pending": 2,
                "owner_fallback_rate": 0.1,
                "root_cause_top": [{"code": "cmd_tag_missing", "count": 9, "share": 0.5}],
                "owner_group_distribution": {"prompt-team": 12},
                "overturn": {
                    "overturn_count": 3,
                    "overturn_rate": 0.1667,
                    "by_type": {"eval_rule_issue": 2, "judge_false_positive": 1},
                    "cases": [],
                },
                "confidence": {"avg": 0.82, "low_rate": 0.05},
                "evidence": {"evidence_rate": 0.9},
                "human_review": {
                    "count": 1,
                    "cases": [
                        {
                            "trace_id": "t-9",
                            "suite": "车机-Max",
                            "case_id": "max-3",
                            "root_cause": "工具超时",
                        }
                    ],
                },
                "pending_backlog": {
                    "count": 2,
                    "cases": [
                        {
                            "trace_id": "t-7",
                            "suite": "车机-Max",
                            "case_id": "max-1",
                            "triage_level": "severe_bad",
                        }
                    ],
                },
            },
            performance={
                "collected": 93,
                "missing": 2,
                "missing_reasons": {"timeout": 2},
                "e2e_ms": {
                    "count": 93,
                    "avg": 3200.5,
                    "p50": 3000.0,
                    "p95": 8800.0,
                    "p99": 9900.0,
                    "max": 12000.0,
                },
                "first_text_ms": {
                    "count": 93,
                    "avg": 980.0,
                    "p50": 900.0,
                    "p95": 1800.0,
                    "p99": 2100.0,
                    "max": 2500.0,
                },
                "buckets": {
                    "react": {
                        "cases": 93,
                        "calls": {"avg": 2.1, "total": 195},
                        "latency_ms": {"count": 93, "avg": 1500.0, "p95": 3000.0},
                        "prompt_tokens": {"avg": 100.0, "avg_per_call": 47.6},
                        "completion_tokens": {"avg": 20.0, "avg_per_call": 9.5},
                    },
                    "cgi": {"cases": 0},
                },
                "total_tokens": 123456,
                "total_cost": 1.23,
            },
            crosstab={
                "checker_root_cause_gap": [
                    {"checker": "poi_checker", "hits": 4, "unmapped": 4, "unmapped_rate": 1.0},
                ],
                "suite_overturn": [
                    {
                        "suite": "车机-Max",
                        "total": 30,
                        "low_score": 12,
                        "attributed": 10,
                        "overturned": 3,
                        "overturn_rate": 0.3,
                    },
                ],
            },
        ),
        llm_summary={
            "headline": "CMD 缺失是本批次主要问题",
            "clusters": [
                {
                    "title": "CMD 标签缺失簇",
                    "root_cause_codes": ["cmd_tag_missing"],
                    "affected_count": 9,
                    "owner_group": "prompt-team",
                    "summary": "导航类请求未生成 CMD",
                    "evidence": "root_cause_top 首位 9 条",
                    "related_cases": ["t-1", "t-2"],
                }
            ],
            "feedback_suggestions": [
                {"target": "prompt", "action": "补充 CMD 生成指令", "rationale": "failure_modes"},
            ],
            "aone_suggestions": [
                {
                    "title": "修复 CMD 缺失",
                    "priority": "P1",
                    "owner_group": "prompt-team",
                    "acceptance": "该 failure_mode 归零",
                    "related_cases": ["t-1"],
                },
            ],
            "caveats": ["归因覆盖率 90%，结论存在盲区"],
        },
        generated_at="2026-07-29T16:00:00+08:00",
    )
