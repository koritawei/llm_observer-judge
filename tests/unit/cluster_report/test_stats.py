"""T20-T23 unit tests: the four L1 stat blocks (a fixed fixture asserts exact numbers).

L1 is a pure arithmetic layer, so every number must be asserted exactly — it is the baseline the L2 LLM soft numeric check uses.
A weak assertion would hide errors in the report above it.
"""

from benchmark.reporting.cluster.schema import (
    AttributionView,
    BatchCaseRecord,
    BucketPerf,
    CaseKey,
    CasePerf,
)
from benchmark.reporting.cluster.stats.attribution import build_attribution_stats, root_cause_label
from benchmark.reporting.cluster.stats.crosstab import build_crosstab_stats
from benchmark.reporting.cluster.stats.effectiveness import build_effectiveness_stats
from benchmark.reporting.cluster.stats.enhancements import build_enhancement_stats
from benchmark.reporting.cluster.stats.observations import build_observation_stats
from benchmark.reporting.cluster.stats.performance import build_performance_stats


def _record(  # noqa: PLR0913 - fixture exposes independent report inputs
    *,
    trace_id: str = "t-1",
    score: float | None = 1.0,
    suite: str = "车机-Standard",
    case_id: str = "c-1",
    dimensions: list[tuple[str, str]] | None = None,
    checkers: list[tuple[str, str]] | None = None,
    checker_hits: list[list[str]] | None = None,
    hard_rules: list[str] | None = None,
    risk_flags: list[str] | None = None,
    failure_modes: list[str] | None = None,
    triage_level: str = "",
    attribution: AttributionView | None = None,
    perf: CasePerf | None = None,
    notification_workitem_id: str = "",
    metrics: list[dict] | None = None,
    diagnostics: list[dict] | None = None,
    business_id: str = "",
    scenario: str = "",
    profile_id: str = "",
    profile_version: str = "",
) -> BatchCaseRecord:
    """``checker_hits`` lines up with ``checkers`` by index: checker_hits[i] is the hard_rule_hits of
    checkers[i] itself (not mixed with the LLM judge), and it checks that :func:`build_effectiveness_stats`
    ``hard_rule_hit_rate`` counts only checker hits. When omitted, each checker is treated as having no hits.
    """
    return BatchCaseRecord(
        trace_id=trace_id,
        score=score,
        triage_level=triage_level,
        case_meta={"suite": suite, "case_id": case_id},
        case_key=CaseKey(suite=suite, case_id=case_id),
        dimension_results=[{"dimension": d, "status": s} for d, s in (dimensions or [])],
        checker_results=[
            {
                "source_name": n,
                "status": s,
                "hard_rule_hits": (
                    checker_hits[i] if checker_hits and i < len(checker_hits) else []
                ),
            }
            for i, (n, s) in enumerate(checkers or [])
        ],
        hard_rule_hits=hard_rules or [],
        risk_flags=risk_flags or [],
        failure_modes=[{"mode": m} for m in (failure_modes or [])],
        attribution=attribution or AttributionView(),
        notification_workitem_id=notification_workitem_id,
        perf=perf,
        metric_observations=metrics or [],
        diagnostic_findings=diagnostics or [],
        business_id=business_id,
        scenario=scenario,
        profile_id=profile_id,
        profile_version=profile_version,
    )


def _completed(
    *,
    root_cause_code: str = "cmd_tag_missing",
    category: str = "prompt",
    owner_group: str = "prompt-team",
    conclusion_type: str = "agent_badcase",
    confidence: float = 0.9,
    corrected: str = "",
    needs_review: bool = False,
    evidence: bool = True,
) -> AttributionView:
    return AttributionView(
        status="completed",
        root_cause="根因描述",
        root_cause_code=root_cause_code,
        category=category,
        owner_group=owner_group,
        conclusion_type=conclusion_type,
        confidence=confidence,
        corrected_triage_level=corrected,
        needs_human_review=needs_review,
        evidence_details=[{"field": "cmd_list"}] if evidence else [],
    )


class TestEffectivenessStats:
    def test_score_distribution_and_rates(self) -> None:
        records = [
            _record(trace_id="t-1", score=2.0),
            _record(trace_id="t-2", score=2.0),
            _record(trace_id="t-3", score=1.0),
            _record(trace_id="t-4", score=0.0),
        ]

        stats = build_effectiveness_stats(records, expected_total=5)

        assert stats["total"] == 4
        assert stats["scored"] == 4
        assert stats["score_distribution"] == {"0": 1, "1": 1, "2": 2}
        assert stats["pass_rate"] == 0.5
        assert stats["partial_rate"] == 0.25
        assert stats["fail_rate"] == 0.25
        assert stats["avg_score"] == 1.25
        # eval_coverage = recalled 4 / expected 5
        assert stats["eval_coverage"] == 0.8

    def test_attribution_correction_drives_effective_statistics(self) -> None:
        records = [
            _record(trace_id="t-1", score=0.0),
            _record(
                trace_id="t-2",
                score=0.0,
                attribution=_completed(
                    conclusion_type="judge_false_positive",
                    corrected="good",
                ),
            ),
        ]

        stats = build_effectiveness_stats(records)

        assert stats["score_distribution"] == {"0": 1, "1": 0, "2": 1}
        assert stats["avg_score"] == 1.0
        assert stats["score_views"] == {
            "judge_score": {
                "scored": 2,
                "avg_score": 0.0,
                "distribution": {"0": 2, "1": 0, "2": 0},
            },
            "attribution_score": {
                "scored": 1,
                "avg_score": 2.0,
                "distribution": {"0": 0, "1": 0, "2": 1},
            },
            "effective_score": {
                "scored": 2,
                "avg_score": 1.0,
                "distribution": {"0": 1, "1": 0, "2": 1},
            },
        }

    def test_fractional_score_bucketing(self) -> None:
        """The A/B two-round average (1.5) is filed in the 1-point bucket and is not rounded to 2."""
        stats = build_effectiveness_stats([_record(score=1.5), _record(score=1.9)])
        assert stats["score_distribution"] == {"0": 0, "1": 2, "2": 0}

    def test_unscored_records_excluded_from_rates(self) -> None:
        """A missing score (scoring failed) is not in the score-distribution denominator, but it is in total."""
        stats = build_effectiveness_stats([_record(score=2.0), _record(score=None)])
        assert stats["total"] == 2
        assert stats["scored"] == 1
        assert stats["pass_rate"] == 1.0

    def test_dimension_fail_rate_excludes_not_applicable(self) -> None:
        """not_applicable is not in the denominator — a dimension that does not apply to a conditional scenario must not understate the fail rate."""
        records = [
            _record(dimensions=[("D4", "fail"), ("D3", "not_applicable")]),
            _record(dimensions=[("D4", "pass"), ("D3", "not_applicable")]),
            _record(dimensions=[("D4", "partial")]),
        ]

        stats = build_effectiveness_stats(records)
        d4 = stats["dimension_fail_rate"]["D4"]

        assert d4 == {"judged": 3, "failed": 2, "fail_rate": round(2 / 3, 4)}
        # D3 is entirely not_applicable → it does not appear in the result
        assert "D3" not in stats["dimension_fail_rate"]

    def test_dimension_partial_and_ambiguous_count_as_failure(self) -> None:
        records = [
            _record(dimensions=[("D2", "partial")]),
            _record(dimensions=[("D2", "ambiguous")]),
            _record(dimensions=[("D2", "pass")]),
        ]
        assert build_effectiveness_stats(records)["dimension_fail_rate"]["D2"]["failed"] == 2

    def test_hard_rule_and_checker_and_risk_aggregation(self) -> None:
        records = [
            _record(
                hard_rules=["cmd_missing"],
                risk_flags=["cmd_sensitive", "low_confidence"],
                checkers=[("cmd_checker", "fail"), ("schema_checker", "pass")],
                failure_modes=["cmd_tag_missing", "cmd_param_destination"],
            ),
            _record(
                hard_rules=["cmd_missing", "poi_missing"],
                risk_flags=["cmd_sensitive"],
                checkers=[("cmd_checker", "partial")],
                failure_modes=["cmd_tag_missing"],
            ),
            _record(),
        ]

        stats = build_effectiveness_stats(records)

        # The hard_rule_hit_rate definition is in test_hard_rule_hit_rate_checker_only_over_low_score_denominator;
        # here only the other aggregation dimensions, which do not depend on it, are covered.
        assert stats["hard_rule_hits"] == {"cmd_missing": 2, "poi_missing": 1}
        # A checker in the pass state does not count as a failure hit
        assert stats["checker_fail_hits"] == {"cmd_checker": 2}
        assert stats["risk_flags"] == {"cmd_sensitive": 2, "low_confidence": 1}
        assert stats["failure_modes"] == {
            "cmd_tag_missing": 2,
            "cmd_param_destination": 1,
        }

    def test_hard_rule_hit_rate_checker_only_over_low_score_denominator(self) -> None:
        """The numerator counts only checker (deterministic rule) hits, not LLM sub-judge hits;
        the denominator is the number of low-score cases (score < 2), not all cases."""
        records = [
            # Low score + the checker really hit → counts in the numerator
            _record(
                trace_id="t-1",
                score=1.0,
                checkers=[("cmd_checker", "fail")],
                checker_hits=[["cmd_missing"]],
                hard_rules=["cmd_missing"],
            ),
            # Low score, but the hard rule comes only from the LLM sub judge (the checker side did not hit) → not in the numerator
            _record(
                trace_id="t-2",
                score=0.0,
                checkers=[("cmd_checker", "fail")],
                checker_hits=[[]],
                hard_rules=["judge_only_hit"],
            ),
            # A full score → not in the denominator (even if the checker hit a hard rule)
            _record(
                trace_id="t-3",
                score=2.0,
                checkers=[("cmd_checker", "fail")],
                checker_hits=[["cmd_missing"]],
                hard_rules=["cmd_missing"],
            ),
        ]

        stats = build_effectiveness_stats(records)

        # Denominator = number of low-score cases = 2 (t-1, t-2); t-3 is a full score and is excluded
        # Numerator = low-score cases the checker hit = 1 (only t-1; t-2 was hit only by the judge)
        assert stats["hard_rule_hit_rate"] == 0.5

    def test_suite_and_triage_counts(self) -> None:
        records = [
            _record(suite="车机-Standard", triage_level="severe_bad"),
            _record(suite="车机-Max", triage_level="good"),
            _record(suite="车机-Max", triage_level="good"),
        ]
        stats = build_effectiveness_stats(records)
        assert stats["suites"] == {"车机-Max": 2, "车机-Standard": 1}
        assert stats["triage_levels"] == {"good": 2, "severe_bad": 1}

    def test_malformed_dimension_entries_ignored(self) -> None:
        record = BatchCaseRecord(
            score=1.0,
            dimension_results=[
                {"dimension": "X9", "status": "fail"},
                {"dimension": "D4"},
                {"status": "fail"},
                {},
            ],
        )

        assert build_effectiveness_stats([record])["dimension_fail_rate"] == {}

    def test_non_dict_dimension_entries_tolerated_by_stats(self) -> None:
        record = BatchCaseRecord.model_construct(
            score=1.0,
            effective_score=1.0,
            dimension_results=["bad", None, {"dimension": "D4", "status": "fail"}],
        )

        stats = build_effectiveness_stats([record])

        assert stats["dimension_fail_rate"]["D4"]["failed"] == 1

    def test_empty_input_is_safe(self) -> None:
        stats = build_effectiveness_stats([])
        assert stats["total"] == 0
        assert stats["pass_rate"] == 0.0
        assert stats["eval_coverage"] == 0.0
        assert stats["dimension_fail_rate"] == {}


class TestEnhancementStats:
    def test_adapts_cluster_cases_to_shared_waterfall_and_matrix(self) -> None:
        records = [
            BatchCaseRecord(
                score=0.0,
                attribution=AttributionView(status="completed", root_cause="工具超时"),
                dimension_results=[
                    {"dimension": "D1", "status": "fail", "score_hint": 0.0},
                    {"dimension": "D4", "status": "pass", "score_hint": 2.0},
                ],
            ),
            BatchCaseRecord(
                score=1.0,
                dimension_results=[{"dimension": "D1", "status": "partial", "score_hint": 1.0}],
            ),
        ]

        stats = build_enhancement_stats(records)

        waterfall = stats["score_loss_waterfall"]
        assert waterfall is not None
        assert waterfall["end"] == 0.5
        assert {step["label"] for step in waterfall["steps"]} == {"工具超时", "未归因"}
        matrix = stats["root_cause_dim_matrix"]
        assert matrix is not None
        assert matrix["cells"]["工具超时"]["D1"] == 2.0
        assert matrix["cells"]["未归因"]["D1"] == 1.0

    def test_malformed_dimension_entries_are_ignored(self) -> None:
        record = BatchCaseRecord(
            score=1.0,
            dimension_results=[
                {"dimension": "X9", "status": "fail", "score_hint": 0.0},
                {"dimension": "D4", "status": "fail", "score_hint": float("nan")},
                {"status": "fail", "score_hint": 0.0},
                {},
            ],
        )

        stats = build_enhancement_stats([record])

        assert stats["root_cause_dim_matrix"] is None

    def test_non_dict_dimension_entries_are_tolerated(self) -> None:
        record = BatchCaseRecord.model_construct(
            score=1.0,
            effective_score=1.0,
            attribution=AttributionView(),
            dimension_results=["bad", None, {"dimension": "D4", "status": "fail", "score_hint": 0.0}],
        )

        stats = build_enhancement_stats([record])

        assert stats["root_cause_dim_matrix"]["cells"]["未归因"]["D4"] == 2.0

    def test_non_finite_scores_are_ignored(self) -> None:
        records = [
            BatchCaseRecord(score=float("nan")),
            BatchCaseRecord(score=float("inf")),
        ]

        stats = build_enhancement_stats(records)

        assert stats["score_loss_waterfall"] is None
        assert stats["root_cause_dim_matrix"] is None


class TestAttributionStats:
    def test_coverage_uses_pending_plus_completed(self) -> None:
        """Denominator = the number triage actually let through (pending + completed), excluding ones that were not triggered."""
        records = [
            _record(trace_id="t-1", attribution=_completed()),
            _record(trace_id="t-2", attribution=_completed()),
            _record(trace_id="t-3", attribution=AttributionView(status="pending")),
            _record(trace_id="t-4"),  # not triggered
            _record(trace_id="t-5"),
        ]

        stats = build_attribution_stats(records)

        assert stats["should_attribute"] == 3
        assert stats["attributed"] == 2
        assert stats["pending"] == 1
        assert stats["attribution_coverage"] == round(2 / 3, 4)

    def test_root_cause_top_with_share(self) -> None:
        records = [
            _record(trace_id=f"t-{i}", attribution=_completed(root_cause_code="cmd_tag_missing"))
            for i in range(3)
        ] + [
            _record(trace_id="t-x", attribution=_completed(root_cause_code="poi_tag_grafting")),
        ]

        top = build_attribution_stats(records)["root_cause_top"]

        assert top[0] == {"code": "cmd_tag_missing", "count": 3, "share": 0.75}
        assert top[1] == {"code": "poi_tag_grafting", "count": 1, "share": 0.25}

    def test_non_defect_conclusions_are_not_unclassified(self) -> None:
        """Overturning an evaluation and an objective no-fault result are successful attribution outputs, and must not fall into unclassified and pollute the primary-root-cause Top."""
        records = [
            _record(
                trace_id="t-fp",
                attribution=AttributionView(
                    status="completed",
                    conclusion_type="judge_false_positive",
                ),
            ),
            _record(
                trace_id="t-na",
                attribution=AttributionView(
                    status="completed",
                    conclusion_type="non_attributable_outcome",
                ),
            ),
            _record(
                trace_id="t-ins",
                attribution=AttributionView(
                    status="completed",
                    conclusion_type="insufficient_evidence",
                    root_cause="未逐项覆盖全部 material failure modes（缺：fact_contradiction）。",
                ),
            ),
        ]

        assert root_cause_label(records[0]) == "judge_false_positive"
        assert root_cause_label(records[1]) == "non_attributable_outcome"
        assert root_cause_label(records[2]) == "unclassified"

        stats = build_attribution_stats(records)
        top_codes = [item["code"] for item in stats["root_cause_top"]]
        assert "judge_false_positive" in top_codes
        assert "non_attributable_outcome" in top_codes
        assert stats["unclassified_breakdown"]["count"] == 1
        assert stats["unclassified_breakdown"]["buckets"] == {
            "incomplete_mode_coverage": 1
        }

    def test_unclassified_breakdown_buckets_parse_failure_and_trace(self) -> None:
        records = [
            _record(
                trace_id="t-parse",
                attribution=AttributionView(
                    status="completed",
                    root_cause="LLM attribution decision failed: 1 validation error for "
                    "BadcaseAttributionDecision",
                ),
            ),
            _record(
                trace_id="t-trace",
                attribution=AttributionView(
                    status="completed",
                    conclusion_type="insufficient_evidence",
                    root_cause="Base Snapshot 状态为 failed_generation，且 fetch_trace_skeleton 失败",
                ),
            ),
        ]

        breakdown = build_attribution_stats(records)["unclassified_breakdown"]

        assert breakdown["count"] == 2
        assert breakdown["buckets"] == {
            "decision_parse_failed": 1,
            "trace_evidence_unavailable": 1,
        }
        buckets_by_case = {
            case["trace_id"]: case["bucket"] for case in breakdown["cases"]
        }
        assert buckets_by_case["t-parse"] == "decision_parse_failed"
        assert buckets_by_case["t-trace"] == "trace_evidence_unavailable"

    def test_top_n_limit(self) -> None:
        records = [
            _record(trace_id=f"t-{i}", attribution=_completed(root_cause_code=f"code_{i}"))
            for i in range(5)
        ]
        assert len(build_attribution_stats(records, top_n=2)["root_cause_top"]) == 2

    def test_missing_root_cause_code_falls_into_unclassified(self) -> None:
        """Old data with no root_cause_code: count it as unclassified, and do not drop it silently."""
        records = [_record(attribution=_completed(root_cause_code=""))]
        assert build_attribution_stats(records)["root_cause_top"][0]["code"] == "unclassified"

    def test_all_root_cause_top_expands_multi_root_cases(self) -> None:
        """One case, multiple root causes: the all-root-causes Top expands the count; the primary-root-cause Top stays the same."""
        multi = _completed(root_cause_code="tool_param_error").model_copy(
            update={"all_root_cause_codes": ["tool_param_error", "cmd_missing"]}
        )
        records = [
            _record(trace_id="t-1", attribution=multi),
            _record(trace_id="t-2", attribution=_completed(root_cause_code="cmd_missing")),
        ]

        stats = build_attribution_stats(records)

        primary = {item["code"]: item["count"] for item in stats["root_cause_top"]}
        assert primary == {"tool_param_error": 1, "cmd_missing": 1}
        all_codes = {item["code"]: item["count"] for item in stats["all_root_cause_top"]}
        assert all_codes == {"cmd_missing": 2, "tool_param_error": 1}

    def test_all_root_cause_top_falls_back_to_primary_for_legacy_data(self) -> None:
        """An old bucket with no all_root_cause_codes: fall back to the primary root cause, matching the primary-root-cause Top."""
        records = [_record(attribution=_completed(root_cause_code="cmd_missing"))]

        stats = build_attribution_stats(records)

        assert stats["all_root_cause_top"] == stats["root_cause_top"]

    def test_owner_fallback_rate(self) -> None:
        """Hit rate of the RemoteConfig `vehicle-agent/*` P2 fallback — a high rate means owner_mapping is missing config."""
        records = [
            _record(trace_id="t-1", attribution=_completed(owner_group="vehicle-agent/*")),
            _record(trace_id="t-2", attribution=_completed(owner_group="prompt-team")),
        ]
        assert build_attribution_stats(records)["owner_fallback_rate"] == 0.5

    def test_root_cause_cases_indexes_by_all_root_causes(self) -> None:
        """root_cause_cases is indexed by every root cause, so L2 can fill traces back in code (not through the LLM)."""
        multi = _completed(root_cause_code="poi_tag_grafting").model_copy(
            update={"all_root_cause_codes": ["poi_tag_grafting", "info_insufficient_code"]}
        )
        records = [
            _record(trace_id="t-1", case_id="c-1", attribution=multi),
            _record(
                trace_id="t-2",
                case_id="c-2",
                attribution=_completed(root_cause_code="poi_tag_grafting"),
            ),
        ]

        root_cause_cases = build_attribution_stats(records)["root_cause_cases"]

        assert root_cause_cases["poi_tag_grafting"]["count"] == 2
        assert {c["trace_id"] for c in root_cause_cases["poi_tag_grafting"]["cases"]} == {
            "t-1",
            "t-2",
        }
        assert root_cause_cases["info_insufficient_code"]["count"] == 1
        assert root_cause_cases["info_insufficient_code"]["cases"][0]["trace_id"] == "t-1"

    def test_overturn_stats_by_type(self) -> None:
        """Three counter-evidence kinds are listed separately — eval_rule_issue clustering means the evaluation rule itself has a bug."""
        records = [
            _record(
                trace_id="t-1",
                attribution=_completed(conclusion_type="judge_false_positive", corrected="good"),
            ),
            _record(
                trace_id="t-2",
                attribution=_completed(conclusion_type="eval_rule_issue", corrected="watch"),
            ),
            _record(trace_id="t-3", attribution=_completed(conclusion_type="agent_badcase")),
        ]

        overturn = build_attribution_stats(records)["overturn"]

        assert overturn["overturn_count"] == 2
        assert overturn["overturn_rate"] == round(2 / 3, 4)
        assert overturn["by_type"] == {
            "judge_false_positive": 1,
            "eval_rule_issue": 1,
        }
        assert overturn["corrected_triage_count"] == 2
        assert overturn["corrected_levels"] == {"good": 1, "watch": 1}
        assert {c["trace_id"] for c in overturn["cases"]} == {"t-1", "t-2"}

    def test_confidence_distribution(self) -> None:
        records = [
            _record(trace_id="t-1", attribution=_completed(confidence=0.95)),
            _record(trace_id="t-2", attribution=_completed(confidence=0.8)),
            _record(trace_id="t-3", attribution=_completed(confidence=0.6)),
            _record(trace_id="t-4", attribution=_completed(confidence=0.3)),
        ]

        conf = build_attribution_stats(records)["confidence"]

        assert conf["counted"] == 4
        assert conf["high"] == 2  # >= 0.8
        assert conf["mid"] == 1  # 0.5 ~ 0.8
        assert conf["low"] == 1  # < 0.5
        assert conf["avg"] == round((0.95 + 0.8 + 0.6 + 0.3) / 4, 4)
        assert conf["low_rate"] == 0.25

    def test_evidence_rate(self) -> None:
        records = [
            _record(trace_id="t-1", attribution=_completed(evidence=True)),
            _record(trace_id="t-2", attribution=_completed(evidence=False)),
        ]
        evidence = build_attribution_stats(records)["evidence"]
        assert evidence == {"with_evidence": 1, "without_evidence": 1, "evidence_rate": 0.5}

    def test_human_review_list(self) -> None:
        records = [
            _record(trace_id="t-1", score=0.0, attribution=_completed(needs_review=True)),
            _record(trace_id="t-2", attribution=_completed(needs_review=False)),
        ]

        review = build_attribution_stats(records)["human_review"]

        assert review["count"] == 1
        assert review["cases"][0]["trace_id"] == "t-1"
        assert review["cases"][0]["score"] == 0.0

    def test_pending_backlog_lists_cases(self) -> None:
        """Attribution backlog: a case whose acceptance is not finished must be enumerable, and must not be dropped silently."""
        records = [
            _record(
                trace_id="t-1",
                score=0.0,
                triage_level="severe_bad",
                attribution=AttributionView(status="pending"),
            ),
            _record(trace_id="t-2", attribution=_completed()),
        ]

        backlog = build_attribution_stats(records)["pending_backlog"]

        assert backlog["count"] == 1
        assert backlog["cases"][0]["trace_id"] == "t-1"
        assert backlog["cases"][0]["triage_level"] == "severe_bad"

    def test_empty_input_is_safe(self) -> None:
        stats = build_attribution_stats([])
        assert stats["should_attribute"] == 0
        assert stats["attribution_coverage"] == 0.0
        assert stats["root_cause_top"] == []
        assert stats["all_root_cause_top"] == []
        assert stats["overturn"]["overturn_rate"] == 0.0


def _perf(
    *,
    buckets: dict[str, BucketPerf] | None = None,
    models: dict[str, BucketPerf] | None = None,
    e2e: float | None = 1000.0,
    first_text: float | None = 500.0,
    error: str = "",
    route_mode: str = "",
) -> CasePerf:
    return CasePerf(
        buckets=buckets or {},
        models=models or {},
        e2e_ms=e2e,
        first_text_ms=first_text,
        error=error,
        route_mode=route_mode,
    )


class TestPerformanceStats:
    def test_percentiles_of_e2e(self) -> None:
        records = [
            _record(trace_id=f"t-{i}", perf=_perf(e2e=float(v)))
            for i, v in enumerate([100, 200, 300, 400, 1000])
        ]

        e2e = build_performance_stats(records)["e2e_ms"]

        assert e2e["count"] == 5
        assert e2e["avg"] == 400.0
        assert e2e["p50"] == 300.0
        # Linear interpolation: position = 4 * 0.95 = 3.8 → 400 + 0.8*(1000-400) = 880
        assert e2e["p95"] == 880.0
        assert e2e["max"] == 1000.0

    def test_single_value_percentile(self) -> None:
        stats = build_performance_stats([_record(perf=_perf(e2e=250.0))])
        assert stats["e2e_ms"]["p50"] == 250.0
        assert stats["e2e_ms"]["p99"] == 250.0

    def test_bucket_stats_uses_per_bucket_denominator(self) -> None:
        """A case that did not trigger a bucket is not in that bucket's denominator, so it does not pull the mean down."""
        records = [
            _record(
                trace_id="t-1",
                perf=_perf(
                    buckets={
                        "react": BucketPerf(
                            calls=2,
                            prompt_tokens=100,
                            completion_tokens=20,
                            total_tokens=120,
                            latency_ms=2000.0,
                            cost=0.01,
                        ),
                    }
                ),
            ),
            _record(
                trace_id="t-2",
                perf=_perf(
                    buckets={
                        "react": BucketPerf(
                            calls=1,
                            prompt_tokens=50,
                            completion_tokens=10,
                            total_tokens=60,
                            latency_ms=500.0,
                            cost=0.005,
                        ),
                        "verify": BucketPerf(
                            calls=1,
                            prompt_tokens=8,
                            completion_tokens=1,
                            total_tokens=9,
                            latency_ms=50.0,
                        ),
                    }
                ),
            ),
        ]

        buckets = build_performance_stats(records)["buckets"]

        assert buckets["react"]["cases"] == 2
        assert buckets["react"]["calls"] == {"avg": 1.5, "total": 3}
        assert buckets["react"]["prompt_tokens"]["avg"] == 75.0
        # Per-call mean = Σtokens / Σcalls = (100+50) / (2+1) = 50, not the per-case mean 75
        assert buckets["react"]["prompt_tokens"]["avg_per_call"] == 50.0
        # Per-call mean latency: t-1 = 2000/2 = 1000, t-2 = 500/1 = 500 → mean 750
        assert buckets["react"]["latency_ms"]["avg"] == 750.0
        assert buckets["react"]["cost"]["total"] == 0.015
        # verify has data for only 1 case
        assert buckets["verify"]["cases"] == 1
        # cgi/summary have no data
        assert buckets["cgi"] == {"cases": 0}

    def test_model_stats_use_actual_model_denominator(self) -> None:
        records = [
            _record(
                trace_id="t-1",
                perf=_perf(
                    models={"qwen-max": BucketPerf(calls=2, total_tokens=120, latency_ms=2000)}
                ),
            ),
            _record(
                trace_id="t-2",
                perf=_perf(
                    models={"qwen-max": BucketPerf(calls=1, total_tokens=60, latency_ms=500)}
                ),
            ),
            _record(
                trace_id="t-3",
                perf=_perf(
                    models={"qwen-plus": BucketPerf(calls=1, total_tokens=30, latency_ms=300)}
                ),
            ),
        ]

        models = build_performance_stats(records)["models"]

        assert models["qwen-max"]["cases"] == 2
        assert models["qwen-max"]["calls"] == {"avg": 1.5, "total": 3}
        assert models["qwen-max"]["total_tokens"]["avg"] == 90.0
        assert models["qwen-max"]["latency_ms"]["avg"] == 750.0
        assert models["qwen-plus"]["cases"] == 1

    def test_model_allowlist_limits_model_token_stats(self) -> None:
        records = [
            _record(
                perf=_perf(
                    models={
                        "vui-model": BucketPerf(calls=1, total_tokens=120, latency_ms=300),
                        "other-model": BucketPerf(calls=1, total_tokens=80, latency_ms=200),
                    }
                )
            )
        ]

        models = build_performance_stats(
            records,
            model_allowlist=frozenset({"vui-model"}),
        )["models"]

        assert set(models) == {"vui-model"}
        assert models["vui-model"]["total_tokens"]["avg"] == 120.0

    def test_failed_perf_excluded_from_stats(self) -> None:
        """A collection failure does not enter the mean (otherwise a 0 pulls the number down), but it counts as missing and the reason is labeled."""
        records = [
            _record(trace_id="t-ok", perf=_perf(e2e=1000.0)),
            _record(trace_id="t-timeout", perf=_perf(error="timeout")),
            _record(trace_id="t-notfound", perf=_perf(error="trace_not_found")),
            _record(trace_id="t-none", perf=None),
        ]

        stats = build_performance_stats(records)

        assert stats["collected"] == 1
        assert stats["missing"] == 3
        assert stats["missing_reasons"] == {
            "timeout": 1,
            "trace_not_found": 1,
            "not_collected": 1,
        }
        assert stats["e2e_ms"]["avg"] == 1000.0

    def test_none_latency_not_counted(self) -> None:
        records = [
            _record(trace_id="t-1", perf=_perf(e2e=None, first_text=None)),
            _record(trace_id="t-2", perf=_perf(e2e=500.0, first_text=200.0)),
        ]
        stats = build_performance_stats(records)
        assert stats["e2e_ms"]["count"] == 1
        assert stats["first_text_ms"]["count"] == 1

    def test_totals_across_buckets(self) -> None:
        records = [
            _record(
                perf=_perf(
                    buckets={
                        "react": BucketPerf(calls=1, total_tokens=100, cost=0.01),
                        "summary": BucketPerf(calls=1, total_tokens=50, cost=0.005),
                    }
                )
            ),
        ]
        stats = build_performance_stats(records)
        assert stats["total_tokens"] == 150
        assert stats["total_tokens_per_case"]["avg"] == 150.0
        assert stats["total_cost"] == 0.015

    def test_route_modes_and_notification_workitems_are_deduplicated(self) -> None:
        records = [
            _record(perf=_perf(route_mode="lite_sdm"), notification_workitem_id="AONE-1"),
            _record(trace_id="t-2", perf=_perf(route_mode="lite_dag"), notification_workitem_id="AONE-1"),
            _record(trace_id="t-3", perf=_perf(route_mode="standard"), notification_workitem_id="AONE-2"),
            _record(trace_id="t-4", perf=_perf()),
        ]

        stats = build_performance_stats(records)

        assert stats["route_modes"] == {
            "classified": 3,
            "unclassified": 1,
            "modes": {
                "lite_sdm": {"count": 1, "share": 0.3333},
                "lite_dag": {"count": 1, "share": 0.3333},
                "standard": {"count": 1, "share": 0.3333},
            },
        }
        assert stats["aone_defects"] == {"unique_workitems": 2, "related_cases": 3}

    def test_empty_input_is_safe(self) -> None:
        stats = build_performance_stats([])
        assert stats["collected"] == 0
        assert stats["e2e_ms"] == {"count": 0}
        assert stats["buckets"]["react"] == {"cases": 0}


class TestCrosstabStats:
    def test_checker_root_cause_gap(self) -> None:
        """A checker hit with no root_cause_code on the attribution → the ruled root-cause mapping is missing."""
        records = [
            _record(
                trace_id="t-1",
                checkers=[("poi_checker", "fail")],
                attribution=_completed(root_cause_code=""),
            ),
            _record(
                trace_id="t-2",
                checkers=[("poi_checker", "fail")],
                attribution=_completed(root_cause_code=""),
            ),
            _record(
                trace_id="t-3",
                checkers=[("cmd_checker", "fail")],
                attribution=_completed(root_cause_code="cmd_tag_missing"),
            ),
        ]

        gap = build_crosstab_stats(records)["checker_root_cause_gap"]
        by_name = {row["checker"]: row for row in gap}

        assert by_name["poi_checker"]["unmapped_rate"] == 1.0
        assert by_name["cmd_checker"]["unmapped_rate"] == 0.0

    def test_checker_root_cause_gap_excludes_untriaged_cases(self) -> None:
        records = [
            _record(checkers=[("poi_checker", "fail")]),
            _record(
                trace_id="t-pending",
                checkers=[("cmd_checker", "fail")],
                attribution=AttributionView(status="pending"),
            ),
        ]

        gap = build_crosstab_stats(records)["checker_root_cause_gap"]

        assert gap == [{"checker": "cmd_checker", "hits": 1, "unmapped": 1, "unmapped_rate": 1.0}]

    def test_dimension_x_owner_group(self) -> None:
        records = [
            _record(
                trace_id="t-1",
                dimensions=[("D4", "fail")],
                attribution=_completed(owner_group="prompt-team"),
            ),
            _record(
                trace_id="t-2",
                dimensions=[("D4", "partial")],
                attribution=_completed(owner_group="prompt-team"),
            ),
            _record(
                trace_id="t-3",
                dimensions=[("D3", "fail")],
                attribution=_completed(owner_group="tool-team"),
            ),
        ]

        rows = build_crosstab_stats(records)["dimension_x_owner_group"]
        by_dim = {row["dimension"]: row for row in rows}

        assert by_dim["D4"]["owners"] == {"prompt-team": 2}
        assert by_dim["D3"]["owners"] == {"tool-team": 1}
        # Ordered as D0-D6
        assert [row["dimension"] for row in rows] == ["D3", "D4"]

    def test_suite_overturn_rate(self) -> None:
        records = [
            _record(
                trace_id="t-1",
                suite="车机-Max",
                score=0.0,
                attribution=_completed(conclusion_type="eval_rule_issue"),
            ),
            _record(
                trace_id="t-2",
                suite="车机-Max",
                score=1.0,
                attribution=_completed(conclusion_type="agent_badcase"),
            ),
            _record(trace_id="t-3", suite="车机-Standard", score=2.0),
        ]

        rows = build_crosstab_stats(records)["suite_overturn"]
        by_suite = {row["suite"]: row for row in rows}

        assert by_suite["车机-Max"]["overturned"] == 1
        assert by_suite["车机-Max"]["overturn_rate"] == 0.5
        assert by_suite["车机-Max"]["low_score"] == 2
        assert by_suite["车机-Standard"]["overturned"] == 0

    def test_suite_low_score_uses_effective_score(self) -> None:
        record = _record(
            score=0.0,
            attribution=_completed(corrected="good"),
        )

        rows = build_crosstab_stats([record])["suite_overturn"]

        assert rows[0]["low_score"] == 0

    def test_empty_input_is_safe(self) -> None:
        stats = build_crosstab_stats([])
        assert stats["checker_root_cause_gap"] == []
        assert stats["dimension_x_owner_group"] == []
        assert stats["suite_overturn"] == []

    def test_no_failure_mode_root_cause_matrix(self) -> None:
        """The co-occurrence matrix was removed in T5: crosstab no longer emits failure_mode_x_root_cause."""
        records = [
            _record(
                trace_id="t-1",
                failure_modes=["cmd_tag_missing"],
                attribution=_completed(root_cause_code="cmd_tag_missing"),
            ),
        ]
        assert "failure_mode_x_root_cause" not in build_crosstab_stats(records)


class TestObservationStats:
    """T34 envelope observation splits: Metric over limit / Diagnostic hits / business, scenario, and Profile versions.

    Independent observations are only split and do not enter the effectiveness score (design §5.3). A historical batch
    with no envelope rows must return an empty dict; the render layer skips the block, and the report matches the baseline.
    """

    def test_empty_when_no_envelope_observations(self) -> None:
        records = [_record(trace_id="t-1"), _record(trace_id="t-2", case_id="c-2")]

        assert build_observation_stats(records) == {}

    def test_empty_input_is_safe(self) -> None:
        assert build_observation_stats([]) == {}

    def test_metric_breakdown_exceeded_rate(self) -> None:
        records = [
            _record(metrics=[{"metric_key": "first_text_ms", "status": "exceeded"}]),
            _record(
                trace_id="t-2",
                case_id="c-2",
                metrics=[
                    {"metric_key": "first_text_ms", "status": "within_target"},
                    {"metric_key": "e2e_ms", "status": "within_target"},
                ],
            ),
        ]

        stats = build_observation_stats(records)

        assert stats["metric_observations"] == {
            "e2e_ms": {"observed": 1, "exceeded": 0, "exceeded_rate": 0.0},
            "first_text_ms": {"observed": 2, "exceeded": 1, "exceeded_rate": 0.5},
        }

    def test_diagnostic_breakdown_by_key_and_severity(self) -> None:
        records = [
            _record(
                diagnostics=[
                    {"finding_key": "llm_retry", "severity": "warning"},
                    {"finding_key": "loop_guard", "severity": "error"},
                ]
            ),
            _record(
                trace_id="t-2",
                case_id="c-2",
                diagnostics=[{"finding_key": "llm_retry", "severity": "warning"}],
            ),
        ]

        stats = build_observation_stats(records)

        assert stats["diagnostic_findings"] == {
            "by_finding_key": {"llm_retry": 2, "loop_guard": 1},
            "by_severity": {"warning": 2, "error": 1},
        }

    def test_profile_breakdown_by_identity_keys(self) -> None:
        records = [
            _record(
                business_id="vehicle_agent",
                scenario="navigation",
                profile_id="vehicle_agent.standard",
                profile_version="v1",
            ),
            _record(
                trace_id="t-2",
                case_id="c-2",
                business_id="vehicle_agent",
                scenario="multi_turn",
                profile_id="vehicle_agent.standard",
                profile_version="v2",
            ),
        ]

        profiles = build_observation_stats(records)["profiles"]

        assert profiles["business_ids"] == {"vehicle_agent": 2}
        assert profiles["scenarios"] == {"navigation": 1, "multi_turn": 1}
        assert profiles["profiles"] == {
            "vehicle_agent.standard@v1": 1,
            "vehicle_agent.standard@v2": 1,
        }

    def test_only_nonempty_blocks_are_present(self) -> None:
        """When there are only diagnostic observations: only the diagnostic_findings key appears, and the other blocks stay at their defaults."""
        records = [
            _record(diagnostics=[{"finding_key": "llm_retry", "severity": "warning"}]),
        ]

        stats = build_observation_stats(records)

        assert set(stats) == {"diagnostic_findings"}
