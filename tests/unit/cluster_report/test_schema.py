"""T10 unit tests: the clustering-report contract (two case_key paths, coverage, and three-bucket constant alignment)."""

from benchmark.reporting.cluster.constants import (
    BATCH_TAG_RE,
    MODEL_BUCKETS,
    OBSERVATION_NAME_TO_BUCKET,
    REPORT_SCHEMA_VERSION,
)
from benchmark.reporting.cluster.schema import (
    AttributionView,
    BatchCaseRecord,
    CaseKey,
    ClusterReport,
    Coverage,
)


class TestCaseKey:
    def test_prefers_case_id_path(self) -> None:
        key = CaseKey(suite="车机-Standard", case_id="standard-10", tag="chat", query="导航回家")
        assert key.normalized() == ("id", "车机-Standard", "standard-10", "")

    def test_falls_back_to_query_path(self) -> None:
        key = CaseKey(suite="车机-Standard", tag="chat", query="导航回家")
        assert key.normalized() == ("q", "车机-Standard", "chat", "导航回家", "")

    def test_turn_index_disambiguates_multi_turn(self) -> None:
        """When a short query repeats across turns of the same scenario, turn_index keeps the keys from colliding."""
        k1 = CaseKey(suite="车机-多轮", tag="multi-turn.s1", query="取消", turn_index="Q2")
        k2 = CaseKey(suite="车机-多轮", tag="multi-turn.s1", query="取消", turn_index="Q4")
        assert k1.normalized() != k2.normalized()

    def test_single_turn_normalizes_empty_turn_index(self) -> None:
        key = CaseKey(suite="车机-Standard", case_id="standard-1")
        assert key.normalized()[-1] == ""


class TestAttributionView:
    def test_three_state_helpers(self) -> None:
        assert AttributionView().status == "none"
        assert AttributionView(status="pending").is_pending
        assert AttributionView(status="completed", root_cause="x").is_completed

    def test_overturn_detection(self) -> None:
        assert AttributionView(conclusion_type="judge_false_positive").is_overturn
        assert AttributionView(conclusion_type="eval_rule_issue").is_overturn
        assert AttributionView(conclusion_type="dataset_issue").is_overturn
        # agent_badcase is a real defect; a no-fault result does not change the score. Neither counts as overturning the evaluation.
        assert not AttributionView(conclusion_type="agent_badcase").is_overturn
        assert not AttributionView(conclusion_type="non_attributable_outcome").is_overturn
        assert not AttributionView().is_overturn

    def test_effective_root_cause_codes_fallback(self) -> None:
        """The new field wins; old data falls back to a single primary-root-cause list; if both are missing, an empty list."""
        view = AttributionView(
            root_cause_code="x", all_root_cause_codes=["a", "b"]
        )
        assert view.effective_root_cause_codes == ["a", "b"]
        assert AttributionView(root_cause_code="x").effective_root_cause_codes == ["x"]
        view = AttributionView(
            root_cause_code="tool_call_dispatch_dropped",
            root_cause="工具调用调度丢失：模型 call-1 未进入 execute_tools。",
        )
        assert view.root_cause.startswith("工具调用调度丢失：")
        assert AttributionView().effective_root_cause_codes == []


class TestCoverage:
    def test_rates(self) -> None:
        cov = Coverage(expected_total=100, judged=98, should_attribute=23, attributed=21, pending=2)
        assert cov.eval_rate == 0.98
        assert cov.attribution_rate == 21 / 23

    def test_zero_denominator_is_safe(self) -> None:
        cov = Coverage()
        assert cov.eval_rate == 0.0
        assert cov.attribution_rate == 0.0


class TestBatchCaseRecord:
    def test_suite_from_case_meta(self) -> None:
        rec = BatchCaseRecord(case_meta={"suite": "车机-Max", "case_id": "max-3"})
        assert rec.suite == "车机-Max"

    def test_missing_case_meta_tolerated(self) -> None:
        """Live traffic / old data with no case_meta: degrade to empty, and do not raise."""
        rec = BatchCaseRecord(trace_id="t-1")
        assert rec.suite == ""
        assert rec.case_meta == {}
        assert rec.attribution.status == "none"
        assert rec.perf is None

    def test_open_case_meta_passthrough(self) -> None:
        """Open envelope: unknown fields are kept as-is, so a new case source needs no contract change."""
        meta = {"source": "sampled", "suite": "pool-A", "case_id": "s-1", "sample_weight": 0.3}
        rec = BatchCaseRecord(case_meta=meta)
        assert rec.case_meta["sample_weight"] == 0.3

    def test_score_views_keep_original_and_apply_attribution_correction(self) -> None:
        plain = BatchCaseRecord(score=1.0)
        corrected = BatchCaseRecord(
            score=0.0,
            attribution=AttributionView(corrected_triage_level="good"),
        )
        unknown = BatchCaseRecord(
            score=1.0,
            attribution=AttributionView(corrected_triage_level="no_decision"),
        )

        assert (plain.judge_score, plain.attribution_score, plain.effective_score) == (
            1.0,
            None,
            1.0,
        )
        assert (corrected.judge_score, corrected.attribution_score, corrected.effective_score) == (
            0.0,
            2.0,
            2.0,
        )
        assert unknown.attribution_score is None
        assert unknown.effective_score == 1.0

    def test_judge_score_backfills_legacy_score(self) -> None:
        rec = BatchCaseRecord(judge_score=1.0)

        assert (rec.score, rec.judge_score, rec.effective_score) == (1.0, 1.0, 1.0)

    def test_serialization_roundtrip(self) -> None:
        rec = BatchCaseRecord(
            trace_id="t-1",
            score=1.0,
            case_key=CaseKey(suite="s", case_id="c"),
            attribution=AttributionView(status="completed", root_cause="幻觉"),
        )
        dumped = rec.model_dump(mode="json")
        assert BatchCaseRecord.model_validate(dumped).attribution.root_cause == "幻觉"

    def test_trace_meta_is_not_serialized(self) -> None:
        rec = BatchCaseRecord(trace_meta={"query": "导航回家", "response": "已规划路线"})

        assert "trace_meta" not in rec.model_dump(mode="json")


class TestClusterReport:
    def test_default_schema_version(self) -> None:
        assert ClusterReport().schema_version == REPORT_SCHEMA_VERSION == "cluster_report.v2"

    def test_json_serializable(self) -> None:
        report = ClusterReport(batch_tag="run-20260729-153000", cases=[BatchCaseRecord()])
        dumped = report.model_dump(mode="json")
        assert dumped["batch_tag"] == "run-20260729-153000"
        assert len(dumped["cases"]) == 1


class TestConstants:
    def test_batch_tag_regex_captures_date_time(self) -> None:
        m = BATCH_TAG_RE.match("run-20260729-153000")
        assert m and m.group(1) == "20260729" and m.group(2) == "153000"

    def test_batch_tag_regex_allows_suffix(self) -> None:
        m = BATCH_TAG_RE.match("run-20260901-134400-whitebox-v3")
        assert m and m.group(1) == "20260901" and m.group(2) == "134400"

    def test_batch_tag_regex_rejects_malformed(self) -> None:
        for bad in ["run-2026729-153000", "run-20260729", "x-run-20260729-153000"]:
            assert BATCH_TAG_RE.match(bad) is None, bad

    def test_bucket_map_values_are_known_buckets(self) -> None:
        assert set(OBSERVATION_NAME_TO_BUCKET.values()) <= set(MODEL_BUCKETS)
