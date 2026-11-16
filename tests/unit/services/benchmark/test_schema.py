"""Unit tests for benchmark.runs.schema — data models and validation."""

import pytest
from benchmark.runs.schema import (
    AggregatedMetrics,
    AggregationConfig,
    AggregationMethod,
    AttributionStats,
    BenchmarkRequest,
    BenchmarkRunCreatedResponse,
    BenchmarkRunResponse,
    BenchmarkRunResult,
    BenchmarkStatus,
    Criterion,
    CriterionStats,
    DatasetRef,
    JudgeCallback,
    TriageMode,
)
from pydantic import ValidationError

# ── Criterion (frozen dataclass) ──────────────────────────────────────────


class TestCriterion:
    def test_valid_criterion(self) -> None:
        c = Criterion(
            id="factual_accuracy",
            name="事实准确性",
            description="Check if facts are correct",
            weight=0.3,
            evaluator_ref="amap_agent",
        )
        assert c.id == "factual_accuracy"
        assert c.weight == 0.3
        assert c.evaluator_ref == "amap_agent"

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="id must not be empty"):
            Criterion(
                id="",
                name="test",
                description="d",
                weight=1.0,
                evaluator_ref="ev",
            )

    def test_empty_evaluator_ref_raises(self) -> None:
        with pytest.raises(ValueError, match="evaluator_ref must not be empty"):
            Criterion(
                id="c1",
                name="test",
                description="d",
                weight=1.0,
                evaluator_ref="",
            )

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight must be >= 0"):
            Criterion(
                id="c1",
                name="test",
                description="d",
                weight=-0.5,
                evaluator_ref="ev",
            )

    def test_frozen(self) -> None:
        c = Criterion(id="c1", name="t", description="d", weight=1.0, evaluator_ref="ev")
        with pytest.raises(AttributeError):
            c.weight = 0.5  # type: ignore[misc]

    def test_same_evaluator_ref_reused(self) -> None:
        """Two criteria can share the same evaluator_ref (key = criterion.id)."""
        c1 = Criterion(
            id="accuracy_v1",
            name="acc",
            description="d",
            weight=0.5,
            evaluator_ref="amap_agent",
        )
        c2 = Criterion(
            id="accuracy_v2",
            name="acc2",
            description="d",
            weight=0.5,
            evaluator_ref="amap_agent",
        )
        assert c1.evaluator_ref == c2.evaluator_ref
        assert c1.id != c2.id


# ── AggregationConfig (frozen dataclass) ──────────────────────────────────


class TestAggregationConfig:
    def test_weighted_avg(self) -> None:
        agg = AggregationConfig(method="weighted_avg")
        assert agg.method == "weighted_avg"
        assert agg.pass_threshold == 0.8
        assert agg.custom_fn_ref is None

    def test_all_pass(self) -> None:
        agg = AggregationConfig(method="all_pass", pass_threshold=0.7)
        assert agg.method == "all_pass"
        assert agg.pass_threshold == 0.7

    def test_majority_vote(self) -> None:
        agg = AggregationConfig(method="majority_vote")
        assert agg.method == "majority_vote"

    def test_custom_requires_fn_ref(self) -> None:
        with pytest.raises(ValueError, match="custom_fn_ref is required"):
            AggregationConfig(method="custom")

    def test_custom_with_fn_ref(self) -> None:
        agg = AggregationConfig(
            method="custom", custom_fn_ref="benchmark.runs.example:aggregate"
        )
        assert agg.method == "custom"
        assert agg.custom_fn_ref == "benchmark.runs.example:aggregate"

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            AggregationConfig(method="bogus")

    def test_pass_threshold_out_of_range(self) -> None:
        # Thresholds use the same scale as business scores (not a normalized assumption; ieval's full score is 2): >1 is legal
        assert AggregationConfig(method="weighted_avg", pass_threshold=1.5).pass_threshold == 1.5
        with pytest.raises(ValueError, match="pass_threshold must be"):
            AggregationConfig(method="weighted_avg", pass_threshold=-0.1)

    def test_frozen(self) -> None:
        agg = AggregationConfig(method="weighted_avg")
        with pytest.raises(AttributeError):
            agg.method = "all_pass"  # type: ignore[misc]


# ── DatasetRef (frozen dataclass) ──────────────────────────────────────────


class TestDatasetRef:
    def test_defaults(self) -> None:
        ref = DatasetRef(dataset_name="ds", project_id="proj")
        assert ref.dataset_name == "ds"
        assert ref.project_id == "proj"
        assert ref.sample_size is None
        assert ref.sample_seed == 42

    def test_with_sample(self) -> None:
        ref = DatasetRef(dataset_name="ds", project_id="proj", sample_size=100, sample_seed=7)
        assert ref.sample_size == 100
        assert ref.sample_seed == 7

    def test_frozen(self) -> None:
        ref = DatasetRef(dataset_name="ds", project_id="proj")
        with pytest.raises(AttributeError):
            ref.dataset_name = "other"  # type: ignore[misc]


# ── BenchmarkRequest (Pydantic) ────────────────────────────────────────────


class TestBenchmarkRequest:
    def test_defaults(self) -> None:
        req = BenchmarkRequest()
        assert req.sample_size is None
        assert req.sample_seed is None
        assert req.max_concurrency == 10
        assert req.parallel_evaluators is True
        assert req.triage_mode == TriageMode.FUSED

    def test_sample_seed_override(self) -> None:
        req = BenchmarkRequest(sample_seed=99)
        assert req.sample_seed == 99

    def test_per_eval_triage(self) -> None:
        req = BenchmarkRequest(triage_mode=TriageMode.PER_EVAL)
        assert req.triage_mode == TriageMode.PER_EVAL

    def test_none_triage(self) -> None:
        req = BenchmarkRequest(triage_mode=TriageMode.NONE)
        assert req.triage_mode == TriageMode.NONE

    def test_sample_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRequest(sample_size=0)

    def test_max_concurrency_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRequest(max_concurrency=0)


# ── CriterionStats (Pydantic) ──────────────────────────────────────────────


class TestCriterionStats:
    def test_valid(self) -> None:
        stats = CriterionStats(
            criterion_id="c1",
            evaluator_ref="amap_agent",
            sample_count=10,
            success_count=8,
            failed_count=2,
            mean=0.75,
            median=0.8,
            p95=0.95,
            min=0.0,
            max=1.0,
            std=0.15,
            pass_rate=0.8,
        )
        assert stats.criterion_id == "c1"
        assert stats.failed_count == 2
        assert stats.mean == 0.75

    def test_serialization(self) -> None:
        stats = CriterionStats(
            criterion_id="c1",
            evaluator_ref="ev",
            sample_count=5,
            success_count=5,
            failed_count=0,
            mean=0.9,
            median=0.9,
            p95=1.0,
            min=0.8,
            max=1.0,
            std=0.05,
            pass_rate=1.0,
        )
        d = stats.model_dump()
        assert d["criterion_id"] == "c1"
        assert d["mean"] == 0.9


# ── AggregatedMetrics ──────────────────────────────────────────────────────


class TestAggregatedMetrics:
    def test_empty(self) -> None:
        m = AggregatedMetrics(
            aggregation_method="weighted_avg",
            total_samples=0,
            total_duration_ms=0,
        )
        assert m.criterion_metrics == {}
        assert m.attribution_metrics == {}
        assert m.react_rounds_summary == {}

    def test_with_data(self) -> None:
        stats = CriterionStats(
            criterion_id="c1",
            evaluator_ref="ev",
            sample_count=5,
            success_count=5,
            failed_count=0,
            mean=0.9,
            median=0.9,
            p95=1.0,
            min=0.8,
            max=1.0,
            std=0.05,
            pass_rate=1.0,
        )
        attr = AttributionStats(
            field_distributions={
                "primary_issue_category": {"intent_error": 4, "tool_error": 1},
            },
            total_attributed=5,
        )
        m = AggregatedMetrics(
            criterion_metrics={"c1": stats},
            attribution_metrics={"c1": attr},
            react_rounds_summary={1: 3, 2: 2},
            aggregation_method="weighted_avg",
            total_samples=5,
            total_duration_ms=12000,
        )
        assert "c1" in m.criterion_metrics
        assert m.criterion_metrics["c1"].mean == 0.9
        assert (
            m.attribution_metrics["c1"].field_distributions["primary_issue_category"][
                "intent_error"
            ]
            == 4
        )
        assert m.attribution_metrics["c1"].total_attributed == 5
        assert m.react_rounds_summary[2] == 2


# ── BenchmarkRunResult / Response ──────────────────────────────────────────


class TestBenchmarkRunResult:
    def test_completed(self) -> None:
        r = BenchmarkRunResult(
            run_id="run-123",
            final_score=0.85,
            aggregation_method="weighted_avg",
            status=BenchmarkStatus.COMPLETED,
        )
        assert r.run_id == "run-123"
        assert r.final_score == 0.85
        assert r.status == BenchmarkStatus.COMPLETED
        assert r.error is None

    def test_failed(self) -> None:
        r = BenchmarkRunResult(
            run_id="run-456",
            status=BenchmarkStatus.FAILED,
            error="sync timeout",
        )
        assert r.final_score is None
        assert r.status == BenchmarkStatus.FAILED
        assert r.error == "sync timeout"


class TestBenchmarkRunResponse:
    def test_sync_success(self) -> None:
        resp = BenchmarkRunResponse(
            run_id="run-1",
            benchmark_name="example_benchmark",
            status=BenchmarkStatus.COMPLETED,
            final_score=0.9,
        )
        assert resp.final_score == 0.9
        assert resp.metrics is None

    def test_async_created(self) -> None:
        resp = BenchmarkRunCreatedResponse(
            run_id="run-2",
            benchmark_name="example_benchmark",
            poll_url="/v1/benchmarks/runs/run-2",
        )
        assert resp.run_id == "run-2"
        assert resp.status == BenchmarkStatus.RUNNING


# ── JudgeCallback ──────────────────────────────────────────────────────────


class TestJudgeCallback:
    def test_preserves_extra_fields_for_perf_metrics(self) -> None:
        """A non-contract field on the judge side (the odps token count) must enter model_dump unchanged."""
        odps = {
            "total_input_tokens": "1200",
            "total_output_tokens": "340",
            "total_tokens": "1540",
        }
        callback = JudgeCallback.model_validate(
            {
                "run_id": "run-1",
                "type": "judge",
                "item_id": "item-1",
                "criterion_scores": {"D1": {"score": 1.5, "reason": "ok"}},
                "odps": odps,
            },
        )
        assert callback.model_dump()["odps"] == odps
        assert callback.is_success


# ── Enums ──────────────────────────────────────────────────────────────────


class TestEnums:
    def test_aggregation_method_values(self) -> None:
        assert AggregationMethod.WEIGHTED_AVG.value == "weighted_avg"
        assert AggregationMethod.ALL_PASS.value == "all_pass"
        assert AggregationMethod.MAJORITY_VOTE.value == "majority_vote"
        assert AggregationMethod.CUSTOM.value == "custom"

    def test_triage_mode_values(self) -> None:
        assert TriageMode.FUSED.value == "fused"
        assert TriageMode.PER_EVAL.value == "per_eval"
        assert TriageMode.NONE.value == "none"

    def test_benchmark_status_values(self) -> None:
        assert BenchmarkStatus.RUNNING.value == "running"
        assert BenchmarkStatus.COMPLETED.value == "completed"
        assert BenchmarkStatus.FAILED.value == "failed"


# ── ExternalBenchmarkRequest: D9 rubric tightening / D5 require_case_id ───────────


class TestExternalBenchmarkRequestGuards:
    def _payload(self, **updates: object) -> dict:

        payload: dict[str, object] = {
            "benchmark_name": "ieval_external",
            "judge_class": "ieval_external_judge",
            "attribution_class": "ieval_external_attribution",
            "rubric_class": "amap_chat_rubric_v2",
        }
        payload.update(updates)
        return payload

    def test_rubric_override_rejected_in_judge_attribution_mode(self) -> None:
        from benchmark.runs.schema import ExternalBenchmarkRequest

        with pytest.raises(ValidationError, match="attribution_only"):
            ExternalBenchmarkRequest.model_validate(
                self._payload(
                    mode="judge_attribution",
                    rubric=[{"id": "c1"}],
                ),
            )

    def test_rubric_override_allowed_in_attribution_only_mode(self) -> None:
        from benchmark.runs.schema import ExternalBenchmarkRequest

        req = ExternalBenchmarkRequest.model_validate(
            self._payload(
                mode="attribution_only",
                rubric=[{"id": "c1"}],
            ),
        )
        assert req.rubric == [{"id": "c1"}]

    def test_rubric_none_is_always_allowed(self) -> None:
        from benchmark.runs.schema import ExternalBenchmarkRequest

        req = ExternalBenchmarkRequest.model_validate(self._payload())
        assert req.rubric is None
        assert req.require_case_id is False

    def test_require_case_id_defaults_false_and_accepts_true(self) -> None:
        from benchmark.runs.schema import ExternalBenchmarkRequest

        req = ExternalBenchmarkRequest.model_validate(
            self._payload(require_case_id=True),
        )
        assert req.require_case_id is True
