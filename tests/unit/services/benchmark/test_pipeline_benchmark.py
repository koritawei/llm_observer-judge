"""Tests for BenchmarkRunner.run_pipeline_benchmark — Kafka retrospective aggregation.

Verifies the no-LLM pipeline: SQL sample → SQL aggregate → AggregationCalculator
→ report generation.  All DB calls are mocked; no real connections needed.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.records.enums import BenchmarkStatus
from benchmark.runs.runner import BenchmarkRunner, BenchmarkRunnerDeps, PipelineRunExec
from benchmark.runs.schema import PipelineBenchmarkRequest
from benchmark.reporting.framework.aggregation_calculator import (
    AggregationCalculator,
    EvaluatorResultEntry,
)
from benchmark.reporting.framework.base import BenchmarkComponentMetadata
from benchmark.reporting.framework.schema import (
    AggregationConfig,
    Criterion,
    DatasetRef,
)

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
            evaluator_ref="amap_chat_judge",
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
def agg_config() -> AggregationConfig:
    return AggregationConfig(method="weighted_avg", pass_threshold=0.8)


@pytest.fixture
def metadata(rubric, agg_config) -> BenchmarkComponentMetadata:
    return BenchmarkComponentMetadata(
        name="test_benchmark",
        service_name="test_service",
        rubric=rubric,
        aggregation=agg_config,
        dataset=DatasetRef(dataset_name="ds", project_id="p1"),
    )


@pytest.fixture
def benchmark_services(metadata) -> MagicMock:
    """Mock BaseBenchmarkServices — passed to run_pipeline_benchmark."""
    bs = MagicMock()
    bs.get_metadata = MagicMock(return_value=metadata)
    bs._build_report_spec = MagicMock(return_value={"benchmark_name": "test"})
    return bs


def _criterion_stats_json(criterion_id: str, evaluator_ref: str) -> dict[str, object]:
    return {
        "criterion_id": criterion_id,
        "evaluator_ref": evaluator_ref,
        "sample_count": 10,
        "success_count": 10,
        "failed_count": 0,
        "mean": 0.85,
        "median": 0.85,
        "p95": 0.9,
        "min": 0.5,
        "max": 1.0,
        "std": 0.1,
        "pass_rate": 0.9,
    }


@pytest.fixture
def db_run() -> SimpleNamespace:
    """Persisted run record: report emission trusts only the DB, so metrics_json / final_score must be complete.

    ``config_json=None`` keeps the production default of "no snapshot → fall back to component metadata"
    for the rubric resolution path (§6.2).
    """
    return SimpleNamespace(
        run_id="test-run",
        benchmark_name="test_benchmark",
        service_name="test_service",
        dataset_name="ds",
        project_id="p1",
        config_json=None,
        final_score=0.82,
        metrics_json={
            "criterion_metrics": {
                "accuracy": _criterion_stats_json("accuracy", "amap_chat_judge"),
                "helpfulness": _criterion_stats_json("helpfulness", "llm_judge"),
            },
            "attribution_metrics": {},
            "react_rounds_summary": {},
            "aggregation_method": "weighted_avg",
            "total_samples": 10,
            "total_duration_ms": 0,
        },
    )


@pytest.fixture
def benchmark_service(db_run) -> MagicMock:
    """Mock BenchmarkService — stored as runner._benchmark_service."""
    svc = MagicMock()
    svc.get_run = AsyncMock(return_value=db_run)
    svc.update_run_status = AsyncMock()
    svc.sample_kafka_pipeline_records = AsyncMock()
    svc.load_kafka_criterion_entries = AsyncMock()
    svc.load_kafka_case_results = AsyncMock(return_value=[])
    svc.aggregate_kafka_attribution = AsyncMock(return_value={})
    svc.load_case_results = AsyncMock(return_value=[])
    svc.attach_daily_report_context = AsyncMock()
    svc.complete_run = AsyncMock()
    svc.fail_run = AsyncMock()
    svc.heartbeat = AsyncMock()
    return svc


@pytest.fixture
def webdev_client() -> MagicMock:
    client = MagicMock()
    client.generate_report = AsyncMock(
        return_value=("<html></html>", "https://oss.example.com/r1.html"),
    )
    return client


@pytest.fixture
def runner(benchmark_service, webdev_client) -> BenchmarkRunner:
    evaluator_service = MagicMock()
    return BenchmarkRunner(
        BenchmarkRunnerDeps(
            evaluator_service=evaluator_service,
            benchmark_service=benchmark_service,
            aggregation_calculator=AggregationCalculator(),
            webdev_client=webdev_client,
        ),
    )


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════


def _entry(
    criterion_id: str,
    evaluator_ref: str,
    *,
    mean: float = 0.85,
    pass_rate: float = 0.9,
    sample_count: int = 10,
) -> EvaluatorResultEntry:
    return EvaluatorResultEntry(
        criterion_id=criterion_id,
        evaluator_ref=evaluator_ref,
        sample_count=sample_count,
        success_count=sample_count,
        failed_count=0,
        criterion_stats_json={
            "mean": mean,
            "median": mean,
            "p95": mean,
            "min": 0.5,
            "max": 1.0,
            "std": 0.1,
            "pass_rate": pass_rate,
        },
    )


def _pipeline_request(**overrides: object) -> PipelineBenchmarkRequest:
    data: dict[str, object] = {
        "date_start": datetime(2025, 1, 1, tzinfo=UTC),
        "date_end": datetime(2025, 1, 31, tzinfo=UTC),
        "sample_size": 100,
        "generate_report": True,
    }
    data.update(overrides)
    return PipelineBenchmarkRequest.model_validate(data)


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════


class TestRunPipelineBenchmarkAggregatesKafkaScores:
    """Verify the happy path: sample → aggregate → report → complete."""

    @pytest.mark.asyncio
    async def test_aggregates_kafka_scores(
        self,
        runner: BenchmarkRunner,
        benchmark_service: MagicMock,
        benchmark_services: MagicMock,
        webdev_client: MagicMock,
    ) -> None:
        trace_ids = [f"trace_{i}" for i in range(100)]
        benchmark_service.sample_kafka_pipeline_records.return_value = trace_ids
        benchmark_service.load_kafka_criterion_entries.return_value = [
            _entry("accuracy", "amap_chat_judge", mean=0.9, pass_rate=0.9),
            _entry("helpfulness", "llm_judge", mean=0.7, pass_rate=0.7),
        ]

        result = await runner.run_pipeline_benchmark(
            PipelineRunExec(
                benchmark_services=benchmark_services,
                run_id="test-run-1",
                request=_pipeline_request(),
                generate_report=True,
            ),
        )

        assert result.status == BenchmarkStatus.COMPLETED
        assert result.run_id == "test-run-1"
        # 0.9 * 0.6 + 0.7 * 0.4 = 0.82
        expected_score = 0.9 * 0.6 + 0.7 * 0.4
        assert pytest.approx(result.final_score) == expected_score
        assert result.report_url == "https://oss.example.com/r1.html"
        assert result.aggregation_method == "weighted_avg"

        # Verify service call sequence
        benchmark_service.sample_kafka_pipeline_records.assert_awaited_once()
        benchmark_service.load_kafka_criterion_entries.assert_awaited_once()
        benchmark_service.complete_run.assert_awaited_once()
        benchmark_service.fail_run.assert_not_awaited()

        # Verify status transitions: RUNNING → AGGREGATING → GENERATING_REPORT
        status_calls = [c.args[1] for c in benchmark_service.update_run_status.call_args_list]
        assert BenchmarkStatus.RUNNING in status_calls
        assert BenchmarkStatus.AGGREGATING in status_calls
        assert BenchmarkStatus.GENERATING_REPORT in status_calls

        # Verify webdev client was called for report
        webdev_client.generate_report.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sample_params_passed_correctly(
        self,
        runner: BenchmarkRunner,
        benchmark_service: MagicMock,
        benchmark_services: MagicMock,
    ) -> None:
        benchmark_service.sample_kafka_pipeline_records.return_value = ["t1", "t2"]
        benchmark_service.load_kafka_criterion_entries.return_value = [
            _entry("accuracy", "amap_chat_judge"),
            _entry("helpfulness", "llm_judge"),
        ]

        req = _pipeline_request(sample_size=50, generate_report=False)
        await runner.run_pipeline_benchmark(
            PipelineRunExec(
                benchmark_services=benchmark_services,
                run_id="test-run-2",
                request=req,
                generate_report=False,
            ),
        )

        sample_query = benchmark_service.sample_kafka_pipeline_records.await_args.args[0]
        assert sample_query.date_start == req.date_start
        assert sample_query.date_end == req.date_end
        assert sample_query.sample_size == 50
        assert "amap_chat_judge" in sample_query.evaluator_names
        assert "llm_judge" in sample_query.evaluator_names


class TestRunPipelineBenchmarkNoData:
    """When SQL returns zero trace_ids, the run should fail gracefully."""

    @pytest.mark.asyncio
    async def test_no_data_returns_failed(
        self,
        runner: BenchmarkRunner,
        benchmark_service: MagicMock,
        benchmark_services: MagicMock,
    ) -> None:
        benchmark_service.sample_kafka_pipeline_records.return_value = []

        result = await runner.run_pipeline_benchmark(
            PipelineRunExec(
                benchmark_services=benchmark_services,
                run_id="test-run-empty",
                request=_pipeline_request(),
                generate_report=True,
            ),
        )

        assert result.status == BenchmarkStatus.FAILED
        assert "No Kafka records" in (result.error or "")
        benchmark_service.fail_run.assert_awaited_once()
        # Should NOT reach aggregation or completion
        benchmark_service.load_kafka_criterion_entries.assert_not_awaited()
        benchmark_service.complete_run.assert_not_awaited()


class TestRunPipelineBenchmarkReportGeneration:
    """Report generation toggle: True → webdev called; False → skipped."""

    @pytest.mark.asyncio
    async def test_report_generated_when_requested(
        self,
        runner: BenchmarkRunner,
        benchmark_service: MagicMock,
        benchmark_services: MagicMock,
        webdev_client: MagicMock,
    ) -> None:
        trace_ids = [f"trace_{i}" for i in range(5)]
        benchmark_service.sample_kafka_pipeline_records.return_value = trace_ids
        benchmark_service.load_kafka_criterion_entries.return_value = [
            _entry("accuracy", "amap_chat_judge", mean=0.9, pass_rate=0.9),
            _entry("helpfulness", "llm_judge", mean=0.85, pass_rate=0.85),
        ]

        result = await runner.run_pipeline_benchmark(
            PipelineRunExec(
                benchmark_services=benchmark_services,
                run_id="test-run-report",
                request=_pipeline_request(),
                generate_report=True,
            ),
        )

        assert result.status == BenchmarkStatus.COMPLETED
        assert result.report_url == "https://oss.example.com/r1.html"
        webdev_client.generate_report.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_report_skipped_when_not_requested(
        self,
        runner: BenchmarkRunner,
        benchmark_service: MagicMock,
        benchmark_services: MagicMock,
        webdev_client: MagicMock,
    ) -> None:
        trace_ids = [f"trace_{i}" for i in range(5)]
        benchmark_service.sample_kafka_pipeline_records.return_value = trace_ids
        benchmark_service.load_kafka_criterion_entries.return_value = [
            _entry("accuracy", "amap_chat_judge", mean=0.9, pass_rate=0.9),
            _entry("helpfulness", "llm_judge", mean=0.85, pass_rate=0.85),
        ]

        result = await runner.run_pipeline_benchmark(
            PipelineRunExec(
                benchmark_services=benchmark_services,
                run_id="test-run-no-report",
                request=_pipeline_request(),
                generate_report=False,
            ),
        )

        assert result.status == BenchmarkStatus.COMPLETED
        assert result.report_url is None
        webdev_client.generate_report.assert_not_awaited()


class TestRunPipelineBenchmarkException:
    """Unexpected exceptions are caught and the run is marked failed."""

    @pytest.mark.asyncio
    async def test_exception_during_aggregation_marks_failed(
        self,
        runner: BenchmarkRunner,
        benchmark_service: MagicMock,
        benchmark_services: MagicMock,
    ) -> None:
        trace_ids = [f"trace_{i}" for i in range(10)]
        benchmark_service.sample_kafka_pipeline_records.return_value = trace_ids
        benchmark_service.load_kafka_criterion_entries.side_effect = RuntimeError(
            "DB connection lost",
        )

        result = await runner.run_pipeline_benchmark(
            PipelineRunExec(
                benchmark_services=benchmark_services,
                run_id="test-run-exc",
                request=_pipeline_request(),
                generate_report=False,
            ),
        )

        assert result.status == BenchmarkStatus.FAILED
        assert "DB connection lost" in (result.error or "")
        benchmark_service.fail_run.assert_awaited_once()
        benchmark_service.complete_run.assert_not_awaited()


class TestRunPipelineBenchmarkAttribution:
    """Kafka ``attribution_detail_json`` aggregation into metrics."""

    @pytest.mark.asyncio
    async def test_attribution_folded_into_metrics(
        self,
        runner: BenchmarkRunner,
        benchmark_service: MagicMock,
        benchmark_services: MagicMock,
    ) -> None:
        trace_ids = [f"trace_{i}" for i in range(3)]
        benchmark_service.sample_kafka_pipeline_records.return_value = trace_ids
        benchmark_service.load_kafka_criterion_entries.return_value = [
            _entry("accuracy", "amap_chat_judge", mean=0.9, pass_rate=0.9),
            _entry("helpfulness", "llm_judge", mean=0.85, pass_rate=0.85),
        ]
        benchmark_service.aggregate_kafka_attribution.return_value = {
            "accuracy": {
                "field_distributions": {
                    "primary_issue_category": {
                        "intent_error": 2,
                        "tool_error": 1,
                    },
                },
                "total_attributed": 3,
            },
            "fused": {
                "field_distributions": {
                    "primary_issue_category": {"intent_error": 3},
                },
                "total_attributed": 3,
            },
        }

        result = await runner.run_pipeline_benchmark(
            PipelineRunExec(
                benchmark_services=benchmark_services,
                run_id="test-run-attr",
                request=_pipeline_request(),
                generate_report=False,
            ),
        )

        assert result.status == BenchmarkStatus.COMPLETED
        # aggregate_kafka_attribution uses legacy kwargs (resolve_kafka_rubric_query)
        benchmark_service.aggregate_kafka_attribution.assert_awaited_once()
        call = benchmark_service.aggregate_kafka_attribution.await_args
        assert call.kwargs.get("trace_ids") == trace_ids
        assert "rubric" in call.kwargs
        # Verify metrics_json passed to complete_run includes attribution_metrics
        metrics_json = benchmark_service.complete_run.await_args.args[0].metrics_json
        attr = metrics_json.get("attribution_metrics", {})
        assert "accuracy" in attr
        assert (
            attr["accuracy"]["field_distributions"]["primary_issue_category"]["intent_error"] == 2
        )
        assert attr["accuracy"]["field_distributions"]["primary_issue_category"]["tool_error"] == 1
        assert attr["accuracy"]["total_attributed"] == 3
        assert "fused" in attr
        assert attr["fused"]["field_distributions"]["primary_issue_category"]["intent_error"] == 3

    @pytest.mark.asyncio
    async def test_empty_attribution_does_not_populate(
        self,
        runner: BenchmarkRunner,
        benchmark_service: MagicMock,
        benchmark_services: MagicMock,
    ) -> None:
        trace_ids = [f"trace_{i}" for i in range(2)]
        benchmark_service.sample_kafka_pipeline_records.return_value = trace_ids
        benchmark_service.load_kafka_criterion_entries.return_value = [
            _entry("accuracy", "amap_chat_judge"),
            _entry("helpfulness", "llm_judge"),
        ]
        benchmark_service.aggregate_kafka_attribution.return_value = {}

        result = await runner.run_pipeline_benchmark(
            PipelineRunExec(
                benchmark_services=benchmark_services,
                run_id="test-run-no-attr",
                request=_pipeline_request(),
                generate_report=False,
            ),
        )

        assert result.status == BenchmarkStatus.COMPLETED
        metrics_json = benchmark_service.complete_run.await_args.args[0].metrics_json
        assert metrics_json.get("attribution_metrics", {}) == {}

    @pytest.mark.asyncio
    async def test_attribution_called_even_when_report_off(
        self,
        runner: BenchmarkRunner,
        benchmark_service: MagicMock,
        benchmark_services: MagicMock,
    ) -> None:
        """Attribution aggregation runs regardless of report generation."""
        trace_ids = ["trace_1"]
        benchmark_service.sample_kafka_pipeline_records.return_value = trace_ids
        benchmark_service.load_kafka_criterion_entries.return_value = [
            _entry("accuracy", "amap_chat_judge"),
            _entry("helpfulness", "llm_judge"),
        ]
        benchmark_service.aggregate_kafka_attribution.return_value = {
            "fused": {
                "field_distributions": {
                    "primary_issue_category": {"intent_error": 1},
                },
                "total_attributed": 1,
            },
        }

        await runner.run_pipeline_benchmark(
            PipelineRunExec(
                benchmark_services=benchmark_services,
                run_id="test-run-attr-no-report",
                request=_pipeline_request(),
                generate_report=False,
            ),
        )

        # Attribution should still be called and metrics should contain it
        benchmark_service.aggregate_kafka_attribution.assert_awaited_once()
        metrics_json = benchmark_service.complete_run.await_args.args[0].metrics_json
        assert "fused" in metrics_json.get("attribution_metrics", {})
