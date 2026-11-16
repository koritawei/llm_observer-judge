"""BenchmarkRun service facade — lifecycle / metrics / case detail / external-item composition.

Implementation is split across ``run_lifecycle`` / ``run_metrics`` / ``run_case_query`` /
``run_external_item``; shared symbols live in ``run_helpers``.

Tests that patch the ORM / transactions should patch ``benchmark.runs.run_helpers.*``
(mixin methods bind symbols from that module). This module still re-exports the existing
public and test symbols.
"""

from tortoise.expressions import F
from tortoise.transactions import in_transaction

from benchmark.records.models import BenchmarkRun, BenchmarkRunItem, TraceRecord
from benchmark.runs.run_case_query import BenchmarkRunCaseQuery
from benchmark.runs.run_external_item import BenchmarkRunExternalItem
from benchmark.runs.run_helpers import (
    BenchmarkRunSpec,
    CompleteRunWrite,
    DailyTrendQuery,
    ExpandDimensionMetricsSpec,
    ExternalFusedRowProcess,
    KafkaRubricQuery,
    KafkaSampleQuery,
    ParseFailedStubWrite,
    TraceRecordSampleQuery,
    _expand_case_dimension_scores,
    _extract_display_case_id,
    _extract_display_input,
    _extract_display_output,
    _resolve_report_case_id,
    _stamp_bucket,
    current_worker_id,
    generate_benchmark_job_id,
    merge_config_json,
    merge_metrics_json,
    merge_report_meta_json,
)
from benchmark.runs.run_lifecycle import BenchmarkRunLifecycle
from benchmark.runs.run_metrics import BenchmarkRunMetrics


class BenchmarkService(
    BenchmarkRunLifecycle,
    BenchmarkRunMetrics,
    BenchmarkRunCaseQuery,
    BenchmarkRunExternalItem,
):
    """``benchmark_runs`` CRUD + lifecycle + incremental JSON bucket merge."""


__all__ = [
    "BenchmarkRun",
    "BenchmarkRunItem",
    "BenchmarkRunSpec",
    "BenchmarkService",
    "CompleteRunWrite",
    "DailyTrendQuery",
    "ExpandDimensionMetricsSpec",
    "ExternalFusedRowProcess",
    "F",
    "KafkaRubricQuery",
    "KafkaSampleQuery",
    "ParseFailedStubWrite",
    "TraceRecord",
    "TraceRecordSampleQuery",
    "_expand_case_dimension_scores",
    "_extract_display_case_id",
    "_resolve_report_case_id",
    "_extract_display_input",
    "_extract_display_output",
    "_stamp_bucket",
    "current_worker_id",
    "generate_benchmark_job_id",
    "in_transaction",
    "merge_config_json",
    "merge_metrics_json",
    "merge_report_meta_json",
]
