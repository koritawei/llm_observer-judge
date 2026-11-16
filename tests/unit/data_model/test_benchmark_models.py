"""Unit tests for BenchmarkRun ORM model (single-table design)."""

from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun, __all__
from benchmark.records.models import BenchmarkRun as _RunModel
from benchmark.libs.models import BaseModel


class TestBenchmarkRunInheritance:
    def test_is_base_model(self):
        assert issubclass(BenchmarkRun, BaseModel)


class TestBenchmarkRunMeta:
    def test_table_name(self):
        assert BenchmarkRun._meta.db_table == "benchmark_runs"

    def test_unique_together(self):
        unique_together = set(BenchmarkRun._meta.unique_together)
        assert ("run_id",) in unique_together

    def test_indexes(self):
        indexes = set(BenchmarkRun._meta.indexes)
        assert ("benchmark_name", "status") in indexes
        assert ("service_name", "status") in indexes
        assert ("status", "created_at") in indexes


class TestBenchmarkRunFields:
    def test_run_id(self):
        field = BenchmarkRun._meta.fields_map["run_id"]
        assert field.max_length == 128

    def test_status_enum(self):
        field = BenchmarkRun._meta.fields_map["status"]
        assert field.enum_type is BenchmarkStatus

    def test_metrics_json(self):
        field = BenchmarkRun._meta.fields_map["metrics_json"]
        assert field.null is True

    def test_final_score(self):
        field = BenchmarkRun._meta.fields_map["final_score"]
        assert field.null is True

    def test_error_message(self):
        field = BenchmarkRun._meta.fields_map["error_message"]
        assert field.null is True

    def test_completed_evaluators_default(self):
        field = BenchmarkRun._meta.fields_map["completed_evaluators"]
        assert field.default == 0


class TestExports:
    def test_all_contains_benchmark_run(self):
        assert "BenchmarkRun" in __all__
        assert "BenchmarkEvaluatorResult" not in __all__
        assert "BenchmarkTraceRecord" not in __all__

    def test_import_alias(self):
        assert BenchmarkRun is _RunModel
