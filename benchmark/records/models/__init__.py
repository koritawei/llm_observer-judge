"""ORM models for agent-eval.

Every model subclasses data_models.BaseModel (UUID primary key + automatic timestamps).
TraceRecord and the lifecycle model BenchmarkRun compose ObserverMixin and support
lifecycle observers (see :mod:`benchmark.records.observers`); the run_items table is
written via bulk_create / update_or_create, does not go through the save() lifecycle,
and has no observer.

Usage:
    from benchmark.records.models import TraceRecord, BenchmarkRun
"""

from benchmark.records.models.benchmark_kafka_task import BenchmarkKafkaTask
from benchmark.records.models.benchmark_report_provider import BenchmarkReportProvider
from benchmark.records.models.benchmark_run import BenchmarkRun
from benchmark.records.models.benchmark_run_item import BenchmarkRunItem
from benchmark.records.models.cluster_report_job import ClusterReportJob
from benchmark.records.models.runtime_config_record import RuntimeConfigRecord
from benchmark.records.models.trace_record import TraceRecord

__all__ = [
    "TraceRecord",
    "BenchmarkKafkaTask",
    "BenchmarkReportProvider",
    "BenchmarkRun",
    "BenchmarkRunItem",
    "RuntimeConfigRecord",
    "ClusterReportJob",
]
