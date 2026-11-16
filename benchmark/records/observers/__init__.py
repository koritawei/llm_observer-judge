"""Lifecycle observers for agent-eval models.

``register_database()`` auto-discovers and registers every ``ModelObserver``
subclass in this module at application startup.

Registered observers:
  - ``TraceRecordObserver`` — TraceRecord save/delete logs
  - ``BenchmarkRunObserver`` — benchmark lifecycle logs
  - ``BenchmarkKafkaTaskObserver`` — streaming task-mirror save/delete logs
  - ``BenchmarkReportProviderObserver`` — external report-service registration save/delete logs
"""

from benchmark.records.observers.benchmark import BenchmarkRunObserver
from benchmark.records.observers.benchmark_kafka_task import BenchmarkKafkaTaskObserver
from benchmark.records.observers.benchmark_report_provider import (
    BenchmarkReportProviderObserver,
)
from benchmark.records.observers.trace_record import TraceRecordObserver

__all__ = [
    "TraceRecordObserver",
    "BenchmarkRunObserver",
    "BenchmarkKafkaTaskObserver",
    "BenchmarkReportProviderObserver",
]
