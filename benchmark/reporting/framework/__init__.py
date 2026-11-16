"""Rubric evaluation framework — shared implementation of declarative offline rubric evaluation.

This package is abstracted from the benchmark domain and reused by batch evaluation and
realtime stream evaluation:

- :mod:`schema` — shared data contracts (rubric, aggregation, dataset references, stats models)
- :mod:`base` — ``@benchmark_component`` decorator and ``BaseBenchmarkServices`` base class
- :mod:`aggregation_calculator` — run-level aggregate statistics
- :mod:`fusion` — item-level score fusion (``ScoreFuser`` / ``EvalFailure``)
- :mod:`webdev_client` — HTML evaluation report client
- :mod:`report_storage` — report persistence

Domain-specific implementations (such as benchmark request/response models, the runner,
and the scheduler) live in ``benchmark.runs``.
"""

from benchmark.reporting.framework.base import (
    BaseBenchmarkServices,
    BenchmarkComponentMetadata,
    benchmark_component,
    get_benchmark_metadata,
)
from benchmark.reporting.framework.schema import (
    AggregatedMetrics,
    AggregationConfig,
    AggregationMethod,
    AttributionStats,
    Criterion,
    CriterionStats,
    DatasetRef,
)

__all__ = [
    # base — decorator and base class
    "benchmark_component",
    "BaseBenchmarkServices",
    "BenchmarkComponentMetadata",
    "get_benchmark_metadata",
    # schema — shared data contracts
    "AggregationMethod",
    "Criterion",
    "AggregationConfig",
    "DatasetRef",
    "CriterionStats",
    "AttributionStats",
    "AggregatedMetrics",
]
