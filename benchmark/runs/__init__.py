"""Benchmark business — declarative rubric benchmark evaluation.

The shared evaluation framework (``@benchmark_component`` / aggregation / fusion / reports)
lives in :mod:`benchmark.reporting.framework`. This package is the benchmark-specific
implementation:

- request/response schemas (:mod:`schema`)
- run orchestration (:mod:`run_service` / :mod:`runner`)
- benchmark service registration (:mod:`registry` / :mod:`example`)
- ieval externally triggered evaluation (:mod:`ieval` subpackage: Aone / local evaluation,
  attribution, and OSS loading; the orchestrator :mod:`benchmark.workers.external_runner`
  sits in the assembly layer)

Public API::

    from benchmark.reporting.framework import benchmark_component, BaseBenchmarkServices
    from benchmark.runs.schema import Criterion, AggregationConfig, DatasetRef
"""
