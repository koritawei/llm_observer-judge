"""Judge raw-result passthrough registry (§8 extension point).

Default behavior (existing businesses): the worker unpacks ``EvaluatorResult`` into
``{score, reason, sub_scores, dimension?, dimensions?}`` and writes the fused row
``judge_json.criterion_scores[cid]`` (the framework-trimmed shape).

Businesses registered for passthrough (for example search_eval): ``criterion_scores[cid]``
is the full raw ``result.model_dump(mode="json")`` — **no field trimming or parsing**
(``score`` / ``reason`` / ``sub_scores`` / ``extra`` are kept as-is, and extra includes
every v3-protocol field such as dimensions / dimension / input_info).
The outer key ``criterion_scores[cid]`` stays the framework read contract; downstream
readers of the same-named fields (score/reason/sub_scores) stay compatible.
"""

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger("agent_eval.benchmark.common.raw_result_store")

#: Benchmark component names that store judge results raw (unregistered = framework-trimmed shape; legacy behavior
#: unchanged).
_RAW_RESULT_STORE_BENCHMARKS: set[str] = set()


def register_raw_result_store(benchmark_name: str) -> None:
    """Register a business for judge raw-result passthrough (idempotent)."""
    _RAW_RESULT_STORE_BENCHMARKS.add(benchmark_name)
    logger.info("raw_result_store.registered benchmark=%s", benchmark_name)


def is_raw_result_store(benchmark_name: str | None) -> bool:
    """Whether this benchmark stores the judge result raw."""
    return bool(benchmark_name) and benchmark_name in _RAW_RESULT_STORE_BENCHMARKS


def clear_raw_result_stores() -> None:
    """Clear the registry (for test isolation)."""
    _RAW_RESULT_STORE_BENCHMARKS.clear()


__all__ = [
    "clear_raw_result_stores",
    "is_raw_result_store",
    "register_raw_result_store",
]
