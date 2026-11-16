"""job_queue.key_prefix unit tests."""

from benchmark.libs.jobqueue.key_prefix import queue_key_prefix

_BENCHMARK_CASES = "benchmark-cases"


def test_queue_key_prefix_wraps_hash_tag() -> None:
    assert queue_key_prefix("pre:bullmq", _BENCHMARK_CASES) == "{pre:bullmq:benchmark-cases}"


def test_queue_key_prefix_colocates_queue_keys() -> None:
    """The hash tag must wrap the qualified name, so wait/meta land in the same slot."""
    prefix = queue_key_prefix("pre:bullmq", "benchmark-cases")
    assert prefix == "{pre:bullmq:benchmark-cases}"
    sample_wait = f"{prefix}:benchmark-cases:wait"
    sample_meta = f"{prefix}:benchmark-cases:meta"
    tag = prefix[1:-1]
    assert f"{{{tag}}}" in sample_wait or tag in sample_wait
    assert tag in sample_meta
