"""Unit tests for eval_queue.queue_factory routing and constants."""

import pytest
from benchmark.libs.jobqueue.key_prefix import queue_key_prefix
from benchmark.queue.queue_factory import case_queue_name
from benchmark.queue.queue_prefix_policy import (
    BULLMQ_KEYSPACE_GENERATION,
    QueuePrefixPolicyError,
    build_bullmq_logic_prefix,
)


def test_case_queue_name_routes_l2_businesses() -> None:
    assert case_queue_name("hotel", ["hotel"]) == "benchmark-cases-l2"
    assert case_queue_name("ticket", ["hotel"]) == "benchmark-cases"
    assert case_queue_name(None, ["hotel"]) == "benchmark-cases"
    assert case_queue_name("", ["hotel"]) == "benchmark-cases"
    assert case_queue_name("hotel", []) == "benchmark-cases"


def test_build_bullmq_logic_prefix_includes_keyspace_generation() -> None:
    """The generation segment isolates the historical ``pre:bullmq`` keyspace from the new ``pre:bullmq:v2`` keyspace."""
    assert BULLMQ_KEYSPACE_GENERATION == "v2"
    assert build_bullmq_logic_prefix("pre") == "pre:bullmq:v2"
    assert build_bullmq_logic_prefix("prod") == "prod:bullmq:v2"
    assert build_bullmq_logic_prefix("daily") == "daily:bullmq:v2"
    assert build_bullmq_logic_prefix("  pre  ") == "pre:bullmq:v2"
    # Explicitly isolated from the old keyspace (an old node still listening on pre:bullmq will not pick up a new job)
    assert build_bullmq_logic_prefix("pre") != "pre:bullmq"
    assert queue_key_prefix(build_bullmq_logic_prefix("pre"), "benchmark-cases") == (
        "{pre:bullmq:v2:benchmark-cases}"
    )


def test_build_bullmq_logic_prefix_rejects_empty() -> None:
    with pytest.raises(QueuePrefixPolicyError, match="eval_queue_prefix"):
        build_bullmq_logic_prefix("")
    with pytest.raises(QueuePrefixPolicyError, match="eval_queue_prefix"):
        build_bullmq_logic_prefix("   ")
