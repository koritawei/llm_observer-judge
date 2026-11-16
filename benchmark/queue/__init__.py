"""eval_queue — BullMQ evaluation-task dispatch infrastructure (design doc §1-§17).

Infrastructure-only package: it depends only on config / data_model / utils / app.libs,
and must not depend on business packages (benchmark / eval_framework). Business workers,
healers, and the external-trigger orchestrator live in :mod:`benchmark.workers`
(the assembly layer).

Module layout:
- ``eval_case`` — unified data abstraction (EvalCase)
- ``queue_factory`` — Redis connection + Queue/Worker factory (EvalQueueRegistry)
- ``inflight_slot`` — Redis counting semaphore (acquire_slot / release_slot)
- ``slot_busy`` — slot-busy delay + jitter + probe rate limit
- ``retry_requeue`` — failure requeue and terminal-failure entry (requeue_or_fail_from_running)
- ``callback_next_submit`` — callback releases the next item + external-wait enqueue
- ``submit_failure`` — submit retryable / force_fail classification (§8.2)
- ``worker_skeleton`` — synchronous evaluation worker skeleton (slot / heartbeat / exception loop)

Zero-domain BullMQ patches live in :mod:`benchmark.libs.jobqueue` (job listing, staging,
hash tags, and the lock-renewing Worker).
"""

from benchmark.queue.callback_next_submit import enqueue_external_cases
from benchmark.queue.eval_case import EvalCase
from benchmark.queue.inflight_slot import (
    acquire_cluster_slot,
    acquire_slot,
    release_cluster_slot,
    release_slot,
)
from benchmark.queue.queue_factory import (
    BENCHMARK_CASES,
    BENCHMARK_CASES_L2,
    BENCHMARK_RUNNING,
    EXTERNAL_RUNNING,
    EXTERNAL_WAIT,
    EvalQueueRegistry,
    case_queue_name,
    load_queue_prefix_from_yaml,
    resolve_redis_url,
)
from benchmark.queue.queue_prefix_policy import (
    QueuePrefixPolicyError,
    QueuePrefixResolveInput,
    RuntimeLaneDetectInput,
    assert_committed_standard_yaml_prefixes,
    detect_runtime_lane,
    has_platform_lane_signal,
    resolve_queue_prefix,
    uses_ambiguous_standard_profile_fallback,
)
from benchmark.queue.retry_requeue import (
    force_fail_from_running,
    requeue_or_fail_from_running,
    retry_policy_from_settings,
)
from benchmark.queue.slot_busy import SlotProbePolicy, delay_for_slot_busy
from benchmark.queue.submit_failure import (
    add_running_with_retry,
    is_retryable_callback_error,
    is_retryable_submit_failure,
    should_release_on_submit_failure,
)

__all__ = [
    "BENCHMARK_CASES",
    "BENCHMARK_CASES_L2",
    "BENCHMARK_RUNNING",
    "case_queue_name",
    "EXTERNAL_RUNNING",
    "EXTERNAL_WAIT",
    "EvalCase",
    "EvalQueueRegistry",
    "SlotProbePolicy",
    "acquire_cluster_slot",
    "acquire_slot",
    "add_running_with_retry",
    "delay_for_slot_busy",
    "enqueue_external_cases",
    "force_fail_from_running",
    "is_retryable_callback_error",
    "is_retryable_submit_failure",
    "QueuePrefixPolicyError",
    "QueuePrefixResolveInput",
    "RuntimeLaneDetectInput",
    "assert_committed_standard_yaml_prefixes",
    "detect_runtime_lane",
    "has_platform_lane_signal",
    "load_queue_prefix_from_yaml",
    "uses_ambiguous_standard_profile_fallback",
    "release_cluster_slot",
    "release_slot",
    "resolve_queue_prefix",
    "resolve_redis_url",
    "requeue_or_fail_from_running",
    "retry_policy_from_settings",
    "should_release_on_submit_failure",
]
