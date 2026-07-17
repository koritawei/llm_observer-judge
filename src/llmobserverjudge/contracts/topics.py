"""Topic names and producer allowlist (SDD-EV-27 / R-02)."""

from __future__ import annotations

from enum import StrEnum


class Topic(StrEnum):
    SPANS = "llmjudge_spans"
    EVAL_JOBS = "llmjudge_eval_jobs"
    EVAL_JOBS_P0 = "llmjudge_eval_jobs_P0"
    EVAL_JOBS_P1 = "llmjudge_eval_jobs_P1"
    EVAL_JOBS_P2 = "llmjudge_eval_jobs_P2"
    EVAL_JOBS_P3 = "llmjudge_eval_jobs_P3"
    ATTRIBUTION_JOBS = "llmjudge_attribution_jobs"
    ANNOTATION_JOBS = "llmjudge_annotation_jobs"
    DLQ_SPANS = "dlq_spans"
    DLQ_EVAL_JOBS = "dlq_eval_jobs"
    DLQ_ANNOTATION_JOBS = "dlq_annotation_jobs"


def eval_jobs_priority_topic(priority: str) -> Topic:
    """Optional priority shard topic (design §7.2)."""
    mapping = {
        "P0": Topic.EVAL_JOBS_P0,
        "P1": Topic.EVAL_JOBS_P1,
        "P2": Topic.EVAL_JOBS_P2,
        "P3": Topic.EVAL_JOBS_P3,
    }
    try:
        return mapping[priority]
    except KeyError as e:
        raise ValueError(f"unknown priority: {priority!r}") from e


class Component(StrEnum):
    GATEWAY = "gateway"
    INGEST = "ingest"
    DISPATCH = "dispatch"
    EVAL_WORKER = "eval_worker"
    ATTRIBUTION_WORKER = "attribution_worker"
    ANNOTATION_PLATFORM = "annotation_platform"
    SDK = "sdk"


# Sole allowed publishers per SDD-EV-27 §4
_DISPATCH_ONLY = frozenset({Component.DISPATCH})

ALLOWED_PRODUCERS: dict[Topic, frozenset[Component]] = {
    Topic.SPANS: frozenset({Component.GATEWAY}),
    Topic.EVAL_JOBS: _DISPATCH_ONLY,
    Topic.EVAL_JOBS_P0: _DISPATCH_ONLY,
    Topic.EVAL_JOBS_P1: _DISPATCH_ONLY,
    Topic.EVAL_JOBS_P2: _DISPATCH_ONLY,
    Topic.EVAL_JOBS_P3: _DISPATCH_ONLY,
    Topic.ATTRIBUTION_JOBS: _DISPATCH_ONLY,
    Topic.ANNOTATION_JOBS: _DISPATCH_ONLY,  # via DispatchClient
}


class ProducerForbiddenError(PermissionError):
    """Raised when a component attempts to publish to a disallowed topic."""


def assert_can_publish(component: Component, topic: Topic) -> None:
    allowed = ALLOWED_PRODUCERS.get(topic, frozenset())
    if component not in allowed:
        raise ProducerForbiddenError(
            f"component={component!s} must not publish topic={topic!s}; "
            f"allowed={sorted(str(c) for c in allowed)}"
        )
