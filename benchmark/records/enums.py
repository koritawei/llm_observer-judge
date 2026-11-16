"""Enums for agent-eval ORM table fields (single source).

Only enums that map to database columns live here. Domain business enums (for example
``CasePriority``, ``MatchType``) live in :mod:`benchmark.domain.enums`.
Before adding an enum, confirm a business caller exists; an enum with zero references
must not be added to this module (guarded by ``tests/unit/data_model/test_enum_lifecycle.py``).

Usage::

    from benchmark.records.enums import TriageLevel, TraceStatus
"""

from enum import StrEnum


class TriageLevel(StrEnum):
    """Four-level triage result, plus an abstain sentinel and an attribution-override marker.

    Used by the ``TraceRecord.triage_level`` scalar column, ``TriageVerdict.level``, and
    badcase ``JudgeFailureSignal.level`` literal parsing.
    ``GOOD`` means no problem, ``SEVERE_BAD`` means a severe bad case,
    ``NO_DECISION`` means the judge could not conclude (abstain).
    ``ATTRIBUTION_OVERRIDDEN`` means attribution overrode the original triage
    (originally bad, corrected by attribution).
    """

    GOOD = "good"
    WATCH = "watch"
    MILD_BAD = "mild_bad"
    SEVERE_BAD = "severe_bad"
    NO_DECISION = "no_decision"
    ATTRIBUTION_OVERRIDDEN = "attribution_overridden"


class TraceStatus(StrEnum):
    """Processing status of a trace-pipeline record.

    Used by the ``TraceRecord.status`` field.
    ``NEW`` is newly stored, ``SCORED`` has finished LLM judging, ``TRIAGED`` has finished
    triage, and ``DISPATCHED`` has dispatched a ticket.
    """

    NEW = "new"
    SCORED = "scored"
    TRIAGED = "triaged"
    DISPATCHED = "dispatched"


class NotificationTaskJobStatus(StrEnum):
    """Execution status of an Aone ticket-creation outbox job.

    Used by the ``TraceRecord.notification_job_status`` scalar column and the ``notification_job_json`` bucket.
    """

    PENDING = "pending"
    RUNNING = "running"
    CREATED = "created"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
    BLOCKED = "blocked"


class BenchmarkStatus(StrEnum):
    """Lifecycle status of a Benchmark run.

    Used by the ``BenchmarkRun.status`` field.
    ``INITIALIZING`` is being created, ``ACCEPTING`` is ingesting streaming data,
    ``RUNNING`` means evaluation is in progress,
    ``AGGREGATING`` is the aggregation phase, ``GENERATING_REPORT`` is generating the report,
    ``COMPLETED`` finished successfully, ``FAILED`` failed,
    and ``STOPPED`` is the terminal state after a manual stop (it may carry a partial report).
    """

    INITIALIZING = "initializing"
    ACCEPTING = "accepting"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


__all__ = [
    "TriageLevel",
    "TraceStatus",
    "NotificationTaskJobStatus",
    "BenchmarkStatus",
]
