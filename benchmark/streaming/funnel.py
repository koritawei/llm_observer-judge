"""On-the-hour funnel reconciliation — filter / sample / eval counts for the streaming pipeline (migrated from ``_JudgeFunnelCounters``).

The reconciliation contract (log prefix ``Judge funnel:``, WARNING level, UTC+8 hour buckets)
matches the JudgeService era bit for bit — the existing production troubleshooting and reconciliation contract stays unchanged.

Reconciliation identities (message granularity)::

    msg_total = skip_record_type + skip_invalid + skip_deleted + received
    received = dedup_skip + filter_* + sample_miss + sample_hit
    eligible = sample_miss + sample_hit

``eval_*`` counts **config calls** (one sample_hit may correspond to several evaluate calls).
``stop_dropped`` is new for streaming: in-flight messages cancelled when instance stop drain times out
(not replayed under early-commit semantics; counted only; see §3.2 drain interaction).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger("agent_eval.benchmark.streaming.funnel")

#: Funnel reconciliation timezone (on the hour, UTC+8); matches the business persist stamp.
_UTC_PLUS_8 = timezone(timedelta(hours=8))


def _hour_key_cn(now: datetime | None = None) -> str:
    """Return ``YYYY-MM-DDTHH+08:00`` for funnel hour buckets."""
    dt = now or datetime.now(_UTC_PLUS_8)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC_PLUS_8)
    else:
        dt = dt.astimezone(_UTC_PLUS_8)
    return dt.strftime("%Y-%m-%dT%H+08:00")


@dataclass
class StreamingFunnelCounters:
    """In-process streaming funnel counts accumulated by hour (fields match the JudgeService era).

    Reconciliation (message granularity)::

        msg_total = skip_record_type + skip_invalid + skip_deleted + received
        received = dedup_skip + filter_* + sample_miss + sample_hit
        eligible = sample_miss + sample_hit

    ``eval_*`` counts **config calls** (one sample_hit may correspond to several evaluate calls).
    """

    msg_total: int = 0
    skip_record_type: int = 0
    skip_invalid: int = 0
    skip_deleted: int = 0
    received: int = 0
    dedup_skip: int = 0
    filter_agent_name: int = 0
    filter_no_config: int = 0
    filter_env: int = 0
    filter_attrs: int = 0
    filter_empty_io: int = 0
    filter_test_adiu: int = 0
    eligible: int = 0
    sample_miss: int = 0
    sample_hit: int = 0
    eval_ok: int = 0
    eval_fail_timeout: int = 0
    eval_fail_permanent: int = 0
    eval_fail_transient: int = 0
    #: In-flight messages dropped when instance stop drain times out (early-commit does not replay them).
    stop_dropped: int = 0


def emit_funnel_log(hour: str, counters: StreamingFunnelCounters) -> None:
    """Emit one WARNING line for ops hour-bucket reconciliation.

    Production INFO is often filtered out, so WARNING keeps funnel reconciliation visible.
    The log prefix ``Judge funnel:`` is the existing ops reconciliation contract and is kept bit for bit after the streaming migration.
    """
    c = counters
    logger.warning(
        "Judge funnel: hour=%s msg_total=%d skip_record_type=%d "
        "skip_invalid=%d skip_deleted=%d received=%d dedup_skip=%d "
        "filter_agent_name=%d filter_no_config=%d filter_env=%d "
        "filter_attrs=%d filter_empty_io=%d filter_test_adiu=%d "
        "eligible=%d sample_miss=%d sample_hit=%d "
        "eval_ok=%d eval_fail_timeout=%d eval_fail_permanent=%d "
        "eval_fail_transient=%d stop_dropped=%d",
        hour,
        c.msg_total,
        c.skip_record_type,
        c.skip_invalid,
        c.skip_deleted,
        c.received,
        c.dedup_skip,
        c.filter_agent_name,
        c.filter_no_config,
        c.filter_env,
        c.filter_attrs,
        c.filter_empty_io,
        c.filter_test_adiu,
        c.eligible,
        c.sample_miss,
        c.sample_hit,
        c.eval_ok,
        c.eval_fail_timeout,
        c.eval_fail_permanent,
        c.eval_fail_transient,
        c.stop_dropped,
    )


__all__ = [
    "StreamingFunnelCounters",
    "emit_funnel_log",
    "_UTC_PLUS_8",
    "_hour_key_cn",
]
