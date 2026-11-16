"""On-the-hour funnel reconciliation — filter / sample / read-back / eval counts for sidecar MetaQ.

Log prefix ``Sidecar funnel:``, WARNING, on the hour in UTC+8. Notifications are sent only when the trace is complete,
so there is no Kafka record_type / deleted / dedup / empty input-output cut.

Reconciliation::

    msg_total = skip_invalid + received
    received = filter_* + sample_miss + sample_hit
    eligible = sample_miss + sample_hit
    sample_hit = eval_ok + eval_fail_* + read_fail
"""

import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger("agent_eval.sidecar_ingest.funnel")

_UTC_PLUS_8 = timezone(timedelta(hours=8))
_PROGRESS_INTERVAL_S = 300.0


def _hour_key_cn(now: datetime | None = None) -> str:
    """Return ``YYYY-MM-DDTHH+08:00`` for funnel hour buckets."""
    dt = now or datetime.now(_UTC_PLUS_8)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC_PLUS_8)
    else:
        dt = dt.astimezone(_UTC_PLUS_8)
    return dt.strftime("%Y-%m-%dT%H+08:00")


@dataclass
class SidecarFunnelCounters:
    """In-process sidecar funnel counts accumulated by hour."""

    msg_total: int = 0
    skip_invalid: int = 0
    received: int = 0
    filter_no_config: int = 0
    filter_agent_name: int = 0
    filter_test_adiu: int = 0
    filter_env: int = 0
    filter_name: int = 0
    filter_tags: int = 0
    filter_time: int = 0
    eligible: int = 0
    sample_miss: int = 0
    sample_hit: int = 0
    read_fail: int = 0
    eval_ok: int = 0
    eval_fail_timeout: int = 0
    eval_fail_permanent: int = 0
    eval_fail_transient: int = 0


class SidecarFunnel:
    """One process-wide hour bucket. Safe for concurrent MetaQ poll loops."""

    def __init__(self) -> None:
        self._counters = SidecarFunnelCounters()
        self._hour = _hour_key_cn()
        self._lock = threading.Lock()
        self._last_progress = time.monotonic()

    def bump(self, field: str) -> None:
        """Increment one counter after rolling the hour if needed."""
        self._roll()
        snapshot: tuple[str, SidecarFunnelCounters] | None = None
        with self._lock:
            setattr(self._counters, field, getattr(self._counters, field) + 1)
            now = time.monotonic()
            if now - self._last_progress >= _PROGRESS_INTERVAL_S:
                self._last_progress = now
                snapshot = (self._hour, replace(self._counters))
        if snapshot is not None:
            _emit_progress(snapshot[0], snapshot[1])

    def flush(self) -> None:
        """Emit the current hour and reset. Called on shutdown."""
        with self._lock:
            hour = self._hour
            prev = self._counters
            self._counters = SidecarFunnelCounters()
        _emit(hour, prev)

    def _roll(self) -> None:
        hour = _hour_key_cn()
        with self._lock:
            if hour == self._hour:
                return
            prev_hour = self._hour
            prev = self._counters
            self._counters = SidecarFunnelCounters()
            self._hour = hour
        _emit(prev_hour, prev)


def _emit_progress(hour: str, counters: SidecarFunnelCounters) -> None:
    """Running totals for the open hour. Does not reset counters."""
    logger.warning("Sidecar funnel progress: %s", _format(hour, counters))


def _emit(hour: str, counters: SidecarFunnelCounters) -> None:
    logger.warning("Sidecar funnel: %s", _format(hour, counters))


def _format(hour: str, counters: SidecarFunnelCounters) -> str:
    c = counters
    return (
        f"hour={hour} msg_total={c.msg_total} skip_invalid={c.skip_invalid} "
        f"received={c.received} filter_no_config={c.filter_no_config} "
        f"filter_agent_name={c.filter_agent_name} "
        f"filter_test_adiu={c.filter_test_adiu} filter_env={c.filter_env} "
        f"filter_name={c.filter_name} filter_tags={c.filter_tags} "
        f"filter_time={c.filter_time} eligible={c.eligible} "
        f"sample_miss={c.sample_miss} sample_hit={c.sample_hit} "
        f"read_fail={c.read_fail} eval_ok={c.eval_ok} "
        f"eval_fail_timeout={c.eval_fail_timeout} "
        f"eval_fail_permanent={c.eval_fail_permanent} "
        f"eval_fail_transient={c.eval_fail_transient}"
    )


__all__ = ["SidecarFunnel", "SidecarFunnelCounters", "_hour_key_cn"]
