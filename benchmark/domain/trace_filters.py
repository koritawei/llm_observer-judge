"""Trace filter + deterministic sampling shared by Kafka streaming and sidecar.

Kafka feeds :class:`~benchmark.streaming.schemas.TraceRecord`;
sidecar feeds the MetaQ notify body. Both expose ``name`` / ``tags`` /
``timestamp``.
"""

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from benchmark.domain.schemas import EvalConfigEntry

_TEST_ADIU_SUFFIX = "_test"
_ADIU_INPUT_PATHS = (
    "adiu",
    "user.adiu",
    "request_body.adiu",
    "request_body.user.adiu",
    "request_body.viewer_context.user.adiu",
)

_UNIX_MS_THRESHOLD = 10_000_000_000


class FilterableTrace(Protocol):
    """Minimal trace view used by :func:`matches_trace_filters`."""

    name: str | None
    tags: list[str] | None
    timestamp: str | int | None


def parse_timestamp(ts: str | int | None) -> datetime | None:
    """Parse a trace timestamp into a timezone-aware ``datetime``.

    Accepts ISO 8601 strings or Unix epoch seconds/millis as ``int``.
    Returns ``None`` when the value is ``None`` or unparseable.
    """
    if ts is None:
        return None
    if isinstance(ts, int):
        try:
            return datetime.fromtimestamp(
                ts / 1000 if ts > _UNIX_MS_THRESHOLD else ts,
                tz=UTC,
            )
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def matches_trace_name(config: EvalConfigEntry, name: str | None) -> bool:
    """Apply exact and substring name constraints; each constraint must pass."""
    if config.name and name not in config.name:
        return False
    if config.trace_name_contains and (
        name is None or not any(part in name for part in config.trace_name_contains)
    ):
        return False
    if config.trace_name_excludes and name is not None:
        if any(part in name for part in config.trace_name_excludes):
            return False
    return True


def matches_trace_filters(config: EvalConfigEntry, trace: FilterableTrace) -> bool:
    """Return whether *trace* passes name / tags / timestamp filters on *config*.

    Each dimension is independent: empty/None means no restriction.
    """
    if not matches_trace_name(config, trace.name):
        return False

    if config.tags:
        trace_tag_set = set(trace.tags) if trace.tags else set()
        if not (trace_tag_set & set(config.tags)):
            return False

    if config.start_time or config.end_time:
        trace_ts = parse_timestamp(trace.timestamp)
        if trace_ts is None:
            return False
        if config.start_time:
            start = parse_timestamp(config.start_time)
            if start and trace_ts < start:
                return False
        if config.end_time:
            end = parse_timestamp(config.end_time)
            if end and trace_ts > end:
                return False

    return True


def should_sample_deterministic(sampling_rate: float, sample_key: str) -> bool:
    """Deterministic sampling based on SHA256(*sample_key*).

    Uses the first 4 bytes of SHA256 as a uniform float in [0, 1).
    """
    if sampling_rate >= 1:
        return True
    if sampling_rate <= 0:
        return False
    digest = hashlib.sha256(sample_key.encode()).digest()
    r = int.from_bytes(digest[:4], "big") / 0x100000000
    return r < sampling_rate


def is_test_adiu(value: object) -> bool:
    """True when *value* is an adiu whose suffix is ``_test`` (case-insensitive)."""
    if not isinstance(value, str):
        return False
    return value.strip().lower().endswith(_TEST_ADIU_SUFFIX)


def should_skip_test_adiu(
    *,
    user_id: str | None = None,
    input_value: object = None,
) -> bool:
    """True when ``user_id`` or a nested input adiu is a test account."""
    if is_test_adiu(user_id):
        return True
    mapping = _as_mapping(input_value)
    if mapping is None:
        return False
    return any(is_test_adiu(_get_path(mapping, path)) for path in _ADIU_INPUT_PATHS)


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed: object = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _get_path(data: Mapping[str, Any], path: str) -> object:
    current: object = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


__all__ = [
    "FilterableTrace",
    "is_test_adiu",
    "matches_trace_filters",
    "matches_trace_name",
    "parse_timestamp",
    "should_sample_deterministic",
    "should_skip_test_adiu",
]
