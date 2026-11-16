"""Parse sidecar-trace-worker MetaQ notify body (protocol §2.1).

``build_notify`` / ``notify_properties`` match sidecar-trace-worker
``aggregate.build_notify`` + ``metaq.apply_notify_envelope``.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

_FINISH_REASONS = frozenset({"finish", "timeout"})
_UNKNOWN_PROJECT_ID = "unknown"
_LEVEL_RANK = {"DEBUG": 0, "DEFAULT": 1, "WARNING": 2, "ERROR": 3}
_PROP_KEYS = ("project_id", "agent_name", "environment", "name")


class SidecarNotifyParseError(ValueError):
    """Notify body is not a valid §2.1 payload."""


class SidecarTraceNotify(BaseModel):
    """Thin complete-trace notify. Observation tree is never in the message."""

    model_config = {"extra": "ignore"}

    trace_id: str
    finish_reason: Literal["finish", "timeout"]
    name: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    environment: str | None = None
    tags: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
    has_error: bool = False
    timestamp: int | None = None
    project_id: str | None = None
    agent_name: str | None = None

    @field_validator("trace_id")
    @classmethod
    def _trace_id_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("trace_id is required")
        return stripped

    @field_validator("tags", mode="before")
    @classmethod
    def _tags_default(cls, value: object) -> object:
        return [] if value is None else value


def parse_notify(raw: bytes | str | dict[str, Any]) -> SidecarTraceNotify:
    """Parse JSON bytes/str/dict into :class:`SidecarTraceNotify`.

    Missing optional fields become ``null`` / ``[]`` / ``false``.
    Unknown keys are ignored. Invalid ``trace_id`` / ``finish_reason`` raise.
    """
    if isinstance(raw, dict):
        payload: Any = raw
    else:
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SidecarNotifyParseError("notify body is not JSON") from exc
    if not isinstance(payload, dict):
        raise SidecarNotifyParseError("notify body is not a JSON object")
    normalized = dict(payload)
    if "finish_reason" in normalized and normalized["finish_reason"] not in _FINISH_REASONS:
        raise SidecarNotifyParseError(f"invalid finish_reason: {normalized['finish_reason']!r}")
    try:
        return SidecarTraceNotify.model_validate(normalized)
    except ValidationError as exc:
        raise SidecarNotifyParseError(str(exc)) from exc


def project_segment(project_id: str | None) -> str:
    if not isinstance(project_id, str):
        return _UNKNOWN_PROJECT_ID
    stripped = project_id.strip()
    return stripped if stripped else _UNKNOWN_PROJECT_ID


def eval_topic_name(prefix: str, project_id: str | None) -> str:
    return f"{prefix}{project_segment(project_id)}"


def notify_fields(observations: list[Any]) -> dict[str, Any]:
    """Filter fields worker copies from the observation tree into the notify."""
    obs = [item for item in observations if isinstance(item, dict)]
    newest_first = sorted(obs, key=_sort_ts, reverse=True)
    latency_ms, timestamp = _latency_and_timestamp(obs)
    return {
        "agent_name": _latest_nonempty(newest_first, "agent_name"),
        "name": _trace_name(newest_first),
        "user_id": _latest_nonempty(newest_first, "user_id"),
        "session_id": _latest_nonempty(newest_first, "session_id"),
        "environment": _latest_nonempty(newest_first, "environment"),
        "tags": _latest_tags(newest_first),
        "latency_ms": latency_ms,
        "has_error": _has_error(obs),
        "timestamp": timestamp,
    }


def build_notify(
    trace_id: str,
    finish_reason: str,
    observations: list[Any],
    *,
    project_id: str,
) -> dict[str, Any]:
    """Rebuild the MetaQ Body worker would send for this finished trace."""
    return {
        "trace_id": trace_id,
        "finish_reason": finish_reason,
        "project_id": project_segment(project_id),
        **notify_fields(observations),
    }


def notify_tag(payload: dict[str, Any]) -> str | None:
    return _prop_str(payload.get("project_id"))


def notify_properties(payload: dict[str, Any]) -> dict[str, str]:
    """User properties for the notify envelope. Omits empty values (worker same)."""
    props: dict[str, str] = {}
    for key in _PROP_KEYS:
        value = _prop_str(payload.get(key))
        if value is not None:
            props[key] = value
    tags = payload.get("tags")
    if isinstance(tags, list):
        joined = ",".join(part for part in (_prop_str(item) for item in tags) if part is not None)
        if joined:
            props["tags"] = joined
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, bool) and isinstance(timestamp, (int, float)):
        props["timestamp"] = str(int(timestamp))
    return props


def notify_console_fields(
    payload: dict[str, Any],
    *,
    topic_prefix: str,
) -> dict[str, Any]:
    """Topic / tag / key / properties / body for MetaQ console replay."""
    return {
        "topic": eval_topic_name(topic_prefix, payload.get("project_id")),
        "tag": notify_tag(payload),
        "key": str(payload["trace_id"]),
        "properties": notify_properties(payload),
        "body": payload,
    }


def _prop_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _field_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value != "" else None


def _ms(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _sort_ts(ob: dict[str, Any]) -> int:
    end = _ms(ob.get("end_time"))
    if end is not None:
        return end
    start = _ms(ob.get("start_time"))
    return start if start is not None else 0


def _latest_nonempty(obs: list[dict[str, Any]], key: str) -> str | None:
    for ob in obs:
        got = _field_str(ob.get(key))
        if got is not None:
            return got
    return None


def _latest_tags(obs: list[dict[str, Any]]) -> list[str]:
    for ob in obs:
        tags = ob.get("tags")
        if isinstance(tags, list) and tags:
            return [str(item) for item in tags]
    return []


def _is_io_root(ob: dict[str, Any]) -> bool:
    parent = ob.get("parent_span_id")
    return parent is None or parent == ""


def _trace_name(obs: list[dict[str, Any]]) -> str | None:
    got = _latest_nonempty(obs, "trace_name")
    if got is not None:
        return got
    for ob in obs:
        if _is_io_root(ob) or ob.get("is_app_root") is True:
            got = _field_str(ob.get("name"))
            if got is not None:
                return got
    return None


def _has_error(obs: list[dict[str, Any]]) -> bool:
    best = _LEVEL_RANK["DEFAULT"]
    for ob in obs:
        level = ob.get("level")
        rank = _LEVEL_RANK.get(level) if isinstance(level, str) else None
        if rank is None:
            rank = _LEVEL_RANK["DEFAULT"]
        best = max(best, rank)
    return best == _LEVEL_RANK["ERROR"]


def _latency_and_timestamp(obs: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    starts: list[int] = []
    highs: list[int] = []
    for ob in obs:
        start = _ms(ob.get("start_time"))
        end = _ms(ob.get("end_time"))
        if start is not None:
            starts.append(start)
            highs.append(end if end is not None and end > start else start)
        elif end is not None:
            highs.append(end)
    if not starts:
        return None, None
    return max(highs) - min(starts), min(starts)


__all__ = [
    "SidecarNotifyParseError",
    "SidecarTraceNotify",
    "build_notify",
    "eval_topic_name",
    "notify_console_fields",
    "notify_fields",
    "notify_properties",
    "notify_tag",
    "parse_notify",
    "project_segment",
]
