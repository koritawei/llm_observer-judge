"""Sidecar observation tree → TRACE-level eval fields and tool projections.

Kafka Path B reads Langfuse ``TraceRecord`` (trace-level input / output / metadata).
Sidecar has no separate TRACE row; the IO-root span (empty ``parent_span_id``)
is that record. Converter and ``trace`` tools both read this module — do not
re-implement tree walks in tools.py.
"""

from typing import Any

from benchmark.streaming.metadata import (
    AGENT_NAME_METADATA_KEY,
    is_discard_agent_name,
    metadata_agent_name,
)
from benchmark.trace.schema import (
    ObservationDetail,
    TraceSpanSkeleton,
    _to_preview,
)

_SIDECAR_OBSERVATIONS_KEY = "sidecar_observations"


def dict_observations(observations: list[Any]) -> list[dict[str, Any]]:
    return [item for item in observations if isinstance(item, dict)]


def latest_nonempty(observations: list[Any], key: str) -> str | None:
    newest_first = _newest_first(observations)
    for item in newest_first:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        nested = item.get("metadata")
        if isinstance(nested, dict):
            nested_value = nested.get(key)
            if isinstance(nested_value, str) and nested_value.strip():
                return nested_value.strip()
    return None


def extract_project_id(notify_project_id: str | None, observations: list[Any]) -> str | None:
    if notify_project_id and notify_project_id.strip():
        return notify_project_id.strip()
    return latest_nonempty(observations, "project_id")


def extract_agent_name(notify_agent_name: str | None, observations: list[Any]) -> str | None:
    if notify_agent_name and not is_discard_agent_name(notify_agent_name):
        return notify_agent_name.strip()
    for item in _newest_first(observations):
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        name = metadata_agent_name(meta) if isinstance(meta, dict) else None
        if isinstance(name, str) and not is_discard_agent_name(name):
            return name.strip()
        top = item.get(AGENT_NAME_METADATA_KEY) or item.get("agentName")
        if isinstance(top, str) and not is_discard_agent_name(top):
            return top.strip()
    return latest_nonempty(observations, "agent_name")


def extract_io(observations: list[Any]) -> tuple[Any, Any]:
    """TRACE-level input/output: IO-root span, else newest span that carries IO."""
    record = _io_root_with_io(observations)
    if record is not None:
        return record.get("input"), record.get("output")
    for item in _newest_first(observations):
        if _has_io(item):
            return item.get("input"), item.get("output")
    return None, None


def extract_metadata(observations: list[Any]) -> dict[str, Any]:
    """TRACE-level metadata from the IO-root span only (not child span metadata)."""
    roots = [item for item in dict_observations(observations) if _is_io_root(item)]
    for item in _newest_first(roots):
        meta = item.get("metadata")
        if isinstance(meta, dict) and meta:
            return dict(meta)
    return {}


def extract_trace_fields(observations: list[Any]) -> tuple[Any, Any, dict[str, Any]]:
    """``(input, output, metadata)`` aligned with Kafka ``TraceRecord``."""
    inp, out = extract_io(observations)
    return inp, out, extract_metadata(observations)


def observations_to_trace_evidence(
    observations: list[Any],
    trace_id: str | None,
) -> dict[str, Any]:
    """Build the skeleton preprocess expects, without calling Langfuse."""
    spans: list[dict[str, Any]] = []
    generation_detail: dict[str, Any] | None = None
    latest_start = ""
    for item in dict_observations(observations):
        span_id = _span_id(item)
        span_type = _type_name(item) or ""
        span = {
            "id": span_id,
            "name": item.get("name"),
            "type": span_type,
            "input_preview": item.get("input"),
            "output_preview": item.get("output"),
            "start_time": item.get("start_time"),
            "parent_span_id": item.get("parent_span_id"),
        }
        spans.append(span)
        type_name = str(span_type).lower()
        if type_name == "generation" and span_id:
            start = item.get("start_time")
            start_key = str(start) if start is not None else ""
            if generation_detail is None or start_key > latest_start:
                generation_detail = item
                latest_start = start_key
    evidence: dict[str, Any] = {
        "business_trace_id": trace_id,
        "spans": spans,
        "observations": observations,
    }
    if generation_detail is not None:
        evidence["generation_detail"] = generation_detail
    return evidence


def find_observation(observations: list[Any], span_id: str) -> dict[str, Any] | None:
    wanted = span_id.strip()
    if not wanted:
        return None
    for item in dict_observations(observations):
        if _span_id(item) == wanted:
            return item
    return None


def observations_to_full_trace(observations: list[Any], trace_id: str) -> dict[str, Any]:
    mapped: list[dict[str, Any]] = []
    for item in dict_observations(observations):
        row = dict(item)
        sid = _span_id(item)
        if sid:
            row.setdefault("id", sid)
        mapped.append(row)
    return {"id": trace_id, "observations": mapped}


def to_span_skeleton(item: dict[str, Any], preview_chars: int) -> TraceSpanSkeleton:
    return TraceSpanSkeleton(
        id=_span_id(item),
        name=_optional_str(item.get("name")),
        type=_type_name(item),
        status=_optional_str(item.get("level") or item.get("status")),
        parent_observation_id=_parent_id(item),
        start_time=_as_time(item.get("start_time") or item.get("startTime")),
        end_time=_as_time(item.get("end_time") or item.get("endTime")),
        input_preview=_to_preview(item.get("input"), preview_chars),
        output_preview=_to_preview(item.get("output"), preview_chars),
    )


def to_observation_detail(item: dict[str, Any], trace_id: str) -> ObservationDetail:
    return ObservationDetail.model_validate(
        {
            **item,
            "id": _span_id(item),
            "trace_id": item.get("trace_id") or item.get("traceId") or trace_id,
            "name": item.get("name"),
            "type": _type_name(item),
            "input": item.get("input"),
            "output": item.get("output"),
            "metadata": item.get("metadata"),
        }
    )


def filter_observations(
    observations: list[Any],
    *,
    name: str | None,
    type: str | None,
    status: str | None,
) -> list[dict[str, Any]]:
    wanted_name = (name or "").strip().lower()
    wanted_type = (type or "").strip().lower()
    wanted_status = (status or "").strip().lower()
    matched: list[dict[str, Any]] = []
    for item in dict_observations(observations):
        if wanted_name and str(item.get("name") or "").strip().lower() != wanted_name:
            continue
        if wanted_type and str(_type_name(item) or "").strip().lower() != wanted_type:
            continue
        actual_status = str(item.get("level") or item.get("status") or "").strip().lower()
        if wanted_status and actual_status != wanted_status:
            continue
        matched.append(item)
    return matched


def _is_io_root(item: dict[str, Any]) -> bool:
    parent = item.get("parent_span_id")
    return parent is None or parent == ""


def _has_io(item: dict[str, Any]) -> bool:
    return item.get("input") is not None or item.get("output") is not None


def _io_root_with_io(observations: list[Any]) -> dict[str, Any] | None:
    roots = [item for item in dict_observations(observations) if _is_io_root(item)]
    for item in _newest_first(roots):
        if _has_io(item):
            return item
    return None


def _newest_first(observations: list[Any]) -> list[dict[str, Any]]:
    items = dict_observations(observations)

    def _sort_key(item: dict[str, Any]) -> int:
        end = item.get("end_time")
        start = item.get("start_time")
        if isinstance(end, (int, float)):
            return int(end)
        if isinstance(start, (int, float)):
            return int(start)
        return 0

    return sorted(items, key=_sort_key, reverse=True)


def _span_id(item: dict[str, Any]) -> str:
    raw = item.get("span_id") or item.get("id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if raw is None or isinstance(raw, (dict, list)):
        return ""
    return str(raw).strip()


def _type_name(item: dict[str, Any]) -> str | None:
    return _optional_str(item.get("type") or item.get("observation_type"))


def _parent_id(item: dict[str, Any]) -> str | None:
    raw = (
        item.get("parent_span_id")
        or item.get("parentObservationId")
        or item.get("parent_observation_id")
    )
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _as_time(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(int(value))
    return str(value)


__all__ = [
    "_SIDECAR_OBSERVATIONS_KEY",
    "_span_id",
    "dict_observations",
    "extract_agent_name",
    "extract_io",
    "extract_metadata",
    "extract_project_id",
    "extract_trace_fields",
    "filter_observations",
    "find_observation",
    "observations_to_full_trace",
    "observations_to_trace_evidence",
    "to_observation_detail",
    "to_span_skeleton",
]
