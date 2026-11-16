"""Compatibility adapter from the unified evaluation envelope to the existing business extra read view."""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from benchmark.pipeline.envelope import AgentEvalEnvelope
from benchmark.pipeline.extensions import (
    VEHICLE_EXTENSION_TYPE,
    VUI_EXTENSION_TYPE,
    VUI_EXTENSION_VERSION,
)

ENVELOPE_SCHEMA_VERSION = "agent_eval_extra.v1"


def normalize_vehicle_extra(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """Restore a Vehicle unified envelope to the legacy top-level read view; return any other payload unchanged."""

    if not isinstance(extra, Mapping):
        return {}
    if extra.get("schema_version") != ENVELOPE_SCHEMA_VERSION:
        return dict(extra)
    extension = extra.get("business_extension")
    if (
        not isinstance(extension, Mapping)
        or extension.get("extension_type") != VEHICLE_EXTENSION_TYPE
    ):
        return dict(extra)
    payload = extension.get("payload")
    if not isinstance(payload, Mapping):
        return dict(extra)

    signals = extra.get("evaluation_signals") or []
    view: dict[str, Any] = dict(payload)
    view["score"] = payload.get("score", extra.get("primary_score"))
    view["reason"] = payload.get("reason") or extra.get("summary_reason") or ""

    # Use an `is not None` check so an explicit empty list [] in the payload is not skipped through an `or` chain to the fallback
    dr = payload.get("dimension_results")
    view["dimension_results"] = dr if dr is not None else (extra.get("dimension_results") or [])

    cr = payload.get("checker_results")
    view["checker_results"] = (
        cr
        if cr is not None
        else [
            signal
            for signal in signals
            if isinstance(signal, Mapping) and signal.get("source_type") == "checker"
        ]
    )

    rf = payload.get("risk_flags")
    view["risk_flags"] = (
        rf
        if rf is not None
        else list(
            dict.fromkeys(
                flag
                for signal in signals
                if isinstance(signal, Mapping)
                for flag in signal.get("risk_flags") or []
            )
        )
    )

    hr = payload.get("hard_rule_hits")
    view["hard_rule_hits"] = (
        hr
        if hr is not None
        else list(
            dict.fromkeys(
                hit
                for signal in signals
                if isinstance(signal, Mapping)
                for hit in signal.get("hard_rule_hits") or []
            )
        )
    )
    return view


def normalize_vui_extra(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """Restore a VUI unified envelope to the legacy top-level read view."""

    payload = _payload_of(extra, VUI_EXTENSION_TYPE)
    if payload is None:
        return dict(extra) if isinstance(extra, Mapping) else {}
    try:
        # Runner observation channels accompany the canonical envelope in EvaluatorResult.extra.
        canonical = {
            key: value
            for key, value in extra.items()
            if key not in {"stage_report", "envelope_identity"}
        }
        # Older stored VUI envelopes omitted business_id; the extension type identifies them.
        canonical.setdefault("business_id", "vui_agent")
        envelope = AgentEvalEnvelope.model_validate(canonical)
    except ValidationError:
        return dict(extra)
    extension = envelope.business_extension
    if envelope.business_id != "vui_agent" or extension.extension_version != VUI_EXTENSION_VERSION:
        return dict(extra)
    view = dict(payload)
    view["schema_version"] = "vui_eval_extra.v1"
    checkers = [signal for signal in envelope.evaluation_signals if signal.source_type == "checker"]
    view["checker_results"] = [signal.model_dump(mode="json") for signal in checkers]
    view["hard_rule_hits"] = list(
        dict.fromkeys(hit for signal in checkers for hit in signal.hard_rule_hits)
    )
    view["risk_flags"] = list(
        dict.fromkeys(flag for signal in envelope.evaluation_signals for flag in signal.risk_flags)
    )
    view.setdefault("score", envelope.primary_score)
    view.setdefault("reason", envelope.summary_reason)
    snapshot = view.get("output_snapshot")
    if isinstance(snapshot, Mapping):
        view["output_snapshot"] = json.dumps(snapshot, ensure_ascii=False)
    return view


def normalize_eval_extra(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """Dispatch by business-extension type to the matching compatibility read view."""

    extension_type = _extension_type_of(extra)
    if extension_type == VEHICLE_EXTENSION_TYPE:
        return normalize_vehicle_extra(extra)
    if extension_type == VUI_EXTENSION_TYPE:
        return normalize_vui_extra(extra)
    return dict(extra) if isinstance(extra, Mapping) else {}


def _extension_type_of(extra: Mapping[str, Any] | None) -> str:
    if not isinstance(extra, Mapping) or extra.get("schema_version") != ENVELOPE_SCHEMA_VERSION:
        return ""
    extension = extra.get("business_extension")
    if not isinstance(extension, Mapping):
        return ""
    return str(extension.get("extension_type") or "")


def _payload_of(extra: Mapping[str, Any] | None, extension_type: str) -> Mapping[str, Any] | None:
    if not isinstance(extra, Mapping) or extra.get("schema_version") != ENVELOPE_SCHEMA_VERSION:
        return None
    extension = extra.get("business_extension")
    if not isinstance(extension, Mapping) or extension.get("extension_type") != extension_type:
        return None
    payload = extension.get("payload")
    return payload if isinstance(payload, Mapping) else None


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "normalize_eval_extra",
    "normalize_vehicle_extra",
    "normalize_vui_extra",
]
