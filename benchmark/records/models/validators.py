"""Validators for TraceRecord bucket fields.

Light structural checks for TraceRecord JSONField buckets.
Validators are inlined on the model field as ``validators=[...]``,
and Tortoise ORM calls them on assign/save.

Policy: check only the key fields of each bucket (core ids, ranged values,
and the core result dict). Other fields may be written freely (forward compatible).

Common rules:
  - None is valid (means unset)
  - a non-dict type is rejected
  - key fields are type/range checked
  - unknown keys are allowed (forward compatible)
"""

from collections.abc import Callable
from typing import Any


class TraceRecordValidationError(ValueError):
    """Raised when a TraceRecord bucket field fails validation."""

    field_name: str
    detail: str

    def __init__(self, field_name: str, detail: str) -> None:
        self.field_name = field_name
        self.detail = detail
        super().__init__(f"{field_name}: {detail}")


# ── Internal type-check helpers ──────────────────────────────────────────


def _ensure_dict(field_name: str, value: Any) -> dict[str, Any]:
    """Require a dict, otherwise raise TraceRecordValidationError."""
    if not isinstance(value, dict):
        raise TraceRecordValidationError(
            field_name,
            f"期望 dict，实际 {type(value).__name__}",
        )
    return value


def _check_str(field_name: str, key: str, value: Any) -> None:
    if value is not None and not isinstance(value, str):
        raise TraceRecordValidationError(
            field_name, f"{key} 期望 str，实际 {type(value).__name__}",
        )


def _check_dict(field_name: str, key: str, value: Any) -> None:
    if value is not None and not isinstance(value, dict):
        raise TraceRecordValidationError(
            field_name, f"{key} 期望 dict，实际 {type(value).__name__}",
        )


def _check_float_range(
    field_name: str, key: str, value: Any, lo: float, hi: float,
) -> None:
    """Check that a float/int is inside [lo, hi] (bool excluded)."""
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int | float)
    ):
        raise TraceRecordValidationError(
            field_name, f"{key} 期望 float，实际 {type(value).__name__}",
        )
    if value is not None and not (lo <= value <= hi):
        raise TraceRecordValidationError(
            field_name, f"{key} 期望 {lo}-{hi}，实际 {value}",
        )


def _check_int_nonneg(field_name: str, key: str, value: Any) -> None:
    """Check a non-negative int (bool excluded)."""
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise TraceRecordValidationError(
            field_name, f"{key} 期望非负 int，实际 {value!r}",
        )


# ── Bucket validators (key fields only) ─────────────────────────────────


def validate_judge_json(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate the judge_json bucket — judge result.

    Key field: score (the score result, str)
    """
    if value is None:
        return None
    d = _ensure_dict("judge_json", value)
    _check_str("judge_json", "score", d.get("score"))
    return d


def validate_attribution_detail_json(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the attribution_detail_json bucket — attribution detail.

    Key fields:
      - confidence (confidence, float 0-1)
      - needs_human_review (bool, if present)
    """
    if value is None:
        return None
    d = _ensure_dict("attribution_detail_json", value)
    _check_float_range(
        "attribution_detail_json", "confidence", d.get("confidence"), 0.0, 1.0,
    )
    needs_human_review = d.get("needs_human_review")
    if needs_human_review is not None and not isinstance(needs_human_review, bool):
        raise TraceRecordValidationError(
            "attribution_detail_json",
            f"needs_human_review 期望 bool，实际 {type(needs_human_review).__name__}",
        )
    return d


def validate_notification_dispatch_json(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate the notification_dispatch_json bucket — dispatch info.

    Key fields:
      - notification_ref (dispatch reference, str)
      - notification_skip_reason (why it was not dispatched, str; written only when should_dispatch=False and similar cases)
    """
    if value is None:
        return None
    d = _ensure_dict("notification_dispatch_json", value)
    _check_str("notification_dispatch_json", "notification_ref", d.get("notification_ref"))
    _check_str("notification_dispatch_json", "notification_skip_reason", d.get("notification_skip_reason"))
    return d


def validate_notification_job_json(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate the notification_job_json bucket — Aone ticket detail.

    Key fields:
      - notification_job_status (ticket-job status, str)
      - notification_job_id (ticket id, str)
      - notification_job_attempt_count (retry count, non-negative int)
    """
    if value is None:
        return None
    d = _ensure_dict("notification_job_json", value)
    _check_str("notification_job_json", "notification_job_status", d.get("notification_job_status"))
    _check_str("notification_job_json", "notification_job_id", d.get("notification_job_id"))
    _check_int_nonneg(
        "notification_job_json", "notification_job_attempt_count",
        d.get("notification_job_attempt_count"),
    )
    return d


def validate_label_studio_attribution_annotation_json(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the label_studio_attribution_annotation_json bucket — attribution-result annotation.

    Key field: attribution_result (the LS annotation result, dict)
    """
    if value is None:
        return None
    d = _ensure_dict("label_studio_attribution_annotation_json", value)
    _check_dict(
        "label_studio_attribution_annotation_json",
        "attribution_result", d.get("attribution_result"),
    )
    return d


def validate_label_studio_badcase_judge_annotation_json(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the label_studio_badcase_judge_annotation_json bucket — badcase judge-result annotation.

    Key field: triage_result (the LS annotation result, dict)
    """
    if value is None:
        return None
    d = _ensure_dict("label_studio_badcase_judge_annotation_json", value)
    _check_dict(
        "label_studio_badcase_judge_annotation_json",
        "triage_result", d.get("triage_result"),
    )
    return d


# ── Registry ──────────────────────────────────────────────────────────

# Bucket field name → validator
TRACE_RECORD_VALIDATORS: dict[
    str, Callable[[dict[str, Any] | None], dict[str, Any] | None]
] = {
    "judge_json": validate_judge_json,
    "attribution_detail_json": validate_attribution_detail_json,
    "notification_dispatch_json": validate_notification_dispatch_json,
    "notification_job_json": validate_notification_job_json,
    "label_studio_attribution_annotation_json": validate_label_studio_attribution_annotation_json,
    "label_studio_badcase_judge_annotation_json": validate_label_studio_badcase_judge_annotation_json,
}

# All bucket field names (for a fast membership check)
TRACE_RECORD_BUCKET_FIELDS: frozenset[str] = frozenset(TRACE_RECORD_VALIDATORS.keys())


def validate_trace_record_bucket(field_name: str, value: Any) -> Any:
    """Validate one bucket field. If the field is not in the registry, return it unchanged."""
    validator = TRACE_RECORD_VALIDATORS.get(field_name)
    if validator is None:
        return value
    return validator(value)


def validate_trace_record_buckets(**kwargs: Any) -> dict[str, Any]:
    """Validate every bucket field in kwargs and return the validated kwargs.

    Usage:
        kwargs = validate_trace_record_buckets(**kwargs)
    """
    result = dict(kwargs)
    for field_name, validator in TRACE_RECORD_VALIDATORS.items():
        if field_name in result:
            result[field_name] = validator(result[field_name])
    return result


__all__ = [
    "TraceRecordValidationError",
    "TRACE_RECORD_VALIDATORS",
    "TRACE_RECORD_BUCKET_FIELDS",
    "validate_trace_record_bucket",
    "validate_trace_record_buckets",
    "validate_judge_json",
    "validate_attribution_detail_json",
    "validate_notification_dispatch_json",
    "validate_notification_job_json",
    "validate_label_studio_attribution_annotation_json",
    "validate_label_studio_badcase_judge_annotation_json",
]
