"""TraceRecord bucket validators — type and range checks for the critical fields."""

import pytest
from benchmark.records.models.validators import (
    TraceRecordValidationError,
    validate_notification_job_json,
    validate_attribution_detail_json,
    validate_judge_json,
    validate_trace_record_bucket,
    validate_trace_record_buckets,
)


def test_validate_judge_json_accepts_none_and_valid_dict() -> None:
    assert validate_judge_json(None) is None
    out = validate_judge_json({"score": "7.0", "extra": "ok"})
    assert out == {"score": "7.0", "extra": "ok"}


def test_validate_judge_json_rejects_non_dict() -> None:
    with pytest.raises(TraceRecordValidationError, match="judge_json"):
        validate_judge_json("bad")


def test_validate_judge_json_rejects_non_string_score() -> None:
    with pytest.raises(TraceRecordValidationError, match="score"):
        validate_judge_json({"score": 7})


def test_validate_attribution_detail_json_rejects_confidence_out_of_range() -> None:
    with pytest.raises(TraceRecordValidationError, match="confidence"):
        validate_attribution_detail_json({"confidence": 1.5})


def test_validate_attribution_detail_json_rejects_bool_confidence() -> None:
    with pytest.raises(TraceRecordValidationError, match="confidence"):
        validate_attribution_detail_json({"confidence": True})


def test_validate_attribution_detail_json_rejects_non_bool_needs_human_review() -> None:
    with pytest.raises(TraceRecordValidationError, match="needs_human_review"):
        validate_attribution_detail_json({"needs_human_review": "yes"})


def test_validate_attribution_detail_json_accepts_bool_needs_human_review() -> None:
    out = validate_attribution_detail_json(
        {"confidence": 0.5, "needs_human_review": False},
    )
    assert out is not None
    assert out["needs_human_review"] is False


def test_validate_notification_job_json_rejects_negative_attempt_count() -> None:
    with pytest.raises(TraceRecordValidationError, match="notification_job_attempt_count"):
        validate_notification_job_json({"notification_job_attempt_count": -1})


def test_validate_trace_record_bucket_unknown_field_passthrough() -> None:
    assert validate_trace_record_bucket("trace_meta_json", {"x": 1}) == {"x": 1}


def test_validate_trace_record_buckets_validates_present_fields_only() -> None:
    out = validate_trace_record_buckets(
        judge_json={"score": "1"},
        attribution_detail_json={"confidence": 0.5},
        trace_meta_json={"note": "keep"},
    )
    assert out["judge_json"]["score"] == "1"
    assert out["attribution_detail_json"]["confidence"] == 0.5
    assert out["trace_meta_json"] == {"note": "keep"}
