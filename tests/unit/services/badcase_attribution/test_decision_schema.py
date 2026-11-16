"""Unit tests for the BadcaseAttributionDecision contract."""

import pytest
from benchmark.attribution.badcase.schemas import (
    AttributionResult,
    BadcaseAttributionDecision,
    EvidenceDetail,
    FailureAssessment,
    TraceToolCall,
)
from pydantic import ValidationError


def test_tool_calls_decision_requires_tool_calls() -> None:
    with pytest.raises(ValidationError, match="must include tool_calls"):
        BadcaseAttributionDecision(
            decision_type="tool_calls",
            rationale_summary="need data",
            tool_calls=[],
        )


def test_tool_calls_decision_rejects_final() -> None:
    with pytest.raises(ValidationError, match="must not include final"):
        BadcaseAttributionDecision(
            decision_type="tool_calls",
            rationale_summary="need data",
            tool_calls=[TraceToolCall(name="fetch_trace_skeleton")],
            final=AttributionResult(root_cause="x"),
        )


def test_final_decision_requires_final() -> None:
    with pytest.raises(ValidationError, match="must include final"):
        BadcaseAttributionDecision(
            decision_type="final",
            rationale_summary="done",
        )


def test_final_decision_rejects_tool_calls() -> None:
    with pytest.raises(ValidationError, match="must not include tool_calls"):
        BadcaseAttributionDecision(
            decision_type="final",
            rationale_summary="done",
            tool_calls=[TraceToolCall(name="fetch_trace_skeleton")],
            final=AttributionResult(root_cause="x"),
        )


def test_valid_tool_calls_and_final_shapes() -> None:
    tool = BadcaseAttributionDecision(
        decision_type="tool_calls",
        rationale_summary="skeleton first",
        tool_calls=[
            TraceToolCall(name="fetch_trace_skeleton", arguments={"trace_id": "t1"}),
        ],
    )
    assert tool.tool_calls[0].name == "fetch_trace_skeleton"

    final = BadcaseAttributionDecision(
        decision_type="final",
        rationale_summary="done",
        final=AttributionResult(
            trace_id="t1",
            case_id="t1",
            category="prompt",
            root_cause="ok",
            confidence=0.8,
        ),
    )
    assert final.final is not None
    assert final.final.root_cause == "ok"


def test_category_normalizes_to_closed_set() -> None:
    """Legal enum values pass through unchanged (including case and whitespace normalization)."""
    assert AttributionResult(category="tool").category == "tool"
    assert AttributionResult(category=" Infra ").category == "infra"


def test_category_invalid_falls_back_to_unknown() -> None:
    """Dirty values (for example the model copies an entire enum string) fall back to unknown, and the whole result is not discarded."""
    dirty = AttributionResult(
        category="prompt|tool|model|data|orchestration|infra|unknown",
        root_cause="still useful",
    )
    assert dirty.category == "unknown"
    assert dirty.root_cause == "still useful"
    assert AttributionResult(category="").category == "unknown"
    assert AttributionResult(category=None).category == "unknown"


def test_non_attributable_outcome_is_a_valid_conclusion() -> None:
    result = AttributionResult(conclusion_type="non_attributable_outcome")

    assert result.conclusion_type == "non_attributable_outcome"


def test_corrected_triage_level_normalizes() -> None:
    from benchmark.records.enums import TriageLevel

    assert (
        AttributionResult(corrected_triage_level="GOOD").corrected_triage_level == TriageLevel.GOOD
    )
    assert (
        AttributionResult(corrected_triage_level=" watch ").corrected_triage_level
        == TriageLevel.WATCH
    )
    assert AttributionResult(corrected_triage_level="").corrected_triage_level is None
    assert AttributionResult(corrected_triage_level="not_a_level").corrected_triage_level is None
    assert AttributionResult(corrected_triage_level=None).corrected_triage_level is None


def test_tool_calls_hard_max_length() -> None:
    from benchmark.attribution.badcase.schemas import HARD_MAX_TOOL_CALLS_PER_REQUEST

    too_many = [
        TraceToolCall(name="fetch_observation_detail", arguments={"observation_id": f"o{i}"})
        for i in range(HARD_MAX_TOOL_CALLS_PER_REQUEST + 1)
    ]
    with pytest.raises(ValidationError):
        BadcaseAttributionDecision(
            decision_type="tool_calls",
            rationale_summary="flood",
            tool_calls=too_many,
        )


def test_runtime_options_reject_max_tool_calls_per_round() -> None:
    """per-round cap is a code constant, not a BadcaseAttributionRuntimeOptions field."""
    from benchmark.attribution.badcase.schemas import BadcaseAttributionRuntimeOptions

    with pytest.raises(ValidationError):
        BadcaseAttributionRuntimeOptions.model_validate(
            {"max_react_rounds": 2, "max_tool_calls_per_round": 2}
        )


def test_evidence_detail_source_accepts_closed_set() -> None:
    """Legal enum values pass through unchanged."""
    assert EvidenceDetail(source="extra").source == "extra"
    assert EvidenceDetail(source="trace_tool").source == "trace_tool"
    assert EvidenceDetail(source="").source == ""


def test_evidence_detail_source_invalid_falls_back_to_empty() -> None:
    """Dirty values (for example the model associates the "Judge Failure Signal" section
    title in the prompt with a source category and invents judge_signal) normalize
    to an empty string, and do not raise ValidationError that would discard the
    whole evidence_details."""
    dirty = EvidenceDetail(field="score", expected="2", actual="0", source="judge_signal")
    assert dirty.source == ""
    assert dirty.field == "score"
    assert EvidenceDetail(source=None).source == ""


def test_runtime_options_reject_legacy_trace_context() -> None:
    from benchmark.attribution.badcase.schemas import BadcaseAttributionRuntimeOptions

    with pytest.raises(ValidationError):
        BadcaseAttributionRuntimeOptions.model_validate(
            {
                "max_react_rounds": 2,
                "trace_context": {"chunk_size": 2000},
            }
        )


def test_all_root_cause_codes_defaults_empty_for_legacy_data() -> None:
    """Legacy data (no new fields) defaults to an empty list; do not raise and do not fabricate values."""
    assert AttributionResult(root_cause="x").all_root_cause_codes == []


def test_all_root_cause_codes_dedup_and_filter() -> None:
    """Deduplicate while preserving order, and drop empty values and custom (other:) codes (same policy as normalize_root_cause_code)."""
    result = AttributionResult(
        root_cause="x",
        all_root_cause_codes=[
            "tool_param_error",
            "cmd_missing",
            "tool_param_error",
            "",
            None,
            "other:新失败",
        ],
    )
    assert result.all_root_cause_codes == ["tool_param_error", "cmd_missing"]


def test_all_root_cause_codes_rejects_non_list() -> None:
    """When the LLM mistakenly passes a scalar or string, normalize to an empty list and do not fail the whole result."""
    assert AttributionResult(all_root_cause_codes="tool_param_error").all_root_cause_codes == []
    assert AttributionResult(all_root_cause_codes=None).all_root_cause_codes == []


def test_failure_assessment_invalid_disposition_falls_back_to_unresolved() -> None:
    """When the model writes a conclusion_type value into disposition, normalize to unresolved and do not fail the whole decision."""
    assessment = FailureAssessment(
        failure_mode="cmd_missing",
        disposition="non_attributable_outcome",
    )

    assert assessment.disposition == "unresolved"
    assert FailureAssessment(failure_mode="cmd_missing").disposition == "unresolved"
