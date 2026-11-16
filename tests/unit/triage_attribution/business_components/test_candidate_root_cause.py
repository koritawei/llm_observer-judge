"""Shared contract and conservative gate tests for an `other:` candidate root cause."""

import pytest
from benchmark.attribution.badcase.schemas import AttributionResult
from benchmark.attribution.components.candidate_root_cause import CandidateRootCause
from pydantic import ValidationError


def _candidate(**updates: object) -> CandidateRootCause:
    payload: dict[str, object] = {
        "raw_code": "other:tool_schema_gap",
        "title": "工具 schema 未覆盖关键属性",
        "surface": "tool_data",
        "component": "search_place",
        "symptom": "missing_price_attribute",
        "details": {"missing_fields": ["price"]},
        "evidence_refs": ["extra:failure_modes"],
        "reasoning": "工具返回成功，但数据中不存在用户所需属性。",
        "agent_scope": "vehicle",
    }
    payload.update(updates)
    return CandidateRootCause.model_validate(payload)


def test_candidate_key_is_stable_and_scope_isolated() -> None:
    vehicle = _candidate()
    vui = vehicle.with_agent_scope("vui")

    assert vehicle.candidate_key() == "v1:vehicle:tool_data:search_place:missing_price_attribute"
    assert vehicle.candidate_key() != vui.candidate_key()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_code", "novel:tool_schema_gap"),
        ("raw_code", "other:ToolSchemaGap"),
        ("surface", "unbounded"),
        ("component", "search place"),
        ("symptom", "missing-price"),
        ("evidence_refs", []),
    ],
)
def test_candidate_rejects_invalid_contract(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _candidate(**{field: value})


def test_candidate_rejects_non_json_safe_details() -> None:
    with pytest.raises(ValidationError):
        _candidate(details={"invalid": {"value"}})


def test_candidate_forces_human_review_and_zero_dispatch() -> None:
    result = AttributionResult(
        root_cause="工具数据覆盖不足",
        should_dispatch=True,
        corrected_triage_level="good",
        candidate_root_cause=_candidate(),
    )

    assert result.root_cause_code is None
    assert result.all_root_cause_codes == []
    assert result.should_dispatch is False
    assert result.needs_human_review is True
    assert result.corrected_triage_level is None
    assert result.candidate_root_cause is not None


def test_registered_root_cause_discards_duplicate_candidate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        result = AttributionResult(
            root_cause="工具成功但缺少字段",
            root_cause_code="tool_data_missing",
            all_root_cause_codes=["tool_data_missing"],
            candidate_root_cause=_candidate(),
        )

    assert result.root_cause_code == "tool_data_missing"
    assert result.candidate_root_cause is None
    assert "candidate_root_cause_discarded_formal_code_present" in caplog.text
