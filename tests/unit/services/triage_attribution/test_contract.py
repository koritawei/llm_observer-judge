"""Triage contract: TriageInput / TriageVerdict / TriageLevel + abstract TriageClassifier."""
# ruff: noqa: PLR2004

import pytest
from benchmark.attribution.service import (
    TriageClassifier,
    TriageInput,
    TriageLevel,
    TriageVerdict,
)
from pydantic import ValidationError


def test_levels_are_plain_strings() -> None:
    assert TriageLevel.MILD_BAD == "mild_bad"
    assert TriageLevel.NO_DECISION.value == "no_decision"


def test_input_minimal() -> None:
    request = TriageInput(trace_id="t", evaluator_name="e", project_id="p")
    assert request.score is None
    assert request.reason is None
    assert request.label is None


def test_input_carries_label() -> None:
    request = TriageInput(trace_id="t", evaluator_name="e", project_id="p", label="unsafe")
    assert request.label == "unsafe"


def test_input_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        TriageInput(trace_id="t", evaluator_name="e", project_id="p", bogus=1)


def test_input_rejects_empty_identity() -> None:
    with pytest.raises(ValidationError):
        TriageInput(trace_id="t", evaluator_name="", project_id="p")


def test_verdict_is_level_plus_attribute() -> None:
    verdict = TriageVerdict(level=TriageLevel.SEVERE_BAD, should_attribute=True)
    assert verdict.level == TriageLevel.SEVERE_BAD
    assert verdict.should_attribute is True


def test_verdict_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        TriageVerdict(level=TriageLevel.GOOD, should_attribute=False, bogus=1)


def test_classifier_requires_classify() -> None:
    with pytest.raises(TypeError):
        TriageClassifier()  # abstract class cannot be instantiated

    class _Concrete(TriageClassifier):
        def classify(self, request: TriageInput) -> TriageVerdict:
            return TriageVerdict(level=TriageLevel.GOOD, should_attribute=False)

    assert isinstance(_Concrete(), TriageClassifier)
