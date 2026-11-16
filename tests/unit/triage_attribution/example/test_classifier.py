"""Classification behavior of the example ExampleTriageClassifier."""
# ruff: noqa: PLR2004

from benchmark.attribution.service import TriageInput, TriageLevel
from business.agents.example.classifier import ExampleTriageClassifier


def _classify(score: float | None) -> tuple[TriageLevel, bool]:
    verdict = ExampleTriageClassifier().classify(
        TriageInput(trace_id="t", evaluator_name="example", project_id="p", score=score)
    )
    return verdict.level, verdict.should_attribute


def test_good_does_not_attribute() -> None:
    assert _classify(9.0) == (TriageLevel.GOOD, False)


def test_watch_does_not_attribute() -> None:
    assert _classify(6.0) == (TriageLevel.WATCH, False)


def test_mild_bad_attributes() -> None:
    assert _classify(3.0) == (TriageLevel.MILD_BAD, True)


def test_severe_bad_attributes() -> None:
    assert _classify(1.0) == (TriageLevel.SEVERE_BAD, True)


def test_no_score_is_no_decision() -> None:
    assert _classify(None) == (TriageLevel.NO_DECISION, False)
