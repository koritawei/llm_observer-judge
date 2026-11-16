"""The new triage_decide signature — call classifier.classify() to decide whether to attribute."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from benchmark.attribution.service import (
    TriageClassifier,
    TriageInput,
    TriageLevel,
    TriageVerdict,
)
from benchmark.attribution.service.triage import TriageConfig, triage_decide


@dataclass
class _FakeEvaluatorRequest:
    trace_id: str | None = "tr-1"
    agent_name: str | None = "amap-agent"
    project_id: str | None = "proj-x"


@dataclass
class _FakeEvaluatorResult:
    score: float | None = 2.5
    reason: str | None = "bad answer"


class _StubClassifier(TriageClassifier):
    def __init__(self, verdict: TriageVerdict) -> None:
        self._verdict = verdict

    def classify(self, request: TriageInput) -> TriageVerdict:
        return self._verdict


@pytest.mark.asyncio
async def test_good_level_skips_attribution() -> None:
    verdict = TriageVerdict(level=TriageLevel.GOOD, should_attribute=False)
    decision = await triage_decide(
        evaluator_name="example",
        evaluator_request=_FakeEvaluatorRequest(),
        evaluator_result=_FakeEvaluatorResult(score=9.0),
        classifier=_StubClassifier(verdict),
        config=TriageConfig(),
    )
    assert decision.should_attribute is False
    assert decision.final_level == "good"


@pytest.mark.asyncio
async def test_severe_bad_triggers_attribution() -> None:
    verdict = TriageVerdict(level=TriageLevel.SEVERE_BAD, should_attribute=True)
    decision = await triage_decide(
        evaluator_name="example",
        evaluator_request=_FakeEvaluatorRequest(),
        evaluator_result=_FakeEvaluatorResult(score=1.0),
        classifier=_StubClassifier(verdict),
        config=TriageConfig(),
    )
    assert decision.should_attribute is True
    assert decision.final_level == "severe_bad"


@pytest.mark.asyncio
async def test_classifier_exception_returns_should_attribute_false() -> None:
    """When the classifier raises, triage_decide falls back to not attributing (so the main path is not blocked)."""

    class _BoomClassifier(TriageClassifier):
        def classify(self, request: TriageInput) -> TriageVerdict:
            raise RuntimeError("classifier boom")

    decision = await triage_decide(
        evaluator_name="example",
        evaluator_request=_FakeEvaluatorRequest(),
        evaluator_result=_FakeEvaluatorResult(),
        classifier=_BoomClassifier(),
        config=TriageConfig(),
    )
    assert decision.should_attribute is False
    assert "classifier_failed" in decision.reason


@pytest.mark.asyncio
async def test_missing_trace_id_raises_runtime_error() -> None:
    """When trace_id / project_id is missing, do not fall back to a literal anymore, and raise RuntimeError directly.

    This replaces the old 'or unknown' fallback — the entry guard guarantees both fields are non-empty, and the main path cannot reach it.
    """
    with pytest.raises(RuntimeError, match="triage_decide called with missing identity"):
        await triage_decide(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(trace_id=None, project_id=None),
            evaluator_result=_FakeEvaluatorResult(),
            classifier=_StubClassifier(
                TriageVerdict(level=TriageLevel.MILD_BAD, should_attribute=True)
            ),
            config=TriageConfig(),
        )
