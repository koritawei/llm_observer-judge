"""TriageBeforeRunHook looks up the classifiers table by evaluator_name."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from benchmark.evaluators.base import EvaluatorResult
from benchmark.attribution.service import (
    TriageClassifier,
    TriageInput,
    TriageLevel,
    TriageVerdict,
)
from benchmark.attribution.service.service import BeforeRunInput, TriageBeforeRunHook
from benchmark.attribution.service.triage import TriageConfig


@dataclass
class _FakeRequest:
    trace_id: str | None = "tr-1"
    project_id: str | None = "p"


@dataclass
class _FakeResult:
    score: float | None = 1.0


class _SevereClassifier(TriageClassifier):
    def classify(self, request: TriageInput) -> TriageVerdict:
        return TriageVerdict(level=TriageLevel.SEVERE_BAD, should_attribute=True)


class _GoodClassifier(TriageClassifier):
    def classify(self, request: TriageInput) -> TriageVerdict:
        return TriageVerdict(level=TriageLevel.GOOD, should_attribute=False)


@pytest.mark.asyncio
async def test_known_evaluator_uses_matched_classifier() -> None:
    hook = TriageBeforeRunHook(
        classifiers={"example": _SevereClassifier(), "other": _GoodClassifier()},
        triage_config=TriageConfig(),
    )
    decision = await hook(
        BeforeRunInput(
            evaluator_name="example",
            evaluator_request=_FakeRequest(),
            evaluator_result=_FakeResult(),
        ),
    )
    assert decision.should_proceed is True


@pytest.mark.asyncio
async def test_dynamic_threshold_uses_server_computed_score() -> None:
    hook = TriageBeforeRunHook(
        classifiers={"example": _GoodClassifier()},
        triage_config=TriageConfig(),
    )
    result = EvaluatorResult(
        score=99.0,
        sub_scores={"a": 5.0},
        extra={
            "dynamic_rubric_opt_in": True,
            "declared_weights": {"a": 1.0},
            "weights": {"a": 1.0},
            "attribution_threshold": 6.0,
            "sub_score_scale": [0.0, 10.0],
        },
    )
    decision = await hook(
        BeforeRunInput(
            evaluator_name="example",
            evaluator_request=_FakeRequest(),
            evaluator_result=result,
        ),
    )
    assert decision.should_proceed is True
    assert decision.reason == "dynamic_attribution_threshold"
    assert decision.metadata is not None
    assert decision.metadata["computed_fused_score"] == 5.0


async def test_unknown_evaluator_skips() -> None:
    """When evaluator_name is not found in the classifiers table, skip attribution (do not raise)."""
    hook = TriageBeforeRunHook(
        classifiers={"example": _SevereClassifier()},
        triage_config=TriageConfig(),
    )
    decision = await hook(
        BeforeRunInput(
            evaluator_name="missing",
            evaluator_request=_FakeRequest(),
            evaluator_result=_FakeResult(),
        ),
    )
    assert decision.should_proceed is False
    assert "no_classifier" in (decision.reason or "")


@pytest.mark.asyncio
async def test_dynamic_attribution_threshold_bypasses_classifier() -> None:
    """When the opt-in dynamic threshold is in effect, use dynamic_attribution_threshold, and do not look up the classifier."""
    from benchmark.evaluators.base import EvaluatorResult

    hook = TriageBeforeRunHook(
        classifiers={"example": _SevereClassifier()},
        triage_config=TriageConfig(),
    )
    low = EvaluatorResult(
        score=3.0,
        sub_scores={"a": 3.0},
        extra={
            "dynamic_rubric_opt_in": True,
            "declared_weights": {"a": 1.0},
            "weights": {"a": 1.0},
            "attribution_threshold": 6.0,
            "sub_score_scale": [0.0, 10.0],
        },
    )
    decision = await hook(
        BeforeRunInput(
            evaluator_name="example",
            evaluator_request=_FakeRequest(),
            evaluator_result=low,
        ),
    )
    assert decision.should_proceed is True
    assert decision.reason == "dynamic_attribution_threshold"
    assert decision.metadata["attribution_threshold"] == 6.0

    high = EvaluatorResult(
        score=9.0,
        sub_scores={"a": 9.0},
        extra={
            "dynamic_rubric_opt_in": True,
            "declared_weights": {"a": 1.0},
            "weights": {"a": 1.0},
            "attribution_threshold": 6.0,
            "sub_score_scale": [0.0, 10.0],
        },
    )
    decision_skip = await hook(
        BeforeRunInput(
            evaluator_name="example",
            evaluator_request=_FakeRequest(),
            evaluator_result=high,
        ),
    )
    assert decision_skip.should_proceed is False
    assert decision_skip.reason == "dynamic_attribution_threshold"
