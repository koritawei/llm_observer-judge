"""Merge multiple criteria that share an evaluator_ref / sub_scores fan-out."""

import pytest
from benchmark.evaluators.base import EvaluatorResult
from benchmark.runs.common.criterion_eval import (
    can_fanout_sub_scores,
    evaluate_criteria_grouped,
    fanout_criterion_scores,
    group_criteria_by_evaluator_ref,
)
from benchmark.reporting.framework.schema import Criterion


def test_group_criteria_by_evaluator_ref_preserves_order() -> None:
    criteria = [
        Criterion(id="a", name="A", weight=0.5, evaluator_ref="ex"),
        Criterion(id="b", name="B", weight=0.5, evaluator_ref="ex"),
        Criterion(id="c", name="C", weight=1.0, evaluator_ref="other"),
    ]
    groups = group_criteria_by_evaluator_ref(criteria)
    assert list(groups.keys()) == ["ex", "other"]
    assert [c.id for c in groups["ex"]] == ["a", "b"]
    assert [c.id for c in groups["other"]] == ["c"]


def test_can_fanout_requires_all_criterion_ids() -> None:
    result = EvaluatorResult(score=8.0, reason="ok", sub_scores={"a": 7.0, "b": 9.0})
    criteria = [
        {"id": "a", "evaluator_ref": "ex"},
        {"id": "b", "evaluator_ref": "ex"},
    ]
    assert can_fanout_sub_scores(result, criteria) is True
    assert (
        can_fanout_sub_scores(
            EvaluatorResult(score=8.0, reason="ok", sub_scores={"a": 7.0}),
            criteria,
        )
        is False
    )


def test_fanout_criterion_scores_splits_sub_scores() -> None:
    result = EvaluatorResult(
        score=8.0,
        reason="overall",
        sub_scores={"准确度": 7.0, "相关性": 9.0},
        extra={"k": 1},
    )
    criteria = [
        {"id": "准确度", "evaluator_ref": "example"},
        {"id": "相关性", "evaluator_ref": "example"},
    ]
    out = fanout_criterion_scores(result, criteria)
    assert out["准确度"].score == 7.0
    assert out["相关性"].score == 9.0
    assert out["准确度"].reason == "overall"
    assert out["准确度"].sub_scores == {"准确度": 7.0, "相关性": 9.0}
    assert out["准确度"].extra == {"k": 1}


@pytest.mark.asyncio
async def test_evaluate_criteria_grouped_fans_out_one_call() -> None:
    calls: list[str] = []

    async def evaluate(ref: str) -> EvaluatorResult:
        calls.append(ref)
        return EvaluatorResult(
            score=8.0,
            reason="ok",
            sub_scores={"a": 6.0, "b": 9.0},
        )

    criteria = [
        {"id": "a", "evaluator_ref": "ex"},
        {"id": "b", "evaluator_ref": "ex"},
    ]
    out = await evaluate_criteria_grouped(criteria, evaluate)
    assert calls == ["ex"]
    assert out["a"].score == 6.0
    assert out["b"].score == 9.0


@pytest.mark.asyncio
async def test_evaluate_criteria_grouped_falls_back_without_aligned_sub_scores() -> None:
    calls: list[str] = []

    async def evaluate(ref: str) -> EvaluatorResult:
        calls.append(ref)
        return EvaluatorResult(score=float(len(calls)), reason="ok")

    criteria = [
        {"id": "D0", "evaluator_ref": "amap_agent"},
        {"id": "D1", "evaluator_ref": "amap_agent"},
    ]
    out = await evaluate_criteria_grouped(criteria, evaluate)
    assert calls == ["amap_agent", "amap_agent"]
    assert out["D0"].score == 1.0
    assert out["D1"].score == 2.0
