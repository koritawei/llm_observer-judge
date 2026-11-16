"""Merge evaluation of multiple criteria that share one ``evaluator_ref``.

Background: if a rubric writes several dimensions as separate Criterion rows that share
one ``evaluator_ref``, the worker ``evaluate``s each dimension by default and reruns the
same LLM/graph N times. A holistic judge (for example ``judge_example`` /
``judge_search_ai_card``) can emit every dimension score in one ``sub_scores`` call.

This module groups by ``evaluator_ref`` and calls the evaluator once per group. When
``result.sub_scores`` **covers every criterion.id in the group**, scores are fanned out
per dimension; otherwise the group falls back to one call per criterion (compatible with
evaluators that do not yet emit aligned sub_scores, such as the local ``amap_agent``
aggregate score).
"""

from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.evaluators.base import EvaluatorResult

logger = get_logger("agent_eval.benchmark.common.criterion_eval")

#: Criterion shape: a dict (worker job) or an object with ``id`` / ``evaluator_ref``.
CriterionLike = Any
EvaluateFn = Callable[[str], Awaitable[EvaluatorResult]]


def _criterion_id(criterion: CriterionLike) -> str:
    if isinstance(criterion, dict):
        return str(criterion["id"])
    return str(criterion.id)


def _criterion_ref(criterion: CriterionLike) -> str:
    if isinstance(criterion, dict):
        return str(criterion.get("evaluator_ref") or "")
    return str(getattr(criterion, "evaluator_ref", "") or "")


def group_criteria_by_evaluator_ref(
    criteria: Sequence[CriterionLike],
) -> OrderedDict[str, list[CriterionLike]]:
    """Group by ``evaluator_ref``, preserving order (consecutive identical refs merge into one group)."""
    groups: OrderedDict[str, list[CriterionLike]] = OrderedDict()
    for criterion in criteria:
        ref = _criterion_ref(criterion)
        groups.setdefault(ref, []).append(criterion)
    return groups


def can_fanout_sub_scores(
    result: EvaluatorResult,
    criteria: Sequence[CriterionLike],
) -> bool:
    """Whether ``sub_scores`` covers every criterion.id in the group (safe to fan out once)."""
    sub = result.sub_scores
    if not isinstance(sub, dict) or not criteria:
        return False
    return all(_criterion_id(c) in sub for c in criteria)


def fanout_criterion_scores(
    result: EvaluatorResult,
    criteria: Sequence[CriterionLike],
) -> dict[str, EvaluatorResult]:
    """Split one evaluate result into per-dimension results by criterion.id from ``sub_scores``.

    The caller must confirm fan-out with :func:`can_fanout_sub_scores` first; a missing key
    scores 0 for that dimension. The top-level ``reason`` / ``extra`` / full ``sub_scores``
    are passed through to every dimension unchanged.
    """
    sub = dict(result.sub_scores or {})
    out: dict[str, EvaluatorResult] = {}
    for criterion in criteria:
        cid = _criterion_id(criterion)
        raw = sub.get(cid)
        score = float(raw) if raw is not None else 0.0
        out[cid] = EvaluatorResult(
            score=score,
            reason=result.reason,
            sub_scores=dict(sub) if sub else None,
            extra=dict(result.extra or {}),
        )
    return out


async def evaluate_criteria_grouped(
    criteria: Sequence[CriterionLike],
    evaluate: EvaluateFn,
) -> dict[str, EvaluatorResult]:
    """Evaluate by evaluator_ref groups: one call when fan-out is possible, otherwise once per dimension.

    ``evaluate(evaluator_ref) -> EvaluatorResult`` is injected by the caller as a closure
    (it already closes over ``EvaluatorRequest`` / ``EvalRunOptions`` and other context).
    """
    scores: dict[str, EvaluatorResult] = {}
    for ref, group in group_criteria_by_evaluator_ref(criteria).items():
        if len(group) == 1:
            cid = _criterion_id(group[0])
            scores[cid] = await evaluate(ref)
            continue

        first = await evaluate(ref)
        if can_fanout_sub_scores(first, group):
            logger.info(
                "criterion_eval.fanout ref=%s criteria=%s",
                ref,
                [_criterion_id(c) for c in group],
            )
            scores.update(fanout_criterion_scores(first, group))
            continue

        # sub_scores not aligned: keep the first result for the first dimension, and call again for the other dimensions
        # (legacy evaluators).
        logger.info(
            "criterion_eval.fallback_per_criterion ref=%s criteria=%s sub_score_keys=%s",
            ref,
            [_criterion_id(c) for c in group],
            sorted((first.sub_scores or {}).keys()),
        )
        scores[_criterion_id(group[0])] = first
        for criterion in group[1:]:
            scores[_criterion_id(criterion)] = await evaluate(ref)
    return scores


__all__ = [
    "can_fanout_sub_scores",
    "evaluate_criteria_grouped",
    "fanout_criterion_scores",
    "group_criteria_by_evaluator_ref",
]
