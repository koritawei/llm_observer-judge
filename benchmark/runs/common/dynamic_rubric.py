"""Dynamic Rubric: read weights / the attribution gate from the judge result and override static rubric defaults.

Design rules (do not break a custom Rubric / threshold):

- **Only when** the judge result carries both a valid ``extra.weights`` and ``sub_scores``
  that cover those keys is ``fused_score`` replaced by the weighted score; otherwise keep
  :class:`~benchmark.reporting.framework.fusion.ScoreFuser` / the static Criterion.weight.
- **Only when** the result carries a valid ``extra.attribution_threshold`` (or the projected
  stored field) does a case trigger attribution when ``fused < threshold``; otherwise fall
  back to :func:`~benchmark.attribution.badcase.attribution_trigger_policy.should_trigger_attribution`
  (the business-registered policy / default rubric_v2).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.evaluators.base import EvaluatorResult
from business.evaluators.example.grading import grade_fused_score
from benchmark.reporting.framework.fusion import EvalFailure, is_failure

logger = get_logger("agent_eval.benchmark.common.dynamic_rubric")


def _as_float_map(raw: object) -> dict[str, float] | None:
    """Strictly parse a numeric dict; if any value is illegal the whole dict is invalid and nothing is silently dropped."""
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            return None
        out[str(key)] = number
    return out or None


@dataclass(frozen=True)
class DynamicRubricResolution:
    """Parsed dynamic score of one judge call (shared by execution and the snapshot)."""

    fused_score: float | None = None
    attribution_threshold: float | None = None
    fallback_reason: str | None = None
    model_score: float | None = None
    score_mismatch: bool = False


def _result_parts(
    result: EvaluatorResult | Mapping[str, Any],
) -> tuple[Mapping[str, float] | None, Mapping[str, Any]]:
    if isinstance(result, EvaluatorResult):
        return result.sub_scores, result.extra or {}
    if not isinstance(result, Mapping):
        return None, {}
    sub_scores = result.get("sub_scores")
    extra = result.get("extra")
    return (
        sub_scores if isinstance(sub_scores, Mapping) else None,
        extra if isinstance(extra, Mapping) else result,
    )


def dynamic_rubric_opt_in(result: EvaluatorResult | Mapping[str, Any]) -> bool:
    """Explicit switch for the dynamic protocol; a judge that did not opt in does not interpret same-named fields."""
    _sub_scores, extra = _result_parts(result)
    return extra.get("dynamic_rubric_opt_in") is True


_SCORE_SCALE_LENGTH = 2


def _score_scale(extra: Mapping[str, Any]) -> tuple[float, float] | None:
    raw = extra.get("sub_score_scale")
    if not isinstance(raw, (list, tuple)) or len(raw) != _SCORE_SCALE_LENGTH:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
        return None
    low, high = float(raw[0]), float(raw[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        return None
    return low, high


def _threshold_in_scale(raw: object, scale: tuple[float, float] | None) -> float | None:
    threshold = _coerce_threshold(raw)
    if threshold is None or not math.isfinite(threshold):
        return None
    if scale is not None and not (scale[0] <= threshold <= scale[1]):
        return None
    return threshold


def extract_dynamic_weights(result: EvaluatorResult | Mapping[str, Any]) -> dict[str, float] | None:
    """Read returned weights only from an opt-in judge result."""
    if not dynamic_rubric_opt_in(result):
        return None
    _sub_scores, extra = _result_parts(result)
    return _as_float_map(extra.get("weights"))


def _coerce_threshold(raw: object) -> float | None:
    """Loosely parse the attribution threshold (int/float/numeric string); illegal → None."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def extract_attribution_threshold(
    result: EvaluatorResult | Mapping[str, Any],
) -> float | None:
    """Parse an effective threshold from an opt-in judge; on an illegal value fall back to the example declared value."""
    return resolve_dynamic_rubric(result).attribution_threshold


def fuse_from_weights(  # noqa: PLR0911 — every illegal branch of the dynamic parameters must fall back as a whole
    sub_scores: Mapping[str, float] | None,
    weights: Mapping[str, float] | None,
    *,
    declared_weights: Mapping[str, float] | None = None,
    score_scale: tuple[float, float] | None = None,
) -> float | None:
    """Weight strictly as the prompt declares; any bad dimension or number rejects this calculation."""
    if not sub_scores or not weights or set(sub_scores) != set(weights):
        return None
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights.values()):
        return None
    total_w = sum(float(value) for value in weights.values())
    if not math.isclose(total_w, 1.0, rel_tol=0.0, abs_tol=1e-6):
        return None
    if declared_weights is not None and dict(weights) != dict(declared_weights):
        return None
    for value in sub_scores.values():
        score = float(value)
        if not math.isfinite(score):
            return None
        if score_scale is not None and not (score_scale[0] <= score <= score_scale[1]):
            return None
    return float(sum(float(sub_scores[key]) * float(weight) for key, weight in weights.items()))


def _model_score(result: EvaluatorResult | Mapping[str, Any]) -> float | None:
    """Prefer ``extra.model_score`` (result.score may already be the server-side fused score)."""
    extra = result.extra if isinstance(result, EvaluatorResult) else result.get("extra")
    if isinstance(extra, Mapping):
        audited = extra.get("model_score")
        if (
            not isinstance(audited, bool)
            and isinstance(audited, (int, float))
            and math.isfinite(float(audited))
        ):
            return float(audited)
    raw = result.score if isinstance(result, EvaluatorResult) else result.get("score")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        return None
    return float(raw)


def _score_mismatch(model_score: float | None, computed_score: float | None) -> bool:
    return (
        model_score is not None
        and computed_score is not None
        and not math.isclose(model_score, computed_score, rel_tol=0.0, abs_tol=1e-6)
    )


def resolve_dynamic_rubric(
    result: EvaluatorResult | Mapping[str, Any],
) -> DynamicRubricResolution:
    """Parse the dynamic score and the threshold; the two are validated independently and each illegal branch falls back on its own.

    The threshold uses only the ``attribution_threshold`` returned by the judge (it applies as
    soon as it is legal on the score scale) and is not dragged down by a bad weight. An illegal
    weight only falls back to the declared weights and recomputes the fused score. The default
    threshold registered on the rubric is used internally by the caller and is not a separate
    field in parsing or persistence.
    """
    if not dynamic_rubric_opt_in(result):
        return DynamicRubricResolution()
    sub_scores, extra = _result_parts(result)
    declared = _as_float_map(extra.get("declared_weights"))
    returned = _as_float_map(extra.get("weights"))
    scale = _score_scale(extra)
    threshold = _threshold_in_scale(extra.get("attribution_threshold"), scale)
    model_score = _model_score(result)
    computed = fuse_from_weights(
        sub_scores,
        returned,
        declared_weights=declared,
        score_scale=scale,
    )
    if computed is not None:
        return DynamicRubricResolution(
            fused_score=computed,
            attribution_threshold=threshold,
            model_score=model_score,
            score_mismatch=_score_mismatch(model_score, computed),
        )
    reason = (
        "weights_mismatch"
        if returned is not None and declared is not None and returned != declared
        else "invalid_dynamic_parameters"
    )
    fallback_score = fuse_from_weights(sub_scores, declared, score_scale=scale)
    return DynamicRubricResolution(
        fused_score=fallback_score,
        attribution_threshold=threshold,
        fallback_reason=reason,
        model_score=model_score,
        score_mismatch=_score_mismatch(model_score, fallback_score),
    )


def maybe_override_fused_score(
    scores: Mapping[str, EvaluatorResult | EvalFailure | Any],
) -> float | None:
    """Scan each criterion result; the first result that can be weighted dynamically overrides fused, otherwise None."""
    for result in scores.values():
        if result is None or is_failure(result):
            continue
        if not isinstance(result, EvaluatorResult):
            continue
        weights = extract_dynamic_weights(result)
        resolved = resolve_dynamic_rubric(result)
        fused = resolved.fused_score
        if fused is not None:
            logger.info(
                "dynamic_rubric.fused_override fused=%s weights=%s fallback=%s",
                fused,
                weights,
                resolved.fallback_reason,
            )
            return fused
    return None


def extract_attribution_threshold_from_criterion_scores(
    criterion_scores: Mapping[str, Any],
) -> float | None:
    """Take the first valid attribution_threshold from the fused row's ``criterion_scores`` projection."""
    for payload in criterion_scores.values():
        if not isinstance(payload, Mapping):
            continue
        threshold = extract_attribution_threshold(payload)
        if threshold is not None:
            return threshold
    return None


def should_trigger_with_optional_threshold(
    *,
    fused_score: float | None,
    attribution_threshold: float | None,
) -> bool | None:
    """When a dynamic threshold is present, return whether to trigger; with no threshold return None (the caller uses the static policy).

    ``fused_score is None`` → trigger (same missing-score rule as RubricV2TriggerPolicy).
    """
    if attribution_threshold is None:
        return None
    if fused_score is None:
        return True
    return float(fused_score) < float(attribution_threshold)


#: Aligned with ``rubric_v2_grading.RUBRIC_V2_SCORE_SCALE_MAX``: a pass_threshold
#: above this value is a large scale such as 0-10 and may be filled back as the attribution gate (avoid the default <1.4
#: fallback).
_LARGE_SCALE_PASS_THRESHOLD_MIN = 2.0


def resolve_gate_attribution_threshold(
    criterion_scores: Mapping[str, Any] | None,
    *,
    pass_threshold: float | None = None,
) -> float | None:
    """Attribution-gate threshold: prefer the value written back by the judge; on a large score scale also fall back to ``pass_threshold``.

    Fixes the case where criterion_scores did not project ``attribution_threshold`` and the code
    fell back to default RubricV2 (<1.4), so an example composite score of 2.9 was wrongly treated
    as not triggering attribution.
    """
    if criterion_scores:
        found = extract_attribution_threshold_from_criterion_scores(criterion_scores)
        if found is not None:
            return found
    if pass_threshold is None:
        return None
    try:
        threshold = float(pass_threshold)
    except (TypeError, ValueError):
        return None
    if threshold > _LARGE_SCALE_PASS_THRESHOLD_MIN:
        return threshold
    return None


def apply_dynamic_fused_override(
    scores: Mapping[str, EvaluatorResult | EvalFailure | Any],
    fused_score: float | None,
) -> float | None:
    """Override fused when weights+sub_scores are valid; otherwise return the ScoreFuser result unchanged."""
    dynamic = maybe_override_fused_score(scores)
    return dynamic if dynamic is not None else fused_score


def first_attribution_threshold_from_scores(
    scores: Mapping[str, EvaluatorResult | EvalFailure | Any],
) -> float | None:
    """Scan EvaluatorResult.extra on each criterion and take the first valid attribution_threshold."""
    for result in scores.values():
        if result is None or is_failure(result) or not isinstance(result, EvaluatorResult):
            continue
        threshold = extract_attribution_threshold(result)
        if threshold is not None:
            return threshold
    return None


def project_criterion_score_row(result: EvaluatorResult) -> dict[str, Any]:
    """Project one EvaluatorResult into a fused-row criterion_scores entry.

    Write ``weights`` / ``attribution_threshold`` / ``grade`` only when they have a value.
    A custom static rubric that lacks these fields projects the same way it did before the change.
    """
    row: dict[str, Any] = {
        "score": result.score,
        "reason": result.reason,
    }
    if result.sub_scores is not None:
        row["sub_scores"] = result.sub_scores
    extra = result.extra or {}
    if extra.get("dynamic_rubric_opt_in") is True:
        row["dynamic_rubric_opt_in"] = True
    declared_weights = extra.get("declared_weights")
    if isinstance(declared_weights, dict) and declared_weights:
        row["declared_weights"] = declared_weights
    resolved = resolve_dynamic_rubric(result)
    # result.score may already be fused; the audited model score prefers resolved/extra.
    row["model_score"] = resolved.model_score if resolved.model_score is not None else result.score
    if resolved.fused_score is not None:
        row["computed_fused_score"] = resolved.fused_score
    if resolved.score_mismatch:
        row["score_mismatch"] = True
    if resolved.fallback_reason is not None:
        row["dynamic_rubric_fallback_reason"] = resolved.fallback_reason
    for key in ("dynamic_rubric_opt_in_source", "raw_judge_json"):
        value = extra.get(key)
        if isinstance(value, str) and value:
            row[key] = value
    weights = extra.get("weights")
    if isinstance(weights, dict) and weights:
        row["weights"] = weights
    # Persist only a validated threshold (same as resolve_dynamic_rubric), so a later read does not accept an
    # out-of-range gate.
    if resolved.attribution_threshold is not None:
        row["attribution_threshold"] = resolved.attribution_threshold
    validated_scale = _score_scale(extra)
    if validated_scale is not None:
        row["sub_score_scale"] = [validated_scale[0], validated_scale[1]]
    grade = extra.get("grade")
    if isinstance(grade, str) and grade.strip():
        row["grade"] = grade.strip()
    return row


def stamp_fused_grade(
    judge_json: dict[str, Any],
    *,
    fused_score: float | None,
    pass_threshold: float | None,
) -> None:
    """Write the framework grade ``grade`` at the top of fused ``judge_json`` (plan A, update in place).

    Also fill ``grade`` on any ``criterion_scores[*]`` that is missing it, from that dimension's score.
    """
    if fused_score is not None:
        judge_json["grade"] = grade_fused_score(
            float(fused_score),
            pass_threshold=pass_threshold,
        )
    cs = judge_json.get("criterion_scores")
    if not isinstance(cs, dict):
        return
    for payload in cs.values():
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("grade"), str) and payload["grade"].strip():
            continue
        raw = payload.get("score")
        if raw is None:
            continue
        try:
            payload["grade"] = grade_fused_score(
                float(raw),
                pass_threshold=pass_threshold,
            )
        except (TypeError, ValueError):
            continue


def project_criterion_scores(
    scores: Mapping[str, EvaluatorResult | EvalFailure | Any],
) -> dict[str, dict[str, Any]]:
    """Project criterion results that did not fail, in batch."""
    out: dict[str, dict[str, Any]] = {}
    for cid, result in scores.items():
        if result is None or is_failure(result) or not isinstance(result, EvaluatorResult):
            continue
        out[str(cid)] = project_criterion_score_row(result)
    return out


__all__ = [
    "apply_dynamic_fused_override",
    "extract_attribution_threshold",
    "extract_attribution_threshold_from_criterion_scores",
    "extract_dynamic_weights",
    "first_attribution_threshold_from_scores",
    "fuse_from_weights",
    "maybe_override_fused_score",
    "project_criterion_score_row",
    "project_criterion_scores",
    "resolve_dynamic_rubric",
    "resolve_gate_attribution_threshold",
    "should_trigger_with_optional_threshold",
    "stamp_fused_grade",
]
