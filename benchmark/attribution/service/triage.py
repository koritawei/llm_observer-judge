"""TriageAttributionService submodule — upfront triage decision (classifier-based).

Business extension point: implement :class:`TriageClassifier` (pure CPU, synchronous) and register it on
:func:`benchmark.attribution.classifiers_registry.triage_classifiers`;
``triage_decide`` takes a classifier instance, calls ``classify``, and returns :class:`TriageDecision`.
"""

from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, ConfigDict, Field

from benchmark.records.enums import TriageLevel
from benchmark.pipeline.normalize import normalize_eval_extra
from benchmark.attribution.service.classifier import TriageClassifier
from benchmark.attribution.service.schema import TriageInput

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Triage decision result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriageDecision:
    """Public result of one ``triage_decide`` call."""

    should_attribute: bool
    reason: str
    base_level: str | None = None
    final_level: str | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TriageConfig(BaseModel):
    """Runtime config for the triage decision.

    Legacy fields kept for backward-compatible yaml; the classifier carries its own thresholds, and these two fields are currently only for logs/observation.
    """

    model_config = ConfigDict(extra="forbid")

    badcase_score_threshold: float = Field(
        default=0.5,
        description="遗留字段(classifier 自带阈值),当前仅用于日志/观测",
    )
    policy_match_strict: bool = Field(
        default=True,
        description="遗留字段,保留供未来策略引擎扩展",
    )
    label_studio_dispatcher_enabled: bool = Field(
        default=False,
        description="LS dispatcher 运行时开关(Settings.label_studio_enabled 为进程级总开关)",
    )


# ---------------------------------------------------------------------------
# Business methods
# ---------------------------------------------------------------------------


def _build_triage_input(
    *,
    evaluator_name: str,
    evaluator_request: Any,  # noqa: ANN401 - evaluator extension boundary
    evaluator_result: Any,  # noqa: ANN401 - evaluator extension boundary
) -> TriageInput:
    """Build a TriageInput from the evaluator request/result.

    The caller guarantees trace_id / project_id are non-empty and not None; this function does not fill them in.
    A missing field is an upstream contract violation and raises RuntimeError.
    """

    trace_id = getattr(evaluator_request, "trace_id", None) or getattr(
        evaluator_result, "trace_id", None
    )
    project_id = getattr(evaluator_request, "project_id", None)
    if not trace_id or not project_id:
        raise RuntimeError(
            f"triage_decide called with missing identity fields "
            f"(trace_id={trace_id!r}, project_id={project_id!r}); "
            f"caller must validate metadata before invocation."
        )
    score = getattr(evaluator_result, "score", None)
    reason = getattr(evaluator_result, "reason", None)

    # Extract extended fields from evaluator extra (v3 adds no_decision, needs_human_review).
# Migration compatibility: the unified envelope is restored to the legacy read view by the normalized adapter (fields are not lost after they move into
# business_extension.payload); a historical envelope passes through unchanged.
    extra = normalize_eval_extra(getattr(evaluator_result, "extra", None) or {})
    risk_flags = extra.get("risk_flags", []) or []
    no_decision = bool(extra.get("no_decision", False))
    needs_human_review = bool(extra.get("needs_human_review", False))

    return TriageInput(
        trace_id=trace_id,
        evaluator_name=evaluator_name,
        project_id=project_id,
        score=score,
        reason=reason,
        risk_flags=list(risk_flags),
        no_decision=no_decision,
        needs_human_review=needs_human_review,
    )


def _level_value(level: TriageLevel | None) -> str | None:
    """TriageLevel(StrEnum) → str value; None → None。"""
    if level is None:
        return None
    return level.value


async def triage_decide(
    *,
    evaluator_name: str,
    evaluator_request: Any,  # noqa: ANN401 - evaluator extension boundary
    evaluator_result: Any,  # noqa: ANN401 - evaluator extension boundary
    classifier: TriageClassifier,
    config: TriageConfig | None = None,
) -> TriageDecision:
    """Call classifier.classify() to decide whether to attribute.

    Args:
        classifier: the :class:`TriageClassifier` injected by the business (looked up by evaluator_name).
        config: legacy config (does not affect the decision now; the classifier carries its own thresholds).

    Returns:
        :class:`TriageDecision`. If the classifier raises, return ``should_attribute=False``
        so the main path is not blocked.
    """

    triage_input = _build_triage_input(
        evaluator_name=evaluator_name,
        evaluator_request=evaluator_request,
        evaluator_result=evaluator_result,
    )
    try:
        verdict = classifier.classify(triage_input)
    except Exception as exc:
        logger.exception(
            "triage.classifier_failed evaluator=%s err=%s",
            evaluator_name,
            exc,
        )
        return TriageDecision(
            should_attribute=False,
            reason=f"classifier_failed: {exc}",
        )

    level_str = _level_value(verdict.level)
    return TriageDecision(
        should_attribute=verdict.should_attribute,
        reason=level_str or "no_decision",
        final_level=level_str,
        base_level=level_str,
    )


__all__ = [
    "TriageConfig",
    "TriageDecision",
    "triage_decide",
]
