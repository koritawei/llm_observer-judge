"""Cross-business Checker protocol and defensive execution orchestration."""

import json
from typing import Protocol, TypeVar

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import JsonValue

from benchmark.pipeline.signals import (
    CheckerResult,
    DimensionResult,
    EvaluationSignal,
    SignalStatus,
)

logger = get_logger("agent_eval.evaluator.business_components.checker")
ContextT = TypeVar("ContextT")


class BaseChecker(Protocol[ContextT]):
    name: str
    dimensions: tuple[str, ...]

    def check(self, context: ContextT) -> CheckerResult: ...


def make_checker_result(
    *,
    name: str,
    dimensions: tuple[str, ...],
    status: SignalStatus,
    reason: str,
    **details: object,
) -> CheckerResult:
    """Build a Checker result and keep the business domain's optional signal fields."""

    for field in (
        "dimension_results",
        "hard_rule_hits",
        "risk_flags",
        "evidence_refs",
        "failure_modes",
    ):
        if details.get(field) is None:
            details.pop(field, None)

    return CheckerResult(
        source_name=name,
        dimensions=list(dimensions),
        status=status,
        reason=reason,
        **details,
    )


def dimension_result(
    *,
    dimension: str,
    status: SignalStatus,
    reason: str,
    **details: object,
) -> DimensionResult:
    """Build a single-dimension result and keep optional fields such as evidence and risk."""

    return DimensionResult(dimension=dimension, status=status, reason=reason, **details)


def compact_json(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def contains_any(value: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in value for keyword in keywords)


def has_cmd_payload(value: JsonValue) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(compact_json(value) not in ("[]", "{}", ""))
    return True


def run_checkers(
    context: ContextT, checkers: tuple[BaseChecker[ContextT], ...]
) -> list[EvaluationSignal]:
    """When one Checker fails, degrade it to a low-confidence signal and do not abort the whole evaluation."""

    results: list[EvaluationSignal] = []
    for checker in checkers:
        try:
            results.append(checker.check(context))
        except Exception as error:  # pragma: no cover
            logger.exception(
                "business_checker.failed checker=%s", getattr(checker, "name", "unknown")
            )
            results.append(
                make_checker_result(
                    name=getattr(checker, "name", "unknown_checker"),
                    dimensions=tuple(getattr(checker, "dimensions", ())),
                    status="unknown",
                    reason=f"checker 执行失败：{error}",
                    confidence=0.0,
                    decision="run_judge",
                    risk_flags=["schema_anomaly"],
                )
            )
    return results
