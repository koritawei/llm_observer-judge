"""response node — form the verdict ``{score, reason, sub_scores?, weights?, ...}``.

When ``sub_score_axes`` is non-empty, validate the parsed ``sub_scores`` /
``weights`` against the axes: missing keys → warning; extra keys → warning.
Never raises — tolerate legacy LLM output; the business relies on prompt design as the backstop.
"""

from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from business.evaluators.example.scoring import parse_evaluation
from business.evaluators.example.state import ERROR, ERROR_CODE, LLM_CONTENT, RESULT
from benchmark.utils.exceptions import LLM_UNPARSEABLE_CODE

logger = get_logger("agent_eval.evaluator.example")


class ResponseNode:
    """Extract the typed :class:`EvaluationResult` from the LLM reply text."""

    #: Graph node key — route returns must equal this (see agent._build_state_graph).
    NODE_NAME = "response"

    def __init__(self, *, sub_score_axes: list[str] | None = None) -> None:
        self._sub_score_axes = set(sub_score_axes or [])

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get(ERROR):
            return {}
        content = state.get(LLM_CONTENT) or ""
        result = parse_evaluation(content)
        if result is None:
            logger.warning(
                "evaluator.example.llm_unparseable llm_content=%r",
                content,
            )
            return {
                ERROR: "LLM reply is not a valid {score, reason} verdict",
                ERROR_CODE: LLM_UNPARSEABLE_CODE,
            }
        self._validate_axes("sub_scores", result.sub_scores)
        self._validate_axes("weights", result.weights)
        return {RESULT: result}

    def _validate_axes(self, field: str, values: dict[str, float] | None) -> None:
        """Warning-level axes check. Applies only when axes are declared and the LLM returned the corresponding field."""
        if not self._sub_score_axes:
            return
        if values is None:
            logger.warning(
                "evaluator.example.%s_missing axes=%s",
                field,
                sorted(self._sub_score_axes),
            )
            return
        got = set(values.keys())
        missing = self._sub_score_axes - got
        extra = got - self._sub_score_axes
        if missing:
            logger.warning(
                "evaluator.example.%s_axes_missing axes=%s missing=%s",
                field,
                sorted(self._sub_score_axes),
                sorted(missing),
            )
        if extra:
            logger.warning(
                "evaluator.example.%s_axes_extra axes=%s extra=%s",
                field,
                sorted(self._sub_score_axes),
                sorted(extra),
            )
