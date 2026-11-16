"""build_prompt node — load the configured prompt template (via ``PromptManager``)
and render it against the pre-built ``render_context``.

When ``sub_score_axes`` / ``sub_score_scale`` are non-empty / set, also inject
``axes`` (a line-oriented string) / ``score_min`` / ``score_max`` into ``render_context`` so
the prompt template can embed them as guidance for the LLM.

Dynamic Rubric: when ``dimension_weights`` / ``attribution_threshold`` are non-empty, inject
weighted axes lines and ``${attribution_threshold}``.
"""

from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.agenthost.prompts import PromptManager

from business.evaluators.example.schema import render_template
from business.evaluators.example.state import ERROR, ERROR_CODE, PROMPT, PROMPT_OVERRIDE, RENDER_CONTEXT
from benchmark.utils.exceptions import PROMPT_NOT_FOUND_CODE

logger = get_logger("agent_eval.evaluator.example")

#: Fallback for ${axes} — matches the evaluator.prompt convention "show (no per-item scoring configured) and use format A".
_AXES_UNCONFIGURED = "(未配置分项评分)"
#: Fallback for ${expected_output} — matches the evaluator.prompt convention "if unset, score from query+answer".
_EXPECTED_OUTPUT_UNCONFIGURED = "(未配置期望输出)"
#: Display sentinel when ${attribution_threshold} is unset (formats A/B may still use a hardcoded 3 as a narrative fallback).
_ATTR_THRESHOLD_UNCONFIGURED = "(未配置归因门限)"
#: Display fallback for ${score_min} / ${score_max} (when axes are set but scale is not).
#: Guides LLM output only; the Response side does not check the value range.
DEFAULT_SCORE_MIN = 0.0
DEFAULT_SCORE_MAX = 10.0
_TRACE_UNAVAILABLE = "(trace 不可用)"


def format_axes_with_weights(
    axes: list[str],
    weights: dict[str, float] | None,
) -> str:
    """Render per-item dimension lines; append ``(weight=…)`` when a weight is present."""
    if not axes:
        return _AXES_UNCONFIGURED
    weight_map = weights or {}
    lines: list[str] = []
    for axis in axes:
        if axis in weight_map:
            lines.append(f"{axis} (weight={weight_map[axis]})")
        else:
            lines.append(axis)
    return "\n".join(lines)


class BuildPromptNode:
    """Load the evaluator prompt via ``PromptManager`` and render it."""

    #: Graph node key — route returns must equal this (see agent._build_state_graph).
    NODE_NAME = "build_prompt"

    def __init__(
        self,
        prompt_manager: PromptManager,
        prompt_name: str,
        *,
        sub_score_axes: list[str] | None = None,
        sub_score_scale: tuple[float, float] | None = None,
        dimension_weights: dict[str, float] | None = None,
        attribution_threshold: float | None = None,
        tool_prompt: str | None = None,
    ) -> None:
        self._prompt_manager = prompt_manager
        self._prompt_name = prompt_name
        self._sub_score_axes = list(sub_score_axes or [])
        self._sub_score_scale = sub_score_scale
        self._dimension_weights = dict(dimension_weights or {})
        self._attribution_threshold = attribution_threshold
        self._tool_prompt = tool_prompt

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        # RemoteConfig prompt override: when present, skip prompt_manager entirely.
        override = state.get(PROMPT_OVERRIDE)
        if override:
            variables = dict(state.get(RENDER_CONTEXT) or {})
            self._inject_dynamic_vars(variables)
            variables.setdefault("expected_output", _EXPECTED_OUTPUT_UNCONFIGURED)
            variables.setdefault("attribution_threshold", _ATTR_THRESHOLD_UNCONFIGURED)
            variables.setdefault("axes", _AXES_UNCONFIGURED)
            variables.setdefault("score_min", DEFAULT_SCORE_MIN)
            variables.setdefault("score_max", DEFAULT_SCORE_MAX)
            prompt = render_template(override, variables)
            final = _append_tool_prompt(prompt, variables, self._tool_prompt)
            _log_rendered_prompt(final, source="override")
            return {PROMPT: final}

        try:
            template = await self._prompt_manager.load_prompt(self._prompt_name)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "evaluator.example.prompt_not_found name=%r error=%s",
                self._prompt_name,
                type(exc).__name__,
            )
            return {ERROR: str(exc), ERROR_CODE: PROMPT_NOT_FOUND_CODE}

        render_context: dict[str, Any] = dict(state.get(RENDER_CONTEXT) or {})
        self._inject_dynamic_vars(render_context)
        # Fallback: inject a sentinel/default for unset items so a literal ${...} placeholder is not rendered to the LLM.
        render_context.setdefault("axes", _AXES_UNCONFIGURED)
        render_context.setdefault("expected_output", _EXPECTED_OUTPUT_UNCONFIGURED)
        render_context.setdefault("score_min", DEFAULT_SCORE_MIN)
        render_context.setdefault("score_max", DEFAULT_SCORE_MAX)
        render_context.setdefault("attribution_threshold", _ATTR_THRESHOLD_UNCONFIGURED)

        prompt = render_template(template, render_context)
        final = _append_tool_prompt(prompt, render_context, self._tool_prompt)
        _log_rendered_prompt(final, source="file_template")
        return {PROMPT: final}

    def _inject_dynamic_vars(self, render_context: dict[str, Any]) -> None:
        if self._sub_score_axes:
            render_context["axes"] = format_axes_with_weights(
                self._sub_score_axes,
                self._dimension_weights,
            )
        if self._sub_score_scale is not None:
            lo, hi = self._sub_score_scale
            render_context["score_min"] = float(lo)
            render_context["score_max"] = float(hi)
        if self._attribution_threshold is not None:
            render_context["attribution_threshold"] = float(self._attribution_threshold)


def _log_rendered_prompt(prompt: str, *, source: str) -> None:
    """Log metadata of the rendered prompt; do not log the full text (it can be tens of thousands of characters and flood the log)."""
    logger.info(
        "evaluator.example.prompt_rendered source=%s chars=%d",
        source,
        len(prompt),
    )


def _append_tool_prompt(
    body: str,
    variables: dict[str, Any],
    tool_prompt: str | None,
) -> str:
    """After the body is chosen and rendered, append the tool notes passed by the caller. If none were passed, append nothing."""

    if not tool_prompt:
        return body
    context = {key: "" if value is None else str(value) for key, value in variables.items()}
    context.setdefault("used_tool", "")
    context.setdefault("trace_evidence", _TRACE_UNAVAILABLE)
    tool = render_template(tool_prompt, context)
    return f"{body.rstrip()}\n\n{tool}".strip()
