"""evaluate node — run the evaluation by asking the LLM to judge the prompt."""

from collections.abc import Callable
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from business.evaluators.example.llm import EvaluatorLLM
from business.evaluators.example.state import ERROR, ERROR_CODE, LLM_CONTENT, MODEL_OVERRIDE, PROMPT
from benchmark.utils.exceptions import LLM_FAILED_CODE

logger = get_logger("agent_eval.evaluator.example")


class EvaluateNode:
    """Send the rendered prompt to the LLM, capture the raw reply text.

    When ``remote_config_model_builder`` is set and ``state[MODEL_OVERRIDE]`` is
    present, the RemoteConfig-sourced LLM takes precedence over the default.
    """

    #: Graph node key — route returns must equal this (see agent._build_state_graph).
    NODE_NAME = "evaluate"

    def __init__(
        self,
        llm: EvaluatorLLM,
        *,
        remote_config_model_builder: Callable[[Any], EvaluatorLLM] | None = None,
    ) -> None:
        self._llm = llm
        self._remote_config_model_builder = remote_config_model_builder

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get(ERROR):
            return {}
        prompt = state.get(PROMPT) or ""
        # RemoteConfig model override: build (or reuse cached) LLM from inline config.
        # When all critical fields are None, the override carries no usable
        # config — fall back to the default YAML-assembled LLM.
        override = state.get(MODEL_OVERRIDE)
        use_remote_config = (
            override is not None
            and not override.is_empty
            and self._remote_config_model_builder is not None
        )
        if use_remote_config:
            try:
                llm = self._remote_config_model_builder(override)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "evaluator.example.remote_config_llm_build_failed error_type=%s",
                    type(exc).__name__,
                )
                return {
                    ERROR: f"remote_config llm build failed: {type(exc).__name__}",
                    ERROR_CODE: LLM_FAILED_CODE,
                }
        else:
            llm = self._llm
        try:
            content = await llm.complete(prompt)
        except Exception as exc:  # noqa: BLE001 - surface as a structured graph error
            logger.warning("evaluator.example.llm_failed error_type=%s", type(exc).__name__)
            return {
                ERROR: f"llm call failed: {type(exc).__name__}",
                ERROR_CODE: LLM_FAILED_CODE,
            }
        return {LLM_CONTENT: content}
