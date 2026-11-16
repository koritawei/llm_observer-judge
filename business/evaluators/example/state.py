"""Graph state for the example evaluator's StateGraph.

Nodes are stateless: each reads ``state`` and returns a dict patch. ``error`` /
``error_code`` short-circuit the remaining nodes (they route straight to END),
and ``error_code`` is what the agent maps onto an OpenAI error code.
"""

from typing import Any, TypedDict

from business.evaluators.example.scoring import EvaluationResult


class ExampleState(TypedDict, total=False):
    # inputs: named template variables the agent extracted (query / answer / …)
    render_context: dict[str, str]
    # RemoteConfig prompt override (when present, bypasses prompt_manager)
    prompt_override: str | None
    # build_prompt output
    prompt: str
    # RemoteConfig model override (when present, bypasses default LLM)
    model_override: Any | None
    # evaluate output
    llm_content: str
    # response output
    result: EvaluationResult | None
    # error channel (any node may set; routes to END)
    error: str | None
    error_code: str | None


# Channel-key constants — use for every state read/write (`state.get(ERROR)` /
# `{PROMPT: …}`) instead of string literals, so the names live in one place.
RENDER_CONTEXT = "render_context"
PROMPT_OVERRIDE = "prompt_override"
PROMPT = "prompt"
LLM_CONTENT = "llm_content"
MODEL_OVERRIDE = "model_override"
RESULT = "result"
ERROR = "error"
ERROR_CODE = "error_code"


__all__ = [
    "ERROR",
    "ERROR_CODE",
    "LLM_CONTENT",
    "MODEL_OVERRIDE",
    "PROMPT",
    "PROMPT_OVERRIDE",
    "RENDER_CONTEXT",
    "RESULT",
    "EvaluationResult",
    "ExampleState",
]
