"""Model-client call adapter.

The attribution LLM is injected through an explicit ``AttributionLLM`` seam; this module only provides generic
``call_model`` / response parsing and does not discover a runtime client.
"""

import inspect
import json
from collections.abc import Callable
from typing import Literal, TypeVar, cast

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, ValidationError

from benchmark.utils.llm_json import parse_and_validate_llm_json, strip_json_fence

logger = get_logger("agent_eval.badcase_attribution.model_client")

ModelT = TypeVar("ModelT", bound=BaseModel)

#: JSON text preview cap for truncation/empty-content diagnostic logs. Attribution is a low-frequency path,
#: and 200 characters are enough to spot the four symptoms "empty / preamble / markdown / cut off" by eye.
_DIAGNOSTIC_PREVIEW_CHARS = 200


class ModelMessage(BaseModel):
    """Minimal chat message sent to the model client."""

    role: Literal["system", "user"]
    content: str


async def call_model(model_client: object, system_prompt: str, user_prompt: str) -> object:
    """Support several model-client call styles and return the raw response."""

    messages = [
        ModelMessage(role="system", content=system_prompt).model_dump(mode="json"),
        ModelMessage(role="user", content=user_prompt).model_dump(mode="json"),
    ]
    if callable(model_client):
        fn = cast(Callable[..., object], model_client)
        return await maybe_await(fn(messages=messages))

    for method_name in ("ainvoke", "agenerate", "generate"):
        method = getattr(model_client, method_name, None)
        if callable(method):
            fn = cast(Callable[..., object], method)
            return await maybe_await(fn(messages))
    raise TypeError(f"Unsupported model client: {type(model_client).__name__}")


async def maybe_await(value: object) -> object:
    """Support both a sync return and an awaitable return."""

    if inspect.isawaitable(value):
        return await value
    return value


def response_text(response: object) -> str:
    """Extract text content from different response objects."""

    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    return str(response)


def parse_model_response(response: object, model_type: type[ModelT]) -> ModelT:
    """Parse a model response into a Pydantic model.

    Three shapes are supported: a target model instance, a dict, and other (wrapper object / string).
    Wrapper objects (LangChain ``AIMessage`` and similar) take the
    ``content`` field via :func:`response_text` and hand it to the shared LLM JSON parser, which handles common markdown fences and recoverable
    JSON syntax errors.

    On parse failure, log ``finish_reason`` / content length / content preview so the four symptoms
    "hit max_tokens truncation" / "empty content" / "noise around the JSON" / "model refusal"
    can be told apart.
    """

    if isinstance(response, model_type):
        return response
    if isinstance(response, dict):
        return model_type.model_validate(response)
    content = response_text(response)
    parsed = parse_and_validate_llm_json(content, model_type)
    if parsed is not None:
        return parsed

    try:
        # Preserve the existing error type and diagnostic logging when the shared
        # parser rejects a truncated, schema-invalid, or unrecoverable response.
        return model_type.model_validate_json(strip_json_fence(content))
    except ValidationError as exc:
        _log_parse_failure(response, model_type, exc)
        raise


def _log_parse_failure(
    response: object, model_type: type[ModelT], exc: ValidationError
) -> None:
    """Log a diagnostic when ``parse_model_response`` fails, and do not raise.

    No decision logic — only evidence for later root-cause analysis.
    """

    content = getattr(response, "content", None)
    content_repr = (
        content if isinstance(content, str) else str(content) if content is not None else None
    )
    content_len = len(content_repr) if content_repr is not None else 0
    content_preview = (content_repr or "")[:_DIAGNOSTIC_PREVIEW_CHARS]
    finish_reason = getattr(response, "response_metadata", {}).get("finish_reason")
    usage = getattr(response, "usage_metadata", None) or getattr(
        response, "response_metadata", {}
    ).get("usage", {})
    tool_calls = getattr(response, "tool_calls", None) or []
    logger.warning(
        "model_client.parse_failed model=%s finish_reason=%s content_len=%d "
        "tool_calls=%d usage=%s preview=%s err=%s",
        model_type.__name__,
        finish_reason,
        content_len,
        len(tool_calls),
        json.dumps(usage, default=str) if usage else "{}",
        content_preview,
        exc,
    )


__all__ = [
    "ModelMessage",
    "call_model",
    "maybe_await",
    "parse_model_response",
    "response_text",
]
