"""Protocol schemas for the eval API (OpenAI-compatible) — request + response.

Inbound: FastAPI validates the body against :class:`ChatCompletionRequest` instead
of the route hand-checking a raw ``dict``; validation failures are translated to the
stable OpenAI error envelope by the ``RequestValidationError`` handler in
:mod:`benchmark.utils.error_handler`.

Outbound: the success verdict is a typed pydantic model built by
:func:`build_chat_completion` (``chat.completion``, OpenAI-compatible since clients
hard-parse it). The **error** envelope (``{"error": {...}}``) is an error concern and
lives with the exceptions — :meth:`benchmark.utils.exceptions.APIError.to_envelope`.

Conversions: :func:`to_evaluator_request` / :func:`to_chat_completion` translate the
OpenAI protocol ↔ evaluator domain types (``EvaluatorRequest`` / ``EvaluatorResult``)
so the route stays a thin shell and the service speaks only domain types.
"""

import json
import time
import uuid
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from benchmark.evaluators.base import EvaluatorRequest, EvaluatorResult

logger = get_logger("agent_eval.api.schema")

# ══════════════════════════════════════════════════════════════════════════════
# Inbound request
# ══════════════════════════════════════════════════════════════════════════════


class TextContentPart(BaseModel):
    """An OpenAI ``text`` content part — only ``text`` is consumed.

    ``type`` and other sibling keys are ignored (pydantic default ``extra="ignore"``);
    a non-text part (e.g. ``image_url``) has no ``text`` and fails validation, so the
    extractor skips it instead of reaching for a magic ``"text"`` key.
    """

    text: str


class ChatMessage(BaseModel):
    """One OpenAI chat message.

    ``content`` stays loosely typed (a string or a list of content parts) — it
    carries the JSON :class:`EvalEnvelope` the converter parses, so the API does not
    constrain its inner shape. Unknown sibling fields are ignored (pydantic default)
    to stay OpenAI-client compatible.
    """

    role: str = ""
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    """Inbound ``POST /v1/chat/completions`` body.

    ``model`` selects the evaluator (model = evaluator name); ``messages`` carries
    the tagged eval payload. ``stream`` is accepted but must be falsy — evaluation
    is non-streaming, so ``stream: true`` is rejected up front with a stable 400
    rather than silently returning a single JSON body an SSE client would
    mis-parse. ``tools`` / ``response_format`` are accepted for the proxy fallback
    (Langfuse connectivity probe, function-calling clients) and forwarded to the
    LLM verbatim. Extra OpenAI fields (``temperature`` …) are ignored, not rejected.
    """

    model: str = Field(min_length=1)
    messages: list[ChatMessage]
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    # Native Langfuse id: skip the business-id lookup; still read the trace if input/output is missing.
    skip_trace_resolve: bool = False
    # Optional request-level prompt version alias: each evaluator's allowlist maps it to a template
    # (for example vui's ``vui_agent_evidence_boundary``). Missing, blank, or unknown names fall
    # back to the default template, so existing callers are unaffected.
    prompt_version: str | int | None = None

    @field_validator("stream")
    @classmethod
    def _reject_stream(cls, value: bool) -> bool:
        if value:
            raise ValueError("Streaming responses are not supported by the eval endpoint.")
        return value


# ══════════════════════════════════════════════════════════════════════════════
# Inbound eval envelope — the JSON object carried inside the message content
# ══════════════════════════════════════════════════════════════════════════════


class EnvelopeMetadata(BaseModel, extra="allow"):
    """``metadata`` block — ``trace_id`` is typed; other debug keys kept (extra allow)."""

    trace_id: str | None = None


class EvalEnvelope(BaseModel):
    """The JSON payload carried in the OpenAI message content.

    ``input`` / ``output`` are :class:`pydantic.JsonValue` — the agent
    request/response snapshots, whose format is not fixed (a structured object, or
    plain markdown/text). ``agent_name`` / ``project_id`` are top-level fields
    matching the real Langfuse payload; ``trace_id`` lives under ``metadata``.
    The whole envelope is one ``model_validate_json`` away; no tag scraping /
    escaping.
    """

    input: JsonValue = None
    output: JsonValue = None
    metadata: EnvelopeMetadata = Field(default_factory=EnvelopeMetadata)
    agent_name: str | None = None
    project_id: str | None = None
    judge_prompt_text: str | None = None
    attribution_prompt_text: str | None = None
    # Consumed only by judge_example. Unset means no check; other evaluators ignore it.
    tool_check: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_stringified_json(cls, data: Any) -> Any:  # noqa: ANN401 — before-validator takes any raw input
        """Restore object fields that Langfuse injected as JSON strings.

        Langfuse LLM-as-a-Judge substitutes prompt variables as text, so the trace's
        ``input`` / ``output`` / ``metadata`` (structured objects) arrive as (escaped)
        JSON strings. That makes ``metadata`` (typed :class:`EnvelopeMetadata`) fail
        validation and leaves ``input`` / ``output`` as opaque strings. Parse the
        object/array-looking strings back into objects; leave genuine plain text
        (e.g. an ``output`` that really is a string) untouched.
        """
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        for key in ("input", "output", "metadata", "tool_check"):
            value = coerced.get(key)
            if isinstance(value, str) and value.strip()[:1] in ("{", "["):
                try:
                    coerced[key] = json.loads(value)
                except ValueError:
                    # Keep the original string and continue validation; the log locates a broken envelope.
                    logger.warning(
                        "envelope.coerce_json_failed key=%s value_len=%d head=%r",
                        key,
                        len(value),
                        value[:80],
                    )
        return coerced


# ══════════════════════════════════════════════════════════════════════════════
# Outbound success — OpenAI chat.completion carrying the evaluator verdict
# ══════════════════════════════════════════════════════════════════════════════


class Verdict(BaseModel):
    """The evaluator verdict carried (JSON-encoded) in the assistant message ``content``.

    A typed projection of :class:`EvaluatorResult` — only ``score`` + ``reasoning``
    reach the client. Field names are fixed by the Langfuse Numeric evaluator
    contract (``{"score": number, "reasoning": string}``); ``extra`` stays internal.
    """

    score: float
    reasoning: str | None = None


class ToolCallFunction(BaseModel):
    """OpenAI ``tool_calls[i].function`` — ``arguments`` is a JSON-encoded string."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """One OpenAI ``tool_calls`` entry on an assistant message."""

    id: str
    type: str = "function"
    function: ToolCallFunction


class ResponseMessage(BaseModel):
    """The assistant message.

    ``content`` is the JSON-encoded :class:`Verdict` for evaluator verdicts; for
    the proxy fallback it is the raw LLM text (or ``null`` when the LLM replied
    via ``tool_calls`` only).
    """

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    trace_id: str | None = None


def build_chat_completion(
    *,
    model: str,
    score: float,
    reasoning: str | None,
    request_id: str | None = None,
    created: int | None = None,
    trace_id: str | None = None,
) -> ChatCompletion:
    """Build an OpenAI ``chat.completion`` carrying the evaluator verdict."""

    verdict = Verdict(score=score, reasoning=reasoning).model_dump_json()
    return ChatCompletion(
        id=request_id or f"chatcmpl-{uuid.uuid4().hex}",
        created=created if created is not None else int(time.time()),
        model=model,
        choices=[Choice(message=ResponseMessage(content=verdict))],
        trace_id=trace_id,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Conversions — OpenAI protocol ↔ evaluator domain types
# ══════════════════════════════════════════════════════════════════════════════


def to_evaluator_request(request: ChatCompletionRequest) -> EvaluatorRequest:
    """Parse the message content's JSON :class:`EvalEnvelope` into an EvaluatorRequest.

    ``trace_id`` comes from ``metadata``; ``agent_name`` / ``project_id`` from
    ``extra``. ``prompt_version`` comes from the request top level (optional
    opt-in alias; evaluator whitelist maps it to a template, unknown → default).
    The evaluator name is not parsed here — routing already used
    ``request.model``. Stamps ``pipeline_source=http`` for trace_meta filtering.
    """
    from benchmark.domain.enums import (  # noqa: PLC0415
        PIPELINE_SOURCE_METADATA_KEY,
        PipelineSource,
    )

    envelope = _parse_envelope(_message_content_text(request.messages))
    metadata = envelope.metadata.model_dump(exclude_none=True)
    metadata[PIPELINE_SOURCE_METADATA_KEY] = PipelineSource.HTTP.value
    return EvaluatorRequest(
        input=envelope.input,
        output=envelope.output,
        metadata=metadata,
        trace_id=envelope.metadata.trace_id,
        agent_name=envelope.agent_name,
        project_id=envelope.project_id,
        judge_prompt_text=envelope.judge_prompt_text,
        attribution_prompt_text=envelope.attribution_prompt_text,
        tool_check=envelope.tool_check,
        skip_trace_resolve=request.skip_trace_resolve,
        prompt_version=request.prompt_version,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Required-metadata guard — fail fast before langfuse judge burns a request
# ══════════════════════════════════════════════════════════════════════════════


_REQUIRED_METADATA_FIELDS: tuple[str, ...] = ("trace_id", "project_id", "agent_name")


def validate_evaluator_request(request: EvaluatorRequest) -> list[str]:
    """Return the list of required metadata fields missing on ``request``.

    The check is intentionally lenient on the request body (``input`` / ``output``
    are :class:`JsonValue` — format is not fixed), but strict on the identity
    fields the rest of the pipeline (triage, attribution, DB writes) keys off.
    Returns ``[]`` when all required fields are present and non-empty.
    """

    missing: list[str] = []
    for field in _REQUIRED_METADATA_FIELDS:
        value = getattr(request, field, None)
        if not value or not isinstance(value, str) or not value.strip():
            missing.append(field)
    return missing


def to_chat_completion(
    model: str, result: EvaluatorResult, *, trace_id: str | None = None
) -> ChatCompletion:
    """Project an :class:`EvaluatorResult` onto an OpenAI ``chat.completion``."""

    return build_chat_completion(
        model=model,
        score=result.score,
        reasoning=result.reason,
        trace_id=trace_id,
    )


def has_eval_envelope(request: ChatCompletionRequest) -> bool:
    """Recognize evaluation IO or an explicit trace identity.

    The proxy fallback triggers when this is False — i.e. the inbound content is
    not the evaluator's JSON envelope (Langfuse connectivity probe, ad-hoc chat,
    function-calling test, …). A trace ID routes to evaluation even without IO;
    the service validates identity fields and fetches missing IO from Langfuse.
    Empty JSON without IO or a trace identity still falls through to the proxy.
    """

    envelope = _try_parse_envelope(_message_content_text(request.messages))
    if envelope is None:
        return False
    return (
        envelope.input is not None
        or envelope.output is not None
        or envelope.metadata.trace_id is not None
    )


def build_proxy_chat_completion(
    *,
    model: str,
    content: str | None,
    tool_calls: list[ToolCall] | None = None,
    request_id: str | None = None,
    created: int | None = None,
) -> ChatCompletion:
    """Build an OpenAI ``chat.completion`` for the proxy fallback path.

    Mirrors :func:`build_chat_completion` but carries a raw LLM reply (text or
    ``tool_calls``) instead of a typed verdict, so function-calling / structured
    -output clients (Langfuse save probe via LangChain.js) get the shape they
    negotiated with the LLM.
    """

    if tool_calls:
        message = ResponseMessage(content=None, tool_calls=tool_calls)
        finish_reason = "tool_calls"
    else:
        message = ResponseMessage(content=content or "")
        finish_reason = "stop"
    return ChatCompletion(
        id=request_id or f"chatcmpl-{uuid.uuid4().hex}",
        created=created if created is not None else int(time.time()),
        model=model,
        choices=[Choice(message=message, finish_reason=finish_reason)],
    )


def _parse_envelope(content: str) -> EvalEnvelope:
    """Parse the JSON envelope, tolerating malformed / non-JSON content."""

    parsed = _try_parse_envelope(content)
    return parsed if parsed is not None else EvalEnvelope()


def _try_parse_envelope(content: str) -> EvalEnvelope | None:
    """Parse the JSON envelope; return None on malformed / non-JSON content."""

    try:
        return EvalEnvelope.model_validate_json(content)
    except ValidationError:
        return None


def _message_content_text(messages: list[ChatMessage]) -> str:
    """Concatenate the text content across OpenAI-style messages."""

    parts: list[str] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                try:
                    parts.append(TextContentPart.model_validate(item).text)
                except ValidationError:
                    continue  # non-text part (image_url, …) or non-object — skip
    return "\n".join(parts)
