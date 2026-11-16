"""Parse the example evaluator's LLM reply into a typed :class:`EvaluationResult`.

The LLM returns JSON ``{"score": <0-10>, "reasoning": ...}`` (the Langfuse Numeric
evaluator contract — field names fixed); the ``reasoning`` key maps to the ``reason``
field here. Validated straight into :class:`EvaluationResult` (no hand-rolled
json / dict.get). Pure function.

Optional extensions (format B / dynamic Rubric):
- ``sub_scores``: per-item scores;
- ``weights``: per-item weights (written back from the prompt declaration, for the framework to weight);
- ``attribution_threshold``: attribution threshold (written back from the prompt declaration).

``grade`` is **not** parsed from the LLM JSON: plan A is mapped by the framework from the fused score and written to
``EvaluatorResult.extra`` / ``judge_json`` (see ``grade_fused_score``).
If the LLM mistakenly writes extra fields such as ``grade``, ``extra='ignore'`` drops them and does not cause
``llm_unparseable``.
"""

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from benchmark.utils.llm_json import parse_and_validate_llm_json

logger = get_logger("agent_eval.evaluator.example.scoring")


def _coerce_float(value: object) -> float | None:
    """Lenient numeric parse: int/float/numeric string → float; anything else → None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _optional_float_dict(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("must be a dict, not a non-dict type")
    out: dict[str, float] = {}
    for k, v in value.items():
        coerced = _coerce_float(v)
        if coerced is not None:
            out[str(k)] = coerced
    return out


class EvaluationResult(BaseModel, frozen=True, populate_by_name=True):
    """Parsed verdict from the LLM JSON reply.

    ``score`` rejects a JSON ``bool`` (so ``true`` isn't read as ``1.0``);
    ``reason`` reads ``reason``|``explanation``|``reasoning``, non-string → ``None``.
    Missing or non-dict ``sub_scores`` / ``weights`` → ``None`` (backward compatible with older LLM output).
    Missing ``attribution_threshold`` → ``None``; a numeric string (such as ``"6.0"``) can be parsed.
    Extra fields (including ``grade``) are ignored and do not fail parsing.
    """

    model_config = ConfigDict(extra="ignore")

    score: float
    reason: str | None = Field(
        default=None, validation_alias=AliasChoices("reason", "reasoning", "explanation")
    )
    sub_scores: dict[str, float] | None = None
    weights: dict[str, float] | None = None
    attribution_threshold: float | None = None
    #: Raw judge JSON text: still usable for audit and replay when a dynamic field is malformed.
    raw_json: str | None = None

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, value: object) -> object:
        # bool subclasses int — reject so a JSON `true` is not coerced to 1.0.
        if isinstance(value, bool):
            raise ValueError("score must be a number, not a bool")
        coerced = _coerce_float(value)
        if coerced is None:
            return value
        return coerced

    @field_validator("reason", mode="before")
    @classmethod
    def _reason_str_or_none(cls, value: object) -> str | None:
        return value if isinstance(value, str) else None

    @field_validator("sub_scores", mode="before")
    @classmethod
    def _sub_scores_dict_or_none(cls, value: object) -> dict[str, float] | None:
        try:
            return _optional_float_dict(value)
        except ValueError:
            return None

    @field_validator("weights", mode="before")
    @classmethod
    def _weights_dict_or_none(cls, value: object) -> dict[str, float] | None:
        try:
            return _optional_float_dict(value)
        except ValueError:
            return None

    @field_validator("attribution_threshold", mode="before")
    @classmethod
    def _threshold_float_or_none(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        coerced = _coerce_float(value)
        return coerced


def parse_evaluation(content: str) -> EvaluationResult | None:
    """Parse verdict from an LLM reply (strips a ``` fence), or None if malformed."""

    result = parse_and_validate_llm_json(content, EvaluationResult)
    if result is None:
        preview = (content or "")[:500]
        logger.warning(
            "evaluator.example.parse_evaluation_failed content_len=%d preview=%r",
            len(content or ""),
            preview,
        )
        return None
    return result.model_copy(update={"raw_json": content})
