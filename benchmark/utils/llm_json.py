"""Helpers for parsing JSON from LLM output."""

from typing import TypeVar

from benchmark.libs.agenthost.observability.logger import get_logger
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)
logger = get_logger("agent_eval.llm_json")


def strip_json_fence(value: str) -> str:
    """Strip the markdown ```json fence wrapper common in LLM output and keep the inner JSON text."""

    text = value.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(value: str) -> str:
    """Cut out the first complete ``{...}`` object from surrounding noise (tolerates prefixes such as a lead-in colon).

    When there are no object braces, return the text unchanged; do not try to parse nesting inside strings (later
    ``json_repair`` / pydantic re-validate).
    """
    text = value.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return text
    return text[start : end + 1]


def has_unclosed_json_structure(value: str) -> bool:
    """Return whether JSON text appears to end before a string or container closes."""

    opening_by_closing = {"}": "{", "]": "["}
    containers: list[str] = []
    in_string = False
    escaped = False

    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character in "{[":
            containers.append(character)
        elif character in opening_by_closing:
            if containers and containers[-1] == opening_by_closing[character]:
                containers.pop()

    return in_string or bool(containers)


def parse_and_validate_llm_json(
    content: str,
    model_type: type[ModelT],
) -> ModelT | None:
    """Parse LLM JSON, repairing syntax errors before validating the same model again.

    Valid JSON that fails the target model's schema is deliberately not repaired, so
    this helper only tolerates serialization mistakes rather than changing a verdict.
    """

    normalized = extract_json_object(strip_json_fence(content))
    try:
        return model_type.model_validate_json(normalized)
    except ValidationError as error:
        if not any(item["type"] == "json_invalid" for item in error.errors()):
            logger.info(
                "llm_json.schema_reject model=%s errors=%s",
                model_type.__name__,
                error.errors(),
            )
            return None

    # json-repair can synthetically close a truncated string or container. Those
    # outputs are incomplete model responses, not recoverable serialization errors.
    if has_unclosed_json_structure(normalized):
        return None

    try:
        repaired = repair_json(
            normalized,
            return_objects=False,
            skip_json_loads=True,
        )
        if not isinstance(repaired, str):
            return None
        parsed = model_type.model_validate_json(repaired)
        logger.info(
            "llm_json.repaired model=%s content_len=%d",
            model_type.__name__,
            len(normalized),
        )
        return parsed
    except (TypeError, ValidationError, ValueError) as exc:
        logger.info(
            "llm_json.repair_failed model=%s err=%s",
            model_type.__name__,
            type(exc).__name__,
        )
        return None


__all__ = [
    "extract_json_object",
    "has_unclosed_json_structure",
    "parse_and_validate_llm_json",
    "strip_json_fence",
]
