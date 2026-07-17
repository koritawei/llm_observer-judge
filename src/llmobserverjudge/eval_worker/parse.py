"""Parse LLM judge responses into scored / model_error results."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_judge_response(
    content: str,
    *,
    agent_id: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Extract score JSON from model text. Unparseable → model_error."""
    scale_max = float(cfg.get("scale_max", 10) or 10)
    threshold = float(cfg.get("threshold", 0.5))
    weight = float(cfg.get("weight", 1.0))

    data = _extract_json(content)
    if data is None:
        return {
            "agent_id": agent_id,
            "score_status": "model_error",
            "error": "unparseable_judge_response",
            "error_type": "ParseError",
            "label": "PARSE_ERROR",
            "raw_content": content[:2000],
        }

    try:
        raw = float(data.get("score", data.get("raw_score")))
    except (TypeError, ValueError):
        return {
            "agent_id": agent_id,
            "score_status": "model_error",
            "error": "missing_or_invalid_score",
            "error_type": "ParseError",
            "label": "PARSE_ERROR",
            "raw_content": content[:2000],
        }

    # Allow 0-1 scores if scale_max is 10 and value <= 1
    if raw <= 1.0 and scale_max > 1.0 and data.get("normalized") is True:
        raw = raw * scale_max

    normalized = raw / scale_max if scale_max > 0 else 0.0
    return {
        "agent_id": agent_id,
        "score_status": "scored",
        "score": raw,
        "scale_max": scale_max,
        "threshold": threshold,
        "weight": weight,
        "normalized_score": round(normalized, 4),
        "label": str(data.get("label") or ""),
        "reasoning": str(data.get("reasoning") or data.get("reason") or ""),
    }


def extract_json_object(content: str) -> dict[str, Any] | None:
    """Public alias for JSON blob extraction from model text."""
    return _extract_json(content)


def _extract_json(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    if not text:
        return None
    # fenced
    m = _FENCE_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # raw object
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # first {...} blob
    m2 = _JSON_OBJ_RE.search(text)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None
