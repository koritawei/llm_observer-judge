"""Safe routing condition evaluator — no eval/exec (ADR-0010)."""

from __future__ import annotations

import re
from typing import Any

_TAGS_CONTAINS = re.compile(
    r"""^\s*tags\s+contains\s+['"]([^'"]+)['"]\s*$""", re.IGNORECASE
)
_STATUS_EQ = re.compile(
    r"""^\s*status\s*==\s*['"]([^'"]+)['"]\s*$""", re.IGNORECASE
)


def validate_condition_syntax(condition: str) -> None:
    """Raise ValueError if condition is not in the supported DSL."""
    expr = (condition or "").strip()
    if not expr:
        raise ValueError("empty routing condition")
    if expr.lower() == "always":
        return
    if _TAGS_CONTAINS.match(expr) or _STATUS_EQ.match(expr):
        return
    raise ValueError(f"unsupported routing condition: {condition!r}")


def evaluate_condition(condition: str, trace_summary: dict[str, Any]) -> bool:
    expr = (condition or "").strip()
    validate_condition_syntax(expr)
    if expr.lower() == "always":
        return True

    m = _TAGS_CONTAINS.match(expr)
    if m:
        tag = m.group(1)
        tags = trace_summary.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        return tag in tags

    m = _STATUS_EQ.match(expr)
    if m:
        return str(trace_summary.get("status", "")) == m.group(1)

    raise ValueError(f"unsupported routing condition: {condition!r}")
