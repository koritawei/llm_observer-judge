"""Langfuse trace.name convention for agent-eval.

Format: agent-eval.{agent_name}

Why:
- The `agent-eval.` prefix matches libs `observability.app_name: agent-eval` (conf/app.*.yaml),
  so filtering the Langfuse UI by name prefix returns every agent-eval trace.
- The trailing `{agent_name}` is the business agent under evaluation (for example `vehicle-agent` / `amap-agent`),
  which makes per-business filtering and stats straightforward.
- `trace_id` / `evaluator_name` stay out of the name: the former goes in metadata / span input, and the latter is
  distinguished with tags or metadata; the name stays a fixed two segments so a missing argument does not produce
  a misleading name such as `agent-eval.agent-eval.unknown`.

Whitelist character filter: the Langfuse trace.name field is split on separators in the UI/search,
so non-whitelist characters (`/`, `@`, `#`, `!`, and similar) are turned into `.` or stripped, to avoid breaking search/filter.
"""

from __future__ import annotations

import re

_WHITELIST = re.compile(r"[^a-zA-Z0-9_.-]+")
_SERVICE_PREFIX = "agent-eval"
_UNKNOWN = "unknown"


def _normalize_trace_name_part(part: str) -> str:
    """Strip / replace non-whitelist chars. `/` → `.`; other non-whitelist characters are stripped."""
    return _WHITELIST.sub("", part.replace("/", "."))


def build_eval_trace_name(agent_name: str | None) -> str:
    """Build Langfuse trace.name for an agent-eval evaluation run.

    Args:
        agent_name: name of the business agent under evaluation, such as `vehicle-agent` / `amap-agent`.
            Empty or None is allowed (the trailing segment falls back to `unknown`).

    Returns:
        Langfuse trace.name, in the form `agent-eval.{agent_name}`.
    """
    raw = (agent_name or "").strip()
    normalized = _normalize_trace_name_part(raw) if raw else ""
    business = normalized or _UNKNOWN
    return f"{_SERVICE_PREFIX}.{business}"


__all__ = ["build_eval_trace_name"]
