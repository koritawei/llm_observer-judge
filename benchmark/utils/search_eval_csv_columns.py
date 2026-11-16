"""search_eval OSS wide-table column aliases — the single source of truth shared by the interceptor and v2 ``parse_case``.

Column-name drift (``lng``/``lon``, ``response_text``, ``env_time``, ``traceid``, and similar)
is maintained only in this module; the parse interceptor and the judge-side csv_row fallback must share this table, so adding an alias
in only one place does not make the other side silently read empty.
"""

from typing import Any

#: ctx seven-field target key → CSV column aliases (first non-empty value in order).
CTX_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "trace_id": ("trace_id", "traceid"),
    "adiu": ("adiu",),
    "longitude": ("longitude", "lng", "lon"),
    "latitude": ("latitude", "lat"),
    "address": ("address",),
    "current_time": ("current_time", "env_time", "time"),
    "user_profile": ("user_profile",),
}

QUERY_COLUMNS: tuple[str, ...] = ("query",)
RESULT_COLUMNS: tuple[str, ...] = ("result", "response_text")

#: Tool-call chain columns (passed through into output as-is so judge/report can trace tool evidence).
TOOL_CALL_COLUMNS: tuple[str, ...] = ("tool_call_json", "tool_calls_json")

#: v2 tool evidence: prefer explicit tool_trace, then fall back to the legacy tool-call chain columns.
TOOL_TRACE_COLUMNS: tuple[str, ...] = ("tool_trace", *TOOL_CALL_COLUMNS)

#: Recent user App-behavior column (optional: mounted into ctx only when non-empty).
APP_USE_CONTEXT_COLUMNS: tuple[str, ...] = ("app_use_context",)

#: Langfuse project identity columns (the project_id source for the trace_url factory).
PROJECT_COLUMNS: tuple[str, ...] = ("project", "project_id")


def first_csv_text(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    """Take the first non-empty column value in alias order (coerce to str; dirty values become an empty string)."""
    for col in columns:
        value = row.get(col)
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        if text.strip():
            return text.strip()
    return ""


def ctx_from_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build ctx from a wide-table row (seven fields + optional app_use_context); empty values are omitted."""
    ctx: dict[str, Any] = {}
    for key, aliases in CTX_COLUMN_ALIASES.items():
        text = first_csv_text(row, aliases)
        if text:
            ctx[key] = text
    app_use_context = first_csv_text(row, APP_USE_CONTEXT_COLUMNS)
    if app_use_context:
        ctx["app_use_context"] = app_use_context
    return ctx


__all__ = [
    "APP_USE_CONTEXT_COLUMNS",
    "CTX_COLUMN_ALIASES",
    "PROJECT_COLUMNS",
    "QUERY_COLUMNS",
    "RESULT_COLUMNS",
    "TOOL_CALL_COLUMNS",
    "TOOL_TRACE_COLUMNS",
    "ctx_from_csv_row",
    "first_csv_text",
]
