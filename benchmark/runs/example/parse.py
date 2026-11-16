"""ExampleParseInterceptor — example dataset CSV row → business item.

The loader only puts the full KV map in ``metadata.csv_row``. Before submit this interceptor:

1. Promotes identity columns ``trace_id`` / ``project`` (and aliases);
2. Promotes and **normalizes** payload columns ``input`` / ``output`` / ``expected_output``
   to the example evaluator contract (the same fields the WebUI requires):

   - ``input`` (required) → ``{"request_body": {"query": "..."}}``
   - ``output`` (required) →
     ``{"component": {"type": "markdown", "text": "..."}, "response_text": "..."}``
   - ``expected_output`` (optional) is lifted **as-is**: JSON text decodes to a dict,
     plain text stays a str; the structure is not rewritten. The example judge substitutes
     it only when the prompt contains ``${expected_output}``.

CSV cells are always ``str``: JSON text is equivalent to a dict; plain-text input/output
is wrapped into the shapes above.
Column aliases: ``query``/``question``→input, ``answer``/``response_text``→output.
"""

import json
from typing import Any

from benchmark.runs.common.parse_interceptor import DataParseContext

_INPUT_COLUMNS: tuple[str, ...] = ("input", "query", "question")
_OUTPUT_COLUMNS: tuple[str, ...] = ("output", "answer", "response_text")
_EXPECTED_COLUMNS: tuple[str, ...] = ("expected_output", "expected", "reference")
_TRACE_COLUMNS: tuple[str, ...] = ("trace_id", "traceid", "source_trace_id")
_PROJECT_COLUMNS: tuple[str, ...] = ("project", "project_id")


def _first_text(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    """Take the first non-empty text in column order."""
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        if text.strip():
            return text.strip()
    return ""


def _first_value(row: dict[str, Any], columns: tuple[str, ...]) -> object | None:
    """Take the first non-empty cell in column order (keep the original type)."""
    for column in columns:
        if column not in row:
            continue
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _coerce_payload(value: object) -> object:
    """CSV payload: a JSON string → dict/list; a structure or plain text is returned as-is."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return value


def _text_or_empty(value: object) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else str(value).strip()


def normalize_example_input(value: object) -> object:
    """Normalize to ``{"request_body": {"query": "..."}}``; an empty value is returned as-is."""
    value = _coerce_payload(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        return {"request_body": {"query": text}}
    if not isinstance(value, dict):
        return value
    request_body = value.get("request_body")
    body = dict(request_body) if isinstance(request_body, dict) else {}
    query = _text_or_empty(body.get("query")) or _text_or_empty(value.get("query"))
    if not query:
        return value
    body["query"] = query
    return {**value, "request_body": body}


def normalize_example_output(value: object) -> object:
    """Normalize to the component.text + response_text double-written shape; an empty value is returned as-is."""
    value = _coerce_payload(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        return {
            "component": {"type": "markdown", "text": text},
            "response_text": text,
        }
    if not isinstance(value, dict):
        return value
    out = dict(value)
    component_raw = out.get("component")
    component = dict(component_raw) if isinstance(component_raw, dict) else {}
    text = _text_or_empty(component.get("text")) or _text_or_empty(out.get("response_text"))
    if not text:
        return out
    component["text"] = _text_or_empty(component.get("text")) or text
    if not _text_or_empty(component.get("type")):
        component["type"] = "markdown"
    out["component"] = component
    if not _text_or_empty(out.get("response_text")):
        out["response_text"] = text
    return out


def normalize_example_expected(value: object) -> object:
    """Optional reference answer: only a JSON string becomes a dict; plain text / dict is kept **as-is**."""
    return _coerce_payload(value)


class ExampleParseInterceptor:
    """example business parse interceptor (``ExampleRubric.parse_interceptor_ref``)."""

    def parse(self, ctx: DataParseContext) -> list[dict[str, Any]]:
        """Full raw KV → business item. Top-level payloads are still normalized when ``metadata.csv_row`` is absent."""
        return [self._map_raw(raw) for raw in ctx.raw_items]

    @staticmethod
    def _map_raw(raw: dict[str, Any]) -> dict[str, Any]:
        """One raw item → business item."""
        raw_metadata = raw.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        raw_row = metadata.get("csv_row")
        row = raw_row if isinstance(raw_row, dict) else {}

        item = dict(raw)

        if row:
            if "input" not in item:
                lifted_input = _first_value(row, _INPUT_COLUMNS)
                if lifted_input is not None:
                    item["input"] = lifted_input
            if "output" not in item:
                lifted_output = _first_value(row, _OUTPUT_COLUMNS)
                if lifted_output is not None:
                    item["output"] = lifted_output
            if "expected_output" not in item:
                lifted_expected = _first_value(row, _EXPECTED_COLUMNS)
                if lifted_expected is not None:
                    item["expected_output"] = lifted_expected

            trace_id = _first_text(row, _TRACE_COLUMNS)
            if trace_id and not item.get("source_trace_id"):
                item["source_trace_id"] = trace_id

            project_id = _first_text(row, _PROJECT_COLUMNS)
            if project_id:
                metadata["project_id"] = project_id
            metadata["csv_row"] = dict(row)
            item["metadata"] = metadata

        if "input" in item:
            item["input"] = normalize_example_input(item["input"])
        if "output" in item:
            item["output"] = normalize_example_output(item["output"])
        if "expected_output" in item:
            item["expected_output"] = normalize_example_expected(item["expected_output"])
        return item


__all__ = [
    "ExampleParseInterceptor",
    "normalize_example_expected",
    "normalize_example_input",
    "normalize_example_output",
]
