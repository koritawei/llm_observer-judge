"""Normalize an entire JSON dataset into the batch-pipeline canonical CSV.

The upload API only chooses a parser. This module does not handle OSS, HTTP,
or benchmark business config. Output column names and values can be handed
straight to ``csv.DictReader`` and DailyDataLoader.
"""

import csv
import io
import json

from benchmark.evaluators.base import JsonValue

_INPUT_COLUMNS = ("input", "query", "question")
_OUTPUT_COLUMNS = ("output", "response_text", "answer")
_EXPECTED_COLUMNS = ("expected_output", "reference", "expected")
_CASE_ID_COLUMNS = ("case_id", "id", "task_id")
_CANONICAL_COLUMNS = ("case_id", "input", "output", "expected_output")


class DatasetNormalizationError(ValueError):
    """The JSON cannot be turned into an executable dataset CSV."""


def _text(value: JsonValue) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _first(row: dict[str, JsonValue], columns: tuple[str, ...]) -> JsonValue:
    for column in columns:
        if column in row and row[column] is not None:
            return row[column]
    return None


def _wrapped(value: JsonValue, key: str) -> str:
    if isinstance(value, dict | list):
        return _text(value)
    return _text({key: _text(value)})


def _row_values(row: JsonValue, index: int) -> dict[str, JsonValue]:
    if not isinstance(row, dict):
        # Compatible with page preview: a scalar row only fills input; other fields stay empty.
        row = {"input": row}
    input_value = _first(row, _INPUT_COLUMNS)
    if input_value is None or not _text(input_value).strip():
        raise DatasetNormalizationError(f"dataset JSON row {index + 1} missing input/query")
    case_id = _first(row, _CASE_ID_COLUMNS)
    output_value = _first(row, _OUTPUT_COLUMNS)
    expected_value = _first(row, _EXPECTED_COLUMNS)
    values = {str(key): _text(value) for key, value in row.items()}
    values.update(
        {
            "case_id": _text(case_id) or f"row-{index + 1}",
            "input": _wrapped(input_value, "query"),
            "output": (_wrapped(output_value, "response_text") if output_value is not None else ""),
            "expected_output": _text(expected_value),
        }
    )
    return values


def normalize_dataset_json(data: bytes) -> bytes:
    """Convert a JSON array of objects, or a rows/data wrapper, into UTF-8 CSV bytes."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetNormalizationError("dataset JSON must be UTF-8") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetNormalizationError(f"invalid dataset JSON: {exc.msg}") from exc
    if isinstance(payload, dict):
        rows = payload.get("rows", payload.get("data"))
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise DatasetNormalizationError(
            "dataset JSON must be a non-empty array or rows/data object"
        )

    normalized = [_row_values(row, index) for index, row in enumerate(rows)]
    columns: list[str] = []
    for name in (*_CANONICAL_COLUMNS, *(key for row in normalized for key in row)):
        if name not in columns:
            columns.append(name)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    return output.getvalue().encode("utf-8")


__all__ = ["DatasetNormalizationError", "normalize_dataset_json"]
