"""Unit tests for JSON dataset normalization."""

import csv
import io
import json

import pytest
from benchmark.runs.common.dataset_normalizer import (
    DatasetNormalizationError,
    normalize_dataset_json,
)


def _rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def test_normalizes_aliases_nested_values_and_identity() -> None:
    payload = [
        {
            "id": "case-1",
            "query": "杭州天气",
            "response_text": "晴天",
            "reference": "杭州晴天",
            "metadata": {"trace_id": "trace-1"},
        },
        {
            "task_id": "case-2",
            "question": "南京天气",
            "answer": "阴天",
            "expected": "南京阴天",
        },
    ]
    rows = _rows(normalize_dataset_json(json.dumps(payload).encode()))
    assert len(rows) == 2
    assert rows[0]["case_id"] == "case-1"
    assert json.loads(rows[0]["input"]) == {"query": "杭州天气"}
    assert json.loads(rows[0]["output"]) == {"response_text": "晴天"}
    assert rows[0]["expected_output"] == "杭州晴天"
    assert json.loads(rows[0]["metadata"]) == {"trace_id": "trace-1"}
    assert rows[1]["case_id"] == "case-2"


def test_supports_rows_and_data_wrappers_and_scalar_rows() -> None:
    wrapped_rows = _rows(normalize_dataset_json(b'{"rows":[{"input":"hello"}]}'))
    assert wrapped_rows[0]["case_id"] == "row-1"
    assert json.loads(wrapped_rows[0]["input"]) == {"query": "hello"}
    scalar_rows = _rows(normalize_dataset_json(b'{"data":["hello"]}'))
    assert json.loads(scalar_rows[0]["input"]) == {"query": "hello"}


def test_preserves_structured_input_output() -> None:
    payload = [{"input": {"query": "hello"}, "output": {"response_text": "world"}}]
    row = _rows(normalize_dataset_json(json.dumps(payload).encode()))[0]
    assert json.loads(row["input"]) == {"query": "hello"}
    assert json.loads(row["output"]) == {"response_text": "world"}


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b"{}",
        b"[]",
        b'{"rows":[]}',
        b'[{"output":"missing input"}]',
    ],
)
def test_rejects_invalid_dataset(payload: bytes) -> None:
    with pytest.raises(DatasetNormalizationError):
        normalize_dataset_json(payload)


def test_rejects_non_utf8() -> None:
    with pytest.raises(DatasetNormalizationError, match="UTF-8"):
        normalize_dataset_json(b"\xff")
