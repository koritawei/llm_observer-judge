"""build_record — keep falsy values in the CSV column mapping, and keep the two paths consistent.

In an Aone record the numeric columns (return_code / tool_count / event_count) treat
0 as a valid value (return_code=0 means success): both the csv_row path and the fallback path must
keep ``"0"``, and must not use ``or ""`` to wipe a falsy value.
"""

from typing import Any

from benchmark.runs.common.parse_interceptor import (
    CaseIdentityScope,
    DataParseContext,
)
AmapChatParseInterceptor = None
build_record = None
DailyDataLoader = None


def _fallback_item(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """A structured item with no csv_row (input to the fallback path)."""
    return {
        "id": "case-1",
        "item_id": "case-1",
        "input": {"query": "用户问题"},
        "output": {"response_text": "回复", "tool_calls": []},
        "expected_output": {"reference": "ref"},
        "metadata": metadata if metadata is not None else {"adiu": "d1"},
    }


def test_csv_row_path_keeps_falsy_values() -> None:
    """csv_row path: None → "", and a falsy string such as "0" is kept."""
    item = _fallback_item({"csv_row": {"task_id": "t1", "return_code": "0", "tool_count": "0"}})
    record = build_record(item)
    assert record["return_code"] == "0"
    assert record["tool_count"] == "0"
    assert record["task_id"] == "t1"


def test_fallback_path_keeps_numeric_zero() -> None:
    """Fallback path: return_code/tool_count/event_count of 0 (the success code) is kept as "0"."""
    item = _fallback_item(
        {
            "adiu": "d1",
            "return_code": 0,
            "tool_count": 0,
            "event_count": 0,
            "city": "北京",
        }
    )
    record = build_record(item)
    assert record["return_code"] == "0"
    assert record["tool_count"] == "0"
    assert record["event_count"] == "0"
    assert record["city"] == "北京"


def test_fallback_path_none_to_empty() -> None:
    """Fallback path: a missing field / None → empty string (reconciliation columns are always str)."""
    record = build_record(_fallback_item())
    assert record["return_code"] == ""
    assert record["tool_count"] == ""
    assert record["adiu"] == "d1"


def test_two_paths_consistent_for_zero() -> None:
    """The same 0 matches on both paths: csv_row "0" == fallback 0 → "0"."""
    via_csv = build_record(_fallback_item({"csv_row": {"task_id": "t", "tool_count": "0"}}))
    via_fallback = build_record(_fallback_item({"adiu": "d", "tool_count": 0}))
    assert via_csv["tool_count"] == via_fallback["tool_count"] == "0"


def test_fallback_query_none_to_empty() -> None:
    """The input.query key exists but the value is None → "" (do not leak None into the record)."""
    item = _fallback_item()
    item["input"] = {"query": None}
    assert build_record(item)["query"] == ""


def test_fallback_location_serialized() -> None:
    """user_location / history serialize normally into string columns."""
    item = _fallback_item(
        {
            "adiu": "d1",
            "user_location": {"longitude": 116.4, "latitude": 39.9},
            "history": [{"role": "user", "content": "q"}],
        }
    )
    record = build_record(item)
    assert record["lng"] == "116.4"
    assert record["lat"] == "39.9"
    assert '"q"' in record["user_history"]


def test_parse_row_smoke_structure() -> None:
    """A generic loader row + the interceptor → the item's four-part structure (the §7.1/§7.7 refactor safety net).

    The loader ``_parse_row`` only produces a full-KV generic row (no business semantics); the 21-column business assembly
    is carried by ``AmapChatParseInterceptor``. This test locks the structure after the two are chained and checks that it
    matches the old ``_parse_row`` output position by position.
    """
    row = {
        "id": "csv-1",
        "adiu": "d1",
        "query": "q",
        "lng": "116.4",
        "lat": "39.9",
        "city": "北京",
        "city_code": "010",
        "user_history": "[]",
        "level": "L1",
        "type": "navi",
        "scene": "drive",
        "task_id": "t1",
        "session_id": "s1",
        "tool_count": "0",
        "reference": "ref",
        "tool_calls_json": "[]",
        "response_text": "resp",
        "result": "ok",
        "return_code": "0",
        "event_count": "1",
        "error": "",
    }
    # (1) loader: a full-KV generic row (it does not interpret business columns)
    # D5 identity normalization: case_id > id > task_id column wins (this row's id="csv-1" wins);
    # item_id is always {run_id}:{seq}.
    raw = DailyDataLoader._parse_row(
        row, list(row), CaseIdentityScope(run_id="bench-20260812", seq=0)
    )
    assert raw["id"] == "csv-1"
    assert raw["case_id"] == "csv-1"
    assert raw["item_id"] == "bench-20260812:0"
    assert raw["metadata"]["csv_row"] == row
    assert raw["metadata"]["title_columns"] == list(row)
    assert "input" not in raw and "output" not in raw
    assert raw["source_trace_id"] is None

    # (2) interceptor: raw KV → business item (the old _parse_row contract; id is taken from csv_row.task_id)
    item = AmapChatParseInterceptor().parse(
        DataParseContext(benchmark_name="ieval_external", raw_items=[raw]),
    )[0]
    assert item["id"] == "t1"
    assert item["item_id"] == "bench-20260812:0"
    assert item["input"]["query"] == "q"
    assert item["input"]["task_id"] == "t1"
    assert item["output"]["return_code"] == "0"
    assert item["expected_output"]["reference"] == "ref"
    assert item["metadata"]["csv_row"] == row
    assert item["metadata"]["user_location"] == {
        "longitude": 116.4,
        "latitude": 39.9,
    }
    assert item["source_trace_id"] is None


def test_judge_record_priority_over_csv_row() -> None:
    """When metadata.judge_record already holds the final record, that value wins (§7.3 optional capability)."""
    item = _fallback_item(
        {
            "csv_row": {"task_id": "t1", "query": "from-csv"},
            "judge_record": {"query": "custom", "k": "v"},
        }
    )
    record = build_record(item)
    # judge_record fully replaces the csv_row passthrough; a missing task_id is filled with the item id
    assert record["query"] == "custom"
    assert record["k"] == "v"
    assert record["task_id"] == "case-1"
