"""search_eval wide-table column aliases: a single source of truth."""

from benchmark.utils.search_eval_csv_columns import (
    RESULT_COLUMNS,
    TOOL_TRACE_COLUMNS,
    ctx_from_csv_row,
    first_csv_text,
)


def test_first_csv_text_prefers_alias_order() -> None:
    assert first_csv_text({"response_text": "a"}, RESULT_COLUMNS) == "a"
    assert first_csv_text({"result": "b", "response_text": "a"}, RESULT_COLUMNS) == "b"
    assert first_csv_text({"tool_calls_json": "[]"}, TOOL_TRACE_COLUMNS) == "[]"


def test_ctx_from_csv_row_maps_aliases() -> None:
    ctx = ctx_from_csv_row(
        {
            "traceid": "t1",
            "lng": "1.0",
            "lat": "2.0",
            "env_time": "now",
            "app_use_context": "recent",
        },
    )
    assert ctx == {
        "trace_id": "t1",
        "longitude": "1.0",
        "latitude": "2.0",
        "current_time": "now",
        "app_use_context": "recent",
    }
