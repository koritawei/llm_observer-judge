"""CSV header normalization: BOM / leftover quotes."""

from benchmark.utils.csv_headers import (
    normalize_csv_header,
    normalize_csv_row,
)


def test_normalize_csv_header_strips_bom_and_quotes() -> None:
    assert normalize_csv_header('\ufeff"query"') == "query"
    assert normalize_csv_header('"query"') == "query"
    assert normalize_csv_header("\ufeffquery") == "query"
    assert normalize_csv_header("  query  ") == "query"


def test_normalize_csv_row_rewrites_bom_first_column() -> None:
    row = normalize_csv_row(
        {
            '\ufeff"query"': "附近咖啡",
            "trace_id": "t1",
            "result": "回答",
        },
    )
    assert row["query"] == "附近咖啡"
    assert "query" in row
    assert '\ufeff"query"' not in row
    assert row["result"] == "回答"
