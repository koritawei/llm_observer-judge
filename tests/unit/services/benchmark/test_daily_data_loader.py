"""DailyDataLoader — unit tests for daily OSS CSV parsing and limit sampling.

Covers:
- ``load_daily`` parses everything (limit omitted = the current-behavior baseline).
- ``limit`` sampling: parse only the first N rows (total_count = N, item indexes 0..N-1);
  when limit is greater than the actual row count, the natural result is the full set.
- The file for the day does not exist → ``DailyNoDataError``.

StorageClient is mocked (download → bytes | None); real OSS is not touched.
"""

from typing import Any

import pytest
DailyDataLoader = None
DailyLoadSpec = None
DailyNoDataError = None

_CSV_HEADER = "id,task_id,query,adiu\n"


def _csv_bytes(rows: int) -> bytes:
    """Build an N-row CSV with a BOM (utf-8-sig, matching the data-pipeline output format)."""
    lines = [_CSV_HEADER.rstrip("\n")]
    for i in range(rows):
        lines.append(f"{100 + i},task-{i},query-{i},adiu-{i}")
    return ("\ufeff" + "\n".join(lines) + "\n").encode("utf-8")


class _FakeStorage:
    """download → preset bytes; None simulates a missing file."""

    def __init__(self, data: bytes | None) -> None:
        self._data = data
        self.downloaded_keys: list[str] = []

    async def download(self, object_key: str) -> bytes | None:
        self.downloaded_keys.append(object_key)
        return self._data


async def _load(
    storage: _FakeStorage,
    *,
    limit: int | None = None,
) -> Any:
    loader = DailyDataLoader(storage)  # type: ignore[arg-type]
    return await loader.load_daily(
        DailyLoadSpec(
            run_id="bm-20260820",
            oss_prefix="ieval-service/benchmark/benchmark_result",
            date="20260820",
            limit=limit,
        ),
    )


@pytest.mark.asyncio
async def test_load_daily_explicit_object_key_used_verbatim() -> None:
    """Frozen path for an already uploaded dataset: an explicit object_key is used verbatim, and the trigger-day partition is not joined."""
    storage = _FakeStorage(_csv_bytes(2))
    loader = DailyDataLoader(storage)  # type: ignore[arg-type]
    result = await loader.load_daily(
        DailyLoadSpec(
            run_id="bm-20261021",
            oss_prefix="search_eval_p",
            date="20261021",
            object_key="search_eval_p/20261018/benchmark_final_results.csv",
        ),
    )
    assert result.total_count == 2
    assert storage.downloaded_keys == [
        "search_eval_p/20261018/benchmark_final_results.csv",
    ]
    assert result.object_key == "search_eval_p/20261018/benchmark_final_results.csv"


@pytest.mark.asyncio
async def test_load_daily_full_without_limit() -> None:
    """limit omitted → parse the full file (the current-behavior baseline)."""
    result = await _load(_FakeStorage(_csv_bytes(3)))
    assert result.total_count == 3
    assert len(result.items) == 3
    assert result.title_columns == ["id", "task_id", "query", "adiu"]
    assert result.object_key.endswith("20260820/benchmark_final_results.csv")


@pytest.mark.asyncio
async def test_load_daily_excel_quoted_bom_header_normalizes_query() -> None:
    """Excel utf-8-sig quotes every field: the first column stays query (not a BOM-prefixed "query")."""
    # Matches the production search_eval wide table: BOM + "query","trace_id",...
    body = (
        '"query","trace_id","result","tool_call_json","project"\n'
        '"济宁市运河文化","acee69b5","回答","[]","proj-1"\n'
    )
    raw = ("\ufeff" + body).encode("utf-8")
    result = await _load(_FakeStorage(raw))
    assert result.title_columns[0] == "query"
    assert result.items[0]["metadata"]["csv_row"]["query"] == "济宁市运河文化"


@pytest.mark.asyncio
async def test_load_daily_items_are_generic_kv_rows() -> None:
    """§7.1 strips business semantics: item = identity normalization + full KV passthrough, and business columns are not assembled."""
    result = await _load(_FakeStorage(_csv_bytes(2)))
    item = result.items[0]
    # Identity normalization (D5): the case_id > id > task_id column wins (this row's id="100" wins);
    # item_id is always {run_id}:{seq}
    assert item["id"] == "100"
    assert item["case_id"] == "100"
    assert item["item_id"] == "bm-20260820:0"
    assert item["source_trace_id"] is None
    # Full KV passthrough is the only source of truth (including the header snapshot)
    assert item["metadata"]["csv_row"] == {
        "id": "100",
        "task_id": "task-0",
        "query": "query-0",
        "adiu": "adiu-0",
    }
    assert item["metadata"]["title_columns"] == ["id", "task_id", "query", "adiu"]
    # Business-column assembly has moved out of the loader (carried by the business interceptor, §7.5)
    assert "input" not in item
    assert "output" not in item
    assert "expected_output" not in item


@pytest.mark.asyncio
async def test_load_daily_limit_truncates_to_first_n() -> None:
    """limit=2 → parse only the first 2 rows; total_count=2, item indexes 0/1."""
    result = await _load(_FakeStorage(_csv_bytes(3)), limit=2)
    assert result.total_count == 2
    assert [item["item_id"] for item in result.items] == [
        "bm-20260820:0",
        "bm-20260820:1",
    ]
    # Row 3 does not build an item (query-2 does not exist)
    assert all("query-2" not in str(item) for item in result.items)


@pytest.mark.asyncio
async def test_load_daily_limit_exceeds_rows_returns_all() -> None:
    """limit greater than the actual row count → break does not fire, and the natural result is the full set."""
    result = await _load(_FakeStorage(_csv_bytes(3)), limit=10)
    assert result.total_count == 3


@pytest.mark.asyncio
async def test_load_daily_no_data_raises() -> None:
    """download returns None (the file does not exist) → DailyNoDataError."""
    with pytest.raises(DailyNoDataError):
        await _load(_FakeStorage(None))


@pytest.mark.asyncio
async def test_load_daily_field_exceeds_csv_default_limit() -> None:
    """A single field over the csv default 128KB limit (the full response_text) → parses normally.

    search-eval 20260915 production incident: the response_text column exceeded 131072 bytes and raised
    _csv.Error, the scheduler precheck treated it as "no data" and retried until cutoff, and that day's scheduled run
    never started; the loader already raises csv.field_size_limit at import, and this case locks
    that behavior so it does not regress.
    """
    big_text = "x" * (131072 + 1024)  # 1KB over the default limit
    header = "id,task_id,response_text\n"
    csv_text = f"\ufeff{header}100,task-0,{big_text}\n1001,task-1,short\n"
    result = await _load(_FakeStorage(csv_text.encode("utf-8")))

    assert result.total_count == 2
    assert result.items[0]["metadata"]["csv_row"]["response_text"] == big_text
    assert result.items[1]["metadata"]["csv_row"]["response_text"] == "short"


# ── ExternalBenchmarkRequest.limit argument check ────────────────────


def test_request_limit_accepts_none_and_positive() -> None:
    """limit omitted as None (full set) and a positive integer are both legal."""
    from benchmark.runs.schema import ExternalBenchmarkRequest

    base = {
        "benchmark_name": "ieval_external",
        "judge_class": "ieval_external_judge",
        "attribution_class": "ieval_external_attribution",
        "rubric_class": "amap_chat_rubric_v2",
    }
    assert ExternalBenchmarkRequest(**base).limit is None
    assert ExternalBenchmarkRequest(**base, limit=5).limit == 5


def test_request_limit_rejects_non_positive() -> None:
    """limit ≤ 0 → ValidationError（ge=1）。"""
    from benchmark.runs.schema import ExternalBenchmarkRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExternalBenchmarkRequest(
            benchmark_name="ieval_external",
            judge_class="ieval_external_judge",
            attribution_class="ieval_external_attribution",
            rubric_class="amap_chat_rubric_v2",
            limit=0,
        )
    with pytest.raises(ValidationError):
        ExternalBenchmarkRequest(
            benchmark_name="ieval_external",
            judge_class="ieval_external_judge",
            attribution_class="ieval_external_attribution",
            rubric_class="amap_chat_rubric_v2",
            limit=-3,
        )
