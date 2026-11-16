"""Unit tests for ``sample_trace_pipeline_records`` (mock ORM, in the style of test_benchmark_service).

Locks: configurable source matching (exact fixed channel / dataset composite prefix),
the judge-score guard, sample_size truncation and early stop of paging, paging offset iteration,
fail-fast on an empty source, and pushing agent_name / project_id scalar filters down.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs import run_helpers as rh
from benchmark.runs.run_helpers import TraceRecordSampleQuery
from benchmark.runs.run_service import BenchmarkService

_START = datetime(2026, 9, 22, tzinfo=UTC)
_END = datetime(2026, 9, 23, tzinfo=UTC)


def _await(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def _rec(
    trace_id: str,
    source: str,
    *,
    with_score: bool = True,
    content_version: int | str | None = None,
) -> SimpleNamespace:
    judge: dict[str, Any] = {"score": 1.5} if with_score else {"reason": "pending"}
    if content_version is not None:
        judge["extra"] = {"prompt_customization": {"content_version": content_version}}
    return SimpleNamespace(
        trace_id=trace_id,
        trace_meta_json={"source": source},
        judge_json=judge,
    )


def _run_sample(
    pages: list[list[Any]],
    query: TraceRecordSampleQuery,
) -> tuple[list[str], MagicMock]:
    """Drive paged sampling and return (trace_ids, qs_mock) for assertions.

    ``pages`` is the list of records returned page by page, and an empty page is appended at the end to stop iteration;
    paging-chain assertions are taken from ``qs_mock.order_by.return_value`` (offset / limit).
    """
    qs_mock = MagicMock()
    qs_mock.filter = MagicMock(return_value=qs_mock)
    ob_mock = MagicMock()
    off_mock = MagicMock()
    qs_mock.order_by = MagicMock(return_value=ob_mock)
    ob_mock.offset = MagicMock(return_value=off_mock)
    off_mock.limit = AsyncMock(side_effect=[*pages, []])
    with patch("benchmark.runs.run_helpers.TraceRecord") as mock_trace:
        mock_trace.filter = MagicMock(return_value=qs_mock)
        service = BenchmarkService()
        result = _await(service.sample_trace_pipeline_records(query))
    return result, qs_mock


def _query(**overrides: Any) -> TraceRecordSampleQuery:
    base: dict[str, Any] = {
        "date_start": _START,
        "date_end": _END,
        "evaluator_names": ["e1"],
    }
    base.update(overrides)
    return TraceRecordSampleQuery(**base)


class TestSampleTracePipelineRecords:
    def test_exact_source_match(self) -> None:
        pages = [
            [
                _rec("t-kafka", "kafka"),
                _rec("t-metaq", "metaq"),
                _rec("t-Kafka", "Kafka"),  # A case-normalized hit
            ],
        ]
        result, _ = _run_sample(pages, _query(source="kafka"))
        assert result == ["t-kafka", "t-Kafka"]

    def test_dataset_prefix_match(self) -> None:
        pages = [
            [
                _rec("t-ds", "dataset-amap_chat_eval-run1"),
                _rec("t-bare", "dataset"),
                _rec("t-kafka", "kafka"),
            ],
        ]
        result, _ = _run_sample(pages, _query(source="dataset"))
        assert result == ["t-ds"]

    def test_judge_without_score_skipped(self) -> None:
        pages = [
            [
                _rec("t1", "kafka"),
                _rec("t2", "kafka", with_score=False),
            ],
        ]
        result, _ = _run_sample(pages, _query(source="kafka"))
        assert result == ["t1"]

    def test_sample_size_truncates(self) -> None:
        pages = [[_rec(f"t{i}", "kafka") for i in range(5)]]
        result, _ = _run_sample(pages, _query(source="kafka", sample_size=2))
        assert result == ["t0", "t1"]

    def test_sample_size_stops_pagination(self) -> None:
        # The first page already hits sample_size → do not request a second page (limit is awaited only once)
        pages = [
            [_rec("t0", "kafka"), _rec("t1", "kafka")],
            [_rec("t2", "kafka")],
        ]
        result, qs_mock = _run_sample(pages, _query(source="kafka", sample_size=2))
        assert result == ["t0", "t1"]
        limit_mock = qs_mock.order_by.return_value.offset.return_value.limit
        assert limit_mock.await_count == 1

    def test_pagination_iterates_pages(self) -> None:
        pages = [
            [_rec("t0", "kafka"), _rec("t1", "kafka")],
            [_rec("t2", "kafka")],
        ]
        result, qs_mock = _run_sample(pages, _query(source="kafka"))
        assert result == ["t0", "t1", "t2"]
        # Two pages of data + a terminating empty page → offset advances 0 / 500 / 1000
        size = rh._TRACE_SAMPLE_PAGE_SIZE
        ob_mock = qs_mock.order_by.return_value
        ob_mock.offset.assert_any_call(0)
        ob_mock.offset.assert_any_call(size)
        ob_mock.offset.assert_any_call(2 * size)

    def test_agent_name_and_project_filters_pushed_down(self) -> None:
        query = _query(source="kafka", agent_name="amap_chat", project_id="p1")
        result, qs_mock = _run_sample([[]], query)
        assert result == []
        # project_id + agent_name are pushed down as two scalar filters (the initial filter is a class call)
        assert qs_mock.filter.call_count == 2
        qs_mock.filter.assert_any_call(project_id="p1")
        qs_mock.filter.assert_any_call(agent_name="amap_chat")

    def test_optional_filters_absent(self) -> None:
        result, qs_mock = _run_sample([[]], _query(source="kafka"))
        assert result == []
        qs_mock.filter.assert_not_called()

    def test_empty_result(self) -> None:
        result, _ = _run_sample([[]], _query(source="kafka"))
        assert result == []

    def test_blank_source_raises(self) -> None:
        # Fail-fast on a blank source: letting it through silently would sample every trace that is missing a source
        service = BenchmarkService()
        with pytest.raises(ValueError, match="source"):
            _await(service.sample_trace_pipeline_records(_query(source="  ")))

    def test_judge_content_version_filters_rows(self) -> None:
        """example dual chain: sample only the lineage rows of the given content_version."""
        pages = [
            [
                _rec("t-v56", "kafka", content_version=56),
                _rec("t-v57", "kafka", content_version=57),
                _rec("t-plain", "kafka"),
            ],
        ]
        result, _ = _run_sample(
            pages,
            _query(source="kafka", judge_content_version="56"),
        )
        assert result == ["t-v56"]
