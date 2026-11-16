"""Unit tests for BenchmarkRunner EvalFailure pipeline stubs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.runner import BenchmarkRunContext, BenchmarkRunner
from benchmark.runs.schema import BenchmarkRequest
from benchmark.reporting.framework.schema import Criterion, DatasetRef


@pytest.mark.asyncio
async def test_write_eval_failure_stubs_upserts_eval_failed_rows() -> None:
    runner = object.__new__(BenchmarkRunner)
    criterion = Criterion(
        id="c1",
        name="C1",
        description="d",
        weight=1.0,
        evaluator_ref="amap_agent",
    )
    items = [
        SimpleNamespace(id="item-1", source_trace_id=None),
        SimpleNamespace(id="item-2", source_trace_id="lf-1"),
    ]
    services = MagicMock()
    services.get_metadata.return_value = SimpleNamespace(name="example_benchmark")
    dataset_ref = DatasetRef(
        dataset_name="ds",
        project_id="proj-1",
    )

    ctx = BenchmarkRunContext(
        benchmark_services=services,
        run_id="bm_test_20260101120000_abcdef",
        service_name="amap_chat",
        dataset_ref=dataset_ref,
        rubric=[criterion],
        items=items,
        max_concurrency=1,
        parallel_evaluators=False,
        triage_mode="none",
        request=BenchmarkRequest(),
    )

    writer = MagicMock()
    writer.upsert_score = AsyncMock()
    with patch(
        "benchmark.runs.runner.BenchmarkTraceRecordWriter",
        return_value=writer,
    ):
        await runner._write_eval_failure_stubs(ctx, criterion, RuntimeError("boom"))

    assert writer.upsert_score.await_count == 2
    first = writer.upsert_score.await_args_list[0].args[0]
    assert first.score == 0.0
    assert first.judge_json["eval_failed"] is True
    assert first.judge_json["score"] == "0.0"
    assert first.dataset_item_id == "item-1"
    assert first.evaluator_ref == "amap_agent"
    second = writer.upsert_score.await_args_list[1].args[0]
    assert second.trace_id == "lf-1"
    assert second.dataset_item_id == "item-2"


@pytest.mark.asyncio
async def test_write_eval_failure_stubs_continues_on_upsert_error() -> None:
    runner = object.__new__(BenchmarkRunner)
    criterion = Criterion(
        id="c1",
        name="C1",
        description="d",
        weight=1.0,
        evaluator_ref="e1",
    )
    items = [
        SimpleNamespace(id="a", source_trace_id=None),
        SimpleNamespace(id="b", source_trace_id=None),
    ]
    services = MagicMock()
    services.get_metadata.return_value = SimpleNamespace(name="bm")
    dataset_ref = DatasetRef(dataset_name="ds", project_id="p")

    ctx = BenchmarkRunContext(
        benchmark_services=services,
        run_id="bm_x_20260101120000_abcdef",
        service_name="agent",
        dataset_ref=dataset_ref,
        rubric=[criterion],
        items=items,
        max_concurrency=1,
        parallel_evaluators=False,
        triage_mode="none",
        request=BenchmarkRequest(),
    )

    writer = MagicMock()
    writer.upsert_score = AsyncMock(side_effect=[RuntimeError("db"), None])
    with patch(
        "benchmark.runs.runner.BenchmarkTraceRecordWriter",
        return_value=writer,
    ):
        await runner._write_eval_failure_stubs(ctx, criterion, ValueError("x"))

    assert writer.upsert_score.await_count == 2
