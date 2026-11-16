"""Code-attribution worker: write back and then dispatch on success; on failure still advance received."""

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.attribution.code.pipeline import CodeAttributionOutcome
from benchmark.workers.code_attribution_worker import (
    CodeAttributionJobDeps,
    _bind_executor_close,
    process_code_attribution_job,
)


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="code-attribution-test")
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


def _job(data: dict[str, object]) -> MagicMock:
    job = MagicMock()
    job.data = data
    return job


def _row() -> SimpleNamespace:
    return SimpleNamespace(
        input={"query": "附近加油站"},
        output={"tool_calls_json": "[]"},
        source_trace_id="judge-1",
        attribution_detail_json={
            "primary_issue_layer": "tool",
            "primary_issue_category": "tool_unavailable",
            "root_cause_chain": "没有工具",
            "raw_json": {
                "primary_issue_layer": "tool",
                "primary_issue_category": "tool_unavailable",
                "root_cause": "缺工具",
                "root_cause_chain": "没有工具",
            },
        },
    )


@pytest.mark.asyncio
async def test_worker_writes_replacement_then_dispatches(
    executor: ThreadPoolExecutor,
) -> None:
    row = _row()
    replaced = {
        **row.attribution_detail_json,
        "primary_issue_category": "执行失败/超时/5xx",
        "attribution_md": "新稿",
        "raw_json": {
            **row.attribution_detail_json["raw_json"],
            "primary_issue_category": "执行失败/超时/5xx",
        },
    }
    step = MagicMock()
    step.run.return_value = CodeAttributionOutcome(applied=True, document=replaced)
    query = MagicMock()
    query.first = AsyncMock(return_value=row)
    update = AsyncMock()
    run_service = MagicMock()
    run_service.increment_attribution_received_once = AsyncMock(
        return_value=(1, {"attribution_expected_count": 1}, True),
    )
    run_service.heartbeat = AsyncMock()

    with (
        patch(
            "benchmark.workers.code_attribution_worker.BenchmarkRunItem.filter",
            side_effect=[query, update],
        ),
        patch(
            "benchmark.workers.code_attribution_worker.schedule_notification_dispatch",
        ) as dispatch,
        patch(
            "benchmark.workers.code_attribution_worker.try_finalize_external",
            new_callable=AsyncMock,
        ) as finalize,
    ):
        await process_code_attribution_job(
            _job({"run_id": "run1", "item_id": "item1", "advance_received": True}),
            "token",
            CodeAttributionJobDeps(
                run_service=run_service,
                executor=executor,
                step=step,
            ),
        )

    update.update.assert_awaited_once()
    assert update.update.await_args.kwargs["attribution_detail_json"]["primary_issue_category"] == (
        "执行失败/超时/5xx"
    )
    dispatch.assert_called_once()
    assert dispatch.call_args.kwargs["data"].primary_issue_category == "执行失败/超时/5xx"
    run_service.increment_attribution_received_once.assert_awaited_once_with("run1", "item1")
    finalize.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_keeps_document_and_still_advances_received(
    executor: ThreadPoolExecutor,
) -> None:
    row = _row()
    step = MagicMock()
    step.run.side_effect = RuntimeError("aoned down")
    query = MagicMock()
    query.first = AsyncMock(return_value=row)
    run_service = MagicMock()
    run_service.increment_attribution_received_once = AsyncMock(
        return_value=(1, {"attribution_expected_count": 2}, True),
    )
    run_service.heartbeat = AsyncMock()

    with (
        patch(
            "benchmark.workers.code_attribution_worker.BenchmarkRunItem.filter",
            return_value=query,
        ),
        patch(
            "benchmark.workers.code_attribution_worker.schedule_notification_dispatch",
        ) as dispatch,
        patch(
            "benchmark.workers.code_attribution_worker.try_finalize_external",
            new_callable=AsyncMock,
        ) as finalize,
    ):
        await process_code_attribution_job(
            _job({"run_id": "run1", "item_id": "item1", "advance_received": True}),
            "token",
            CodeAttributionJobDeps(
                run_service=run_service,
                executor=executor,
                step=step,
            ),
        )

    dispatch.assert_called_once()
    assert dispatch.call_args.kwargs["data"].primary_issue_category == "tool_unavailable"
    run_service.increment_attribution_received_once.assert_awaited_once()
    finalize.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_runs_analysis_on_given_executor(executor: ThreadPoolExecutor) -> None:
    row = _row()
    pool_thread = executor.submit(threading.get_ident).result()
    seen: dict[str, int] = {}

    def run(_case: object) -> CodeAttributionOutcome:
        seen["thread"] = threading.get_ident()
        return CodeAttributionOutcome(applied=False, document=row.attribution_detail_json)

    step = MagicMock()
    step.run.side_effect = run
    query = MagicMock()
    query.first = AsyncMock(return_value=row)
    run_service = MagicMock()

    with patch(
        "benchmark.workers.code_attribution_worker.BenchmarkRunItem.filter",
        return_value=query,
    ):
        await process_code_attribution_job(
            _job({"run_id": "run1", "item_id": "item1"}),
            "token",
            CodeAttributionJobDeps(
                run_service=run_service,
                executor=executor,
                step=step,
            ),
        )

    assert seen["thread"] == pool_thread


@pytest.mark.asyncio
async def test_worker_close_shuts_down_executor() -> None:
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="code-attribution-test")
    original_close = AsyncMock()
    worker = MagicMock()
    worker.close = original_close
    bound = _bind_executor_close(worker, pool)

    await bound.close(force=True)

    original_close.assert_awaited_once_with(force=True)
    with pytest.raises(RuntimeError):
        pool.submit(threading.get_ident)
