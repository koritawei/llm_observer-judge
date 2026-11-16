"""Unit tests for benchmark_worker — the run-rubric cache / failure truncation and the stack-log contract.

Locked fix contract:
- the in-process cache of ``_get_run_rubric`` removes the per-case N+1 read (one run queries the DB only once)
- ``_update_to_failed`` stores a truncated error_message in the DB column; the full stack goes into the error log
  (the agentgraph logger's exc_info → structured_fields.exception)
"""

import logging
from dataclasses import asdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.reporting.framework.schema import AggregationConfig, Criterion
from benchmark.queue.eval_case import EvalCase
from benchmark.workers import benchmark_worker
from benchmark.workers.benchmark_worker import (
    _RUN_RUBRIC_CACHE_MAX,
    _FailedRowWrite,
    _FusedSuccessRowWrite,
    _get_run_rubric,
    _resolved_case_source_trace_id,
    _update_fused_success_row,
    _update_to_failed,
)


@pytest.fixture(autouse=True)
def _clear_rubric_cache():
    """Isolate the module-level per-run cache (so tests do not pollute each other)."""
    benchmark_worker._run_rubric_cache.clear()
    yield
    benchmark_worker._run_rubric_cache.clear()


def _run_service(run: Any = None) -> MagicMock:
    svc = MagicMock()
    svc.get_run = AsyncMock(return_value=run)
    return svc


@pytest.mark.asyncio
async def test_get_run_rubric_caches_per_run() -> None:
    """Two calls for the same run query the DB only once (the per-case N+1 is gone) and return the same object."""
    run = MagicMock()
    run.config_json = {"rubric": []}
    svc = _run_service(run)

    first = await _get_run_rubric("run-1", svc)
    second = await _get_run_rubric("run-1", svc)

    assert svc.get_run.await_count == 1
    assert first is second


@pytest.mark.asyncio
async def test_get_run_rubric_run_not_found_raises() -> None:
    """The run does not exist → RuntimeError (the original get_run empty check is kept)."""
    svc = _run_service(None)
    with pytest.raises(RuntimeError, match="benchmark run not found"):
        await _get_run_rubric("run-x", svc)


@pytest.mark.asyncio
async def test_get_run_rubric_fifo_eviction() -> None:
    """When capacity is reached, evict the oldest entry (FIFO, so a long-lived worker does not grow without bound)."""
    run = MagicMock()
    run.config_json = {}
    svc = _run_service(run)

    for i in range(_RUN_RUBRIC_CACHE_MAX + 1):
        await _get_run_rubric(f"run-{i}", svc)

    assert len(benchmark_worker._run_rubric_cache) == _RUN_RUBRIC_CACHE_MAX
    assert "run-0" not in benchmark_worker._run_rubric_cache
    assert f"run-{_RUN_RUBRIC_CACHE_MAX}" in benchmark_worker._run_rubric_cache


def _mock_run_item_chain(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the BenchmarkRunItem.filter().exclude().update() chain, and return the chain."""
    chain = MagicMock()
    chain.exclude.return_value = chain
    chain.update = AsyncMock(return_value=2)
    monkeypatch.setattr(
        benchmark_worker,
        "BenchmarkRunItem",
        MagicMock(filter=MagicMock(return_value=chain)),
    )
    return chain


def test_resolved_case_source_trace_id_drops_search_eval_item_id() -> None:
    """The search_eval family: source_trace_id == item_id resolves to None (a dirty value is not written)."""
    item_id = "search_eval-20261030:7"
    case = EvalCase(item_id=item_id, source_trace_id=item_id, metadata={})
    assert _resolved_case_source_trace_id(case, benchmark_name="search_eval") is None
    assert _resolved_case_source_trace_id(case, benchmark_name="search_eval_v2") is None
    assert _resolved_case_source_trace_id(case, benchmark_name="example") == item_id


@pytest.mark.asyncio
async def test_update_fused_success_row_does_not_fallback_to_dirty_case_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When write.source_trace_id=None, do not fall back to the case's dirty item_id when writing the DB."""
    chain = _mock_run_item_chain(monkeypatch)
    item_id = "search_eval-20261030:7"
    case = EvalCase(item_id=item_id, source_trace_id=item_id, input={}, output={})

    await _update_fused_success_row(
        _FusedSuccessRowWrite(
            run_id="r",
            item_id=item_id,
            case=case,
            fused_score=1.0,
            judge_json={},
            source_trace_id=None,
        ),
    )

    kwargs = chain.update.call_args.kwargs
    assert "trace_id" not in kwargs
    assert "source_trace_id" not in kwargs


@pytest.mark.asyncio
async def test_update_to_failed_truncates_db_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DB error_message column is truncated to 2000 (the column-width limit)."""
    chain = _mock_run_item_chain(monkeypatch)
    err = ValueError("x" * 3000)

    updated = await _update_to_failed(
        _FailedRowWrite(run_id="r", item_id="i", criteria=[{"id": "c1"}], error=err)
    )

    assert updated == 2
    kwargs = chain.update.call_args.kwargs
    assert kwargs["error_message"] == "x" * 2000


@pytest.mark.asyncio
async def test_update_to_failed_logs_full_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The full exception stack goes into the error log (the structured exception field); debugging relies on the log."""
    _mock_run_item_chain(monkeypatch)
    err = ValueError("boom")

    with caplog.at_level(
        logging.ERROR,
        logger="agent_eval.eval_workers.benchmark_worker",
    ):
        await _update_to_failed(
            _FailedRowWrite(run_id="r", item_id="i", criteria=[{"id": "c1"}], error=err)
        )

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any(
        "item_marked_failed" in r.getMessage()
        and "ValueError: boom" in str(getattr(r, "structured_fields", {}).get("exception", ""))
        for r in error_records
    )


@pytest.mark.asyncio
async def test_update_to_failed_excludes_terminal_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotency guard: exclude success + failed, so failed→failed is not counted twice.

    ieval_external-20260909 regression: a stalled redelivery / several terminal failures hitting the same
    item over-collected judge_received_count (275/200).
    """
    chain = _mock_run_item_chain(monkeypatch)

    await _update_to_failed(
        _FailedRowWrite(run_id="r", item_id="i", criteria=[{"id": "c1"}], error=ValueError("x"))
    )
    exclude_kwargs = chain.exclude.call_args.kwargs
    assert set(exclude_kwargs["status__in"]) == {"success", "failed"}

    # The fused branch is guarded the same way
    chain2 = _mock_run_item_chain(monkeypatch)
    await _update_to_failed(
        _FailedRowWrite(
            run_id="r", item_id="i", criteria=[{"id": "c1"}], error=ValueError("x"), is_fused=True
        )
    )
    assert set(chain2.exclude.call_args.kwargs["status__in"]) == {"success", "failed"}


@pytest.mark.asyncio
async def test_local_judge_without_optional_extra_uses_empty_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criterion = Criterion(id="c1", name="c1", weight=1.0, evaluator_ref="search_ai_card")
    monkeypatch.setattr(
        benchmark_worker,
        "_get_run_rubric",
        AsyncMock(
            return_value=(
                [criterion],
                AggregationConfig(method="weighted_avg", pass_threshold=2.0),
            )
        ),
    )
    monkeypatch.setattr(
        benchmark_worker, "prewrite_running_fused_item", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(benchmark_worker, "build_trace_url_for_item", AsyncMock(return_value=None))
    persist = AsyncMock(return_value=1)
    monkeypatch.setattr(benchmark_worker, "_update_fused_success_row", persist)
    local_judge = MagicMock(
        judge_item=AsyncMock(
            return_value={
                "criterion_scores": {
                    "c1": {"score": 2.0, "reason": "ok", "sub_scores": {"relevance": 2.0}}
                }
            }
        )
    )
    run_service = MagicMock(
        increment_progress=AsyncMock(),
        get_run=AsyncMock(return_value=None),
    )

    outcome = await benchmark_worker._process_fused_case(
        benchmark_worker.FusedCaseContext(
            run_id="search_eval-20261029-a3",
            item_id="search_eval-20261029-a3:33",
            case=benchmark_worker.EvalCase(
                item_id="search_eval-20261029-a3:33",
                input={"query": "咖啡", "result": "咖啡店"},
                output={},
                metadata={},
            ),
            criteria=[asdict(criterion)],
            service_name="search_ai_card",
            project_id=None,
            evaluator_service=MagicMock(),
            run_service=run_service,
            local_judge=local_judge,
        )
    )

    assert outcome is not None
    assert outcome.fused_score == 2.0
    saved = persist.call_args.args[0].judge_json
    assert saved["raw_json"]["c1"]["extra"] == {}
