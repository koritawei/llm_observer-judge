"""Unit tests for cas_advance_external_judge_item — conditional advance is idempotent inside one transaction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.run_service import BenchmarkService
from benchmark.runs.schema import ExternalJudgeItemResult


@pytest.mark.asyncio
async def test_cas_advance_first_success_bumps_evaluated_and_received() -> None:
    service = BenchmarkService.__new__(BenchmarkService)

    run = MagicMock()
    run.config_json = {"judge_received_count": 2, "judge_expected_count": 10}
    run.evaluated_items = 2
    run.failed_items = 0
    run.save = AsyncMock()

    item = MagicMock()
    item.status = "running"
    item.save = AsyncMock()

    run_qs = MagicMock()
    run_qs.select_for_update.return_value = run_qs
    run_qs.using_db.return_value = run_qs
    run_qs.first = AsyncMock(return_value=run)

    item_qs = MagicMock()
    item_qs.select_for_update.return_value = item_qs
    item_qs.using_db.return_value = item_qs
    item_qs.first = AsyncMock(return_value=item)

    def _filter(**kwargs):
        if "item_id" in kwargs:
            return item_qs
        return run_qs

    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=MagicMock())
    tx.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "benchmark.runs.run_helpers.in_transaction",
            return_value=tx,
        ),
        patch(
            "benchmark.runs.run_helpers.BenchmarkRun.filter",
            side_effect=_filter,
        ),
        patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem.filter",
            side_effect=_filter,
        ),
        patch(
            "benchmark.runs.run_helpers._stamp_bucket",
            side_effect=lambda x: x,
        ),
    ):
        advanced = await service.cas_advance_external_judge_item(
            result=ExternalJudgeItemResult(
                run_id="run1",
                item_id="item1",
                failed=False,
                score=0.9,
                judge_json={"score": 0.9},
                source_trace_id="job-1",
                trace_id="langfuse-trace-1",
                trace_url="https://langfuse.test/project/p1/traces/t1",
            ),
        )

    assert advanced is True
    assert item.status == "success"
    assert item.trace_id == "langfuse-trace-1"
    assert item.trace_url == "https://langfuse.test/project/p1/traces/t1"
    assert run.evaluated_items == 3
    assert run.failed_items == 0
    assert run.config_json["judge_received_count"] == 3
    # judge_json bucket passthrough (including the raw_json callback body injected by the caller)
    assert item.judge_json.get("score") == "0.9"
    item.save.assert_awaited()
    run.save.assert_awaited()


@pytest.mark.asyncio
async def test_cas_advance_already_terminal_is_idempotent() -> None:
    service = BenchmarkService.__new__(BenchmarkService)

    run = MagicMock()
    run.config_json = {"judge_received_count": 5}
    run.evaluated_items = 5
    run.failed_items = 0
    run.save = AsyncMock()

    item = MagicMock()
    item.status = "success"
    item.save = AsyncMock()

    run_qs = MagicMock()
    run_qs.select_for_update.return_value = run_qs
    run_qs.using_db.return_value = run_qs
    run_qs.first = AsyncMock(return_value=run)

    item_qs = MagicMock()
    item_qs.select_for_update.return_value = item_qs
    item_qs.using_db.return_value = item_qs
    item_qs.first = AsyncMock(return_value=item)

    def _filter(**kwargs):
        if "item_id" in kwargs:
            return item_qs
        return run_qs

    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=MagicMock())
    tx.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "benchmark.runs.run_helpers.in_transaction",
            return_value=tx,
        ),
        patch(
            "benchmark.runs.run_helpers.BenchmarkRun.filter",
            side_effect=_filter,
        ),
        patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem.filter",
            side_effect=_filter,
        ),
    ):
        advanced = await service.cas_advance_external_judge_item(
            result=ExternalJudgeItemResult(
                run_id="run1",
                item_id="item1",
                failed=False,
                score=0.5,
                judge_json={},
            ),
        )

    assert advanced is False
    assert run.evaluated_items == 5
    assert run.config_json["judge_received_count"] == 5
    item.save.assert_not_awaited()
    run.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_cas_advance_failed_bumps_failed_items() -> None:
    service = BenchmarkService.__new__(BenchmarkService)

    run = MagicMock()
    run.config_json = {}
    run.evaluated_items = 0
    run.failed_items = 1
    run.save = AsyncMock()

    item = MagicMock()
    item.status = "running"
    item.save = AsyncMock()

    run_qs = MagicMock()
    run_qs.select_for_update.return_value = run_qs
    run_qs.using_db.return_value = run_qs
    run_qs.first = AsyncMock(return_value=run)

    item_qs = MagicMock()
    item_qs.select_for_update.return_value = item_qs
    item_qs.using_db.return_value = item_qs
    item_qs.first = AsyncMock(return_value=item)

    def _filter(**kwargs):
        if "item_id" in kwargs:
            return item_qs
        return run_qs

    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=MagicMock())
    tx.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "benchmark.runs.run_helpers.in_transaction",
            return_value=tx,
        ),
        patch(
            "benchmark.runs.run_helpers.BenchmarkRun.filter",
            side_effect=_filter,
        ),
        patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem.filter",
            side_effect=_filter,
        ),
        patch(
            "benchmark.runs.run_helpers._stamp_bucket",
            side_effect=lambda x: x,
        ),
    ):
        advanced = await service.cas_advance_external_judge_item(
            result=ExternalJudgeItemResult(
                run_id="run1",
                item_id="item1",
                failed=True,
                score=0.0,
                judge_json={"error": "boom"},
                error_message="boom",
            ),
        )

    assert advanced is True
    assert item.status == "failed"
    assert run.failed_items == 2
    assert run.evaluated_items == 0
    assert run.config_json["judge_received_count"] == 1
