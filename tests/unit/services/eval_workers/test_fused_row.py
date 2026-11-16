"""Fused-row prewrite: must not roll success/failed back to running."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.workers.fused_row import (
    FusedRunningPrewrite,
    prewrite_running_fused_item,
)


@pytest.mark.asyncio
async def test_prewrite_creates_running_row() -> None:
    item = MagicMock()
    item.status = "running"
    with patch(
        "benchmark.workers.fused_row.BenchmarkRunItem.get_or_create",
        new_callable=AsyncMock,
        return_value=(item, True),
    ):
        still_open = await prewrite_running_fused_item(
            FusedRunningPrewrite(run_id="r1", item_id="i1"),
        )
    assert still_open is True
    item.save.assert_not_called()


@pytest.mark.asyncio
async def test_prewrite_skips_success_without_overwrite() -> None:
    item = MagicMock()
    item.status = "success"
    item.save = AsyncMock()
    with patch(
        "benchmark.workers.fused_row.BenchmarkRunItem.get_or_create",
        new_callable=AsyncMock,
        return_value=(item, False),
    ):
        still_open = await prewrite_running_fused_item(
            FusedRunningPrewrite(
                run_id="r1",
                item_id="i1",
                source_trace_id="job-new",
            ),
        )
    assert still_open is False
    item.save.assert_not_called()
    assert item.status == "success"


@pytest.mark.asyncio
async def test_prewrite_skips_failed_without_overwrite() -> None:
    item = MagicMock()
    item.status = "failed"
    item.save = AsyncMock()
    with patch(
        "benchmark.workers.fused_row.BenchmarkRunItem.get_or_create",
        new_callable=AsyncMock,
        return_value=(item, False),
    ):
        still_open = await prewrite_running_fused_item(
            FusedRunningPrewrite(run_id="r1", item_id="i1"),
        )
    assert still_open is False
    item.save.assert_not_called()


@pytest.mark.asyncio
async def test_prewrite_refreshes_job_id_on_running_row() -> None:
    item = MagicMock()
    item.status = "running"
    item.save = AsyncMock()
    with patch(
        "benchmark.workers.fused_row.BenchmarkRunItem.get_or_create",
        new_callable=AsyncMock,
        return_value=(item, False),
    ):
        still_open = await prewrite_running_fused_item(
            FusedRunningPrewrite(
                run_id="r1",
                item_id="i1",
                source_trace_id="job-2",
                item_input={"q": "x"},
            ),
        )
    assert still_open is True
    assert item.source_trace_id == "job-2"
    item.save.assert_awaited_once()
    assert "status" not in item.save.await_args.kwargs["update_fields"]
