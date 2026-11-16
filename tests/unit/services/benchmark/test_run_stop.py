"""The stop-this-run latch and partial meta (§6.8)."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.run_stop import (
    STOP_RECONCILE_GRACE_S,
    FinalizeStoppedWrite,
    align_expectations_for_stop,
    count_running_benchmark_items,
    finalize_as_stopped,
    inspect_stop_report_readiness,
    is_stop_requested,
    mark_stop_requested,
    parse_stop_requested_at,
    partial_report_meta,
    reconcile_stop_target_to_processed,
    stop_attribution_pending,
    stop_items_ready_for_report,
    stop_reconcile_in_grace,
    stop_rows_per_item,
)


def test_is_stop_requested_reads_config_flag() -> None:
    assert is_stop_requested({"stop_requested_at": "2026-11-03T08:00:00+00:00"})
    assert not is_stop_requested({})
    assert not is_stop_requested(None)
    assert not is_stop_requested("bad")


def test_stop_reconcile_in_grace_covers_mark_align_window() -> None:
    now = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)
    fresh = (now - timedelta(seconds=5)).isoformat()
    aged = (now - timedelta(seconds=STOP_RECONCILE_GRACE_S + 1)).isoformat()
    assert stop_reconcile_in_grace({"stop_requested_at": fresh}, now=now)
    assert not stop_reconcile_in_grace({"stop_requested_at": aged}, now=now)
    assert not stop_reconcile_in_grace({}, now=now)
    assert not stop_reconcile_in_grace({"stop_requested_at": "not-a-time"}, now=now)
    assert parse_stop_requested_at({"stop_requested_at": fresh}) is not None


def test_stop_rows_per_item_matches_triage_mode() -> None:
    run = MagicMock()
    run.total_evaluators = 4
    assert stop_rows_per_item(run, {"triage_mode": "fused"}) == 1
    assert stop_rows_per_item(run, {"triage_mode": "per_eval"}) == 4
    assert stop_rows_per_item(run, {"triage_mode": "none"}) == 4
    run.total_evaluators = 0
    assert stop_rows_per_item(run, {"triage_mode": "per_eval", "total_evaluators": 2}) == 2
    assert stop_rows_per_item(run, {"triage_mode": "per_eval"}) == 1


def test_partial_report_meta_marks_skipped_when_no_cases() -> None:
    meta = partial_report_meta(
        completed_cases=0,
        stop_requested_at="2026-11-03T08:00:00+00:00",
        existing={"report_site_url": "https://pages.example/r.html"},
    )
    assert meta["partial"] is True
    assert meta["stop_kind"] == "manual"
    assert meta["report_status"] == "skipped_no_cases"
    assert meta["report_site_url"] == "https://pages.example/r.html"
    assert meta["completed_cases"] == 0


def test_partial_report_meta_ready_when_cases_exist() -> None:
    meta = partial_report_meta(
        completed_cases=12,
        stop_requested_at="2026-11-03T08:00:00+00:00",
    )
    assert meta["report_status"] == "ready"
    assert meta["completed_cases"] == 12


@pytest.mark.asyncio
async def test_mark_stop_requested_preserves_total_samples() -> None:
    """The stop latch writes the latch + a provisional stop_target; the expected count is left for align."""
    run = MagicMock()
    run.status = "running"
    run.config_json = {
        "judge_expected_count": 200,
        "judge_received_count": 77,
        "attribution_expected_count": 200,
        "attribution_received_count": 70,
    }
    run.evaluated_items = 70
    run.failed_items = 7
    run.total_samples = 200
    run.total_evaluators = 1

    update_mock = AsyncMock()
    filter_chain = MagicMock()
    filter_chain.first = AsyncMock(return_value=run)
    filter_chain.update = update_mock

    with patch(
        "benchmark.runs.run_stop.BenchmarkRun.filter",
        return_value=filter_chain,
    ) as filter_fn:
        result = await mark_stop_requested(run_id="run-1")

    assert result["ok"] is True
    assert result["noop"] is False
    assert result["completed_cases"] == 77
    assert filter_fn.call_count >= 2
    update_kwargs: dict[str, Any] = update_mock.await_args.kwargs
    assert set(update_kwargs.keys()) == {"config_json"}
    assert "total_samples" not in update_kwargs
    cfg = update_kwargs["config_json"]
    assert cfg["stop_requested_at"]
    # fused：provisional = max(total_samples, processed+rows_per+1)=200
    assert cfg["stop_target_count"] == 200
    assert result["stop_target_count"] == 200
    # The latch stage does not shrink expected (otherwise an in-flight running row with no result yet would emit the report early)
    assert cfg["judge_expected_count"] == 200
    assert cfg["attribution_expected_count"] == 200


@pytest.mark.asyncio
async def test_mark_stop_provisional_target_unreachable_when_partial() -> None:
    """Before the run finishes, provisional = max(full, processed+rows_per+1), and CAS does not release before align."""
    run = MagicMock()
    run.status = "running"
    run.config_json = {}
    run.evaluated_items = 10
    run.failed_items = 0
    run.total_samples = 10  # When processed == total it still has to be raised, so the latch window does not release immediately
    run.total_evaluators = 1

    update_mock = AsyncMock()
    filter_chain = MagicMock()
    filter_chain.first = AsyncMock(return_value=run)
    filter_chain.update = update_mock

    with patch(
        "benchmark.runs.run_stop.BenchmarkRun.filter",
        return_value=filter_chain,
    ):
        result = await mark_stop_requested(run_id="run-edge")

    cfg = update_mock.await_args.kwargs["config_json"]
    assert cfg["stop_target_count"] == 12  # max(10, 10+1+1)
    assert result["stop_target_count"] == 12


@pytest.mark.asyncio
async def test_mark_stop_provisional_per_eval_uses_row_threshold() -> None:
    """per_eval provisional is raised by the total_samples×evaluators row count."""
    run = MagicMock()
    run.status = "running"
    run.config_json = {"triage_mode": "per_eval"}
    run.evaluated_items = 6
    run.failed_items = 0
    run.total_samples = 10
    run.total_evaluators = 3

    update_mock = AsyncMock()
    filter_chain = MagicMock()
    filter_chain.first = AsyncMock(return_value=run)
    filter_chain.update = update_mock

    with patch(
        "benchmark.runs.run_stop.BenchmarkRun.filter",
        return_value=filter_chain,
    ):
        result = await mark_stop_requested(run_id="run-per-eval")

    # max(10*3, 6+3+1) = 30
    assert result["stop_target_count"] == 30


@pytest.mark.asyncio
async def test_align_expectations_for_stop_includes_inflight() -> None:
    """align (fused): judge expected=received+inflight; attribution expected takes the max so it is not pushed down."""
    run = MagicMock()
    run.config_json = {
        "triage_mode": "fused",
        "judge_expected_count": 200,
        "judge_received_count": 77,
        "attribution_expected_count": 200,
        "attribution_received_count": 70,
    }
    run.evaluated_items = 70
    run.failed_items = 7
    run.total_evaluators = 1

    update_mock = AsyncMock()
    filter_chain = MagicMock()
    filter_chain.first = AsyncMock(return_value=run)
    filter_chain.update = update_mock

    with patch(
        "benchmark.runs.run_stop.BenchmarkRun.filter",
        return_value=filter_chain,
    ):
        aligned = await align_expectations_for_stop(
            "run-1",
            inflight_judge=3,
            inflight_attribution=1,
            processed_snapshot=77,
        )

    assert aligned["stop_target_count"] == 81  # 77 + 3*1 + 1
    assert aligned["inflight_total"] == 4
    assert aligned["processed"] == 77
    assert aligned["rows_per_item"] == 1
    cfg = update_mock.await_args.kwargs["config_json"]
    assert cfg["stop_target_count"] == 81
    assert cfg["judge_expected_count"] == 80  # 77 + 3
    # Must not push down an already registered expected (the initial 200); it may only be raised to received+inflight
    assert cfg["attribution_expected_count"] == 200


@pytest.mark.asyncio
async def test_align_per_eval_scales_inflight_jobs_to_rows() -> None:
    """per_eval: stop_target = processed + judge_jobs×N + attr_jobs (so CAS does not fire early)."""
    run = MagicMock()
    run.config_json = {"triage_mode": "per_eval"}
    run.evaluated_items = 30
    run.failed_items = 0
    run.total_evaluators = 3

    update_mock = AsyncMock()
    filter_chain = MagicMock()
    filter_chain.first = AsyncMock(return_value=run)
    filter_chain.update = update_mock

    with patch(
        "benchmark.runs.run_stop.BenchmarkRun.filter",
        return_value=filter_chain,
    ):
        aligned = await align_expectations_for_stop(
            "run-per-eval",
            inflight_judge=2,
            inflight_attribution=1,
            processed_snapshot=30,
        )

    # 30 + 2*3 + 1 = 37; if the job count were added by mistake it would become 33 → the first job (+3) would CAS
    assert aligned["rows_per_item"] == 3
    assert aligned["stop_target_count"] == 37
    assert update_mock.await_args.kwargs["config_json"]["stop_target_count"] == 37


@pytest.mark.asyncio
async def test_align_uses_processed_snapshot_not_stale_row() -> None:
    """processed_snapshot wins over evaluated/failed at row-read time, so they are not double-counted."""
    run = MagicMock()
    run.config_json = {"triage_mode": "fused"}
    # By the time the row is read the callback has landed, which is 1 more than the snapshot — it must not be added into stop_target again
    run.evaluated_items = 78
    run.failed_items = 0
    run.total_evaluators = 1

    update_mock = AsyncMock()
    filter_chain = MagicMock()
    filter_chain.first = AsyncMock(return_value=run)
    filter_chain.update = update_mock

    with patch(
        "benchmark.runs.run_stop.BenchmarkRun.filter",
        return_value=filter_chain,
    ):
        aligned = await align_expectations_for_stop(
            "run-1",
            inflight_judge=1,
            processed_snapshot=77,
        )

    assert aligned["processed"] == 77
    assert aligned["stop_target_count"] == 78
    assert update_mock.await_args.kwargs["config_json"]["stop_target_count"] == 78


@pytest.mark.asyncio
async def test_reconcile_stop_target_clamps_to_processed() -> None:
    """Queue-already-empty fallback: inflight=0, and stop_target is pushed down to the current processed."""
    run = MagicMock()
    run.config_json = {"stop_target_count": 99, "stop_requested_at": "t"}
    run.evaluated_items = 40
    run.failed_items = 2
    run.total_evaluators = 1

    update_mock = AsyncMock()
    filter_chain = MagicMock()
    filter_chain.first = AsyncMock(return_value=run)
    filter_chain.update = update_mock

    with patch(
        "benchmark.runs.run_stop.BenchmarkRun.filter",
        return_value=filter_chain,
    ):
        aligned = await reconcile_stop_target_to_processed("run-1")

    assert aligned["stop_target_count"] == 42
    assert aligned["inflight_total"] == 0
    assert update_mock.await_args.kwargs["config_json"]["stop_target_count"] == 42


@pytest.mark.asyncio
async def test_count_running_benchmark_items_and_ready_gate() -> None:
    """stop-run report gate: ready only when the running-item row count is back to zero."""
    filter_chain = MagicMock()
    filter_chain.count = AsyncMock(return_value=3)
    run_filter = MagicMock()
    run_filter.first = AsyncMock(return_value=MagicMock(config_json={}))
    with (
        patch(
            "benchmark.runs.run_stop.BenchmarkRunItem.filter",
            return_value=filter_chain,
        ) as filt,
        patch(
            "benchmark.runs.run_stop.BenchmarkRun.filter",
            return_value=run_filter,
        ),
    ):
        assert await count_running_benchmark_items("bm-1") == 3
        assert await stop_items_ready_for_report("bm-1") is False
    filt.assert_called_with(run_id="bm-1", status="running")

    filter_chain.count = AsyncMock(return_value=0)
    with (
        patch(
            "benchmark.runs.run_stop.BenchmarkRunItem.filter",
            return_value=filter_chain,
        ),
        patch(
            "benchmark.runs.run_stop.BenchmarkRun.filter",
            return_value=run_filter,
        ),
    ):
        assert await count_running_benchmark_items("bm-1") == 0
        assert await stop_items_ready_for_report("bm-1") is True


def test_stop_attribution_pending_counts_unreceived() -> None:
    """When attribution is configured, pending is the expected-received gap / the id set difference."""
    assert stop_attribution_pending({}) == 0
    assert (
        stop_attribution_pending(
            {
                "attribution_expected_count": 3,
                "attribution_received_count": 1,
                "attribution_expected_item_ids": ["a", "b", "c"],
                "attribution_received_item_ids": ["a"],
            },
        )
        == 2
    )
    assert (
        stop_attribution_pending(
            {
                "attribution_expected_count": 2,
                "attribution_received_count": 2,
                "attribution_expected_item_ids": ["a", "b"],
                "attribution_received_item_ids": ["a", "b"],
            },
        )
        == 0
    )


@pytest.mark.asyncio
async def test_inspect_stop_report_readiness_waits_attribution() -> None:
    """running=0 but attribution is not complete: a stop-run still must not emit the report."""
    item_filter = MagicMock()
    item_filter.count = AsyncMock(return_value=0)
    run = MagicMock()
    run.config_json = {
        "attribution_expected_count": 2,
        "attribution_received_count": 0,
        "attribution_expected_item_ids": ["i1", "i2"],
        "attribution_received_item_ids": [],
    }
    run_filter = MagicMock()
    run_filter.first = AsyncMock(return_value=run)
    with (
        patch(
            "benchmark.runs.run_stop.BenchmarkRunItem.filter",
            return_value=item_filter,
        ),
        patch(
            "benchmark.runs.run_stop.BenchmarkRun.filter",
            return_value=run_filter,
        ),
    ):
        snap = await inspect_stop_report_readiness("bm-attr")
        assert snap.ready is False
        assert snap.running_items == 0
        assert snap.attribution_pending == 2
        assert await stop_items_ready_for_report("bm-attr") is False


@pytest.mark.asyncio
async def test_align_stop_does_not_lower_attribution_expected() -> None:
    """stop align / reconcile must not push attribution expected down to received (that would stop waiting for Aone)."""
    run = MagicMock()
    run.config_json = {
        "attribution_expected_count": 5,
        "attribution_received_count": 2,
        "triage_mode": "fused",
    }
    run.evaluated_items = 2
    run.failed_items = 0
    run.total_evaluators = 1
    update_mock = AsyncMock()
    filter_chain = MagicMock()
    filter_chain.first = AsyncMock(return_value=run)
    filter_chain.update = update_mock
    with patch(
        "benchmark.runs.run_stop.BenchmarkRun.filter",
        return_value=filter_chain,
    ):
        aligned = await align_expectations_for_stop(
            "run-attr",
            inflight_judge=0,
            inflight_attribution=0,
            processed_snapshot=2,
        )
    cfg = update_mock.await_args.kwargs["config_json"]
    assert cfg["attribution_expected_count"] == 5
    assert aligned["stop_target_count"] == 2


@pytest.mark.asyncio
async def test_finalize_as_stopped_defers_while_items_running() -> None:
    """While running items remain, do not persist stopped / emit the report."""
    run = MagicMock()
    run.status = "running"
    run.config_json = {"stop_requested_at": "2026-11-11T10:00:00+00:00"}
    run.evaluated_items = 2
    run.failed_items = 0
    run.report_meta_json = {}
    filter_chain = MagicMock()
    filter_chain.first = AsyncMock(return_value=run)
    filter_chain.update = AsyncMock(return_value=1)
    with (
        patch(
            "benchmark.runs.run_stop.BenchmarkRun.filter",
            return_value=filter_chain,
        ),
        patch(
            "benchmark.runs.run_stop.stop_items_ready_for_report",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "benchmark.runs.run_stop.count_running_benchmark_items",
            new=AsyncMock(return_value=3),
        ),
    ):
        ok = await finalize_as_stopped(FinalizeStoppedWrite(run_id="bm-wait"))

    assert ok is False
    filter_chain.update.assert_not_awaited()
