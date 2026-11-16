"""The finalize_trigger callback contract: a missing complete_run argument must not leave it stuck in aggregating."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.workers import finalize_trigger as ft


@pytest.fixture(autouse=True)
def _clear_callbacks() -> None:
    ft._finalize_callbacks.clear()
    yield
    ft._finalize_callbacks.clear()


def _external_run_service(**extra: object) -> MagicMock:
    run_service = MagicMock()
    run_service.get_run = AsyncMock(
        return_value=MagicMock(config_json={}),
    )
    run_service.fail_run = AsyncMock()
    for key, value in extra.items():
        setattr(run_service, key, value)
    return run_service


@pytest.mark.asyncio
async def test_try_finalize_external_invokes_registered_callback() -> None:
    callback = AsyncMock()
    ft.register_finalize_callback("external", callback)
    run_service = _external_run_service()

    fake_rows = (1, [{"run_id": "r1"}])
    conn = MagicMock()
    conn.execute_query = AsyncMock(return_value=fake_rows)

    with patch(
        "benchmark.workers.finalize_trigger.Tortoise.get_connection",
        return_value=conn,
    ):
        won = await ft.try_finalize_external("r1", run_service)

    assert won is True
    callback.assert_awaited_once_with("r1")
    run_service.fail_run.assert_not_awaited()
    cas_sql = str(conn.execute_query.await_args_list[0].args[0])
    assert "stop_requested_at" in cas_sql
    assert "benchmark_run_items" in cas_sql


@pytest.mark.asyncio
async def test_try_finalize_external_callback_error_marks_failed() -> None:
    async def _boom(_run_id: str) -> None:
        raise TypeError("complete_run() missing final_score")

    ft.register_finalize_callback("external", _boom)
    run_service = _external_run_service()

    fake_rows = (1, [{"run_id": "r1"}])
    conn = MagicMock()
    conn.execute_query = AsyncMock(return_value=fake_rows)

    with patch(
        "benchmark.workers.finalize_trigger.Tortoise.get_connection",
        return_value=conn,
    ):
        won = await ft.try_finalize_external("r1", run_service)

    assert won is True
    run_service.fail_run.assert_awaited_once()
    assert run_service.fail_run.await_args.kwargs["reason"] == "finalize failed"


@pytest.mark.asyncio
async def test_try_finalize_external_no_callback_completes_with_score() -> None:
    run_service = _external_run_service(
        load_fused_score_average=AsyncMock(return_value=0.99),
        complete_run=AsyncMock(),
    )

    fake_rows = (1, [{"run_id": "r1"}])
    conn = MagicMock()
    conn.execute_query = AsyncMock(return_value=fake_rows)

    with patch(
        "benchmark.workers.finalize_trigger.Tortoise.get_connection",
        return_value=conn,
    ):
        won = await ft.try_finalize_external("r1", run_service)

    assert won is True
    run_service.complete_run.assert_awaited_once()
    # The complete_run signature is narrowed to a CompleteRunWrite parameter object (FUNC-PARAMS-001)
    write = run_service.complete_run.await_args.args[0]
    assert write.run_id == "r1"
    assert write.final_score == 0.99
    assert write.metrics_json is not None
    run_service.fail_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_try_finalize_external_stop_latch_blocks_before_cas() -> None:
    """The stop latch is set and running items remain: do not enter CAS / emit the report from the callback."""
    callback = AsyncMock()
    ft.register_finalize_callback("external", callback)
    run_service = _external_run_service()
    run_service.get_run = AsyncMock(
        return_value=MagicMock(
            config_json={"stop_requested_at": "2026-11-11T10:00:00+00:00"},
        ),
    )
    conn = MagicMock()
    conn.execute_query = AsyncMock()

    with (
        patch(
            "benchmark.workers.finalize_trigger.Tortoise.get_connection",
            return_value=conn,
        ),
        patch(
            "benchmark.workers.finalize_trigger.stop_items_ready_for_report",
            new=AsyncMock(return_value=False),
        ),
    ):
        won = await ft.try_finalize_external("r1", run_service)

    assert won is False
    conn.execute_query.assert_not_awaited()
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_try_finalize_external_stuck_aggregating_rolls_back_on_running_items() -> None:
    """Recover from a stuck aggregating: under stop, when running items remain, pull it back to running."""
    callback = AsyncMock()
    ft.register_finalize_callback("external", callback)
    run_service = _external_run_service()
    run_service.get_run = AsyncMock(
        return_value=MagicMock(
            config_json={"stop_requested_at": "2026-11-11T10:00:00+00:00"},
        ),
    )
    conn = MagicMock()
    # 1) Phase1 CAS did not win 2) it is already aggregating 3) pull it back to running
    conn.execute_query = AsyncMock(
        side_effect=[
            (0, []),
            (1, [{"ok": 1}]),
            (1, [{"ok": 1}]),
        ],
    )

    with (
        patch(
            "benchmark.workers.finalize_trigger.Tortoise.get_connection",
            return_value=conn,
        ),
        patch(
            "benchmark.workers.finalize_trigger.stop_items_ready_for_report",
            new=AsyncMock(return_value=False),
        ),
    ):
        # The entry latch blocks first; to test the stuck path the entry must be let through once, then stuck blocks again
        with patch(
            "benchmark.workers.finalize_trigger._stop_latch_blocks_report",
            new=AsyncMock(side_effect=[False, True]),
        ):
            won = await ft.try_finalize_external("r1", run_service)

    assert won is False
    callback.assert_not_awaited()
    assert conn.execute_query.await_count == 3
    rollback_sql = str(conn.execute_query.await_args_list[2].args[0])
    assert "status = 'running'" in rollback_sql
    assert "aggregating" in rollback_sql
