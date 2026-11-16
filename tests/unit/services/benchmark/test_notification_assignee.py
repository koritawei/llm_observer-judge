"""Change a case assignee: employee-alias check, case ownership, a1 update, and bucket write-back."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from benchmark.runs.notification_assignee import NotificationAssigneeError, reassign_case_notification_assignee
from benchmark.runs.run_case_page import NOTIFICATION_SOURCE_ITEMS, NOTIFICATION_SOURCE_TRACE


def _run() -> SimpleNamespace:
    return SimpleNamespace(
        run_id="run-1",
        config_json={"agent_name": "chat-agent"},
    )


@pytest.mark.asyncio
async def test_reassign_rejects_invalid_assignee_pattern() -> None:
    with pytest.raises(NotificationAssigneeError) as exc:
        await reassign_case_notification_assignee(
            object(),
            _run(),
            case_id="case-1",
            assignee="bad name!",
        )
    assert exc.value.status_code == 400
    assert "花名" in exc.value.message


@pytest.mark.asyncio
async def test_reassign_rejects_case_not_in_items() -> None:
    """items source: load_item_notification_row is empty → 404, and an out-of-scope reassignment is forbidden."""
    with (
        patch(
            "benchmark.runs.notification_assignee.resolve_notification_source",
            return_value=NOTIFICATION_SOURCE_ITEMS,
        ),
        patch(
            "benchmark.runs.notification_assignee.load_item_notification_row",
            AsyncMock(return_value=None),
        ),
        patch(
            "benchmark.runs.notification_assignee._load_trace_row",
            AsyncMock(),
        ) as load_trace,
    ):
        with pytest.raises(NotificationAssigneeError) as exc:
            await reassign_case_notification_assignee(
                object(),
                _run(),
                case_id="foreign-item",
                assignee="八鹤",
            )
    assert exc.value.status_code == 404
    load_trace.assert_not_awaited()


@pytest.mark.asyncio
async def test_reassign_rejects_trace_not_in_run() -> None:
    with (
        patch(
            "benchmark.runs.notification_assignee.resolve_notification_source",
            return_value=NOTIFICATION_SOURCE_TRACE,
        ),
        patch(
            "benchmark.runs.notification_assignee.trace_case_in_run",
            AsyncMock(return_value=False),
        ),
        patch(
            "benchmark.runs.notification_assignee._load_trace_row",
            AsyncMock(),
        ) as load_trace,
    ):
        with pytest.raises(NotificationAssigneeError) as exc:
            await reassign_case_notification_assignee(
                object(),
                _run(),
                case_id="foreign-trace",
                assignee="529504",
            )
    assert exc.value.status_code == 404
    load_trace.assert_not_awaited()


@pytest.mark.asyncio
async def test_reassign_rejects_missing_notification_url() -> None:
    row = SimpleNamespace(notification_dispatch_json={"notification_ref": "W-1"}, save=AsyncMock())
    with (
        patch(
            "benchmark.runs.notification_assignee.resolve_notification_source",
            return_value=NOTIFICATION_SOURCE_ITEMS,
        ),
        patch(
            "benchmark.runs.notification_assignee.load_item_notification_row",
            AsyncMock(return_value=row),
        ),
    ):
        with pytest.raises(NotificationAssigneeError) as exc:
            await reassign_case_notification_assignee(
                object(),
                _run(),
                case_id="case-1",
                assignee="八鹤",
            )
    assert exc.value.status_code == 400
    assert "aone url" in exc.value.message


@pytest.mark.asyncio
async def test_reassign_updates_notification_and_writes_bucket() -> None:
    row = SimpleNamespace(
        notification_dispatch_json={
            "notification_ref": "W-9",
            "notification_url": "https://aone.example/W-9",
            "notification_assigned_to": "青冬",
        },
        save=AsyncMock(),
    )
    update = AsyncMock()
    with (
        patch(
            "benchmark.runs.notification_assignee.resolve_notification_source",
            return_value=NOTIFICATION_SOURCE_ITEMS,
        ),
        patch(
            "benchmark.runs.notification_assignee.load_item_notification_row",
            AsyncMock(return_value=row),
        ),
        patch(
            "benchmark.runs.notification_assignee.load_agent_notification_context",
            AsyncMock(return_value=("chat-agent", None)),
        ),
        patch(
            "benchmark.runs.notification_assignee.resolve_effective_notification_targets",
            return_value=("2165583", "八鹤", [], True),
        ),
        patch(
            "benchmark.runs.notification_assignee.update_workitem_assignee",
            update,
        ),
    ):
        result = await reassign_case_notification_assignee(
            object(),
            _run(),
            case_id="case-1",
            assignee="路人甲",
        )

    assert result == {
        "notification_ref": "W-9",
        "notification_url": "https://aone.example/W-9",
        "notification_assigned_to": "路人甲",
    }
    assert row.notification_dispatch_json["notification_assigned_to"] == "路人甲"
    row.save.assert_awaited_once()
    update.assert_awaited_once()
    assert update.await_args.args[0] == "W-9"
    assert update.await_args.args[1] == "路人甲"
    notification_config = update.await_args.args[2]
    assert notification_config.project == "2165583"


@pytest.mark.asyncio
async def test_reassign_accepts_employee_id() -> None:
    row = SimpleNamespace(
        notification_dispatch_json={
            "notification_ref": "W-9",
            "notification_url": "https://aone.example/W-9",
        },
        save=AsyncMock(),
    )
    update = AsyncMock()
    with (
        patch(
            "benchmark.runs.notification_assignee.resolve_notification_source",
            return_value=NOTIFICATION_SOURCE_ITEMS,
        ),
        patch(
            "benchmark.runs.notification_assignee.load_item_notification_row",
            AsyncMock(return_value=row),
        ),
        patch(
            "benchmark.runs.notification_assignee.load_agent_notification_context",
            AsyncMock(return_value=("chat-agent", None)),
        ),
        patch(
            "benchmark.runs.notification_assignee.resolve_effective_notification_targets",
            return_value=("2165583", "八鹤", [], True),
        ),
        patch(
            "benchmark.runs.notification_assignee.update_workitem_assignee",
            update,
        ),
    ):
        result = await reassign_case_notification_assignee(
            object(),
            _run(),
            case_id="case-1",
            assignee="529504",
        )
    assert result["notification_assigned_to"] == "529504"
    update.assert_awaited_once()


@pytest.mark.asyncio
async def test_reassign_works_without_notification_project() -> None:
    """update does not depend on --project; the assignee can still be changed when project is not configured."""
    row = SimpleNamespace(
        notification_dispatch_json={
            "notification_ref": "W-9",
            "notification_url": "https://aone.example/W-9",
        },
        save=AsyncMock(),
    )
    update = AsyncMock()
    with (
        patch(
            "benchmark.runs.notification_assignee.resolve_notification_source",
            return_value=NOTIFICATION_SOURCE_ITEMS,
        ),
        patch(
            "benchmark.runs.notification_assignee.load_item_notification_row",
            AsyncMock(return_value=row),
        ),
        patch(
            "benchmark.runs.notification_assignee.load_agent_notification_context",
            AsyncMock(return_value=("chat-agent", None)),
        ),
        patch(
            "benchmark.runs.notification_assignee.resolve_effective_notification_targets",
            return_value=("", "", [], True),
        ),
        patch(
            "benchmark.runs.notification_assignee.update_workitem_assignee",
            update,
        ),
    ):
        result = await reassign_case_notification_assignee(
            object(),
            _run(),
            case_id="case-1",
            assignee="八鹤",
        )
    assert result["notification_assigned_to"] == "八鹤"
    update.assert_awaited_once()
