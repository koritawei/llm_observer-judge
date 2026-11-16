"""Aone fields on the agent document, and fallback of the dispatch target."""

from unittest.mock import AsyncMock, patch

import pytest
from benchmark.domain.notification_dispatch_config import NotificationDispatchProfile
from benchmark.domain.notification_dispatch_resolve import (
    AgentNotificationSnapshot,
    load_agent_notification_context,
    overlay_agent_notification_fields,
    resolve_effective_notification_targets,
)
from benchmark.domain.benchmark_console import AgentDocument, registration_snapshot
from pydantic import ValidationError


def _agent(**overrides: object) -> AgentDocument:
    payload: dict[str, object] = {
        "id": "agent-1",
        "name": "chat-agent",
        "owner": "owner",
        "registered_at": "2026-01-01 00:00:00",
    }
    payload.update(overrides)
    return AgentDocument.model_validate(payload)


def test_agent_document_defaults_notification_enabled() -> None:
    document = _agent()
    assert document.notification_dispatch_enabled is True
    assert document.notification_project_id is None
    assert document.notification_assignee is None
    assert document.notification_participants == []
    dumped = document.model_dump(by_alias=True)
    assert dumped["notificationDispatchEnabled"] is True
    snapshot = registration_snapshot(document)
    assert snapshot["notification_dispatch_enabled"] is True
    assert snapshot["notification_participants"] == []


def test_agent_document_accepts_numeric_emp_ids() -> None:
    document = _agent(
        notificationAssignee="529504",
        notificationParticipants=["407544", " 888 ", ""],
        notificationProjectId="2165583",
        notificationDispatchEnabled=False,
    )
    assert document.notification_dispatch_enabled is False
    assert document.notification_assignee == "529504"
    assert document.notification_participants == ["407544", "888"]
    assert document.notification_project_id == "2165583"


def test_agent_document_accepts_nickname_assignee_and_owners() -> None:
    document = _agent(
        notificationAssignee="八鹤",
        notificationParticipants=["青冬", "529504"],
        owners=["八鹤", "青冬"],
    )
    assert document.notification_assignee == "八鹤"
    assert document.notification_participants == ["青冬", "529504"]
    assert document.owners == ["八鹤", "青冬"]
    assert document.owner == "八鹤"


def test_agent_document_legacy_owner_backfills_owners_without_person_regex() -> None:
    """A legacy owner may be an email or a team name containing spaces; the read path must not raise ValidationError."""
    document = _agent(owner="eval-team@alibaba-inc.com")
    assert document.owner == "eval-team@alibaba-inc.com"
    assert document.owners == ["eval-team@alibaba-inc.com"]
    spaced = _agent(owner="C4 Team")
    assert spaced.owners == ["C4 Team"]
    assert spaced.owner == "C4 Team"


def test_agent_document_rejects_invalid_owners_entry() -> None:
    with pytest.raises(ValidationError, match="owners 只能包含花名或数字工号"):
        _agent(owners=["bad name!"])


def test_agent_document_rejects_invalid_assignee() -> None:
    with pytest.raises(ValidationError, match="花名或数字工号"):
        _agent(notification_assignee="bad name!")


def test_agent_document_rejects_invalid_participant() -> None:
    with pytest.raises(ValidationError, match="花名或数字工号"):
        _agent(notification_participants=["529504", "bad name!"])


def test_empty_assignee_falls_back_to_none() -> None:
    document = _agent(notification_assignee="  ")
    assert document.notification_assignee is None


def test_resolve_uses_remote_config_agent_then_default_ignoring_document() -> None:
    profiles = {
        "chat-agent": NotificationDispatchProfile(
            notification_project_id="111",
            assignee="222",
            participants=("333",),
        ),
        "agent-eval": NotificationDispatchProfile(
            notification_project_id="2165583",
            assignee="529504",
            participants=("407544",),
        ),
    }
    with patch(
        "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
        return_value=profiles,
    ):
        project, assignee, participants, enabled = resolve_effective_notification_targets(
            "chat-agent",
            AgentNotificationSnapshot(name="chat-agent", notification_assignee="999"),
        )
        fallback_project, fallback_assignee, _, fallback_enabled = resolve_effective_notification_targets(
            "unknown-agent",
            None,
        )
    # The document notification_assignee no longer overrides RemoteConfig
    assert (project, assignee, participants, enabled) == ("111", "222", ["333"], True)
    assert (fallback_project, fallback_assignee, fallback_enabled) == (
        "2165583",
        "529504",
        True,
    )


def test_overlay_agent_notification_fields_is_noop_deprecated() -> None:
    """Deprecated API: document fields must no longer override RemoteConfig defaults."""
    project, assignee, parts = overlay_agent_notification_fields(
        "111",
        "222",
        ["333"],
        AgentNotificationSnapshot(
            notification_project_id="999",
            notification_assignee="888",
            notification_participants=["777"],
        ),
    )
    assert (project, assignee, parts) == ("111", "222", ["333"])


def test_resolve_ignores_document_disable_and_project_override() -> None:
    profiles = {
        "agent-eval": NotificationDispatchProfile(notification_project_id="2165583", assignee="529504"),
    }
    with patch(
        "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
        return_value=profiles,
    ):
        project, assignee, _parts, enabled = resolve_effective_notification_targets(
            "missing",
            AgentNotificationSnapshot(notification_dispatch_enabled=False, notification_project_id="42"),
        )
    assert enabled is True
    assert project == "2165583"
    assert assignee == "529504"


@pytest.mark.asyncio
async def test_load_context_uses_registered_agent_table_by_name() -> None:
    """When the snapshot is missing aone fields, look up the evaluation registry by name, and do not reverse-look-up AgentPark."""
    document = _agent(
        notificationAssignee="10086",
        notificationProjectId="9001",
        notificationDispatchEnabled=True,
    )
    with patch(
        "benchmark.domain.notification_dispatch_resolve.find_agent_document_by_name",
        new=AsyncMock(return_value=document),
    ) as find_by_name:
        name, snapshot = await load_agent_notification_context({"agent_name": "chat-agent"})
    find_by_name.assert_awaited_once_with("chat-agent")
    assert name == "chat-agent"
    assert snapshot is not None
    assert snapshot.notification_assignee == "10086"
    assert snapshot.notification_project_id == "9001"


@pytest.mark.asyncio
async def test_load_context_name_miss_falls_through_to_remote_config() -> None:
    with patch(
        "benchmark.domain.notification_dispatch_resolve.find_agent_document_by_name",
        new=AsyncMock(return_value=None),
    ):
        name, snapshot = await load_agent_notification_context({"agentName": "unknown-agent"})
    assert name == "unknown-agent"
    assert snapshot is None
