"""aone_dispatch_config domain service — parsing RemoteConfig ticket-dispatch profiles."""

from unittest.mock import patch

from benchmark.domain.notification_dispatch_config import (
    NOTIFICATION_DISPATCH_DATA_ID,
    NOTIFICATION_DISPATCH_GROUP,
    NotificationDispatchProfile,
    fetch_notification_dispatch_profiles,
    resolve_notification_dispatch_profile,
)


def test_notification_dispatch_profile_from_dict() -> None:
    profile = NotificationDispatchProfile.from_dict(
        {
            "notification_project_id": "2165583",
            "assignee": "529504",
            "project_url": "https://example.com/p/2165583",
        }
    )
    assert profile.notification_project_id == "2165583"
    assert profile.assignee == "529504"


def test_notification_dispatch_profile_ignores_legacy_project_url_field() -> None:
    """The project_url field was removed from the schema; a legacy yaml that still sets it is ignored silently and does not raise."""
    profile = NotificationDispatchProfile.from_dict(
        {
            "notification_project_id": "2165583",
            "assignee": "529504",
            "project_url": "https://example.com/p/2165583",
        }
    )
    assert profile.notification_project_id == "2165583"
    assert profile.assignee == "529504"
    # project_url is no longer a schema field
    assert "project_url" not in profile.__dataclass_fields__


def test_notification_dispatch_profile_from_dict_accepts_project_id_alias() -> None:
    profile = NotificationDispatchProfile.from_dict({"project_id": "999"})
    assert profile.notification_project_id == "999"


def test_notification_dispatch_profile_parses_participants_list() -> None:
    """The primary participants format: list[str] (nickname / employee id / email)."""
    profile = NotificationDispatchProfile.from_dict(
        {
            "notification_project_id": "2165583",
            "assignee": "529504",
            "participants": ["八鹤", "407544", "bob@alibaba-inc.com"],
        },
    )
    assert profile.participants == ("八鹤", "407544", "bob@alibaba-inc.com")


def test_notification_dispatch_profile_parses_participants_csv_string() -> None:
    """A comma-separated string is tolerated (handy for hand-written config); blank elements are skipped."""
    profile = NotificationDispatchProfile.from_dict(
        {
            "notification_project_id": "2165583",
            "participants": "八鹤, 407544 ,",
        },
    )
    assert profile.participants == ("八鹤", "407544")


def test_notification_dispatch_profile_dirty_participants_falls_back_empty() -> None:
    """A dirty type or a missing value → an empty tuple, and the whole profile is not discarded."""
    profile = NotificationDispatchProfile.from_dict(
        {
            "notification_project_id": "2165583",
            "participants": 123,
        },
    )
    assert profile.participants == ()
    assert profile.notification_project_id == "2165583"

    missing = NotificationDispatchProfile.from_dict({"notification_project_id": "2165583"})
    assert missing.participants == ()
    assert missing.root_cause_groups == ()


def test_notification_dispatch_profile_parses_root_cause_groups() -> None:
    """Root-cause codes map to a group of people; an empty group and dirty values are dropped."""
    profile = NotificationDispatchProfile.from_dict(
        {
            "notification_project_id": "2165583",
            "root_cause_groups": {
                "tool_error": ["alice", " bob "],
                "knowledge_gap": "carol, dave",
                "empty": [],
                "dirty": 1,
            },
        },
    )
    assert dict(profile.root_cause_groups) == {
        "tool_error": ("alice", "bob"),
        "knowledge_gap": ("carol", "dave"),
    }


def test_fetch_notification_dispatch_profiles_parses_remote_config_payload() -> None:
    raw = {
        "agent-eval": {
            "notification_project_id": "2165583",
            "assignee": "529504",
        },
        "bad-key": "not-a-dict",
    }
    with patch(
        "benchmark.domain.notification_dispatch_config.fetch_cached_settings",
        return_value=raw,
    ):
        profiles = fetch_notification_dispatch_profiles(
            data_id="test.id",
            group="test.group",
            use_cache=False,
        )

    assert set(profiles) == {"agent-eval"}
    assert profiles["agent-eval"].notification_project_id == "2165583"


def test_resolve_notification_dispatch_profile_returns_entry() -> None:
    raw = {"agent-eval": {"notification_project_id": "2165583", "assignee": "529504"}}
    with patch(
        "benchmark.domain.notification_dispatch_config.fetch_cached_settings",
        return_value=raw,
    ):
        entry = resolve_notification_dispatch_profile("agent-eval")

    assert entry is not None
    assert entry.notification_project_id == "2165583"
    assert entry.assignee == "529504"


def test_resolve_notification_dispatch_profile_missing_returns_none() -> None:
    with patch(
        "benchmark.domain.notification_dispatch_config.fetch_cached_settings",
        return_value={},
    ):
        assert resolve_notification_dispatch_profile("missing") is None


def test_resolve_notification_dispatch_profile_empty_key_returns_none() -> None:
    assert resolve_notification_dispatch_profile("") is None


def test_remote_config_constants() -> None:
    assert NOTIFICATION_DISPATCH_DATA_ID == "config.agent.eval.notification"
    assert NOTIFICATION_DISPATCH_GROUP == "group.agent_eval"
