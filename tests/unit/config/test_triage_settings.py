"""Settings parses the custom.triage_attribution section of conf/app.<env>.yaml."""

from pathlib import Path
from unittest.mock import _patch, patch

import pytest
from benchmark.settings.triage_loader import build_notification_config, load_triage_configs
from benchmark.domain.notification_dispatch_config import NotificationDispatchProfile


def _mock_remote_config_profile() -> _patch:
    return patch(
        "benchmark.settings.triage_loader.resolve_notification_dispatch_profile",
        return_value=NotificationDispatchProfile(
            notification_project_id="2165583",
            assignee="529504",
        ),
    )


def test_load_triage_configs_from_daily_yaml() -> None:
    service_root = Path(__file__).resolve().parents[3]
    yaml_path = service_root / "conf" / "app.daily.yaml"
    with _mock_remote_config_profile():
        triage_config, notification_config = load_triage_configs(yaml_path)

    assert triage_config.badcase_score_threshold == 0.5
    assert notification_config.enabled is False  # daily turns realtime ticket dispatch off by default
    assert notification_config.dispatch_profile == "agent-eval"
    assert notification_config.project == "2165583"
    assert notification_config.assignee == "529504"
    assert notification_config.category == "bug"


@pytest.mark.parametrize(
    "env,notification_enabled,notification_dry_run",
    [
        ("daily", False, False),
        ("pre", False, False),
        ("prod", True, False),
    ],
)
def test_load_triage_configs_across_envs(
    env: str,
    notification_enabled: bool,
    notification_dry_run: bool,
) -> None:
    """All three environments parse; daily/pre turn realtime ticket dispatch off by default, and prod turns it on.

    Check custom.triage_attribution in conf/app.{daily,pre,prod}.yaml so they do not drift.
    """

    service_root = Path(__file__).resolve().parents[3]
    yaml_path = service_root / "conf" / f"app.{env}.yaml"
    with _mock_remote_config_profile():
        triage_config, notification_config = load_triage_configs(yaml_path)

    # triage fields are the same in all three environments; aone.enabled differs by environment
    assert triage_config.badcase_score_threshold == 0.5
    assert triage_config.policy_match_strict is True
    assert triage_config.label_studio_dispatcher_enabled is False
    assert notification_config.enabled is notification_enabled
    assert notification_config.a1_bin == "a1"
    assert notification_config.dispatch_profile == "agent-eval"
    assert notification_config.project == "2165583"
    assert notification_config.category == "bug"
    # An empty workitem_type uses the project's default subtype (avoid a misconfigured --type=bug)
    assert notification_config.workitem_type == ""
    assert notification_config.title_prefix == "[badcase]"
    assert notification_config.timeout_seconds == 30
    # dry_run is env-specific
    assert notification_config.dry_run is notification_dry_run


def test_build_notification_config_yaml_assignee_overrides_remote_config() -> None:
    with patch(
        "benchmark.settings.triage_loader.resolve_notification_dispatch_profile",
        return_value=NotificationDispatchProfile(notification_project_id="2165583", assignee="remote_config-user"),
    ):
        cfg = build_notification_config(
            {
                "project": "agent-eval",
                "assignee": "yaml-user",
                "category": "bug",
            }
        )

    assert cfg.project == "2165583"
    assert cfg.assignee == "yaml-user"


def test_build_notification_config_without_remote_config_leaves_project_empty() -> None:
    with patch(
        "benchmark.settings.triage_loader.resolve_notification_dispatch_profile",
        return_value=None,
    ):
        cfg = build_notification_config({"project": "agent-eval", "category": "bug"})

    assert cfg.dispatch_profile == "agent-eval"
    assert cfg.project == ""
