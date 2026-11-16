"""Parse triage config from the custom.triage_attribution section of conf/app.<env>.yaml.

Called by the app/main.py lifespan when constructing TriageAttributionService.
AppConfigManager allows this section via register_section("custom").
This loader reads it independently with yaml.safe_load and skips AppConfigManager schema checks.
"""

from pathlib import Path

import yaml

from benchmark.domain.notification_dispatch_config import resolve_notification_dispatch_profile
from benchmark.attribution.badcase.publisher import NotificationConfig
from benchmark.attribution.badcase.schemas import NotificationRuntimeOptions
from benchmark.attribution.service.triage import TriageConfig


def build_notification_config(raw: dict | None) -> NotificationConfig:
    """Merge the yaml dispatch policy with RemoteConfig runtime fields.

    yaml ``project`` is the RemoteConfig lookup key (for example ``agent-eval``), not the
    ``a1 --project`` argument. RemoteConfig supplies the former env fields such as
    ``notification_project_id`` and ``assignee``.
    """
    yaml_cfg = NotificationRuntimeOptions.model_validate(raw or {})
    dispatch_profile = yaml_cfg.project
    notification_project_id = ""
    assignee = yaml_cfg.assignee

    if dispatch_profile:
        remote_config_entry = resolve_notification_dispatch_profile(dispatch_profile)
        if remote_config_entry is not None:
            notification_project_id = remote_config_entry.notification_project_id
            if not assignee:
                assignee = remote_config_entry.assignee

    return NotificationConfig(
        enabled=yaml_cfg.enabled,
        dry_run=yaml_cfg.dry_run,
        dispatch_profile=dispatch_profile,
        project=notification_project_id,
        category=yaml_cfg.category,
        workitem_type=yaml_cfg.workitem_type,
        title_prefix=yaml_cfg.title_prefix,
        assignee=assignee,
        tags=yaml_cfg.tags,
        a1_bin=yaml_cfg.a1_bin,
        timeout_seconds=yaml_cfg.timeout_seconds,
    )


def load_triage_configs(
    yaml_path: Path,
) -> tuple[TriageConfig, NotificationConfig]:
    """Build config objects from the custom.triage_attribution section of conf/app.<env>.yaml.

    Args:
        yaml_path: Absolute path of conf/app.<env>.yaml.

    Returns:
        ``(TriageConfig, AoneConfig)``; all defaults when the custom section is missing.
    """
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    ta = (data.get("custom") or {}).get("triage_attribution") or {}
    triage_cfg = TriageConfig.model_validate(ta.get("triage") or {})
    notification_cfg = build_notification_config(ta.get("external"))
    return triage_cfg, notification_cfg


__all__ = ["build_notification_config", "load_triage_configs"]
