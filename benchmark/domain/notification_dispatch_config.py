"""Aone dispatch RemoteConfig — looked up by the yaml ``project`` logical name.

yaml ``custom.triage_attribution.aone.project`` (for example ``agent-eval``) is the config
key. RemoteConfig supplies the runtime fields that used to live in env vars
(``notification_project_id``, ``assignee``, ``participants``).

RemoteConfig JSON shape::

    {
        "agent-eval": {
            "notification_project_id": "2165583",
            "assignee": "529504",
            "participants": ["bahe", "407544", "bob@alibaba-inc.com"],
            "root_cause_groups": {
                "tool_error": ["bahe", "407544"],
                "knowledge_gap": ["bob@alibaba-inc.com"]
            }
        }
    }

``participants`` is the CC list (flower name / email / employee id). When a work item is
created they are joined with commas for a1 ``--participant`` (Aone has no separate CC
field; "participant" is the CC role). A comma-separated string (for example
``"bahe,407544"``) is also accepted. The default is empty (no CC).

RemoteConfig is isolated per deploy environment. The JSON top level no longer nests
daily/pre/prod.
"""

from dataclasses import dataclass
from typing import Any, Self

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.domain._cache import get_or_compute

logger = get_logger(__name__)

#: RemoteConfig data_id for the Aone dispatch config.
NOTIFICATION_DISPATCH_DATA_ID = "config.agent.eval.notification"
#: RemoteConfig group for the Aone dispatch config.
NOTIFICATION_DISPATCH_GROUP = "group.agent_eval"


@dataclass(frozen=True)
class NotificationDispatchProfile:
    """One dispatch RemoteConfig profile (moved in from the former env vars).

    ``participants`` is the CC tuple (flower name / email / employee id). Missing means
    an empty tuple (no CC). Dirty types and blank elements are dropped at parse time and
    do not invalidate the whole profile.
    """

    notification_project_id: str
    assignee: str = ""
    participants: tuple[str, ...] = ()
    #: Root-cause code → audience. Key ``*`` is the default audience for unlisted causes.
    root_cause_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        project_id = raw.get("notification_project_id") or raw.get("project_id") or ""
        return cls(
            notification_project_id=str(project_id),
            assignee=str(raw.get("assignee") or ""),
            participants=_parse_participants(raw.get("participants")),
            root_cause_groups=_parse_root_cause_groups(raw.get("root_cause_groups")),
        )


def _parse_participants(raw: object) -> tuple[str, ...]:
    """Parse the CC config: list[str] is primary; a comma-separated string is tolerated.

    Tolerate rather than raise: a dirty participants value must not invalidate the whole
    profile (``notification_project_id`` / ``assignee`` stay valid). Only unparseable
    items are dropped. Blank elements are skipped.
    """
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list):
        return ()
    return tuple(str(item).strip() for item in raw if str(item or "").strip())


def _parse_root_cause_groups(raw: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Parse the root-cause to audience map. Illegal entries are dropped, not the profile."""

    if not isinstance(raw, dict):
        return ()
    groups: list[tuple[str, tuple[str, ...]]] = []
    for key, value in raw.items():
        name = str(key).strip()
        members = _parse_participants(value)
        if not name or not members:
            continue
        groups.append((name, members))
    return tuple(groups)


def fetch_notification_dispatch_profiles(
    *,
    data_id: str = NOTIFICATION_DISPATCH_DATA_ID,
    group: str = NOTIFICATION_DISPATCH_GROUP,
    use_cache: bool = True,
) -> dict[str, NotificationDispatchProfile]:
    """Fetch every dispatch profile, keyed by the yaml ``project`` logical name.

    Results use the parsed-result cache by default (TTL 300s, see :mod:`benchmark.domain._cache`).
    Pass ``use_cache=False`` to force a refresh.
    """

    def _parse() -> dict[str, NotificationDispatchProfile]:
        settings = fetch_cached_settings(data_id, group, use_cache=use_cache)
        profiles: dict[str, NotificationDispatchProfile] = {}
        for key, value in settings.items():
            if not isinstance(key, str) or not key.strip():
                continue
            if not isinstance(value, dict):
                logger.warning(
                    "RemoteConfig aone dispatch %r 非 dict，跳过: %s",
                    key,
                    type(value).__name__,
                )
                continue
            try:
                profiles[key] = NotificationDispatchProfile.from_dict(value)
            except (TypeError, ValueError) as exc:
                logger.warning("RemoteConfig aone dispatch %r 解析失败: %s", key, exc)
        return profiles

    return get_or_compute((data_id, group), _parse, use_cache=use_cache)


def resolve_notification_dispatch_profile(
    dispatch_profile: str,
    *,
    data_id: str = NOTIFICATION_DISPATCH_DATA_ID,
    group: str = NOTIFICATION_DISPATCH_GROUP,
    use_cache: bool = True,
) -> NotificationDispatchProfile | None:
    """Resolve the RemoteConfig dispatch profile by the yaml ``project`` logical name."""
    if not dispatch_profile:
        return None
    profiles = fetch_notification_dispatch_profiles(data_id=data_id, group=group, use_cache=use_cache)
    entry = profiles.get(dispatch_profile)
    if entry is None:
        logger.warning(
            "RemoteConfig aone dispatch profile %r not found in %s/%s",
            dispatch_profile,
            data_id,
            group,
        )
    return entry


__all__ = [
    "NOTIFICATION_DISPATCH_DATA_ID",
    "NOTIFICATION_DISPATCH_GROUP",
    "NotificationDispatchProfile",
    "fetch_notification_dispatch_profiles",
    "resolve_notification_dispatch_profile",
]
