"""Per-agent Aone dispatch-target resolution.

Every agent reads RemoteConfig ``config.agent.eval.notification``:

1. Explicit ``remote_config_key_override`` (YAML ``profiles`` override)
2. RemoteConfig[agent name]
3. RemoteConfig[``agent-eval``]

project / assignee / participants / switches on the registration document **no longer
override** RemoteConfig. Do not look up AgentPark or any other external service.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.notification_dispatch_config import (
    NotificationDispatchProfile,
    fetch_notification_dispatch_profiles,
)
from benchmark.domain.benchmark_console import (
    find_agent_document_by_id,
    find_agent_document_by_name,
)

logger = get_logger(__name__)

#: Default RemoteConfig profile. Used when neither the business name nor the YAML key hits.
DEFAULT_NOTIFICATION_DISPATCH_PROFILE_KEY = "agent-eval"

_SNAPSHOT_NOTIFICATION_KEYS = frozenset(
    {
        "notification_dispatch_enabled",
        "notificationDispatchEnabled",
        "notification_project_id",
        "notificationProjectId",
        "notification_assignee",
        "notificationAssignee",
        "notification_participants",
        "notificationParticipants",
    }
)


class AgentNotificationFields(Protocol):
    """Dispatch fields on an agent document (or an equivalent snapshot)."""

    name: str
    notification_dispatch_enabled: bool
    notification_project_id: str | None
    notification_assignee: str | None
    notification_participants: Sequence[str]


@dataclass(frozen=True)
class AgentNotificationSnapshot:
    """Dispatch fields on a run snapshot or registration document (aligned with :class:`AgentAoneFields`)."""

    name: str = ""
    notification_dispatch_enabled: bool = True
    notification_project_id: str | None = None
    notification_assignee: str | None = None
    notification_participants: tuple[str, ...] = ()


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_parts(value: object) -> list[str]:
    raw: object = value
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip() for item in raw if str(item or "").strip()]


def _candidate_keys(agent_name: str, remote_config_key_override: str | None) -> list[str]:
    ordered: list[str] = []
    for raw in (remote_config_key_override, agent_name, DEFAULT_NOTIFICATION_DISPATCH_PROFILE_KEY):
        key = _clean_text(raw)
        if key and key not in ordered:
            ordered.append(key)
    return ordered


def _select_profile(
    agent_name: str,
    remote_config_key_override: str | None = None,
) -> tuple[str, NotificationDispatchProfile | None]:
    """Take the first existing RemoteConfig profile: override, then agent name, then ``agent-eval``."""
    profiles = fetch_notification_dispatch_profiles()
    keys = _candidate_keys(agent_name, remote_config_key_override)
    chosen = ""
    for key in keys:
        chosen = key
        entry = profiles.get(key)
        if entry is not None:
            return key, entry
    if chosen:
        return chosen, None
    return DEFAULT_NOTIFICATION_DISPATCH_PROFILE_KEY, None


def matched_notification_profile_key(
    agent_name: str,
    remote_config_key_override: str | None = None,
) -> str:
    """Profile key RemoteConfig actually hit; the last key tried when none hit."""
    key, _entry = _select_profile(agent_name, remote_config_key_override)
    return key


def overlay_agent_notification_fields(
    project: str,
    assignee: str,
    participants: Sequence[str],
    agent_document: AgentNotificationFields | None,
) -> tuple[str, str, list[str]]:
    """Deprecated: the registration document no longer overrides RemoteConfig dispatch targets.

    .. deprecated::
        Kept after the RemoteConfig-only change so old imports still type-check. Always
        returns copies of the first three arguments; ``agent_document`` is ignored.
        Use :func:`resolve_effective_aone_targets` instead.
    """
    del agent_document
    return project, assignee, list(participants)


def resolve_effective_notification_targets(
    agent_name: str,
    agent_document: AgentNotificationFields | None = None,
    *,
    remote_config_key_override: str | None = None,
) -> tuple[str, str, list[str], bool]:
    """Resolve the final dispatch targets (RemoteConfig only).

    Returns:
        ``(project_id, assignee, participants, enabled)``.
        ``enabled`` is always True (the registration page no longer exposes a push switch).
        ``agent_document`` is kept for old callers and **does not** take part in the result.
        Callers may still overlay the first three fields with non-empty per-run ``aone_dispatch``.
    """
    del agent_document  # Registration-document aone fields are deprecated; RemoteConfig is the source.
    _key, entry = _select_profile(agent_name, remote_config_key_override)
    remote_config_project = entry.notification_project_id if entry is not None else ""
    remote_config_assignee = entry.assignee if entry is not None else ""
    remote_config_parts = list(entry.participants) if entry is not None else []
    return remote_config_project, remote_config_assignee, remote_config_parts, True


def resolve_notification_root_cause_groups(
    agent_name: str,
    agent_document: AgentNotificationFields | None = None,
    *,
    remote_config_key_override: str | None = None,
) -> dict[str, tuple[str, ...]]:
    """Read the root-cause audiences on the current profile. Empty map when unset."""

    del agent_document
    _key, entry = _select_profile(agent_name, remote_config_key_override)
    if entry is None:
        return {}
    return dict(entry.root_cause_groups)


def _flag(raw: Mapping[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        if key in raw:
            return bool(raw[key])
    return default


def _optional_text(raw: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key not in raw:
            continue
        text = _clean_text(raw.get(key))
        return text or None
    return None


def snapshot_from_mapping(raw: Mapping[str, Any]) -> AgentNotificationSnapshot:
    """run ``agent_registration`` snapshot (snake / camel) → dispatch view."""
    participants = raw.get(
        "notification_participants",
        raw.get("notificationParticipants", ()),
    )
    return AgentNotificationSnapshot(
        name=_clean_text(raw.get("name")),
        notification_dispatch_enabled=_flag(
            raw,
            "notification_dispatch_enabled",
            "notificationDispatchEnabled",
            default=True,
        ),
        notification_project_id=_optional_text(
            raw,
            "notification_project_id",
            "notificationProjectId",
        ),
        notification_assignee=_optional_text(
            raw,
            "notification_assignee",
            "notificationAssignee",
        ),
        notification_participants=tuple(_clean_parts(participants)),
    )


def _registration_id(config: Mapping[str, Any], snap: Mapping[str, Any] | None) -> str:
    if snap is not None:
        reg_id = _clean_text(snap.get("id"))
        if reg_id:
            return reg_id
    return _clean_text(config.get("agent_registration_id") or config.get("agentRegistrationId"))


def _snapshot_from_document(document: AgentNotificationFields) -> AgentNotificationSnapshot:
    return AgentNotificationSnapshot(
        name=document.name,
        notification_dispatch_enabled=document.notification_dispatch_enabled,
        notification_project_id=document.notification_project_id,
        notification_assignee=document.notification_assignee,
        notification_participants=tuple(document.notification_participants),
    )


async def _load_document_snapshot(registration_id: str) -> AgentNotificationSnapshot | None:
    """Load AgentDocument by registration id. None on failure (does not block dispatch)."""
    try:
        document = await find_agent_document_by_id(registration_id)
    except Exception:
        logger.warning(
            "aone_dispatch.agent_document_load_failed id=%s",
            registration_id,
            exc_info=True,
        )
        return None
    if document is None:
        return None
    return _snapshot_from_document(document)


async def _load_document_by_name(agent_name: str) -> AgentNotificationSnapshot | None:
    """Look up the eval registry by business name. None on miss or read failure (continue RemoteConfig)."""
    name = _clean_text(agent_name)
    if not name:
        return None
    try:
        document = await find_agent_document_by_name(name)
    except Exception:
        logger.warning(
            "aone_dispatch.agent_document_name_lookup_failed name=%s",
            name,
            exc_info=True,
        )
        return None
    if document is None:
        return None
    return _snapshot_from_document(document)


async def load_agent_notification_context(
    config: Mapping[str, Any] | None,
) -> tuple[str, AgentNotificationSnapshot | None]:
    """Take the dispatch view from the run config.

    When the snapshot already has aone fields, use the snapshot (a running job does not
    follow later document edits). When only ``agent_registration_id`` is present, load
    AgentDocument by id. Otherwise look up the eval registry by agent name. If none of
    those hit, return ``(name, None)`` and leave it to RemoteConfig. Do not look up AgentPark.
    """
    if not isinstance(config, Mapping):
        return "", None
    snap_raw = config.get("agent_registration")
    snap = snap_raw if isinstance(snap_raw, Mapping) else None
    if snap is not None and any(key in snap for key in _SNAPSHOT_NOTIFICATION_KEYS):
        view = snapshot_from_mapping(snap)
        return view.name, view
    reg_id = _registration_id(config, snap)
    if reg_id:
        loaded = await _load_document_snapshot(reg_id)
        if loaded is not None:
            return loaded.name, loaded
    agent_name = _clean_text(snap.get("name")) if snap is not None else ""
    if not agent_name:
        agent_name = _clean_text(config.get("agent_name") or config.get("agentName"))
    if not agent_name:
        return "", None
    registered = await _load_document_by_name(agent_name)
    if registered is not None:
        return registered.name, registered
    return agent_name, None


__all__ = [
    "DEFAULT_NOTIFICATION_DISPATCH_PROFILE_KEY",
    "AgentNotificationFields",
    "AgentNotificationSnapshot",
    "load_agent_notification_context",
    "matched_notification_profile_key",
    "overlay_agent_notification_fields",
    "resolve_effective_notification_targets",
    "snapshot_from_mapping",
]
