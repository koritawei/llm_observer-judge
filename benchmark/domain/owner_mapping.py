"""Owner-route mapping domain model — loaded from RemoteConfig, no DB storage.

Domain layer. Contains:
  - :class:`OwnerMapping` — domain model (dataclass, not an ORM)
  - :func:`fetch_owner_mappings` — domain service that loads config from RemoteConfig
  - :func:`resolve_owner_route` — domain service for the three-level route match

The RemoteConfig client singleton and cache live in :mod:`benchmark.settings.remote_config_store`.

RemoteConfig JSON shape::

    {
        "mappings": [
            {
                "mapping_id": "m1",
                "match_type": "agent_cause",
                "match_key": "vehicle_agent/tool_error",
                "assigned_to": ["alice", "bob"],
                "assign_type": "random",
                "project_url": "https://example.com",
                "priority": "P0",
                "owner_group": "platform-team",
                "enabled": true
            }
        ]
    }

Usage::

    from benchmark.domain.owner_mapping import (
        OwnerMapping,
        fetch_owner_mappings,
        resolve_owner_route,
    )
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.constant import CACHE_TTL_SEC
from benchmark.domain._cache import FetchListSpec, fetch_and_parse_list
from benchmark.domain.enums import CasePriority, MatchType

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# RemoteConfig config constants.
# ---------------------------------------------------------------------------

#: RemoteConfig data_id for the owner-mapping config.
OWNER_MAPPING_DATA_ID = "config.owner.mapping"
#: RemoteConfig group for the owner-mapping config.
OWNER_MAPPING_GROUP = "group.owner_mapping"

#: Sort key for priority; None is treated as the lowest priority.
_PRIORITY_SORT_KEY: dict[CasePriority | None, int] = {
    CasePriority.P0: 0,
    CasePriority.P1: 1,
    CasePriority.P2: 2,
    None: 3,
}

# ---------------------------------------------------------------------------
# OwnerMapping domain model.
# ---------------------------------------------------------------------------


@dataclass
class OwnerMapping:
    """One owner-route mapping entry (loaded from RemoteConfig, no DB storage).

    Defines how a bad case is routed to a person or team, with three match strategies:
      1. ``AGENT_CAUSE`` — exact match on agent_name/primary_cause (highest priority)
      2. ``OWNER_GROUP`` — match on the owner_group team (middle priority)
      3. ``DEFAULT`` — fallback route (lowest priority)

    :func:`resolve_owner_route` searches in that order.
    ``priority`` (P0/P1/P2) orders entries that share a match_type.
    ``enabled`` turns a mapping on or off without deleting it.
    """

    mapping_id: str
    match_type: MatchType = MatchType.AGENT_CAUSE
    match_key: str = ""
    assigned_to: str = ""
    #: Multi-person assignment pool. A single-person rule leaves this empty and still reads ``assigned_to``.
    assignees: tuple[str, ...] = ()
    #: ``random`` picks one person from the pool; ``all`` assigns the first and CCs the rest. Default ``random``.
    assign_type: str = "random"
    project_url: str | None = None
    priority: CasePriority | None = None
    owner_group: str | None = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        """Build an instance from a RemoteConfig dict, tolerating enums and optional fields.

        ``match_type`` / ``priority`` accept strings and convert them to enums.
        Missing fields use defaults. An illegal enum value raises ``ValueError``.
        """
        match_type_raw = raw.get("match_type", MatchType.AGENT_CAUSE.value)
        priority_raw = raw.get("priority")
        assigned_to, assignees = _parse_assigned_to(raw.get("assigned_to", ""))
        return cls(
            mapping_id=str(raw.get("mapping_id", "")),
            match_type=(MatchType(match_type_raw) if match_type_raw else MatchType.AGENT_CAUSE),
            match_key=str(raw.get("match_key", "")),
            assigned_to=assigned_to,
            assignees=assignees,
            assign_type=_parse_assign_type(raw.get("assign_type")),
            project_url=raw.get("project_url"),
            priority=CasePriority(priority_raw) if priority_raw else None,
            owner_group=raw.get("owner_group"),
            enabled=bool(raw.get("enabled", True)),
        )


# ---------------------------------------------------------------------------
# RemoteConfig load + route resolution.
# ---------------------------------------------------------------------------


def fetch_cached_settings(
    data_id: str,
    group: str,
    *,
    use_cache: bool = True,
    ttl: int = CACHE_TTL_SEC,
) -> dict[str, Any]:
    """Fetch RemoteConfig on demand so config and the triage service do not import each other in a cycle."""
    # Local import: initializing benchmark.settings imports the triage service, which depends on this module.
    from benchmark.settings.remote_config_store import fetch_cached_settings as fetcher  # noqa: PLC0415

    return fetcher(data_id, group, use_cache=use_cache, ttl=ttl)


def fetch_owner_mappings(
    *,
    data_id: str = OWNER_MAPPING_DATA_ID,
    group: str = OWNER_MAPPING_GROUP,
    use_cache: bool = True,
) -> list[OwnerMapping]:
    """Load the owner-route mapping list from RemoteConfig.

    Results use the parsed-result cache by default (TTL 300s, see :mod:`benchmark.domain._cache`).
    Pass ``use_cache=False`` to force a refresh.

    Returns an empty list, and does not raise, when RemoteConfig is unavailable or empty.
    """
    return fetch_and_parse_list(
        data_id,
        group,
        FetchListSpec(
            list_key="mappings",
            parse_item=OwnerMapping.from_dict,
            fetcher=fetch_cached_settings,
            use_cache=use_cache,
        ),
    )


def resolve_owner_route(
    *,
    agent_name: str,
    primary_cause: str,
    owner_group: str | None = None,
    data_id: str = OWNER_MAPPING_DATA_ID,
    group: str = OWNER_MAPPING_GROUP,
    use_cache: bool = True,
) -> OwnerMapping | None:
    """Three-level owner route: exact agent_cause → wildcard agent_cause → owner_group → default.

    Loads every enabled mapping from RemoteConfig and matches in memory. Returns the first
    hit, or ``None`` when nothing matches.
    """
    mappings = fetch_owner_mappings(data_id=data_id, group=group, use_cache=use_cache)
    enabled = sorted(
        [m for m in mappings if m.enabled],
        key=lambda m: _PRIORITY_SORT_KEY.get(m.priority, 3),
    )

    candidates: list[tuple[MatchType, str]] = [
        (MatchType.AGENT_CAUSE, f"{agent_name}/{primary_cause}"),
        (MatchType.AGENT_CAUSE, f"{agent_name}/*"),
    ]
    if owner_group:
        candidates.append((MatchType.OWNER_GROUP, owner_group))
    candidates.append((MatchType.DEFAULT, "*"))

    for match_type, match_key in candidates:
        for m in enabled:
            if m.match_type == match_type and m.match_key == match_key:
                return m
    return None


def _parse_assign_type(raw: object) -> str:
    """``all`` puts everyone on one ticket; every other value (including the default) is ``random``."""
    text = str(raw or "").strip().lower()
    if text == "all":
        return "all"
    if text and text != "random":
        logger.warning("owner mapping assign_type %r invalid, fallback random", text)
    return "random"


def _parse_assigned_to(raw: object) -> tuple[str, tuple[str, ...]]:
    """Parse assignment config. A string or number is one person; a deduped array of more than one enters the pool."""
    if isinstance(raw, str):
        text = raw.strip()
        return text, ()
    if isinstance(raw, int) and not isinstance(raw, bool):
        return str(raw), ()
    if not isinstance(raw, list):
        return "", ()
    people: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and text not in people:
            people.append(text)
    if len(people) <= 1:
        return (people[0] if people else ""), ()
    return "", tuple(people)


def _route_pool(route: OwnerMapping) -> tuple[str, ...]:
    if route.assignees:
        return route.assignees
    text = route.assigned_to.strip()
    return (text,) if text else ()


def _without_one(people: Sequence[str], chosen: str) -> list[str]:
    skipped = False
    rest: list[str] = []
    for person in people:
        if not skipped and person == chosen:
            skipped = True
            continue
        rest.append(person)
    return rest


def resolve_route_assignment(
    route: OwnerMapping | None,
    *,
    fallback_assignee: str = "",
    fallback_participants: Sequence[str] = (),
    pick: Callable[[Sequence[str]], str] | None = None,
) -> tuple[str, list[str]]:
    """Resolve the assignee and participants for this rule.

    A single-person rule overwrites the fallback assignee with that person. When several
    people are set and ``assign_type=random``, only one is drawn and the others are not
    added as participants. ``assign_type=all`` assigns the first person and appends the
    rest as participants. Both multi-person modes remove the assignee from the participants.
    """
    participants = [str(item).strip() for item in fallback_participants if str(item or "").strip()]
    pool = _route_pool(route) if route is not None else ()
    if len(pool) <= 1:
        assignee = pool[0] if pool else fallback_assignee
        return assignee, participants
    if route is not None and route.assign_type == "all":
        chosen = pool[0]
        merged = [person for person in participants if person != chosen]
        for person in _without_one(pool, chosen):
            if person not in merged:
                merged.append(person)
        return chosen, merged
    chosen = (pick or random.choice)(pool)
    return chosen, [person for person in participants if person != chosen]


__all__ = [
    "OWNER_MAPPING_DATA_ID",
    "OWNER_MAPPING_GROUP",
    "OwnerMapping",
    "fetch_owner_mappings",
    "resolve_owner_route",
    "resolve_route_assignment",
]
