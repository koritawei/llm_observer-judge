"""Agent metadata config — loaded from RemoteConfig, no DB storage.

Domain layer. Contains:
  - :class:`AgentConfig` — domain model (dataclass, not an ORM)
  - :func:`fetch_agent_configs` — domain service that loads config from RemoteConfig
  - :func:`get_agent_config` — look up one config by agent_name

project ↔ agent mapping (many-to-many):
  - **One project, many agents** (several ``agents[]`` rows share a project): the registry
    is an admission allowlist, and stream-level ownership is declared by the eval config
    entry's ``agentName``.
  - **One agent, many projects** (the same agent spans several Langfuse projects such as
    prod/pre): one entry uses a ``project_ids`` array (or the legacy ``project_id``
    repeated across entries; they are equivalent, and writing both takes the union).
    owner/team metadata is maintained in one place.

The RemoteConfig client singleton and cache live in :mod:`benchmark.settings.remote_config_store`.

RemoteConfig JSON shape::

    {
        "agents": [
            {
                "agent_name": "vehicle-agent",
                "project_id": "proj-1",
                "owner": "alice",
                "team": "nlp",
                "description": "Vehicle AI assistant",
                "metadata": {"version": "2.0"}
            }
        ]
    }

Usage::

    from benchmark.domain.agent_config import (
        AgentConfig,
        fetch_agent_configs,
        get_agent_config,
    )
"""

from dataclasses import dataclass, field
from typing import Any, Self

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.domain._cache import FetchListSpec, fetch_and_parse_list

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# RemoteConfig config constants.
# ---------------------------------------------------------------------------
#: RemoteConfig data_id for agent metadata config.
AGENT_REGISTRY_DATA_ID = "config.agent.registry"
#: RemoteConfig group for agent metadata config.
AGENT_REGISTRY_GROUP = "group.agent_registry"


# ---------------------------------------------------------------------------
# AgentConfig domain model.
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """One agent metadata entry (loaded from RemoteConfig, no DB storage).

    Stores agent metadata, including owner, team, description, and metadata.
    Look up a config by agent_name via :func:`get_agent_config`.
    """

    agent_name: str
    project_id: str = ""
#: Many-to-many: one agent spans several projects (for example one Langfuse project each
#: for prod/pre). Coexists with the single ``project_id``; matching takes the union (see ``projects``).
    project_ids: list[str] = field(default_factory=list)
    owner: str = ""
    team: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def projects(self) -> list[str]:
        """Normalize the project set: union of ``project_id`` and ``project_ids`` (deduped, blanks dropped)."""
        merged = [p for p in [self.project_id, *self.project_ids] if p]
        return list(dict.fromkeys(merged))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        """Build an instance from a RemoteConfig dict, tolerating optional fields.

        ``project_ids`` accepts a string array (non-strings are coerced with str; illegal
        input becomes an empty list). Backward compatible with the legacy single
        ``project_id``; writing both takes the union.
        """
        raw_ids = raw.get("project_ids")
        project_ids = [str(p) for p in raw_ids if p] if isinstance(raw_ids, (list, tuple)) else []
        return cls(
            agent_name=raw.get("agent_name", ""),
            project_id=raw.get("project_id", ""),
            project_ids=project_ids,
            owner=raw.get("owner", ""),
            team=raw.get("team", ""),
            description=raw.get("description", ""),
            metadata=raw.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Domain services.
# ---------------------------------------------------------------------------


def fetch_agent_configs(
    *,
    data_id: str = AGENT_REGISTRY_DATA_ID,
    group: str = AGENT_REGISTRY_GROUP,
    use_cache: bool = True,
) -> list[AgentConfig]:
    """Load every agent config from RemoteConfig.

    Results use the parsed-result cache by default (TTL 300s, see :mod:`benchmark.domain._cache`).
    Pass ``use_cache=False`` to force a refresh.

    Returns an empty list when RemoteConfig is unavailable or the config is missing, and
    does not interrupt the caller.
    """
    return fetch_and_parse_list(
        data_id,
        group,
        FetchListSpec(
            list_key="agents",
            parse_item=AgentConfig.from_dict,
            fetcher=fetch_cached_settings,
            use_cache=use_cache,
        ),
    )


def get_agent_config(
    agent_name: str,
    *,
    project_id: str | None = None,
    data_id: str = AGENT_REGISTRY_DATA_ID,
    group: str = AGENT_REGISTRY_GROUP,
    use_cache: bool = True,
) -> AgentConfig | None:
    """Look up one agent config by agent_name.

    An optional project_id filters the match. Returns ``None`` when nothing matches.
    """

    configs = fetch_agent_configs(data_id=data_id, group=group, use_cache=use_cache)
    for config in configs:
        if project_id is not None and project_id not in config.projects:
            continue
        if config.agent_name != agent_name:
            continue
        return config
    return None


def list_agent_configs(
    *,
    project_id: str | None = None,
    data_id: str = AGENT_REGISTRY_DATA_ID,
    group: str = AGENT_REGISTRY_GROUP,
    use_cache: bool = True,
) -> list[AgentConfig]:
    """List agent configs, optionally filtered by project_id."""

    configs = fetch_agent_configs(data_id=data_id, group=group, use_cache=use_cache)
    if project_id is None:
        return configs
    return [c for c in configs if project_id in c.projects]


def get_agent_names_by_project_id(
    project_id: str,
    *,
    data_id: str = AGENT_REGISTRY_DATA_ID,
    group: str = AGENT_REGISTRY_GROUP,
    use_cache: bool = True,
) -> list[str]:
    """Reverse-lookup every registered agent_name for a project_id (multi-value, ordered, deduped).

    One project_id may register several agents (several RemoteConfig ``agents[]`` rows share
    a project). The registry is an **admission allowlist**: the returned set is the agent
    identities legally owned by that project. Stream-level ownership is decided by the eval
    config entry's ``agentName`` (it must hit the candidate set); see
    :class:`benchmark.domain.schemas.EvalConfigEntry`.

    Returns an empty list when nothing matches. The caller decides whether to skip or raise.
    """

    configs = fetch_agent_configs(data_id=data_id, group=group, use_cache=use_cache)
    names: list[str] = []
    for config in configs:
        if project_id in config.projects and config.agent_name and config.agent_name not in names:
            names.append(config.agent_name)
    if len(names) > 1:
        logger.info(
            "agent registry has multiple agents for project %s: %s",
            project_id,
            names,
        )
    return names


def get_agent_name_by_project_id(
    project_id: str,
    *,
    data_id: str = AGENT_REGISTRY_DATA_ID,
    group: str = AGENT_REGISTRY_GROUP,
    use_cache: bool = True,
) -> str | None:
    """Reverse-lookup the first agent_name for a project_id (single-value compatibility API).

    Used when consuming traces: if ``metadata.agent_name`` is missing, match an
    ``agent_name`` from the RemoteConfig agent registry by ``project_id`` and inject it
    into metadata, so a missing value does not become a permanent ``metadata_missing`` error.

    When several agents share a project, the first one (array order) is returned. New code
    should use :func:`get_agent_names_by_project_id` for the full candidate set.

    Returns ``None`` when nothing matches. The caller decides whether to skip or raise.
    """

    configs = fetch_agent_configs(data_id=data_id, group=group, use_cache=use_cache)
    for config in configs:
        if project_id in config.projects and config.agent_name:
            return config.agent_name
    return None


__all__ = [
    "AGENT_REGISTRY_DATA_ID",
    "AGENT_REGISTRY_GROUP",
    "AgentConfig",
    "fetch_agent_configs",
    "get_agent_config",
    "get_agent_name_by_project_id",
    "get_agent_names_by_project_id",
    "list_agent_configs",
]
