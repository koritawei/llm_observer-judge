"""agent-eval domain layer.

Holds domain models and domain services that do not depend on DB storage.
ORM models stay in :mod:`benchmark.records.models`; this layer contains business logic only.

Usage:
    from benchmark.domain.agent_config import AgentConfig, get_agent_name_by_project_id
    from benchmark.domain.eval_config import EvalConfigStore
    from benchmark.domain.owner_mapping import OwnerMapping, resolve_owner_route
    from benchmark.domain import clear_parsed_cache  # clear cache in tests / on config refresh
"""

from benchmark.domain._cache import clear_parsed_cache

__all__ = [
    "clear_parsed_cache",
]
