"""Eval-target → Benchmark Agent visibility allowlist (L1).

Source of truth: the repo default seed ∪ ``@benchmark_component(agent_scopes=...)``
code declarations (merged at startup / catalog sync). Tests may inject a whole table
via :func:`set_ownership_policy`. Names are alias-normalized before compare
(for example ``example_agent`` → ``example_benchmark``). Eval-target ids keep the page value.

``example_benchmark`` is visible to every eval target via ``global_agents``; every other
name is shown only when it appears in that scope's allowlist (seed ∪ code declaration).
A code declaration may put the same name in several scopes so one agent is visible in
multiple target catalogs. L2 registration documents stay single-owned; this module does
not change L2.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from benchmark.domain.eval_config import normalize_agent_name

#: Canonical-name aliases (normalized key → canonical Benchmark Agent name).
_DEFAULT_NAME_ALIASES: dict[str, str] = {
    "example_agent": "example_benchmark",
    "example": "example_benchmark",
}

_DEFAULT_GLOBAL_AGENTS: frozenset[str] = frozenset({"example_benchmark"})

#: Historical default owner for registrations that never wrote agentScope (the only tab before console).
_LEGACY_UNBOUND_SCOPE = "other"

#: Per-scope allowlist. example is visible to every eval target via global_agents.
_DEFAULT_SCOPES: dict[str, frozenset[str]] = {
    "other": frozenset(),
    "skill": frozenset(),
}


@dataclass(frozen=True, slots=True)
class OwnershipPolicy:
    """Eval-target ownership policy: scope allowlist ∪ global_agents."""

    scopes: dict[str, frozenset[str]] = field(default_factory=dict)
    global_agents: frozenset[str] = field(default_factory=frozenset)
    name_aliases: dict[str, str] = field(default_factory=dict)

    def canonicalize(self, name: str) -> str:
        """Benchmark Agent name → canonical compare name (alias expand + case/separator normalize)."""
        raw = (name or "").strip()
        if not raw:
            return ""
        lowered = normalize_agent_name(raw)
        aliased = self.name_aliases.get(lowered, lowered)
        return normalize_agent_name(aliased)

    def allows(self, agent_scope: str, agent_name: str) -> bool:
        """Whether this eval target may see / select this Benchmark Agent name."""
        canon = self.canonicalize(agent_name)
        if not canon:
            return False
        global_set = {self.canonicalize(item) for item in self.global_agents}
        if canon in global_set:
            return True
        allowed = self.scopes.get(agent_scope, frozenset())
        return canon in {self.canonicalize(item) for item in allowed}


@dataclass(frozen=True, slots=True)
class RegistrationVisibility:
    """Inputs for the L1+L2 registration visibility check (FUNC-PARAMS-001 fold)."""

    agent_scope: str
    agent_name: str
    customization: object | None
    registration_scope: str | None
    policy: OwnershipPolicy | None = None


#: Mutable slot so we avoid a ``global`` assignment (tests / canary injection).
_POLICY_SLOT: dict[str, OwnershipPolicy | None] = {"override": None}

#: ``@benchmark_component.agent_scopes`` merge layer: scope → agent names (code declarations only).
_COMPONENT_SCOPE_OVERLAY: dict[str, frozenset[str]] = {}


def default_ownership_policy() -> OwnershipPolicy:
    """Repo default seed (example is globally visible)."""
    return OwnershipPolicy(
        scopes=dict(_DEFAULT_SCOPES),
        global_agents=_DEFAULT_GLOBAL_AGENTS,
        name_aliases=dict(_DEFAULT_NAME_ALIASES),
    )


def refresh_component_scope_overlay(
    declarations: Iterable[tuple[str, Sequence[str]]],
) -> None:
    """Refresh the code-declaration layer from static components' ``(name, agent_scopes)``.

    Entries with empty ``agent_scopes`` are not written into the overlay (seed / global_agents
    only). The same name may appear in several scopes. Callers must pass the **full** static
    component snapshot (overwrite).
    """
    buckets: dict[str, set[str]] = {}
    for raw_name, scopes in declarations:
        name = (raw_name or "").strip()
        if not name:
            continue
        for raw_scope in scopes:
            scope = (raw_scope or "").strip()
            if not scope:
                continue
            buckets.setdefault(scope, set()).add(name)
    _COMPONENT_SCOPE_OVERLAY.clear()
    _COMPONENT_SCOPE_OVERLAY.update({key: frozenset(vals) for key, vals in buckets.items()})


def clear_component_scope_overlay() -> None:
    """Clear the code-declaration layer (test isolation)."""
    _COMPONENT_SCOPE_OVERLAY.clear()


def _merge_seed_with_component_overlay(base: OwnershipPolicy) -> OwnershipPolicy:
    """Seed ∪ code declarations; does not change global_agents / aliases."""
    if not _COMPONENT_SCOPE_OVERLAY:
        return base
    merged: dict[str, set[str]] = {key: set(vals) for key, vals in base.scopes.items()}
    for scope, names in _COMPONENT_SCOPE_OVERLAY.items():
        merged.setdefault(scope, set()).update(names)
    return OwnershipPolicy(
        scopes={key: frozenset(vals) for key, vals in merged.items()},
        global_agents=base.global_agents,
        name_aliases=base.name_aliases,
    )


def get_ownership_policy() -> OwnershipPolicy:
    """The policy currently in effect.

    When tests inject via :func:`set_ownership_policy`, that value is returned (no overlay,
    so cases are not polluted). The production path is the default seed ∪ the component
    ``agent_scopes`` overlay.
    """
    override = _POLICY_SLOT["override"]
    if override is not None:
        return override
    return _merge_seed_with_component_overlay(default_ownership_policy())


def set_ownership_policy(policy: OwnershipPolicy | None) -> None:
    """Inject or clear the policy override (tests / canary injection only)."""
    _POLICY_SLOT["override"] = policy


def is_registration_visible(query: RegistrationVisibility) -> bool:
    """L1+L2: whether a registration is visible under an eval target.

    The dropdown catalog (``/benchmarks?agentScope=``) still lists registrable logical names
    by L1 (``example_benchmark`` is globally selectable). The **completed registration list**
    requires ``registration_scope == agent_scope``. Unbound legacy documents are visible only
    under the legacy scope (``other``), so the list is not filtered empty.
    """
    active = query.policy or get_ownership_policy()
    if not active.allows(query.agent_scope, query.agent_name):
        return False
    scope = query.agent_scope.strip()
    bound = (query.registration_scope or "").strip()
    if bound == scope:
        return True
    return not bound and scope == _LEGACY_UNBOUND_SCOPE
