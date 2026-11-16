"""Evaluation-target ownership: the L1 allowlist and aliases (pure policy, not bound to a DB)."""

from benchmark.domain.agent_scope_ownership import (
    OwnershipPolicy,
    RegistrationVisibility,
    clear_component_scope_overlay,
    default_ownership_policy,
    get_ownership_policy,
    is_registration_visible,
    refresh_component_scope_overlay,
    set_ownership_policy,
)


def test_amap_agent_allows_seed_whitelist() -> None:
    policy = default_ownership_policy()
    assert policy.allows("other", "example_benchmark")
    assert not policy.allows("other", "vehicle_agent")
    assert not policy.allows("amap-agent", "amap_agent_benchmark")
    assert not policy.allows("vehicle-agent", "vehicle_agent")
    assert not policy.allows("vui-agent", "vui_agent")


def test_vehicle_agent_allows_vehicle_and_global_example() -> None:
    policy = default_ownership_policy()
    assert policy.allows("other", "example_benchmark")
    assert not policy.allows("other", "vehicle_agent")
    assert not policy.allows("other", "search_eval")


def test_custom_scope_inherits_global_only() -> None:
    policy = default_ownership_policy()
    assert policy.allows("my-custom-biz", "example_benchmark")
    assert not policy.allows("my-custom-biz", "search_eval")
    assert not policy.allows("vui-agent", "vehicle_agent")


def test_example_agent_alias_matches_example_benchmark() -> None:
    policy = default_ownership_policy()
    assert policy.allows("amap-agent", "example_agent")
    assert policy.allows("vehicle-agent", "example-agent")
    assert policy.canonicalize("example_agent") == "example_benchmark"


def test_injected_policy_overrides_seed() -> None:
    policy = OwnershipPolicy(
        scopes={"amap-agent": frozenset({"search_eval"})},
        global_agents=frozenset(),
        name_aliases={},
    )
    assert policy.allows("amap-agent", "search_eval")
    assert not policy.allows("amap-agent", "example_benchmark")


def test_unbound_registration_visible_only_under_legacy_scope() -> None:
    unbound = RegistrationVisibility(
        agent_scope="other",
        agent_name="example_benchmark",
        customization=None,
        registration_scope=None,
    )
    assert is_registration_visible(unbound)
    assert not is_registration_visible(
        RegistrationVisibility(
            agent_scope="skill",
            agent_name="example_benchmark",
            customization=None,
            registration_scope=None,
        )
    )
    assert is_registration_visible(
        RegistrationVisibility(
            agent_scope="skill",
            agent_name="example_benchmark",
            customization=None,
            registration_scope="skill",
        )
    )


def test_component_overlay_unions_into_seed_and_supports_multi_scope() -> None:
    clear_component_scope_overlay()
    set_ownership_policy(None)
    try:
        refresh_component_scope_overlay(
            [
                ("new_multi_agent", ("amap-agent", "vehicle-agent")),
                ("seed_only_peer", ()),
            ]
        )
        policy = get_ownership_policy()
        assert policy.allows("amap-agent", "new_multi_agent")
        assert policy.allows("vehicle-agent", "new_multi_agent")
        assert not policy.allows("vui-agent", "new_multi_agent")
        # An empty agent_scopes does not enter the overlay; a deleted business name no longer appears in the seed
        assert not policy.allows("amap-agent", "search_eval")
        assert not policy.allows("amap-agent", "seed_only_peer")
    finally:
        clear_component_scope_overlay()
        set_ownership_policy(None)


def test_set_ownership_policy_bypasses_component_overlay() -> None:
    clear_component_scope_overlay()
    refresh_component_scope_overlay([("overlay_only", ("amap-agent",))])
    set_ownership_policy(
        OwnershipPolicy(
            scopes={"amap-agent": frozenset({"search_eval"})},
            global_agents=frozenset(),
            name_aliases={},
        )
    )
    try:
        policy = get_ownership_policy()
        assert policy.allows("amap-agent", "search_eval")
        assert not policy.allows("amap-agent", "overlay_only")
    finally:
        set_ownership_policy(None)
        clear_component_scope_overlay()
