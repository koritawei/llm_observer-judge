"""owner_mapping domain service — three-level matching in resolve_owner_route."""

from unittest.mock import patch

from benchmark.domain.enums import CasePriority, MatchType
from benchmark.domain.owner_mapping import (
    OwnerMapping,
    fetch_owner_mappings,
    resolve_owner_route,
    resolve_route_assignment,
)


def _mapping(
    mapping_id: str,
    match_type: MatchType,
    match_key: str,
    *,
    enabled: bool = True,
    owner_group: str | None = None,
    assigned_to: str = "",
    priority: CasePriority | None = None,
    project_url: str | None = None,
) -> OwnerMapping:
    return OwnerMapping(
        mapping_id=mapping_id,
        match_type=match_type,
        match_key=match_key,
        assigned_to=assigned_to,
        owner_group=owner_group,
        priority=priority,
        project_url=project_url,
        enabled=enabled,
    )


def test_resolve_owner_route_prefers_exact_agent_cause() -> None:
    mappings = [
        _mapping("exact", MatchType.AGENT_CAUSE, "amap-agent/tool_error"),
        _mapping("wildcard", MatchType.AGENT_CAUSE, "amap-agent/*"),
        _mapping("default", MatchType.DEFAULT, "*"),
    ]
    with patch("benchmark.domain.owner_mapping.fetch_owner_mappings", return_value=mappings):
        hit = resolve_owner_route(agent_name="amap-agent", primary_cause="tool_error")
    assert hit is not None
    assert hit.mapping_id == "exact"


def test_resolve_owner_route_falls_back_to_agent_wildcard() -> None:
    mappings = [
        _mapping("wildcard", MatchType.AGENT_CAUSE, "amap-agent/*"),
        _mapping("default", MatchType.DEFAULT, "*"),
    ]
    with patch("benchmark.domain.owner_mapping.fetch_owner_mappings", return_value=mappings):
        hit = resolve_owner_route(agent_name="amap-agent", primary_cause="unknown_cause")
    assert hit is not None
    assert hit.mapping_id == "wildcard"


def test_resolve_owner_route_matches_owner_group() -> None:
    mappings = [
        _mapping("team", MatchType.OWNER_GROUP, "platform-team", owner_group="platform-team"),
        _mapping("default", MatchType.DEFAULT, "*"),
    ]
    with patch("benchmark.domain.owner_mapping.fetch_owner_mappings", return_value=mappings):
        hit = resolve_owner_route(
            agent_name="amap-agent",
            primary_cause="x",
            owner_group="platform-team",
        )
    assert hit is not None
    assert hit.mapping_id == "team"


def test_resolve_owner_route_default_fallback() -> None:
    mappings = [_mapping("default", MatchType.DEFAULT, "*")]
    with patch("benchmark.domain.owner_mapping.fetch_owner_mappings", return_value=mappings):
        hit = resolve_owner_route(agent_name="amap-agent", primary_cause="x")
    assert hit is not None
    assert hit.mapping_id == "default"


def test_resolve_owner_route_skips_disabled_mappings() -> None:
    mappings = [
        _mapping("disabled", MatchType.DEFAULT, "*", enabled=False),
    ]
    with patch("benchmark.domain.owner_mapping.fetch_owner_mappings", return_value=mappings):
        assert resolve_owner_route(agent_name="amap-agent", primary_cause="x") is None


def test_fetch_owner_mappings_parses_remote_config_payload() -> None:
    raw = {
        "mappings": [
            {
                "mapping_id": "m1",
                "match_type": "agent_cause",
                "match_key": "a/b",
                "assigned_to": "bob",
                "enabled": True,
            }
        ]
    }
    with patch("benchmark.domain.owner_mapping.fetch_cached_settings", return_value=raw):
        items = fetch_owner_mappings(data_id="test.id", group="test.group", use_cache=False)
    assert len(items) == 1
    assert items[0].mapping_id == "m1"
    assert items[0].match_type == MatchType.AGENT_CAUSE


def test_resolve_owner_route_priority_p0_beats_p1() -> None:
    """When the same match_key is configured as both P0 and P1, P0 hits (independent of list order)."""
    mappings = [
        _mapping(
            "p1_first",
            MatchType.AGENT_CAUSE,
            "amap-agent/tool_error",
            priority=CasePriority.P1,
            assigned_to="bob",
        ),
        _mapping(
            "p0_second",
            MatchType.AGENT_CAUSE,
            "amap-agent/tool_error",
            priority=CasePriority.P0,
            assigned_to="alice",
        ),
    ]
    with patch("benchmark.domain.owner_mapping.fetch_owner_mappings", return_value=mappings):
        hit = resolve_owner_route(agent_name="amap-agent", primary_cause="tool_error")
    assert hit is not None
    assert hit.mapping_id == "p0_second"
    assert hit.assigned_to == "alice"


def test_resolve_owner_route_priority_p2_wins_over_none() -> None:
    """priority=P2 ranks ahead of priority=None."""
    mappings = [
        _mapping(
            "no_priority",
            MatchType.AGENT_CAUSE,
            "amap-agent/tool_error",
            priority=None,
            assigned_to="bob",
        ),
        _mapping(
            "p2",
            MatchType.AGENT_CAUSE,
            "amap-agent/tool_error",
            priority=CasePriority.P2,
            assigned_to="alice",
        ),
    ]
    with patch("benchmark.domain.owner_mapping.fetch_owner_mappings", return_value=mappings):
        hit = resolve_owner_route(agent_name="amap-agent", primary_cause="tool_error")
    assert hit is not None
    assert hit.mapping_id == "p2"


def test_resolve_owner_route_priority_none_acts_as_lowest() -> None:
    """priority=None is the lowest priority (when every entry for the key is None, take the first in the list)."""
    mappings = [
        _mapping(
            "first",
            MatchType.AGENT_CAUSE,
            "amap-agent/tool_error",
            priority=None,
            assigned_to="bob",
        ),
        _mapping(
            "second",
            MatchType.AGENT_CAUSE,
            "amap-agent/tool_error",
            priority=None,
            assigned_to="alice",
        ),
    ]
    with patch("benchmark.domain.owner_mapping.fetch_owner_mappings", return_value=mappings):
        hit = resolve_owner_route(agent_name="amap-agent", primary_cause="tool_error")
    assert hit is not None
    assert hit.mapping_id == "first"


def test_resolve_owner_route_priority_disabled_mapping_still_skipped() -> None:
    """P0 with enabled=False is still skipped; the hit should be the next tier, P1."""
    mappings = [
        _mapping(
            "p0_disabled",
            MatchType.AGENT_CAUSE,
            "amap-agent/tool_error",
            priority=CasePriority.P0,
            enabled=False,
            assigned_to="bob",
        ),
        _mapping(
            "p1_enabled",
            MatchType.AGENT_CAUSE,
            "amap-agent/tool_error",
            priority=CasePriority.P1,
            assigned_to="alice",
        ),
    ]
    with patch("benchmark.domain.owner_mapping.fetch_owner_mappings", return_value=mappings):
        hit = resolve_owner_route(agent_name="amap-agent", primary_cause="tool_error")
    assert hit is not None
    assert hit.mapping_id == "p1_enabled"


def test_from_dict_keeps_string_assigned_to() -> None:
    mapping = OwnerMapping.from_dict({"mapping_id": "m1", "assigned_to": "alice"})
    assert mapping.assigned_to == "alice"
    assert mapping.assignees == ()


def test_from_dict_numeric_assigned_to_stays_staff_id() -> None:
    mapping = OwnerMapping.from_dict({"mapping_id": "m1", "assigned_to": 529504})
    assert mapping.assigned_to == "529504"
    assert mapping.assignees == ()


def test_from_dict_parses_assigned_to_list() -> None:
    mapping = OwnerMapping.from_dict(
        {"mapping_id": "m1", "assigned_to": ["alice", " bob ", "", "alice"]}
    )
    assert mapping.assigned_to == ""
    assert mapping.assignees == ("alice", "bob")


def test_resolve_route_assignment_random_picks_one_and_leaves_participants() -> None:
    mapping = OwnerMapping(mapping_id="m1", assignees=("alice", "bob", "carol"))
    assignee, participants = resolve_route_assignment(
        mapping,
        fallback_assignee="fallback",
        fallback_participants=["dave", "bob"],
        pick=lambda _pool: "bob",
    )
    assert assignee == "bob"
    assert participants == ["dave"]


def test_resolve_route_assignment_all_uses_first_and_ccs_rest() -> None:
    mapping = OwnerMapping(
        mapping_id="m1",
        assignees=("alice", "bob", "carol"),
        assign_type="all",
    )
    assignee, participants = resolve_route_assignment(
        mapping,
        fallback_assignee="fallback",
        fallback_participants=["dave", "bob"],
        pick=lambda _pool: "carol",
    )
    assert assignee == "alice"
    assert participants == ["dave", "bob", "carol"]


def test_resolve_route_assignment_all_drops_assignee_already_in_participants() -> None:
    mapping = OwnerMapping(
        mapping_id="m1",
        assignees=("alice", "bob", "carol"),
        assign_type="all",
    )
    assignee, participants = resolve_route_assignment(
        mapping,
        fallback_assignee="fallback",
        fallback_participants=["dave", "alice"],
    )
    assert assignee == "alice"
    assert participants == ["dave", "bob", "carol"]


def test_from_dict_assign_type() -> None:
    listed = OwnerMapping.from_dict(
        {"mapping_id": "m1", "assigned_to": ["alice", "bob"], "assign_type": "all"}
    )
    defaulted = OwnerMapping.from_dict(
        {"mapping_id": "m2", "assigned_to": ["alice", "bob"]}
    )
    unknown = OwnerMapping.from_dict(
        {"mapping_id": "m3", "assigned_to": "alice", "assign_type": "fanout"}
    )
    assert listed.assign_type == "all"
    assert defaulted.assign_type == "random"
    assert unknown.assign_type == "random"
