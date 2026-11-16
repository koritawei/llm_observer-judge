"""agent_config domain service — fetching and looking up RemoteConfig agent metadata."""

from unittest.mock import patch

from benchmark.domain.agent_config import (
    AgentConfig,
    fetch_agent_configs,
    get_agent_config,
    get_agent_name_by_project_id,
    get_agent_names_by_project_id,
    list_agent_configs,
)

# noqa: E402 — the class below is appended at the end of the file


def test_fetch_agent_configs_parses_agents_list() -> None:
    raw = {
        "agents": [
            {
                "agent_name": "vehicle-agent",
                "project_id": "proj-1",
                "owner": "alice",
                "team": "nlp",
                "description": "Vehicle AI",
                "metadata": {"version": "2.0"},
            }
        ]
    }
    with patch("benchmark.domain.agent_config.fetch_cached_settings", return_value=raw):
        configs = fetch_agent_configs(data_id="test.id", group="test.group")
    assert len(configs) == 1
    assert configs[0] == AgentConfig(
        agent_name="vehicle-agent",
        project_id="proj-1",
        owner="alice",
        team="nlp",
        description="Vehicle AI",
        metadata={"version": "2.0"},
    )


def test_fetch_agent_configs_returns_empty_when_agents_missing() -> None:
    with patch("benchmark.domain.agent_config.fetch_cached_settings", return_value={}):
        assert fetch_agent_configs(data_id="test.id", group="test.group") == []


def test_get_agent_config_by_name() -> None:
    configs = [
        AgentConfig(agent_name="a", project_id="p1"),
        AgentConfig(agent_name="b", project_id="p2"),
    ]
    with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
        hit = get_agent_config("b", data_id="x", group="y")
    assert hit is not None
    assert hit.agent_name == "b"
    assert hit.project_id == "p2"


def test_get_agent_config_filters_by_project_id() -> None:
    configs = [AgentConfig(agent_name="a", project_id="p1")]
    with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
        assert get_agent_config("a", project_id="other", data_id="x", group="y") is None


def test_list_agent_configs_filters_project() -> None:
    configs = [
        AgentConfig(agent_name="a", project_id="p1"),
        AgentConfig(agent_name="b", project_id="p2"),
    ]
    with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
        filtered = list_agent_configs(project_id="p1", data_id="x", group="y")
    assert [c.agent_name for c in filtered] == ["a"]


def test_get_agent_name_by_project_id_returns_match() -> None:
    """When project_id matches, return the corresponding agent_name."""
    configs = [
        AgentConfig(agent_name="vehicle-agent", project_id="proj-vehicle"),
        AgentConfig(agent_name="amap-agent", project_id="proj-amap"),
    ]
    with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
        result = get_agent_name_by_project_id("proj-vehicle", data_id="x", group="y")
    assert result == "vehicle-agent"


def test_get_agent_name_by_project_id_returns_none_when_no_match() -> None:
    """When project_id matches nothing, return None."""
    configs = [AgentConfig(agent_name="a", project_id="p1")]
    with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
        result = get_agent_name_by_project_id("unknown", data_id="x", group="y")
    assert result is None


def test_get_agent_name_by_project_id_skips_empty_agent_name() -> None:
    """An entry with an empty agent_name is skipped."""
    configs = [
        AgentConfig(agent_name="", project_id="p1"),
        AgentConfig(agent_name="real-agent", project_id="p1"),
    ]
    with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
        result = get_agent_name_by_project_id("p1", data_id="x", group="y")
    assert result == "real-agent"


def test_get_agent_name_by_project_id_returns_none_when_empty() -> None:
    """When the RemoteConfig config is empty, return None."""
    with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=[]):
        result = get_agent_name_by_project_id("any", data_id="x", group="y")
    assert result is None


class TestMultiAgentPerProject:
    """Several agents per project (plan C): registry candidate-set semantics."""

    def test_multiple_agents_returned_in_order(self) -> None:
        """Several agent entries for the same project → the candidate set returns all of them, in array order."""
        configs = [
            AgentConfig(agent_name="amap-agent", project_id="p1"),
            AgentConfig(agent_name="search_agent", project_id="p1"),
        ]
        with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
            result = get_agent_names_by_project_id("p1", data_id="x", group="y")
        assert result == ["amap-agent", "search_agent"]

    def test_candidates_deduplicated(self) -> None:
        """Duplicate agent_name entries are deduped (first-seen order is kept)."""
        configs = [
            AgentConfig(agent_name="a", project_id="p1"),
            AgentConfig(agent_name="b", project_id="p2"),
            AgentConfig(agent_name="a", project_id="p1"),
        ]
        with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
            result = get_agent_names_by_project_id("p1", data_id="x", group="y")
        assert result == ["a"]

    def test_empty_when_no_match(self) -> None:
        with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=[]):
            assert get_agent_names_by_project_id("any", data_id="x", group="y") == []

    def test_single_candidate_matches_legacy_single_api(self) -> None:
        """With a single candidate, the multi-value API behaves like the old single-value API (compatibility anchor)."""
        configs = [AgentConfig(agent_name="only-agent", project_id="p1")]
        with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
            names = get_agent_names_by_project_id("p1", data_id="x", group="y")
            single = get_agent_name_by_project_id("p1", data_id="x", group="y")
        assert names == ["only-agent"]
        assert single == "only-agent"


class TestManyProjectsPerAgent:
    """Many-to-many: one agent_name across several project_ids (the project_ids array)."""

    def test_projects_merges_single_and_array_fields(self) -> None:
        """A project_id scalar and a project_ids array written together → the union, deduped."""
        cfg = AgentConfig(
            agent_name="amap-agent",
            project_id="p-prod",
            project_ids=["p-prod", "p-pre", ""],
        )
        assert cfg.projects == ["p-prod", "p-pre"]

    def test_projects_from_dict_parses_array(self) -> None:
        """from_dict parses the project_ids array (non-string elements are tolerated)."""
        cfg = AgentConfig.from_dict(
            {"agent_name": "a", "project_ids": ["p1", "p2", 3, None]},
        )
        assert cfg.projects == ["p1", "p2", "3"]

    def test_projects_invalid_input_falls_back_to_empty(self) -> None:
        """An illegal project_ids type (string/dict) → collapse to an empty list; project_id still applies."""
        cfg = AgentConfig.from_dict(
            {"agent_name": "a", "project_id": "p1", "project_ids": "not-a-list"},
        )
        assert cfg.projects == ["p1"]

    def test_names_lookup_matches_agent_with_project_ids(self) -> None:
        """An agent registered via project_ids can be reverse-looked-up as a candidate in any of those projects."""
        configs = [
            AgentConfig(agent_name="amap-agent", project_ids=["p-prod", "p-pre"]),
        ]
        with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
            assert get_agent_names_by_project_id("p-prod", data_id="x", group="y") == ["amap-agent"]
            assert get_agent_names_by_project_id("p-pre", data_id="x", group="y") == ["amap-agent"]
            assert get_agent_names_by_project_id("p-other", data_id="x", group="y") == []

    def test_legacy_single_value_entry_equivalent_to_array(self) -> None:
        """Repeating an old project_id across several entries is equivalent to a project_ids array."""
        array_style = [AgentConfig(agent_name="a", project_ids=["p1", "p2"])]
        legacy_style = [
            AgentConfig(agent_name="a", project_id="p1"),
            AgentConfig(agent_name="a", project_id="p2"),
        ]
        with (
            patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=array_style),
        ):
            from_array = get_agent_names_by_project_id("p2", data_id="x", group="y")
        with (
            patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=legacy_style),
        ):
            from_legacy = get_agent_names_by_project_id("p2", data_id="x", group="y")
        assert from_array == from_legacy == ["a"]

    def test_list_agent_configs_filters_by_projects(self) -> None:
        """list_agent_configs filters by the union of projects (the dataset_eval consumption path)."""
        configs = [
            AgentConfig(agent_name="a", project_ids=["p1", "p2"]),
            AgentConfig(agent_name="b", project_id="p3"),
        ]
        with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
            matched = list_agent_configs(project_id="p2", data_id="x", group="y")
        assert [c.agent_name for c in matched] == ["a"]

    def test_get_agent_config_filters_by_projects(self) -> None:
        """project filtering in get_agent_config uses the same union."""
        configs = [
            AgentConfig(agent_name="a", project_ids=["p1", "p2"]),
            AgentConfig(agent_name="b", project_id="p2"),
        ]
        with patch("benchmark.domain.agent_config.fetch_agent_configs", return_value=configs):
            got = get_agent_config("b", project_id="p2", data_id="x", group="y")
        assert got is not None and got.agent_name == "b"
