"""Migration accessors preserve legacy mappings and startup class manifests."""

from benchmark.evaluators import registry as evaluators
from benchmark.attribution import registry as attribution


def test_evaluator_mapping_is_a_copy_and_retains_existing_agents() -> None:
    mapping = evaluators.evaluator_agents()
    assert mapping["judge_vui_agent"] is evaluators.VuiAgentEvaluator
    assert mapping["judge_vehicle_agent"] is evaluators.VehicleAgentEvaluatorAgent
    assert set(mapping) == {
        "judge_example",
        "judge_amap_agent",
        "judge_competitor_pairwise",
        "judge_memory_pairwise",
        "judge_vehicle_agent",
        "judge_vui_agent",
    }
    for component_name, agent_class in mapping.items():
        assert agent_class.__agent_component__["name"] == component_name
    mapping.clear()
    assert len(evaluators.evaluator_agents()) == 6


def test_attribution_mapping_is_a_copy_and_retains_existing_agents() -> None:
    mapping = attribution.attribution_agents()
    assert mapping["attribution_vui_agent"] is attribution.VuiAgentAttributionAgent
    assert mapping["attribution_vehicle_agent"] is attribution.VehicleAgentAttributionAgent
    assert set(mapping) == {
        "attribution_example",
        "attribution_amap_agent",
        "attribution_memory_pairwise",
        "attribution_vehicle_agent",
        "attribution_vui_agent",
    }
    for component_name, agent_class in mapping.items():
        assert agent_class.__agent_component__["name"] == component_name
    mapping.clear()
    assert len(attribution.attribution_agents()) == 5


def test_startup_manifests_remain_class_only_and_include_vui() -> None:
    for manifest in (evaluators, attribution):
        assert all(isinstance(getattr(manifest, name), type) for name in manifest.__all__)
    assert "ImageChatEvaluatorAgent" in evaluators.__all__
    assert "SearchAiCardEvaluatorAgent" in evaluators.__all__
    assert "SearchAiCardV2EvaluatorAgent" in evaluators.__all__
    assert "VuiAgentEvaluator" in evaluators.__all__
    assert "VuiAgentAttributionAgent" in attribution.__all__
