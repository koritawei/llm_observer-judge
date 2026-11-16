"""Unified agent names and full evaluator assembly.

Convention: both the benchmark name and evaluator_name normalize to the agent name
(``search_eval`` → ``search_ai_card``, and the ``judge_<name>`` prefix is stripped);
a configured evaluator not covered by a benchmark is assembled as an evaluator-only agent.
"""

from benchmark.runs.common.agent_catalog import (
    EvaluatorComponents,
    canonical_agent_name,
    configured_evaluators,
    evaluator_capabilities,
)


def test_canonical_agent_name_unifies_benchmark_and_evaluator() -> None:
    assert canonical_agent_name("example_benchmark") == "example"
    assert canonical_agent_name("amap_agent_benchmark") == "amap_agent"
    assert canonical_agent_name("search_eval", "search_ai_card") == "search_ai_card"
    assert canonical_agent_name("ieval_external", "amap_chat") == "ieval_external"
    assert canonical_agent_name("gpt-judge-pro") == "gpt-judge-pro"


def test_configured_evaluators_cover_all_registered_stems() -> None:
    evaluators = configured_evaluators()
    assert set(evaluators) == {
        "amap_agent",
        "competitor_pairwise",
        "example",
        "image_chat",
        "memory_pairwise",
        "search_ai_card",
        "search_ai_card_v2",
        "vehicle_agent",
        "vui_agent",
    }
    assert evaluators["vui_agent"].has_attribution is True
    assert evaluators["vui_agent"].has_triage is False
    assert evaluators["competitor_pairwise"].has_attribution is False
    assert evaluators["competitor_pairwise"].has_triage is False
    assert evaluators["search_ai_card"].has_attribution is False
    assert evaluators["search_ai_card_v2"].has_attribution is False
    assert evaluators["search_ai_card_v2"].has_triage is False
    assert evaluators["image_chat"].has_attribution is False
    assert evaluators["image_chat"].has_triage is False
    for stem in ("amap_agent", "example", "memory_pairwise", "vehicle_agent"):
        assert evaluators[stem].has_attribution is True
        assert evaluators[stem].has_triage is True


def test_evaluator_capabilities_reflect_registered_components() -> None:
    judge_only = evaluator_capabilities(
        "competitor_pairwise",
        EvaluatorComponents(judge_description="pairwise judge"),
    )
    assert [item.kind for item in judge_only] == ["judge"]
    assert judge_only[0].description == "pairwise judge"
    assert judge_only[0].alias == "evaluator_service_judge"

    full = evaluator_capabilities(
        "vehicle_agent",
        EvaluatorComponents(has_attribution=True, has_triage=True),
    )
    by_kind = {item.kind: item for item in full}
    assert set(by_kind) == {"judge", "triage", "attribution"}
    assert by_kind["triage"].alias == "vehicle_agent_triage"
    assert by_kind["attribution"].alias == "vehicle_agent_attribution"
    assert all(item.description for item in full)
    assert {item.pipeline for item in full} == {"streaming"}
