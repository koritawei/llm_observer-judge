"""GET /v1/benchmarks unified agent view: static components merged with page-registered dynamic agents.

Chain-ownership convention: ``ieval_external`` / ``search_eval`` are batch,
``amap_agent_benchmark`` is the realtime stream, and ``example_benchmark`` may be dual-chain;
page-registered dynamic agents use the batch chain.
"""

from types import SimpleNamespace

import pytest
from benchmark.routes.benchmark import _definition_from_document, list_benchmarks
from benchmark.domain.benchmark_console import AgentDocument

SEARCH_EVAL_RUBRIC = "business.runs.search_eval.rubric:SearchEvalRubric"


def _document(name: str, rubric_class: str | None = None) -> AgentDocument:
    return AgentDocument.model_validate(
        {
            "id": "ba-1",
            "name": name,
            "owner": "eval-team",
            "registeredAt": "2026-10-16 10:00:00",
            "rubricClass": rubric_class,
        },
    )


@pytest.mark.asyncio
async def test_unifies_static_and_registered_agents(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static components and page-registered agents are merged for the external view; output is sorted by name."""
    monkeypatch.setattr(
        "benchmark.routes.benchmark.is_benchmark_component_enabled",
        lambda name: True,
    )

    async def _documents() -> list[AgentDocument]:
        return [_document("amap agent", SEARCH_EVAL_RUBRIC)]

    monkeypatch.setattr("benchmark.routes.benchmark.list_agent_documents", _documents)

    definitions = await list_benchmarks()
    by_name = {item.name: item for item in definitions}
    # All four static agents are in the catalog (including the new amap_agent_benchmark realtime-stream agent)
    assert set(by_name) >= {
        "amap_agent_benchmark",
        "example_benchmark",
        "ieval_external",
        "search_eval",
    }
    # Unified agent name: benchmark name / evaluator_name are normalized to agent_name
    assert by_name["search_eval"].agent_name == "search_ai_card"
    assert by_name["amap_agent_benchmark"].agent_name == "amap_agent"
    assert by_name["example_benchmark"].agent_name == "example"
    assert by_name["ieval_external"].agent_name == "ieval_external"
    # Full evaluator assembly: a configured evaluator not covered by a benchmark gets its own agent row
    for stem in ("vehicle_agent", "competitor_pairwise", "memory_pairwise"):
        assert by_name[stem].agent_name == stem
        assert by_name[stem].evaluators == [stem]
    # An evaluator already covered by a benchmark is not listed again (agent_name is unique)
    agent_names = [item.agent_name for item in definitions]
    assert len(agent_names) == len(set(agent_names))
    assert "example" not in by_name
    # evaluator-only rows also have a default report; competitor_pairwise has no attribution/triage registration
    competitor = {c.kind for c in by_name["competitor_pairwise"].capabilities}
    assert competitor == {"judge", "report"}
    competitor_report = next(
        c for c in by_name["competitor_pairwise"].capabilities if c.kind == "report"
    )
    assert competitor_report.class_ref is not None
    assert competitor_report.description
    # Chain ownership: most are single-chain; example is dual-chain (batch+streaming, same config source)
    pipelines_amap = {c.pipeline for c in by_name["amap_agent_benchmark"].capabilities}
    assert pipelines_amap == {"streaming"}, "amap_agent_benchmark 应为纯实时流 agent"
    for name in ("ieval_external", "search_eval"):
        pipelines = {c.pipeline for c in by_name[name].capabilities}
        assert pipelines == {"batch"}, f"{name} 应为纯批量 agent"
    example_pipelines = {c.pipeline for c in by_name["example_benchmark"].capabilities}
    assert example_pipelines == {"batch", "streaming"}
    example_batch_judge = next(
        c
        for c in by_name["example_benchmark"].capabilities
        if c.pipeline == "batch" and c.kind == "judge"
    )
    example_streaming_judge = next(
        c
        for c in by_name["example_benchmark"].capabilities
        if c.pipeline == "streaming" and c.kind == "judge"
    )
    assert example_batch_judge.alias == "evaluator_service_judge"
    assert example_batch_judge.execution == "sync"
    assert example_streaming_judge.alias == "evaluator_service_judge"
    assert example_streaming_judge.execution == "sync"
    example_batch_attr = next(
        c
        for c in by_name["example_benchmark"].capabilities
        if c.pipeline == "batch" and c.kind == "attribution"
    )
    assert example_batch_attr.alias == "evaluator_service_attribution"
    # amap_agent_benchmark capability chain: evaluator_service_judge + amap_agent-specific plugins
    amap_streaming = {c.alias for c in by_name["amap_agent_benchmark"].capabilities}
    assert "evaluator_service_judge" in amap_streaming
    assert "amap_agent_triage" in amap_streaming
    assert "amap_agent_attribution" in amap_streaming
    assert by_name["amap_agent_benchmark"].evaluators == ["amap_agent"]
    # Every view has a rubric class; dynamic documents keep the original class reference
    assert all(item.rubric_class for item in definitions)
    assert by_name["search_eval"].rubric_class == SEARCH_EVAL_RUBRIC
    # Descriptions come back with the name: agent-level + rubric scoring standard + per capability
    assert by_name["amap_agent_benchmark"].description
    assert "core_completion" in by_name["amap_agent_benchmark"].rubric_description
    assert all(c.description for c in by_name["amap_agent_benchmark"].capabilities)
    assert by_name["search_eval"].description
    assert "holistic" in by_name["search_eval"].rubric_description
    assert all(c.description for c in by_name["search_eval"].capabilities)
    assert "D0-D6" in by_name["ieval_external"].rubric_description
    # Batch component capability-chain sample: search_eval uses a dedicated local judge and does not run attribution
    search = by_name["search_eval"]
    batch = {c.kind: c for c in search.capabilities}
    assert batch["judge"].alias == "search_eval_judge"
    assert batch["attribution"].execution == "none"
    assert batch["report"].class_ref is not None
    assert search.evaluators == ["search_ai_card"]
    # Page-registered agent (dynamic): batch chain (takes over offline dataset evaluation)
    dynamic = by_name["amap agent"]
    assert dynamic.rubric_class == SEARCH_EVAL_RUBRIC
    assert dynamic.service_name == "amap agent"
    assert dynamic.owner == "eval-team"
    assert len(dynamic.criteria) > 0
    dynamic_pipelines = {c.pipeline for c in dynamic.capabilities}
    assert dynamic_pipelines == {"batch"}
    assert dynamic.evaluators == ["search_ai_card"]
    # Output is sorted by name
    assert [item.name for item in definitions] == sorted(by_name)


@pytest.mark.asyncio
async def test_catalog_customization_policy_and_attribution_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catalog policy shares its source with the framework registry; the attribution toggle is allowed by default, and supported is derived from capability."""
    monkeypatch.setattr(
        "benchmark.routes.benchmark.is_benchmark_component_enabled",
        lambda name: True,
    )

    async def _documents() -> list[AgentDocument]:
        return []

    monkeypatch.setattr("benchmark.routes.benchmark.list_agent_documents", _documents)

    definitions = await list_benchmarks()
    by_name = {item.name: item for item in definitions}
    # prompt/skill customization exists only for example_benchmark (framework registry); the catalog must expose the same source
    example = by_name["example_benchmark"].customization
    assert example.judge.prompt.slots == ["prompt"]
    assert example.judge.skill.allowed
    assert example.attribution.prompt.slots == ["prompt"]
    # Attribution is optional: the toggle is allowed by default (supported is derived from attribution capability)
    assert example.attribution_toggle.allowed
    assert example.attribution_toggle.supported
    amap = by_name["amap_agent_benchmark"].customization
    assert not amap.judge.prompt.allowed
    assert amap.attribution_toggle.allowed
    assert amap.attribution_toggle.supported
    # Agents without attribution capability do not support the toggle (search attribution execution=none)
    search = by_name["search_eval"].customization
    assert not search.attribution_toggle.supported
    # evaluator-only row: no prompt/skill customization, but attribution can be turned off
    vehicle = by_name["vehicle_agent"].customization
    assert not vehicle.judge.prompt.allowed
    assert vehicle.attribution_toggle.allowed
    assert vehicle.attribution_toggle.supported
    competitor = by_name["competitor_pairwise"].customization
    assert not competitor.attribution_toggle.supported


@pytest.mark.asyncio
async def test_static_name_wins_over_registered_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a dynamic document and a static component share a name, the static one wins (same semantics as dynamic registration on the trigger entry)."""

    async def _documents() -> list[AgentDocument]:
        return [_document("search_eval")]

    monkeypatch.setattr("benchmark.routes.benchmark.list_agent_documents", _documents)

    definitions = await list_benchmarks()
    names = [item.name for item in definitions]
    assert names.count("search_eval") == 1
    search = next(item for item in definitions if item.name == "search_eval")
    assert search.owner is None


def test_document_definition_tolerates_unavailable_rubric_class() -> None:
    """When the class named by rubricClass was removed in a refactor, the catalog tolerates an empty rubric (fail-fast only at trigger time)."""
    document = SimpleNamespace(
        name="ghost rubric",
        rubric_class="business.runs.search_eval.rubric:NoSuchRubric",
        description="",
        owner="eval-team",
        registered_at="2026-10-16 10:00:00",
    )
    definition = _definition_from_document(document)  # type: ignore[arg-type]
    assert definition.criteria == []
    assert definition.aggregation_method == "weighted_avg"
    assert definition.rubric_class == "business.runs.search_eval.rubric:NoSuchRubric"
    assert definition.rubric_description
    assert definition.description is None
    # The dynamic agent uses the batch chain
    assert {c.pipeline for c in definition.capabilities} == {"batch"}


def test_document_definition_derives_rubric_description() -> None:
    """Dynamic-agent rubric description: prefer the first paragraph of the rubric class docstring (as implemented)."""

    class _FakeRubric:
        """Search AI-card T+1 report rubric: single overall-quality dimension (0–1 scale)."""

    rubric_cls = _FakeRubric
    from benchmark.routes.benchmark import _rubric_description

    desc = _rubric_description(
        rubric_cls,
        [{"name": "综合质量"}],
        "weighted_avg",
    )
    assert "单维综合质量" in desc
    # When there is no rubric class, assemble the description from dimension names
    fallback = _rubric_description(
        None,
        [{"name": "事实准确性"}, {"name": "格式正确性"}],
        "weighted_avg",
    )
    assert "2 个维度加权" in fallback
    assert "事实准确性、格式正确性" in fallback
