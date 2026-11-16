"""Capability catalog: most agents belong to only one of the batch or streaming chains; example may use both.

Chain ownership: ``ieval_external`` / ``search_eval`` are batch chains,
``amap_agent_benchmark`` is the streaming chain, and ``example_benchmark`` may
declare both batch and streaming (sharing the local example config); a dynamic
agent registered on the page takes offline dataset evaluation → the batch chain.
The routing source of truth for streaming is the criterion ``evaluator_ref``
(the same origin as the streaming ``evaluator_name``), not the benchmark name.
"""

from benchmark.runs.common.agent_capabilities import (
    CapabilityBindingView,
    batch_trigger_execution_aliases,
    capabilities_or_default,
    declared_capabilities,
    default_batch_capabilities,
    streaming_capabilities,
)
AmapChatRubricV2 = None
from benchmark.runs.registry import BENCHMARK_SERVICES_CLASSES
SearchEvalRubric = None


def test_streaming_chain_for_dedicated_evaluator() -> None:
    """amap_agent evaluator: dedicated triage / attribution plugins + default judge + T+1 report (with descriptions)."""
    bindings = streaming_capabilities(["amap_agent"])
    by_kind = {item.kind: item for item in bindings}
    assert all(item.pipeline == "streaming" for item in bindings)
    # Each capability returns a one-sentence description along with its name
    assert all(item.description for item in bindings)
    assert "打分" in by_kind["judge"].description
    assert "分级" in by_kind["triage"].description
    assert "归因" in by_kind["attribution"].description
    assert "T+1" in by_kind["report"].description
    assert by_kind["judge"].alias == "evaluator_service_judge"
    assert "amap_agent" in (by_kind["judge"].note or "")
    assert by_kind["triage"].alias == "amap_agent_triage"
    assert by_kind["triage"].class_ref is not None
    assert by_kind["attribution"].alias == "amap_agent_attribution"
    assert "async" in (by_kind["attribution"].note or "")
    assert by_kind["report"].provider == "webdev"
    assert "T+1" in (by_kind["report"].note or "")


def test_streaming_chain_routes_by_evaluator_ref_not_benchmark_name() -> None:
    """Routing uses the set of criteria evaluator_ref values, not the benchmark name.

    A criterion of ``amap_agent_benchmark`` (streaming agent) points at
    ``amap_agent`` → the streaming chain hits the amap_agent-specific plugins.
    """
    refs = [c.evaluator_ref for c in AmapChatRubricV2.criteria()]
    assert refs and set(refs) == {"amap_agent"}
    bindings = streaming_capabilities(refs)
    by_alias = {item.alias for item in bindings}
    assert "amap_agent_triage" in by_alias
    assert "amap_agent_attribution" in by_alias
    assert "evaluator_service_attribution" not in by_alias


def test_streaming_chain_falls_back_to_evaluator_service() -> None:
    """An evaluator with no dedicated plugin registered follows the default chain (the classifier looks up the registry by name)."""
    bindings = streaming_capabilities(["search_ai_card"])
    by_kind = {item.kind: item for item in bindings}
    assert by_kind["triage"].alias is None
    assert "search_ai_card" in (by_kind["triage"].note or "")
    assert by_kind["attribution"].alias == "evaluator_service_attribution"


def test_streaming_chain_without_refs_uses_task_level_evaluator() -> None:
    """No evaluator_ref (empty rubric) → the all-default chain, routed by the task-level evaluator config."""
    bindings = streaming_capabilities([])
    by_kind = {item.kind: item for item in bindings}
    assert by_kind["triage"].alias is None
    assert by_kind["attribution"].alias == "evaluator_service_attribution"
    assert "judge_<ref>" in (by_kind["judge"].note or "")


def test_default_batch_chain_respects_local_judge_policy() -> None:
    """A rubric that declares judge_execution_policy=local → the judge note switches to the dedicated local evaluator (with a description)."""
    bindings = default_batch_capabilities(SearchEvalRubric)
    by_kind = {item.kind: item for item in bindings}
    assert all(item.pipeline == "batch" for item in bindings)
    assert all(item.description for item in bindings)
    assert "打分" in by_kind["judge"].description
    assert "归因" in by_kind["attribution"].description
    assert "报表" in by_kind["report"].description
    assert by_kind["judge"].execution == "sync"
    assert by_kind["judge"].alias is None
    assert "judge_execution_policy=local" in (by_kind["judge"].note or "")
    assert by_kind["attribution"].execution == "sync"
    assert by_kind["report"].class_ref is not None


def test_default_batch_chain_without_rubric_uses_process_default() -> None:
    """No rubric class → the process-default EvaluatorService chain (overridable by RemoteConfig)."""
    bindings = default_batch_capabilities(None)
    by_kind = {item.kind: item for item in bindings}
    assert by_kind["judge"].alias == "local_judge"
    assert "external-daily" in (by_kind["judge"].note or "")


def test_declared_capabilities_keeps_declared_pipelines() -> None:
    """A single-chain declaration is passed through as-is."""
    batch_declared = tuple(default_batch_capabilities(SearchEvalRubric))
    assert {item.pipeline for item in declared_capabilities(batch_declared)} == {"batch"}
    streaming_declared = tuple(streaming_capabilities(["amap_agent"]))
    assert {item.pipeline for item in declared_capabilities(streaming_declared)} == {
        "streaming",
    }


def test_declared_capabilities_keeps_both_pipelines_and_dedupes() -> None:
    """When both chains are attached, keep both; dedupe on the same pipeline+kind+alias+class_ref."""
    batch_judge = CapabilityBindingView(
        pipeline="batch",
        kind="judge",
        execution="sync",
        provider="local",
        alias="local_judge",
    )
    streaming_judge = CapabilityBindingView(
        pipeline="streaming",
        kind="judge",
        execution="sync",
        provider="local",
        alias="evaluator_service_judge",
    )
    mixed = [batch_judge, batch_judge, streaming_judge]
    kept = declared_capabilities(mixed)
    assert kept[:2] == [batch_judge, streaming_judge]
    assert {item.pipeline for item in kept} == {"batch", "streaming"}
    assert {item.kind for item in kept[2:]} == {"report"}


def test_capabilities_or_default_falls_back_to_batch() -> None:
    """When nothing is declared, use the batch default chain; when something is declared, the declaration wins."""
    assert {i.pipeline for i in capabilities_or_default(None)} == {"batch"}
    declared = streaming_capabilities(["amap_agent"])
    assert capabilities_or_default(declared) == declared_capabilities(declared)


def test_declared_capabilities_adds_default_report_for_every_pipeline() -> None:
    """A chain that does not explicitly declare report attaches the default class; an existing report only fills an empty class_ref."""
    judge = CapabilityBindingView(
        pipeline="streaming",
        kind="judge",
        execution="sync",
        provider="local",
        description="评分",
    )
    report_without_class = CapabilityBindingView(
        pipeline="batch",
        kind="report",
        execution="sync",
        provider="webdev",
        description="报表",
    )
    bindings = declared_capabilities([judge, report_without_class])
    reports = [item for item in bindings if item.kind == "report"]
    assert {item.pipeline for item in reports} == {"batch", "streaming"}
    assert all(item.class_ref is not None for item in reports)
    assert all(item.description for item in reports)


def test_every_component_capability_has_independent_description() -> None:
    """Every part of the agent view has its own description (describing what the business class does).

    Guard: every capability of every static component (judge/triage/attribution/report)
    must write its own functional description, and the same sentence must not be reused
    inside one component — an empty default leaves a blank line on the page capability
    chain, and a semantically confused copy makes the page unable to tell the capabilities apart.
    """
    for component in BENCHMARK_SERVICES_CLASSES:
        bindings = component.benchmark_capabilities
        assert bindings, f"{component.__name__} declares no capabilities"
        for binding in bindings:
            assert binding.description.strip(), (
                f"{component.__name__} {binding.pipeline}/{binding.kind} "
                "missing capability description"
            )
        descriptions = [binding.description for binding in bindings]
        assert len(set(descriptions)) == len(descriptions), (
            f"{component.__name__} reuses capability descriptions: {descriptions}"
        )


def test_declared_capabilities_filters_unknown_pipeline() -> None:
    """An illegal pipeline declaration is filtered out (defense: values other than batch / streaming are not passed through)."""

    class _Odd:
        pipeline = "unknown"
        kind = "judge"
        execution = "sync"
        provider = "local"

    assert declared_capabilities([_Odd()]) == []  # type: ignore[list-item]


def test_batch_trigger_aliases_prefer_batch_chain() -> None:
    """Batch-chain aliases take priority; streaming aliases do not leak across."""
    aliases = batch_trigger_execution_aliases(
        [
            CapabilityBindingView(
                pipeline="batch",
                kind="judge",
                execution="sync",
                provider="local",
                alias="local_judge",
            ),
            CapabilityBindingView(
                pipeline="batch",
                kind="attribution",
                execution="none",
                provider="none",
                alias="no_attribution",
            ),
        ],
    )
    assert aliases == {
        "judge_class": "local_judge",
        "attribution_class": "no_attribution",
    }


def test_batch_trigger_aliases_fallback_streaming_local_to_evaluator_service() -> None:
    """When only the streaming local judge exists, a batch trigger falls back to evaluator_service_judge so Aone is not called silently."""
    aliases = batch_trigger_execution_aliases(streaming_capabilities(["amap_agent"]))
    assert aliases == {"judge_class": "evaluator_service_judge"}


def test_batch_trigger_aliases_empty_without_capabilities() -> None:
    assert batch_trigger_execution_aliases(()) == {}
    assert batch_trigger_execution_aliases(None) == {}


def test_batch_trigger_aliases_prefer_batch_on_dual_pipeline() -> None:
    """Dual-chain agent: a batch trigger takes the unified evaluator_service_* alias and does not cross into streaming."""
    from benchmark.runs.example.services import ExampleBenchmarkServices

    aliases = batch_trigger_execution_aliases(
        ExampleBenchmarkServices.benchmark_capabilities,
    )
    assert aliases == {
        "judge_class": "evaluator_service_judge",
        "attribution_class": "evaluator_service_attribution",
    }


def test_example_dual_pipeline_unified_evaluator_service_aliases() -> None:
    """On example, both the batch and streaming chains display evaluator_service_* for judge and attribution."""
    from benchmark.runs.example.services import ExampleBenchmarkServices

    by_pipe_kind = {
        (item.pipeline, item.kind): item for item in ExampleBenchmarkServices.benchmark_capabilities
    }
    assert by_pipe_kind[("batch", "judge")].alias == "evaluator_service_judge"
    assert by_pipe_kind[("batch", "attribution")].alias == ("evaluator_service_attribution")
    assert by_pipe_kind[("streaming", "judge")].alias == "evaluator_service_judge"
    assert by_pipe_kind[("streaming", "attribution")].alias == ("evaluator_service_attribution")
