"""Unified agent names and assembly of the full evaluator catalog.

Name rule (the only source of truth for the external view):

- **agent name** is the only external name. Both the benchmark name and the streaming
  ``evaluator_name`` normalize to it: ``example_benchmark`` → ``example``,
  ``amap_agent_benchmark`` → ``amap_agent``, ``search_eval`` →
  ``search_ai_card`` (its evaluator stem is the service_name). Every other benchmark
  name, and every dynamic registration name, is already the agent name.
- Execution-side ids do not change: triggers still use the benchmark name (the ``name``
  field is kept), components stay ``judge_<agent>`` / ``attribution_<agent>``, and the
  streaming config ``evaluatorName`` is already the agent name. This module converts
  among the three so each side does not keep its own spelling.

Full assembly: configured evaluators (judge / attribution components + the triage
classifier registry) are grouped by agent name. Names already covered by a benchmark
row are not listed again. Uncovered ones (for example ``vehicle_agent`` /
``competitor_pairwise`` / ``memory_pairwise``) are assembled into agent rows that
expose only evaluator capabilities.
"""

from dataclasses import dataclass

from benchmark.domain.agent_names import canonical_agent_name
from benchmark.runs.common.agent_capabilities import CapabilityBindingView
from benchmark.runs.common.execution_classes import (
    STREAMING_ATTRIBUTION_CLASS_REFS,
    STREAMING_JUDGE_CLASS_REFS,
    STREAMING_TRIAGE_CLASS_REFS,
)

_JUDGE_PREFIX = "judge_"
_ATTRIBUTION_PREFIX = "attribution_"
_GENERIC_JUDGE_DESCRIPTION = "An agent-eval evaluator."


@dataclass(frozen=True)
class EvaluatorComponents:
    """Evaluator components configured under one agent name (judge / attribution / triage)."""

    judge_description: str | None = None
    has_attribution: bool = False
    has_triage: bool = False


def _component_stem(component_name: str, prefix: str) -> str | None:
    if component_name.startswith(prefix) and len(component_name) > len(prefix):
        return component_name[len(prefix) :]
    return None


def configured_evaluators() -> dict[str, EvaluatorComponents]:
    """Group every configured evaluator by agent name (the registries are the only source).

    Judge uses component names and class descriptions from ``business.evaluators.registry``,
    attribution uses ``business.agents.registry``, and triage uses the classifier registry
    keys. Imports are lazy: the registry eager-import chain is heavy and must not run at
    module import time.
    """
    from benchmark.evaluators import registry as evaluator_registry  # noqa: PLC0415
    from benchmark.attribution import registry as attribution_registry  # noqa: PLC0415
    from benchmark.attribution.classifiers_registry import (  # noqa: PLC0415
        triage_classifiers,
    )

    judges: dict[str, str | None] = {}
    for class_name in evaluator_registry.__all__:
        cls = getattr(evaluator_registry, class_name)
        meta = getattr(cls, "__agent_component__", None)
        component = meta.get("name") if isinstance(meta, dict) else None
        stem = _component_stem(component or "", _JUDGE_PREFIX)
        if stem is None:
            continue
        description = getattr(cls, "default_description", None)
        judges[stem] = (
            description
            if isinstance(description, str) and description != _GENERIC_JUDGE_DESCRIPTION
            else None
        )

    attributions: set[str] = set()
    for class_name in attribution_registry.__all__:
        cls = getattr(attribution_registry, class_name)
        meta = getattr(cls, "__agent_component__", None)
        component = meta.get("name") if isinstance(meta, dict) else None
        stem = _component_stem(component or "", _ATTRIBUTION_PREFIX)
        if stem is not None:
            attributions.add(stem)

    triage_stems = set(triage_classifiers())
    stems = sorted(set(judges) | attributions | triage_stems)
    return {
        stem: EvaluatorComponents(
            judge_description=judges.get(stem),
            has_attribution=stem in attributions,
            has_triage=stem in triage_stems,
        )
        for stem in stems
    }


def evaluator_capabilities(
    agent_name: str,
    components: EvaluatorComponents,
) -> list[CapabilityBindingView]:
    """Capability chain for an evaluator-only agent (streaming shape: judge is required; triage/attribution follow
    registration).
    """
    bindings: list[CapabilityBindingView] = [
        CapabilityBindingView(
            pipeline="streaming",
            kind="judge",
            execution="sync",
            provider="local",
            description=(
                components.judge_description
                or f"对 {agent_name} 的线上 trace 执行 judge 评分（组件 judge_{agent_name}）"
            ),
            alias="evaluator_service_judge",
            class_ref=STREAMING_JUDGE_CLASS_REFS.get("evaluator_service_judge"),
            note=f"按 agent name 路由 judge_{agent_name}",
        ),
    ]
    if components.has_triage:
        alias = f"{agent_name}_triage"
        bindings.append(
            CapabilityBindingView(
                pipeline="streaming",
                kind="triage",
                execution="sync",
                provider="local",
                description=f"按 {agent_name} 分流规则对评过 trace 分级",
                alias=alias if alias in STREAMING_TRIAGE_CLASS_REFS else None,
                class_ref=STREAMING_TRIAGE_CLASS_REFS.get(alias),
                note=f"{agent_name} 分流分类器（classifiers_registry）",
            ),
        )
    if components.has_attribution:
        dedicated = f"{agent_name}_attribution"
        alias = (
            dedicated
            if dedicated in STREAMING_ATTRIBUTION_CLASS_REFS
            else "evaluator_service_attribution"
        )
        bindings.append(
            CapabilityBindingView(
                pipeline="streaming",
                kind="attribution",
                execution="sync",
                provider="local",
                description=f"对 {agent_name} 的不满分 trace 取证归因",
                alias=alias,
                class_ref=STREAMING_ATTRIBUTION_CLASS_REFS.get(alias),
                note=f"归因组件 attribution_{agent_name}",
            ),
        )
    return bindings


__all__ = [
    "EvaluatorComponents",
    "canonical_agent_name",
    "configured_evaluators",
    "evaluator_capabilities",
]
