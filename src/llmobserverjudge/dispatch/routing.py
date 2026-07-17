"""Routing rules — YAML-driven match (SDD-EV-27)."""

from __future__ import annotations

from typing import Any

from llmobserverjudge.contracts.models import EvalConfig, Priority
from llmobserverjudge.dispatch.conditions import evaluate_condition
from llmobserverjudge.dispatch.config_loader import default_dispatch_section
from llmobserverjudge.dispatch.config_models import (
    AnnotationTrigger,
    DispatchSection,
    EvalRoutingRule,
)


def match_eval_configs(
    trace_summary: dict[str, Any],
    config: DispatchSection | None = None,
) -> tuple[list[EvalConfig], Priority]:
    """First matching routing_rule wins; falls back to last `always` if present."""
    section = config or default_dispatch_section()
    matched: list[EvalConfig] | None = None
    priority = Priority.P2

    for rule in section.routing_rules:
        if not evaluate_condition(rule.condition, trace_summary):
            continue
        model = rule.model or section.default_model
        matched = [
            EvalConfig(agent_id=agent_id, model=model, prompt_id="prompt_default_judge")
            for agent_id in rule.judge_agents
        ]
        priority = rule.priority
        break

    if matched is None:
        # safety net: last always rule or empty
        for rule in reversed(section.routing_rules):
            if rule.condition.strip().lower() == "always":
                model = rule.model or section.default_model
                matched = [
                    EvalConfig(agent_id=a, model=model, prompt_id="prompt_default_judge")
                    for a in rule.judge_agents
                ]
                priority = rule.priority
                break
        if matched is None:
            matched = [
                EvalConfig(
                    agent_id="judge_quality",
                    model=section.default_model,
                    prompt_id="prompt_default_judge",
                )
            ]
            priority = Priority.P2

    return matched, priority


def find_eval_routing_rule(
    eval_agent_id: str, config: DispatchSection | None = None
) -> EvalRoutingRule | None:
    section = config or default_dispatch_section()
    for rule in section.eval_routing_rules:
        if rule.eval_agent_id == eval_agent_id:
            return rule
    return None


def should_trigger_annotation(
    trigger: AnnotationTrigger,
    *,
    score: float | None,
    confidence: float | None,
    threshold: float,
) -> bool:
    if trigger == AnnotationTrigger.NEVER:
        return False
    if trigger == AnnotationTrigger.ALWAYS:
        return True
    # low_confidence: score below threshold OR confidence < 0.5
    if score is not None and score < threshold:
        return True
    if confidence is not None and confidence < 0.5:
        return True
    return False


def should_trigger_attribution(
    rule: EvalRoutingRule,
    *,
    score: float | None,
    is_bad_case: bool | None = None,
) -> bool:
    """Attribution only for bad cases — good cases are skipped.

    Prefer explicit ``is_bad_case``; fall back to score < threshold when unset.
    """
    if is_bad_case is False:
        return False
    if is_bad_case is True:
        return True
    if score is None:
        return False
    return score < rule.bad_case_threshold
