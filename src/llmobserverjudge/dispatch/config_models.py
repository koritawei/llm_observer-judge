"""Dispatch YAML config models (SDD-EV-27)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from llmobserverjudge.contracts.models import Priority


class AnnotationTrigger(StrEnum):
    ALWAYS = "always"
    LOW_CONFIDENCE = "low_confidence"
    NEVER = "never"


class RoutingRule(BaseModel):
    condition: str
    judge_agents: list[str] = Field(min_length=1)
    priority: Priority = Priority.P2
    model: str | None = None


class EvalRoutingRule(BaseModel):
    eval_agent_id: str
    bad_case_threshold: float = 0.5
    attribution_agent: str = "attributor_default"
    attribution_priority: Priority = Priority.P1
    annotation_trigger: AnnotationTrigger = AnnotationTrigger.LOW_CONFIDENCE


class DispatchSection(BaseModel):
    default_model: str = "default-judge"
    publish_priority_shards: bool = False
    routing_rules: list[RoutingRule] = Field(min_length=1)
    eval_routing_rules: list[EvalRoutingRule] = Field(default_factory=list)

    @field_validator("routing_rules")
    @classmethod
    def _must_have_always_or_rules(cls, rules: list[RoutingRule]) -> list[RoutingRule]:
        if not rules:
            raise ValueError("routing_rules must not be empty")
        return rules


class DispatchFileConfig(BaseModel):
    version: int = 1
    dispatch: DispatchSection
