"""Stable cross-business data contract for the evaluation stage."""

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from benchmark.pipeline.signals import (
    DimensionResult,
    EvaluationSignal,
    JudgePlan,
)

EvidenceConfidence = Literal["high", "medium", "low"]


def generate_fact_id(
    extractor_name: str,
    fact_type: str,
    scope: str | None,
    evidence_refs: list[str],
    neutral_value: object = None,
) -> str:
    """Build a stable id from fact content that does not depend on execution order."""

    canonical = json.dumps(
        {
            "extractor_name": extractor_name,
            "fact_type": fact_type,
            "scope": scope or "",
            "evidence_refs": sorted(evidence_refs),
            "neutral_value": neutral_value,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{fact_type}:{digest}"


class EvidenceFact(BaseModel):
    """Neutral fact produced by an Evidence Extractor, carrying no score or root-cause conclusion."""

    fact_id: str = ""
    extractor_name: str
    fact_type: str
    scope: str | None = None
    neutral_value: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    risk_hints: list[str] = Field(default_factory=list)
    confidence: EvidenceConfidence = "medium"

    @field_validator("evidence_refs", "risk_hints", mode="after")
    @classmethod
    def _dedupe_strings(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in value if item))

    @model_validator(mode="after")
    def _fill_fact_id(self) -> "EvidenceFact":
        if not self.fact_id:
            self.fact_id = generate_fact_id(
                self.extractor_name,
                self.fact_type,
                self.scope,
                self.evidence_refs,
                self.neutral_value,
            )
        return self


class EvaluationSnapshot(BaseModel):
    """Internal standard evaluation snapshot consumed by later processing."""

    schema_version: str
    logic_name: str
    trace_id: str = ""
    score: float | None = None
    reason: str = ""
    dimension_results: list[DimensionResult] = Field(default_factory=list)
    checker_signals: list[EvaluationSignal] = Field(default_factory=list)
    judge_signals: list[EvaluationSignal] = Field(default_factory=list)
    evidence_facts: list[EvidenceFact] = Field(default_factory=list)
    hard_rule_hits: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    route_plan: JudgePlan | None = None
    no_decision: bool = False
    needs_human_review: bool = False

    @field_validator("hard_rule_hits", "risk_flags", "failure_modes", mode="after")
    @classmethod
    def _dedupe_strings(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in value if item))


__all__ = [
    "EvaluationSnapshot",
    "EvidenceConfidence",
    "EvidenceFact",
    "generate_fact_id",
]
