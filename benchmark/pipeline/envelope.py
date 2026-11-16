"""Unified evaluation envelope ``agent_eval_extra.v1``: a stable output contract across businesses."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from benchmark.pipeline.contracts import EvidenceFact
from benchmark.pipeline.signals import DimensionResult, EvaluationSignal, JsonValue

EnvelopeSchemaVersion = Literal["agent_eval_extra.v1"]


class MetricObservation(BaseModel):
    """Independent runtime observation metric that does not take part in quality scoring."""

    model_config = ConfigDict(extra="forbid")

    metric_key: str
    value: float | int | None = None
    unit: str
    status: Literal["within_target", "exceeded", "unknown"]
    threshold: float | int | None = None
    threshold_ref: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    collector_name: str
    source_field: str | None = None
    sample_scope: str | None = None

    @field_validator("evidence_refs", mode="after")
    @classmethod
    def _dedupe_strings(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in value if item))


class DiagnosticFinding(BaseModel):
    """Engineering diagnostic that is not scored directly."""

    model_config = ConfigDict(extra="forbid")

    finding_key: str
    severity: Literal["info", "warning", "error"]
    status: Literal["active", "recovered", "unknown"]
    reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    source_name: str
    related_dimensions: list[str] = Field(default_factory=list)

    @field_validator("evidence_refs", "related_dimensions", mode="after")
    @classmethod
    def _dedupe_strings(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in value if item))


class FailureRelation(BaseModel):
    """Strongly typed link among failure mode, affected dimension, and responsibility domain."""

    model_config = ConfigDict(extra="forbid")

    failure_mode: str
    related_dimensions: list[str] = Field(default_factory=list)
    responsibility: Literal[
        "model", "tool", "orchestration", "data", "rendering", "engineering", "unknown"
    ]
    impact: Literal["blocking", "degraded", "recovered", "observed"]
    evidence_refs: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("related_dimensions", "evidence_refs", "source_refs", mode="after")
    @classmethod
    def _dedupe_strings(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in value if item))


class BusinessExtension(BaseModel):
    """Versioned container for business-specific fields."""

    model_config = ConfigDict(extra="forbid")

    extension_type: str
    extension_version: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentEvalEnvelope(BaseModel):
    """Platform-wide unified evaluation output envelope."""

    model_config = ConfigDict(extra="forbid")

    schema_version: EnvelopeSchemaVersion = "agent_eval_extra.v1"
    business_id: str
    scenario: str = ""
    profile_id: str = ""
    profile_version: str = ""
    catalog_version: str = ""
    config_digest: str = ""
    trace_id: str = ""
    prompt_version: str = ""
    primary_score: float | None = None
    quality_score: float | None = None
    score_policy_ref: str | None = None
    summary_reason: str = ""
    dimension_results: list[DimensionResult] = Field(default_factory=list)
    metric_observations: list[MetricObservation] = Field(default_factory=list)
    diagnostic_findings: list[DiagnosticFinding] = Field(default_factory=list)
    evidence_facts: list[EvidenceFact] = Field(default_factory=list)
    evaluation_signals: list[EvaluationSignal] = Field(default_factory=list)
    failure_relations: list[FailureRelation] = Field(default_factory=list)
    business_extension: BusinessExtension | None = None


__all__ = [
    "AgentEvalEnvelope",
    "BusinessExtension",
    "DiagnosticFinding",
    "EnvelopeSchemaVersion",
    "FailureRelation",
    "JsonValue",
    "MetricObservation",
]
