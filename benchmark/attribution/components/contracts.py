"""Cross-business data contract for the five-stage attribution chain.

Shared models carry only fields that are stable across businesses. A business package narrows the schema, signal source, and gate
enums by inheritance, and does not introduce Vehicle/VUI dimensions, taxonomy, or failure modes at this layer.
"""

from typing import Any, Literal, Protocol

from benchmark.libs.agenthost.agents.request import AgentRequest
from pydantic import BaseModel, Field

from benchmark.attribution.badcase.contracts import ConclusionType
from benchmark.attribution.badcase.schemas import AttributionResult

GateVerdictKind = Literal[
    "rule_attributable",
    "needs_evidence",
    "needs_llm",
    "ambiguous",
    "unreachable",
    "insufficient",
]


InvestigationMode = Literal["none", "single_llm", "react"]
AttributionStage = Literal[
    "build_context",
    "match_rules",
    "gate_evidence",
    "investigate",
    "resolve_decision",
    "validate_final",
]


class AttributionContextBase(BaseModel):
    """Shared fields of the read-only attribution context built from an evaluation result."""

    trace_id: str = ""
    project_id: str = ""
    schema_version: str = ""
    score: float | None = None
    judge_reason: str = ""
    hard_rule_hits: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    degraded: bool = False


class RuleSignalBase(BaseModel):
    """Shared fields of a candidate root-cause signal produced by the rule-matching stage."""

    rule_id: str
    base_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str = ""


class GateVerdictBase(BaseModel):
    """Shared verdict shape of the evidence gate."""

    verdict: GateVerdictKind
    reason: str = ""
    hypotheses: list[str] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    """Business-agnostic output of the investigation stage; raw model content is left to the business parser."""

    mode: InvestigationMode
    evidence_refs: list[str] = Field(default_factory=list)
    observations: dict[str, Any] = Field(default_factory=dict)
    raw_output: Any = None


class AttributionContextAdapter(Protocol):
    """Convert an existing request or evaluation envelope into a domain attribution context."""

    def adapt(self, request: AgentRequest) -> AttributionContextBase: ...


class RuleMatcher(Protocol):
    """Produce candidate root-cause signals from domain rules."""

    def match(self, context: AttributionContextBase) -> list[RuleSignalBase]: ...


class EvidenceGate(Protocol):
    """Decide whether the existing evidence can support a direct rule verdict."""

    def decide(
        self,
        context: AttributionContextBase,
        signals: list[RuleSignalBase],
    ) -> GateVerdictBase: ...


class InvestigationRunner(Protocol):
    """Add investigation evidence in none, single_llm, or react mode."""

    async def investigate(
        self,
        context: AttributionContextBase,
        verdict: GateVerdictBase,
    ) -> InvestigationResult: ...


class DecisionResolver(Protocol):
    """Resolve rules, the gate, and investigation results into a platform attribution result."""

    def resolve(
        self,
        context: AttributionContextBase,
        signals: list[RuleSignalBase],
        verdict: GateVerdictBase,
        investigation: InvestigationResult | None,
    ) -> AttributionResult: ...


class FinalValidationGate(Protocol):
    """Apply reference, confidence, and dispatch gates to the final attribution result."""

    def validate(
        self,
        context: AttributionContextBase,
        result: AttributionResult,
    ) -> AttributionResult: ...


__all__ = [
    "AttributionContextAdapter",
    "AttributionContextBase",
    "AttributionStage",
    "ConclusionType",
    "DecisionResolver",
    "EvidenceGate",
    "FinalValidationGate",
    "GateVerdictBase",
    "GateVerdictKind",
    "InvestigationMode",
    "InvestigationResult",
    "InvestigationRunner",
    "RuleMatcher",
    "RuleSignalBase",
]
