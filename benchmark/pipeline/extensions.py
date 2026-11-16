"""Strongly typed business extensions for the unified envelope."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmark.pipeline.envelope import BusinessExtension

VEHICLE_EXTENSION_TYPE = "vehicle_eval"
VEHICLE_EXTENSION_VERSION = "v3"
VUI_EXTENSION_TYPE = "vui_eval"
VUI_EXTENSION_VERSION = "v2"


class VehicleEvalExtensionV3(BaseModel):
    """Versioned payload that keeps the existing Vehicle extra read view."""

    model_config = ConfigDict(extra="allow")

    score: float
    reason: str = ""
    dimension_results: list[dict[str, Any]] = Field(default_factory=list)
    checker_results: list[dict[str, Any]] = Field(default_factory=list)
    evidence_facts: list[dict[str, Any]] = Field(default_factory=list)
    judge_scores: list[dict[str, Any]] = Field(default_factory=list)
    hard_rule_hits: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    failure_summary: dict[str, Any] = Field(default_factory=dict)
    route_plan: dict[str, Any] | None = None
    no_decision: bool = False
    batch_tag: str = ""
    case_meta: dict[str, Any] = Field(default_factory=dict)
    tag: str = ""
    query: str = ""
    turn_index: str = ""

    def to_business_extension(self) -> BusinessExtension:
        return BusinessExtension(
            extension_type=VEHICLE_EXTENSION_TYPE,
            extension_version=VEHICLE_EXTENSION_VERSION,
            payload=self.model_dump(mode="json"),
        )


class VuiEvalExtensionV2(BaseModel):
    """Keep VUI-specific fields; runtime metrics such as latency live at the top of the unified envelope."""

    model_config = ConfigDict(extra="allow")

    judge_ran: bool = False
    completion_evidence: dict[str, Any] | list[Any] | None = None
    output_snapshot: dict[str, Any] | None = None
    user_query: str = ""

    def to_business_extension(self) -> BusinessExtension:
        return BusinessExtension(
            extension_type=VUI_EXTENSION_TYPE,
            extension_version=VUI_EXTENSION_VERSION,
            payload=self.model_dump(mode="json"),
        )


__all__ = [
    "VEHICLE_EXTENSION_TYPE",
    "VEHICLE_EXTENSION_VERSION",
    "VUI_EXTENSION_TYPE",
    "VUI_EXTENSION_VERSION",
    "VehicleEvalExtensionV3",
    "VuiEvalExtensionV2",
]
