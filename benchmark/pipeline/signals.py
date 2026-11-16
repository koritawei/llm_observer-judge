"""Unified evaluation signal contracts shared by checkers, sub judges and aggregators.

Business-agnostic: ``dimension`` is a plain ``str``. Each business domain declares its own dimension set and keeps it type-safe internally;
this layer makes no assumption about dimension values.
"""

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

SignalStatus = Literal["pass", "fail", "partial", "unknown", "ambiguous", "not_applicable"]
SignalSourceType = Literal["checker", "sub_judge", "aggregate"]
JudgeDecision = Literal["block_judge", "skip_judge", "run_judge", "evidence_only"]
type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


class DimensionResult(BaseModel):
    """Single-dimension conclusion produced by a checker, sub judge, or aggregate."""

    dimension: str
    status: SignalStatus
    score_hint: float | None = Field(
        default=None, validation_alias=AliasChoices("score_hint", "score")
    )
    reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    hard_rule_hits: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class EvaluationSignal(BaseModel):
    """Unified signal shared by rule checkers and LLM judges."""

    source_type: SignalSourceType
    source_name: str = Field(validation_alias=AliasChoices("source_name", "judge_name"))
    dimensions: list[str] = Field(default_factory=list)
    status: SignalStatus = "unknown"
    score_hint: float | None = None
    reason: str = ""
    dimension_results: list[DimensionResult] = Field(default_factory=list)
    hard_rule_hits: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    decision: JudgeDecision | None = None
    failure_modes: list[str] = Field(default_factory=list)
    signals: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("hard_rule_hits", "risk_flags", "evidence_refs", "failure_modes", mode="after")
    @classmethod
    def _dedupe_strings(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in value if item))


class CheckerResult(EvaluationSignal):
    """Checker-specific output type that keeps the unified field shape."""

    source_type: Literal["checker"] = "checker"


class SubJudgeResult(EvaluationSignal):
    """Sub-judge-specific output type that keeps the unified field shape."""

    source_type: Literal["sub_judge"] = "sub_judge"


class JudgePlan(BaseModel):
    """Judge routing plan after checker signals."""

    run_sub_judges: list[str] = Field(default_factory=list)
    skip_sub_judges: list[str] = Field(default_factory=list)
    judge_input_view: dict[str, JsonValue] = Field(default_factory=dict)
    rationale: str = ""
    generic_fallback: bool = False
    execution_mode: Literal["sequential", "fan_out"] = Field(default="sequential", exclude=True)
