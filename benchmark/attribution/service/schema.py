"""Triage field protocol — the input a classifier accepts and the verdict it returns.

Triage maps one evaluation straight onto a ``TriageLevel`` (no 0~10 normalization). The verdict
carries only the matched level plus a ``should_attribute`` flag set by the classifier; the
classification logic and the "why" stay in the classifier and logs, not in the schema. ``label``
is a reserved field (a discriminative tag) with no upstream producer yet.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from benchmark.records.enums import TriageLevel

#: Required non-empty string for fields that form the triage identity.
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class TriageInput(BaseModel):
    """A single evaluation handed to a triage classifier (strict)."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(description="Business trace identifier")
    evaluator_name: NonEmptyStr = Field(description="Evaluator that produced this evaluation")
    project_id: NonEmptyStr = Field(description="Business line identifier")
    score: float | None = Field(default=None, description="Judge score (not normalized)")
    reason: str | None = Field(default=None, description="Judge reasoning")
    label: str | None = Field(default=None, description="Discriminative label (reserved)")
    risk_flags: list[str] = Field(
        default_factory=list, description="Risk flags from evaluator extra"
    )
    no_decision: bool = Field(default=False, description="Evaluator could not reach a decision")
    needs_human_review: bool = Field(
        default=False,
        description=(
            "Evaluator flagged for human review "
            "(transparent pass-through, does not affect triage level)"
        ),
    )


class TriageVerdict(BaseModel):
    """One triage verdict — the matched level + whether it warrants downstream attribution."""

    model_config = ConfigDict(extra="forbid")

    level: TriageLevel = Field(description="The matched level")
    should_attribute: bool = Field(description="Whether it warrants downstream attribution")
