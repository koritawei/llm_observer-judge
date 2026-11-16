"""Shared data contract for badcase attribution.

This module contains only leaf models that the attribution runtime and the result schema can both depend on, so
``badcase_attribution.schemas`` does not depend back on ``triage_attribution``.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ConclusionType = Literal[
    "agent_badcase",
    "judge_false_positive",
    "eval_rule_issue",
    "dataset_issue",
    "non_attributable_outcome",
    "ambiguous_root_cause",
    "insufficient_evidence",
]
CONCLUSION_TYPES: tuple[ConclusionType, ...] = (
    "agent_badcase",
    "judge_false_positive",
    "eval_rule_issue",
    "dataset_issue",
    "non_attributable_outcome",
    "ambiguous_root_cause",
    "insufficient_evidence",
)

_OTHER_CODE_RE = re.compile(r"^other:[a-z][a-z0-9_]{0,79}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,79}$")
_SYMPTOM_RE = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
_MAX_DETAILS_DEPTH = 4
_MAX_DETAILS_ITEMS = 30

CandidateSurface = Literal[
    "tool_data",
    "tool_behavior",
    "model_behavior",
    "prompt_context",
    "orchestration",
    "infra",
    "evaluation",
    "dataset",
    "other",
]


class CandidateRootCause(BaseModel):
    """An auditable `other:` root-cause candidate, not a routable formal root cause."""

    raw_code: str
    title: str = Field(min_length=1, max_length=120)
    surface: CandidateSurface
    component: str = Field(default="", max_length=80)
    symptom: str = Field(min_length=1, max_length=80)
    details: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(min_length=1)
    reasoning: str = Field(default="", max_length=500)
    agent_scope: str = Field(default="", max_length=40)

    @field_validator("raw_code")
    @classmethod
    def validate_raw_code(cls, value: str) -> str:
        normalized = value.strip()
        if not _OTHER_CODE_RE.fullmatch(normalized):
            raise ValueError("raw_code must match other:<snake_case_slug>")
        return normalized

    @field_validator("title", "reasoning", "agent_scope")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("component")
    @classmethod
    def validate_component(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized and not _IDENTIFIER_RE.fullmatch(normalized):
            raise ValueError("component must be a normalized short identifier")
        return normalized

    @field_validator("symptom")
    @classmethod
    def validate_symptom(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SYMPTOM_RE.fullmatch(normalized):
            raise ValueError("symptom must be snake_case")
        return normalized

    @field_validator("evidence_refs")
    @classmethod
    def normalize_evidence_refs(cls, values: list[str]) -> list[str]:
        refs = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
        if not refs:
            raise ValueError("evidence_refs must contain at least one reference")
        return refs

    @field_validator("details")
    @classmethod
    def validate_json_safe_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json_safe(value, depth=0)
        return value

    def candidate_key(self) -> str:
        """Return a deterministic L1 aggregation key inside the scope; do not merge across Agents."""

        scope = self.agent_scope or "unscoped"
        component = self.component or "none"
        return f"v1:{scope}:{self.surface}:{component}:{self.symptom}"

    def with_agent_scope(self, agent_scope: str) -> "CandidateRootCause":
        """The domain adapter injects a scope that must not be merged across domains."""

        return self.model_copy(update={"agent_scope": agent_scope.strip().lower()})


def _validate_json_safe(value: object, *, depth: int) -> None:
    if depth > _MAX_DETAILS_DEPTH:
        raise ValueError("details nesting must not exceed 4 levels")
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, list):
        if len(value) > _MAX_DETAILS_ITEMS:
            raise ValueError("details lists must contain at most 30 items")
        for item in value:
            _validate_json_safe(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_DETAILS_ITEMS:
            raise ValueError("details objects must contain at most 30 keys")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("details keys must be strings")
            _validate_json_safe(item, depth=depth + 1)
        return
    raise ValueError("details must be JSON-safe")


__all__ = [
    "CONCLUSION_TYPES",
    "CandidateRootCause",
    "CandidateSurface",
    "ConclusionType",
]
