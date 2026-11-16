"""Case-display view models and request context (FUNC-PARAMS-001)."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class CaseJudgeDimension(BaseModel):
    """One dimension score row."""

    id: str
    score: float | None = None
    reason: str = ""


class CaseJudgeView(BaseModel):
    """Judge column: total score + overall reason + dimension table."""

    score: float | str | None = None
    reason: str = ""
    dimensions: list[CaseJudgeDimension] = Field(default_factory=list)


class CaseAttributionView(BaseModel):
    """Attribution column: streaming table fields or batch markdown."""

    format: Literal["table", "markdown"] = "table"
    category: str | None = None
    confidence: float | None = None
    evidence: str | None = None
    reasoning: str | None = None
    markdown: str | None = None


@dataclass(frozen=True, slots=True)
class CaseDisplayContext:
    """Normalization context."""

    notification_source: str
    benchmark_name: str = ""


@dataclass(frozen=True, slots=True)
class CaseDisplayRequest:
    """Plugin input (FUNC-PARAMS-001): case and context folded into one object."""

    case: Mapping[str, object]
    context: CaseDisplayContext


@dataclass(frozen=True, slots=True)
class CaseDisplayResult:
    """Plugin output: three columns the WebUI can render directly."""

    output: str
    judge: CaseJudgeView | None
    attribution: CaseAttributionView | None


__all__ = [
    "CaseAttributionView",
    "CaseDisplayContext",
    "CaseDisplayRequest",
    "CaseDisplayResult",
    "CaseJudgeDimension",
    "CaseJudgeView",
]
