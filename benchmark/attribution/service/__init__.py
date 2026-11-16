"""TriageAttributionService submodule entry — exposes the public contract."""

from benchmark.records.enums import TriageLevel

from .classifier import TriageClassifier
from .schema import TriageInput, TriageVerdict
from .service import (
    AfterRunHook,
    NotificationAfterRunHook,
    BeforeRunDecision,
    BeforeRunHook,
    TriageAttributionService,
    TriageBeforeRunHook,
    TriageOutcome,
)
from .triage import (
    TriageConfig,
    TriageDecision,
    triage_decide,
)

__all__ = [
    "TriageAttributionService",
    "TriageOutcome",
    "BeforeRunHook",
    "AfterRunHook",
    "BeforeRunDecision",
    "TriageBeforeRunHook",
    "NotificationAfterRunHook",
    "TriageConfig",
    "TriageDecision",
    "triage_decide",
    "TriageClassifier",
    "TriageInput",
    "TriageVerdict",
    "TriageLevel",
]
