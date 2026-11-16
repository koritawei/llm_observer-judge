"""Evaluation contracts and checker primitives reused across businesses."""

from benchmark.pipeline.checker import BaseChecker, run_checkers
from benchmark.pipeline.checker_registry import (
    CheckerDescriptor,
    CheckerRegistry,
    CheckerSpec,
    ComponentDescriptor,
    ComponentKind,
    LegacyCheckerSetAdapter,
)
from benchmark.pipeline.contracts import EvaluationSnapshot, EvidenceFact
from benchmark.pipeline.envelope import (
    AgentEvalEnvelope,
    BusinessExtension,
    DiagnosticFinding,
    FailureRelation,
    MetricObservation,
)
from benchmark.pipeline.evidence import (
    EvidenceProvider,
    EvidenceProviderRegistry,
)
from benchmark.pipeline.normalize import (
    normalize_eval_extra,
    normalize_vehicle_extra,
    normalize_vui_extra,
)
from benchmark.pipeline.profiles import (
    DimensionBinding,
    ExtensionBinding,
    ProfileSpec,
    apply_runtime_overrides,
    assemble_checkers,
    assemble_evidence_providers,
    compute_config_digest,
    load_profile,
    validate_profile,
)
from benchmark.pipeline.runner import (
    EnvelopeIdentity,
    EvaluationPipelineHooks,
    build_evaluation_graph,
    envelope_identity_from_profile,
    skipped,
)
from benchmark.pipeline.signals import (
    CheckerResult,
    DimensionResult,
    EvaluationSignal,
    JsonValue,
    JudgePlan,
)

__all__ = [
    "AgentEvalEnvelope",
    "BaseChecker",
    "BusinessExtension",
    "CheckerDescriptor",
    "CheckerRegistry",
    "CheckerResult",
    "CheckerSpec",
    "ComponentDescriptor",
    "ComponentKind",
    "DiagnosticFinding",
    "DimensionBinding",
    "DimensionResult",
    "EnvelopeIdentity",
    "EvaluationPipelineHooks",
    "EvaluationSignal",
    "EvaluationSnapshot",
    "EvidenceFact",
    "EvidenceProvider",
    "EvidenceProviderRegistry",
    "ExtensionBinding",
    "FailureRelation",
    "JudgePlan",
    "JsonValue",
    "LegacyCheckerSetAdapter",
    "MetricObservation",
    "ProfileSpec",
    "apply_runtime_overrides",
    "assemble_checkers",
    "assemble_evidence_providers",
    "build_evaluation_graph",
    "compute_config_digest",
    "envelope_identity_from_profile",
    "load_profile",
    "normalize_eval_extra",
    "normalize_vehicle_extra",
    "normalize_vui_extra",
    "run_checkers",
    "skipped",
    "validate_profile",
]
