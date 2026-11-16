"""Shared contract tests for the attribution stage."""

from typing import get_args

from benchmark.attribution.badcase.contracts import CONCLUSION_TYPES
from benchmark.attribution.components.contracts import (
    AttributionContextBase,
    ConclusionType,
    GateVerdictBase,
    InvestigationResult,
    RuleSignalBase,
)


def test_attribution_stage_contracts_support_single_llm_investigation() -> None:
    context = AttributionContextBase(trace_id="trace-1", schema_version="snapshot.v1")
    signal = RuleSignalBase(rule_id="rule-a", evidence_refs=["fact:a"])
    verdict = GateVerdictBase(verdict="needs_llm", hypotheses=["cause-a"])
    investigation = InvestigationResult(
        mode="single_llm",
        evidence_refs=signal.evidence_refs,
        observations={"hypotheses": verdict.hypotheses},
    )

    assert context.trace_id == "trace-1"
    assert investigation.mode == "single_llm"
    assert investigation.evidence_refs == ["fact:a"]


def test_conclusion_type_reuses_schema_contract() -> None:
    assert get_args(ConclusionType) == CONCLUSION_TYPES
