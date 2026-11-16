"""Order, short-circuit, and exception-semantics tests for the shared five-stage attribution Runtime."""

from dataclasses import dataclass, field

import pytest
from benchmark.libs.agenthost.agents.request import AgentRequest
from benchmark.attribution.badcase.schemas import AttributionResult
from benchmark.attribution.components.contracts import (
    AttributionContextBase,
    GateVerdictBase,
    RuleSignalBase,
)
from benchmark.attribution.components.runtime import (
    AttributionDecisionInputs,
    AttributionPipelineTerminalError,
    run_attribution_pipeline,
)


@dataclass
class _Hooks:
    stages: list[str] = field(default_factory=list)
    terminal_at: str = ""
    failure_at: str = ""

    def _enter(self, stage: str) -> None:
        self.stages.append(stage)
        if self.failure_at == stage:
            raise RuntimeError(stage)
        if self.terminal_at == stage:
            raise AttributionPipelineTerminalError(
                AttributionResult(
                    trace_id="trace-1",
                    case_id="trace-1",
                    conclusion_type="insufficient_evidence",
                    needs_human_review=True,
                    should_dispatch=False,
                )
            )

    def normalize_envelope(self, request: AgentRequest) -> AttributionContextBase:
        self._enter("normalize_envelope")
        return AttributionContextBase(trace_id=str(request.traceid or ""))

    def match_rules(self, context: AttributionContextBase) -> list[RuleSignalBase]:
        self._enter("match_rules")
        return [RuleSignalBase(rule_id="rule-a")]

    def evidence_gate(
        self,
        context: AttributionContextBase,
        signals: list[RuleSignalBase],
    ) -> GateVerdictBase:
        self._enter("evidence_gate")
        return GateVerdictBase(verdict="needs_llm")

    async def investigate(
        self,
        context: AttributionContextBase,
        signals: list[RuleSignalBase],
        verdict: GateVerdictBase,
    ) -> str:
        self._enter("investigate")
        return "single_llm"

    def decide_and_validate(
        self,
        context: AttributionContextBase,
        inputs: AttributionDecisionInputs[RuleSignalBase, GateVerdictBase, str],
    ) -> AttributionResult:
        self._enter("decide_and_validate")
        assert inputs.investigation == "single_llm"
        return AttributionResult(
            trace_id=context.trace_id,
            case_id=context.trace_id,
            conclusion_type="agent_badcase",
        )


def _request() -> AgentRequest:
    return AgentRequest(traceid="trace-1", session_id="session-1", messages=[])


@pytest.mark.asyncio
async def test_runtime_runs_fixed_five_stage_order() -> None:
    hooks = _Hooks()

    result = await run_attribution_pipeline(hooks, _request(), agent_name="test")

    assert hooks.stages == [
        "normalize_envelope",
        "match_rules",
        "evidence_gate",
        "investigate",
        "decide_and_validate",
    ]
    assert result.conclusion_type == "agent_badcase"


@pytest.mark.asyncio
async def test_runtime_terminal_result_short_circuits_remaining_stages() -> None:
    hooks = _Hooks(terminal_at="investigate")

    result = await run_attribution_pipeline(hooks, _request(), agent_name="test")

    assert hooks.stages == [
        "normalize_envelope",
        "match_rules",
        "evidence_gate",
        "investigate",
    ]
    assert result.conclusion_type == "insufficient_evidence"
    assert result.needs_human_review is True
    assert result.should_dispatch is False


@pytest.mark.asyncio
async def test_runtime_preserves_unexpected_business_exception() -> None:
    hooks = _Hooks(failure_at="match_rules")

    with pytest.raises(RuntimeError, match="match_rules"):
        await run_attribution_pipeline(hooks, _request(), agent_name="test")
