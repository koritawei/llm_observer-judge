"""Compatibility baseline for cross-business attribution components."""

import asyncio

import pytest
from benchmark.libs.agenthost.agents.config.context import AgentProviderContext
from benchmark.libs.agenthost.agents.request import AgentRequest, ResponseStatus
from benchmark.attribution.badcase.schemas import AttributionResult
from benchmark.attribution.components.agent import BaseAttributionAgent
from benchmark.attribution.components.contracts import (
    AttributionContextBase,
    GateVerdictBase,
    RuleSignalBase,
)
from pydantic import BaseModel


class _AttributionAgent(BaseAttributionAgent):
    @classmethod
    def from_config(
        cls,
        cfg: BaseModel,
        ctx: AgentProviderContext,
    ) -> "_AttributionAgent":
        del cfg
        return cls(name=ctx.name)

    async def attribute(self, request: AgentRequest) -> AttributionResult:
        return AttributionResult(
            trace_id=str(request.traceid or ""),
            case_id="case-1",
            category="unknown",
            root_cause="test",
            confidence=0.0,
            reasoning="test",
            evidence_refs=[],
            action_hint="",
            needs_human_review=False,
            should_dispatch=False,
        )


class _FailingAttributionAgent(_AttributionAgent):
    async def attribute(self, request: AgentRequest) -> AttributionResult:
        del request
        raise RuntimeError("prompt fragment must stay internal")


class _CancelledAttributionAgent(_AttributionAgent):
    async def attribute(self, request: AgentRequest) -> AttributionResult:
        del request
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_base_agent_preserves_response_envelope() -> None:
    request = AgentRequest(traceid="trace-1", session_id="session-1", messages=[])

    response = await _AttributionAgent(name="test").call(request)

    assert response.status == ResponseStatus.SUCCESS
    assert response.metadata["case_id"] == "case-1"
    assert response.metadata["attribution_result"]["root_cause"] == "test"
    assert (
        response.content_parts[0].data
        == AttributionResult.model_validate(
            response.metadata["attribution_result"]
        ).model_dump_json()
    )


@pytest.mark.asyncio
async def test_base_agent_converges_unknown_errors_to_failed_response() -> None:
    request = AgentRequest(traceid="trace-1", session_id="session-1", messages=[])

    response = await _FailingAttributionAgent(name="test").run(request)

    assert response.status == ResponseStatus.FAILED
    assert response.metadata == {"error_code": "attribution_failed"}


@pytest.mark.asyncio
async def test_base_agent_preserves_cancellation() -> None:
    request = AgentRequest(traceid="trace-1", session_id="session-1", messages=[])

    with pytest.raises(asyncio.CancelledError):
        await _CancelledAttributionAgent(name="test").call(request)


def test_business_contracts_keep_domain_fields_open() -> None:
    context = AttributionContextBase(
        trace_id="trace-1",
        schema_version="custom.v1",
        hard_rule_hits=["rule-a"],
    )
    signal = RuleSignalBase(rule_id="rule-a", base_confidence=0.8)
    verdict = GateVerdictBase(verdict="needs_llm", hypotheses=["cause-a"])

    assert context.schema_version == "custom.v1"
    assert signal.base_confidence == 0.8
    assert verdict.verdict == "needs_llm"
