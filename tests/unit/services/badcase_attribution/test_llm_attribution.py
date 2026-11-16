import asyncio
from unittest.mock import AsyncMock

import pytest
from benchmark.attribution.badcase.llm_attribution import (
    EvidenceRoundRequest,
    run_llm_attribution,
    run_llm_evidence_round,
)
from benchmark.attribution.badcase.schemas import TraceContextPack


def _empty_pack(*, trace_id: str, case_id: str) -> TraceContextPack:
    return TraceContextPack(trace_id=trace_id, case_id=case_id)


@pytest.mark.asyncio
async def test_run_llm_attribution_requires_configured_model() -> None:
    """When the model is not configured the call must fail explicitly, and must not fabricate a human-review attribution result."""

    with pytest.raises(RuntimeError, match="LLM is not configured"):
        await run_llm_attribution(
            llm=None,
            system_prompt="system",
            evidence_prompt="evidence",
            context_pack=_empty_pack(trace_id="trace-no-llm", case_id="case-no-llm"),
        )


@pytest.mark.asyncio
async def test_run_llm_attribution_raises_on_call_timeout() -> None:
    """When llm.complete exceeds llm_call_timeout_s, raise an execution failure and do not produce a fallback attribution result."""

    class _SlowLLM:
        async def complete(self, system_prompt: str, evidence_prompt: str) -> object:
            await asyncio.sleep(1.0)
            return "{}"

    with pytest.raises(RuntimeError, match="timed out"):
        await run_llm_attribution(
            llm=_SlowLLM(),
            system_prompt="system",
            evidence_prompt="evidence",
            context_pack=_empty_pack(trace_id="trace-timeout", case_id="case-timeout"),
            llm_call_timeout_s=0.05,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_s", [None, 1.0])
async def test_evidence_round_request_preserves_prompts_and_findings(timeout_s) -> None:
    llm = AsyncMock()
    llm.complete.return_value = {
        "decision_type": "findings",
        "findings": {"observations_summary": "observed", "supports_low_score": True},
    }
    result = await run_llm_evidence_round(
        llm=llm,
        request=EvidenceRoundRequest(
            system_prompt="system",
            evidence_prompt="evidence",
            context_pack=_empty_pack(trace_id="trace", case_id="case"),
            llm_call_timeout_s=timeout_s,
        ),
    )
    llm.complete.assert_awaited_once_with("system", "evidence")
    assert result.findings.observations_summary == "observed"
    assert result.findings.supports_low_score is True


@pytest.mark.asyncio
async def test_evidence_round_request_timeout_and_missing_model() -> None:
    request = EvidenceRoundRequest(
        system_prompt="system",
        evidence_prompt="evidence",
        context_pack=_empty_pack(trace_id="trace", case_id="case"),
        llm_call_timeout_s=0.01,
    )
    with pytest.raises(RuntimeError, match="LLM is not configured"):
        await run_llm_evidence_round(llm=None, request=request)

    llm = AsyncMock()

    async def slow_complete(*_args):
        await asyncio.sleep(1)

    llm.complete.side_effect = slow_complete
    with pytest.raises(RuntimeError, match="timed out"):
        await run_llm_evidence_round(llm=llm, request=request)


@pytest.mark.asyncio
async def test_evidence_round_request_keeps_unparsable_response_incomplete() -> None:
    llm = AsyncMock()
    llm.complete.return_value = "not json"
    result = await run_llm_evidence_round(
        llm=llm,
        request=EvidenceRoundRequest(
            system_prompt="system",
            evidence_prompt="evidence",
            context_pack=_empty_pack(trace_id="trace", case_id="case"),
        ),
    )
    assert result.decision_type == "findings"
    assert result.findings.supports_low_score is None
    assert result.findings.evidence_details == []
    assert result.raw_response == "not json"
