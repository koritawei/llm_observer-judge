"""LLM attribution boundary: AttributionLLM Protocol + single-round decision parsing.

:class:`AttributionLLM` is the only definition of the attribution model; the example side only adapts ChatModel
and re-exports it. ``run_llm_attribution`` calls ``llm.complete`` and parses a
:class:`BadcaseAttributionDecision`; on failure it raises and must not fabricate a human_review.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.attribution.badcase.model_client import (
    parse_model_response,
    response_text,
)
from benchmark.attribution.badcase.schemas import (
    BadcaseAttributionDecision,
    EvidenceRoundDecision,
    ReactFindings,
    TraceContextPack,
)

logger = get_logger("agent_eval.badcase_attribution")


@runtime_checkable
class AttributionLLM(Protocol):
    """Minimal attribution model interface: ``complete(system, evidence)`` → raw response.

    The two prompt segments match the badcase prompt's system / evidence;
    :func:`run_llm_attribution` parses them into :class:`BadcaseAttributionDecision`.
    """

    async def complete(self, system_prompt: str, evidence_prompt: str) -> object:
        """Return the raw model response for the attribution prompt."""


async def run_llm_attribution(
    *,
    llm: AttributionLLM | None,
    system_prompt: str,
    evidence_prompt: str,
    context_pack: TraceContextPack,
    llm_call_timeout_s: float | None = None,
) -> BadcaseAttributionDecision:
    """Call the explicitly injected attribution model and parse JSON.

    This is the main entry of the LLM attribution service; the AttributionAgent ReAct loop only cares about the structured decision,
    not the model-client details.

    When ``llm_call_timeout_s`` > 0, one ``llm.complete`` gets a hang backstop;
    a timeout raises ``RuntimeError`` as an execution failure and does not produce a fallback attribution result.
    """

    if llm is None:
        raise RuntimeError("Badcase attribution LLM is not configured.")

    started = time.perf_counter()
    logger.info(
        "attribution.llm_started trace_id=%s case_id=%s timeout_s=%s",
        context_pack.trace_id,
        context_pack.case_id,
        llm_call_timeout_s,
    )
    try:
        if llm_call_timeout_s and llm_call_timeout_s > 0:
            response = await asyncio.wait_for(
                llm.complete(system_prompt, evidence_prompt),
                timeout=llm_call_timeout_s,
            )
        else:
            response = await llm.complete(system_prompt, evidence_prompt)
    except TimeoutError as exc:
        logger.error(
            "attribution.llm_timeout trace_id=%s case_id=%s timeout_s=%s",
            context_pack.trace_id,
            context_pack.case_id,
            llm_call_timeout_s,
        )
        raise RuntimeError(
            f"Badcase attribution LLM call timed out after {llm_call_timeout_s}s"
        ) from exc
    decision = _parse_response(response, context_pack)
    logger.info(
        "attribution.llm_done trace_id=%s case_id=%s decision_type=%s elapsed_ms=%.2f",
        context_pack.trace_id,
        context_pack.case_id,
        decision.decision_type,
        (time.perf_counter() - started) * 1000,
    )
    return decision


def _parse_response(response: object, context_pack: TraceContextPack) -> BadcaseAttributionDecision:
    """Parse the model's raw response into a `BadcaseAttributionDecision`."""

    if isinstance(response, BadcaseAttributionDecision):
        return _with_decision_identity(response, context_pack)
    if isinstance(response, dict):
        raw_text = json.dumps(response, ensure_ascii=False)
    else:
        raw_text = response_text(response)

    decision = parse_model_response(response, BadcaseAttributionDecision).model_copy(
        update={"raw_response": raw_text}
    )
    return _with_decision_identity(decision, context_pack)


def _with_decision_identity(
    decision: BadcaseAttributionDecision,
    context_pack: TraceContextPack,
) -> BadcaseAttributionDecision:
    """Fill trace/case identity onto the final result inside the decision."""

    if decision.final is None:
        return decision
    return decision.model_copy(
        update={
            "final": decision.final.model_copy(update=_identity_fields(context_pack)),
        }
    )


def _identity_fields(context_pack: TraceContextPack) -> dict[str, str]:
    """Extract trace/case identity from the context and fill it onto the model result."""

    return {
        "trace_id": context_pack.trace_id,
        "case_id": context_pack.case_id,
    }


@dataclass(frozen=True)
class EvidenceRoundRequest:
    """Prompt, trace context, and per-call timeout for an evidence round."""

    system_prompt: str
    evidence_prompt: str
    context_pack: TraceContextPack
    llm_call_timeout_s: float | None = None


async def run_llm_evidence_round(
    *,
    llm: AttributionLLM | None,
    request: EvidenceRoundRequest,
) -> EvidenceRoundDecision:
    """Evidence-round entry: the LLM only outputs tool_calls or ReactFindings, and does not give a final conclusion.

    It coexists with :func:`run_llm_attribution` (the old final mode) and does not change that behavior; a parse failure does not
    fabricate a conclusion, and empty findings make the unified decision take insufficient_evidence.
    """

    system_prompt = request.system_prompt
    evidence_prompt = request.evidence_prompt
    context_pack = request.context_pack
    llm_call_timeout_s = request.llm_call_timeout_s

    if llm is None:
        raise RuntimeError("Badcase attribution LLM is not configured.")

    started = time.perf_counter()
    try:
        if llm_call_timeout_s and llm_call_timeout_s > 0:
            response = await asyncio.wait_for(
                llm.complete(system_prompt, evidence_prompt),
                timeout=llm_call_timeout_s,
            )
        else:
            response = await llm.complete(system_prompt, evidence_prompt)
    except TimeoutError as exc:
        logger.error(
            "attribution.evidence_round_timeout trace_id=%s timeout_s=%s",
            context_pack.trace_id,
            llm_call_timeout_s,
        )
        raise RuntimeError(
            f"Badcase attribution LLM call timed out after {llm_call_timeout_s}s"
        ) from exc

    decision = _parse_evidence_round(response)
    logger.info(
        "attribution.evidence_round_done trace_id=%s decision_type=%s elapsed_ms=%.2f",
        context_pack.trace_id,
        decision.decision_type,
        (time.perf_counter() - started) * 1000,
    )
    return decision


def _parse_evidence_round(response: object) -> EvidenceRoundDecision:
    """Parse an evidence-round output; if it cannot be parsed, degrade to empty findings and do not raise."""

    if isinstance(response, EvidenceRoundDecision):
        return response
    if isinstance(response, dict):
        raw_text = json.dumps(response, ensure_ascii=False)
    else:
        raw_text = response_text(response)
    try:
        parsed = parse_model_response(response, EvidenceRoundDecision)
    except Exception:  # noqa: BLE001 - a parse failure must not fabricate a conclusion
        logger.warning("attribution.evidence_round_unparsable", exc_info=True)
        return EvidenceRoundDecision(
            decision_type="findings",
            rationale_summary="补证输出不可解析",
            findings=ReactFindings(),
            raw_response=raw_text,
        )
    return parsed.model_copy(update={"raw_response": raw_text})


__all__ = [
    "AttributionLLM",
    "EvidenceRoundRequest",
    "run_llm_attribution",
    "run_llm_evidence_round",
]
