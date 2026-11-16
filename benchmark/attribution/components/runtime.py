"""Shared five-stage attribution orchestration runtime (plan §7.1).

It always advances ``normalize envelope → match rules → evidence gate → optional
investigation → decide and validate``. The shared layer only owns stage order, audit, and short-circuit pass-through,
and does not understand business rules, evidence sources, or investigation mode (each business injects its investigation strategy via the ``investigate`` hook,
for example VUI ``single_llm`` and Vehicle ``react_tools``).

Exception semantics stay equivalent to history: a business exception raised by a stage passes through as-is and is
converged into a protocol failure response by ``BaseAttributionAgent.call``; any stage may
``raise AttributionPipelineTerminalError(result)`` to skip later stages and return the final conclusion directly
(for example a direct rule verdict, or insufficient evidence sent to human review).
"""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from benchmark.libs.agenthost.agents.request import AgentRequest
from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.attribution.badcase.schemas import AttributionResult
from benchmark.attribution.components.contracts import (
    AttributionContextBase,
    GateVerdictBase,
    RuleSignalBase,
)

logger = get_logger("agent_eval.attribution.runtime")

ContextT = TypeVar("ContextT", bound=AttributionContextBase)
SignalT = TypeVar("SignalT", bound=RuleSignalBase)
VerdictT = TypeVar("VerdictT", bound=GateVerdictBase)
InvestigationT = TypeVar("InvestigationT")


@dataclass(frozen=True)
class AttributionDecisionInputs(Generic[SignalT, VerdictT, InvestigationT]):
    """Evidence and investigation carried from the gate to the final decision."""

    signals: list[SignalT]
    verdict: VerdictT
    investigation: InvestigationT | None


class AttributionPipelineTerminalError(Exception):
    """A stage already produced a final conclusion: carry the result and skip the remaining stages.

    Used for deterministic early-exit paths (direct rule verdict, gate sends insufficient evidence to human review, investigation failure convergence, and similar).
    It is not an error; the runtime catches it and returns the carried :class:`AttributionResult`.
    """

    def __init__(self, result: AttributionResult) -> None:
        super().__init__("attribution pipeline terminated with a final result")
        self.result = result


@runtime_checkable
class AttributionPipelineHooks(Protocol[ContextT, SignalT, VerdictT, InvestigationT]):
    """Injection points where a business attribution plugin plugs into the five-stage runtime.

    - ``normalize_envelope``: parse the request/eval envelope into a domain attribution context;
    - ``match_rules``: produce candidate root-cause signals from domain rules;
    - ``evidence_gate``: decide whether existing evidence can support a direct rule verdict;
    - ``investigate``: add evidence with the investigation strategy the business chose (may return ``None`` when no investigation is needed,
      for example the direct-rule path);
    - ``decide_and_validate``: produce the final conclusion and run reference/confidence/dispatch checks.

    Any hook may ``raise AttributionPipelineTerminalError(result)`` to skip later stages.
    """

    def normalize_envelope(self, request: AgentRequest) -> ContextT:
        """Stage 1: parse the request into an attribution context."""

    def match_rules(self, context: ContextT) -> list[SignalT]:
        """Stage 2: rule matching produces candidate signals."""

    def evidence_gate(self, context: ContextT, signals: list[SignalT]) -> VerdictT:
        """Stage 3: evidence-gate verdict."""

    async def investigate(
        self,
        context: ContextT,
        signals: list[SignalT],
        verdict: VerdictT,
    ) -> InvestigationT | None:
        """Stage 4: optional investigation (none / single_llm / react_tools is chosen by the business)."""

    def decide_and_validate(
        self,
        context: ContextT,
        inputs: AttributionDecisionInputs[SignalT, VerdictT, InvestigationT],
    ) -> AttributionResult:
        """Stage 5: decision and final checks."""


async def run_attribution_pipeline(
    hooks: AttributionPipelineHooks[ContextT, SignalT, VerdictT, InvestigationT],
    request: AgentRequest,
    *,
    agent_name: str = "attribution",
) -> AttributionResult:
    """Advance the business attribution pipeline in the fixed five-stage order.

    Between stages it only records an audit and passes a short-circuit through: a business exception is re-raised as-is (converged by
    ``BaseAttributionAgent.call``), so behavior stays equivalent to the historical in-process orchestration.
    """

    try:
        context = hooks.normalize_envelope(request)
        logger.info(
            "attribution.runtime.stage agent=%s stage=normalize_envelope trace_id=%s",
            agent_name,
            context.trace_id,
        )
        signals = hooks.match_rules(context)
        verdict = hooks.evidence_gate(context, signals)
        logger.info(
            "attribution.runtime.stage agent=%s stage=evidence_gate verdict=%s signals=%d",
            agent_name,
            verdict.verdict,
            len(signals),
        )
        investigation = await hooks.investigate(context, signals, verdict)
        result = hooks.decide_and_validate(
            context, AttributionDecisionInputs(signals, verdict, investigation)
        )
        logger.info(
            "attribution.runtime.stage agent=%s stage=decided conclusion=%s",
            agent_name,
            result.conclusion_type or "unknown",
        )
        return result
    except AttributionPipelineTerminalError as terminal:
        logger.info(
            "attribution.runtime.terminal agent=%s conclusion=%s",
            agent_name,
            terminal.result.conclusion_type or "unknown",
        )
        return terminal.result


__all__: list[str] = [
    "AttributionPipelineHooks",
    "AttributionPipelineTerminalError",
    "run_attribution_pipeline",
]
