"""Cross-business call boundary for the attribution Agent.

This layer only wraps a business attribution result as a stable ``AgentResponse``. It does not understand business rules,
evidence sources, or investigation mode. A business Agent only implements :meth:`attribute`.
"""

import asyncio
from abc import ABC, abstractmethod

from benchmark.libs.agenthost.agents import AgentCard, BaseAgent
from benchmark.libs.agenthost.agents.request import (
    AgentRequest,
    AgentResponse,
    ContentPart,
    ResponseStatus,
)
from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.attribution.badcase.schemas import AttributionResult

logger = get_logger("agent_eval.attribution.business_components")


class BaseAttributionAgent(BaseAgent, ABC):
    """Call, response serialization, and exception convergence for the unified attribution Agent."""

    def get_agent_card(self) -> AgentCard:
        """Return the attribution platform's base Agent Card; business subclasses should override with a more specific description."""
        return AgentCard(
            name=self.name,
            description=(
                "unified attribution agent: request serialization, evidence response, "
                "and error convergence."
            ),
        )

    async def run(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
        return await self.call(request, **kwargs)

    async def call(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
        del kwargs
        try:
            result = await self.attribute(request)
            return AgentResponse(
                traceid=request.traceid,
                status=ResponseStatus.SUCCESS,
                content_parts=[ContentPart(type="text", data=result.model_dump_json())],
                metadata={
                    "attribution_result": result.model_dump(mode="json"),
                    "case_id": result.case_id,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - converge attribution exceptions into a protocol failure response
            logger.exception(
                "attribution.agent.call_failed agent=%s trace_id=%s",
                self.name,
                request.traceid,
            )
            return AgentResponse(
                traceid=request.traceid,
                status=ResponseStatus.FAILED,
                metadata={"error_code": "attribution_failed"},
            )

    @abstractmethod
    async def attribute(self, request: AgentRequest) -> AttributionResult:
        """Run domain attribution and return the platform's unified result."""


__all__ = ["BaseAttributionAgent"]
