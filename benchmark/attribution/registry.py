"""Attribution registration. Only the example attribution agent is loaded."""

from benchmark.libs.agenthost.agents import BaseAgent
from business.agents.example.attribution.agent import AttributionAgent  # noqa: F401

__all__ = ["AttributionAgent"]

_ATTRIBUTION_AGENTS = {
    "attribution_example": AttributionAgent,
}


def attribution_agents() -> dict[str, type[BaseAgent]]:
    return dict(_ATTRIBUTION_AGENTS)
