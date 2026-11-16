"""attribution submodule entry — exports AttributionAgent only."""

from .agent import AttributionAgent, AttributionAgentConfigSchema

__all__ = [
    "AttributionAgent",
    "AttributionAgentConfigSchema",
]
