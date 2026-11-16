"""Example attribution LLM compatibility re-export; the implementation lives in the shared attribution components."""

from benchmark.attribution.components.llm import AttributionLLM, ChatModelAttributionLLM

__all__ = ["AttributionLLM", "ChatModelAttributionLLM"]
