"""Shared utilities for agent-eval (error handling, LLM JSON parsing, etc.)."""

from benchmark.utils.llm_json import strip_json_fence

__all__ = ["strip_json_fence"]
