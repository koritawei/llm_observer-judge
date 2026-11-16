"""Unit tests for trace.name naming convention (Langfuse)."""

from benchmark.utils.trace_naming import build_eval_trace_name


def test_build_eval_trace_name_normal_case() -> None:
    """A normal input is joined into trace.name."""
    result = build_eval_trace_name("vehicle-agent")
    assert result == "agent-eval.vehicle-agent"


def test_build_eval_trace_name_empty_agent_name() -> None:
    """An empty agent_name → use unknown as the placeholder."""
    assert build_eval_trace_name("") == "agent-eval.unknown"
    assert build_eval_trace_name(None) == "agent-eval.unknown"
    assert build_eval_trace_name("   ") == "agent-eval.unknown"


def test_build_eval_trace_name_normalizes_slashes() -> None:
    """An agent_name containing / must be turned into . (a character on the Langfuse trace.name allowlist)."""
    result = build_eval_trace_name("vehicle-agent/v2")
    assert "/" not in result
    assert result == "agent-eval.vehicle-agent.v2"


def test_build_eval_trace_name_strips_invalid_chars() -> None:
    """Characters outside the allowlist must be stripped."""
    result = build_eval_trace_name("vehicle@agent!")
    assert "@" not in result
    assert "!" not in result
    assert result == "agent-eval.vehicleagent"
