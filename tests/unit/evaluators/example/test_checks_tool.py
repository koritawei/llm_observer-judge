"""Name-level tool checks for judge_example. No agent, no LLM."""

import pytest
from business.evaluators.example.checks_tool import ToolNameCheck, check_tool_names
from pydantic import ValidationError


def test_required_get_order_passes() -> None:
    result = check_tool_names(["get_order"], required=["get_order"])

    assert result.ok is True
    assert result.reason == ""


def test_must_be_empty_fails_when_get_order_used() -> None:
    result = check_tool_names(["get_order"], must_be_empty=True)

    assert result.ok is False
    assert result.reason.startswith("硬检失败：")
    assert "get_order" in result.reason


def test_forbidden_get_order_fails_and_empty_list_with_must_be_empty_passes() -> None:
    failed = check_tool_names(["get_order"], forbidden=["get_order"])
    assert failed.ok is False
    assert failed.reason.startswith("硬检失败：")
    assert "get_order" in failed.reason

    passed = check_tool_names([], must_be_empty=True)
    assert passed.ok is True
    assert passed.reason == ""


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolNameCheck.model_validate({"required_tools": ["get_order"]})
