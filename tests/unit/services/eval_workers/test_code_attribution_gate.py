"""The enqueue gate for code attribution."""

from benchmark.settings.setting import settings
from benchmark.workers.code_attribution_gate import should_run_code_attribution


def test_gate_closed_by_default() -> None:
    previous = settings.code_attribution_enabled
    settings.code_attribution_enabled = False
    try:
        assert should_run_code_attribution({"primary_issue_category": "tool_timeout"}, 0.0) is False
    finally:
        settings.code_attribution_enabled = previous


def test_gate_accepts_fail_and_poor_only() -> None:
    previous = settings.code_attribution_enabled
    settings.code_attribution_enabled = True
    try:
        assert should_run_code_attribution({}, 0.0) is True
        assert should_run_code_attribution({}, 0.5) is True
        assert should_run_code_attribution({}, 1.0) is False
        assert should_run_code_attribution({"analysis_failed": True}, 0.0) is False
        assert (
            should_run_code_attribution({"primary_issue_category": "analysis_failed"}, 0.0) is False
        )
        assert should_run_code_attribution({}, None) is False
    finally:
        settings.code_attribution_enabled = previous
