"""Tests for benchmark.services._stage_key."""

from benchmark.services._stage_key import (
    STAGE_ATTRIBUTION,
    STAGE_JUDGE,
    stage_agent_key,
)


def test_stage_constants():
    assert STAGE_JUDGE == "judge"
    assert STAGE_ATTRIBUTION == "attribution"


def test_stage_agent_key_judge():
    assert stage_agent_key(STAGE_JUDGE, "example") == "judge_example"


def test_stage_agent_key_attribution():
    assert stage_agent_key(STAGE_ATTRIBUTION, "example") == "attribution_example"


def test_stage_agent_key_custom_logic_name():
    # New evaluator logical names (foo, bar) may be added later; the helper must support them
    assert stage_agent_key(STAGE_JUDGE, "foo") == "judge_foo"
    assert stage_agent_key(STAGE_ATTRIBUTION, "bar") == "attribution_bar"