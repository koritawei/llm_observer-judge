"""STEP-05: YAML routing, CoW hot-reload, priority shards, DispatchClient."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from llmobserverjudge.contracts.models import Priority
from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.dispatch.conditions import (
    evaluate_condition,
    validate_condition_syntax,
)
from llmobserverjudge.dispatch.config_loader import (
    DispatchConfigLoader,
    load_dispatch_config,
)
from llmobserverjudge.dispatch.config_models import DispatchSection
from llmobserverjudge.dispatch.publisher import DispatchPublisher
from llmobserverjudge.dispatch.routing import match_eval_configs
from llmobserverjudge.ingest.outbox import InMemoryOutbox
from llmobserverjudge.shared.mq import InMemoryMQ

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "dispatch.yaml"


def test_default_yaml_loads() -> None:
    loader = load_dispatch_config(DEFAULT_YAML)
    assert loader.version >= 1
    assert len(loader.config.routing_rules) >= 2


def test_match_safety_tag_p0() -> None:
    configs, priority = match_eval_configs({"tags": ["safety"]})
    assert priority == Priority.P0
    assert [c.agent_id for c in configs] == ["judge_safety", "judge_code_eval"]


def test_match_default_p2() -> None:
    configs, priority = match_eval_configs({"tags": ["other"]})
    assert priority == Priority.P2
    assert [c.agent_id for c in configs] == ["judge_quality", "judge_fast_vllm"]


def test_condition_status() -> None:
    assert evaluate_condition("status == 'ERROR'", {"status": "ERROR"})
    assert not evaluate_condition("status == 'ERROR'", {"status": "OK"})


def test_unsupported_condition_rejected() -> None:
    with pytest.raises(ValueError):
        validate_condition_syntax("__import__('os')")


def test_invalid_yaml_keeps_previous(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.yaml"
    path.write_text(DEFAULT_YAML.read_text(encoding="utf-8"), encoding="utf-8")
    loader = load_dispatch_config(path)
    v0 = loader.version
    agents0 = loader.config.routing_rules[0].judge_agents

    path.write_text("dispatch:\n  routing_rules: []\n", encoding="utf-8")
    assert loader.load_sync() is False
    assert loader.version == v0
    assert loader.config.routing_rules[0].judge_agents == agents0


@pytest.mark.asyncio
async def test_hot_reload_cow(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.yaml"
    data = yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8"))
    path.write_text(yaml.dump(data), encoding="utf-8")
    loader = load_dispatch_config(path)
    old_cfg = loader.config
    v0 = loader.version

    data["dispatch"]["routing_rules"] = [
        {
            "condition": "always",
            "judge_agents": ["judge_custom"],
            "priority": "P1",
        }
    ]
    path.write_text(yaml.dump(data), encoding="utf-8")
    assert await loader.load() is True
    assert loader.version == v0 + 1
    # CoW: old snapshot unchanged
    assert old_cfg.routing_rules[0].judge_agents != ["judge_custom"]
    assert loader.config.routing_rules[0].judge_agents == ["judge_custom"]

    configs, priority = match_eval_configs({"tags": []}, loader.config)
    assert priority == Priority.P1
    assert configs[0].agent_id == "judge_custom"


@pytest.mark.asyncio
async def test_watch_loop_reloads(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.yaml"
    data = yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8"))
    path.write_text(yaml.dump(data), encoding="utf-8")
    loader = load_dispatch_config(path)
    v0 = loader.version
    task = asyncio.create_task(loader.watch_loop(interval_s=0.05))
    await asyncio.sleep(0.08)
    data["dispatch"]["default_model"] = "hot-model"
    path.write_text(yaml.dump(data), encoding="utf-8")
    for _ in range(40):
        if loader.version > v0:
            break
        await asyncio.sleep(0.05)
    loader.stop()
    await asyncio.wait_for(task, timeout=2)
    assert loader.version > v0
    assert loader.config.default_model == "hot-model"


@pytest.mark.asyncio
async def test_publisher_priority_shard() -> None:
    mq = InMemoryMQ()
    outbox = InMemoryOutbox()
    await outbox.insert_trace_closed(
        tenant_id="t1",
        trace_id="0123456789abcdef0123456789abcdef",
        trace_summary={"tags": ["safety"], "status": "OK"},
    )
    section = DispatchSection.model_validate(
        {
            "default_model": "m",
            "publish_priority_shards": True,
            "routing_rules": [
                {
                    "condition": "tags contains 'safety'",
                    "judge_agents": ["judge_safety"],
                    "priority": "P0",
                },
                {
                    "condition": "always",
                    "judge_agents": ["judge_quality"],
                    "priority": "P2",
                },
            ],
        }
    )
    loader = DispatchConfigLoader(initial=section)
    pub = DispatchPublisher(mq, outbox, config_loader=loader)
    assert await pub.publish_pending() == 1
    assert await mq.depth(Topic.EVAL_JOBS) == 1
    assert await mq.depth(Topic.EVAL_JOBS_P0) == 1
    job = (await mq.drain(Topic.EVAL_JOBS, 1))[0]
    assert job["priority"] == "P0"
    assert job["eval_configs"][0]["agent_id"] == "judge_safety"


@pytest.mark.asyncio
async def test_client_route_post_eval_annotation_always() -> None:
    mq = InMemoryMQ()
    client = DispatchClient(mq)
    result = await client.route_post_eval(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_safety",
        score=0.9,  # above threshold but annotation_trigger=always
        confidence=0.99,
        label="ok",
    )
    assert result["rule_found"] is True
    assert result["annotation_published"] is True
    assert result["attribution_published"] is False  # score above 0.7
    assert await mq.depth(Topic.ANNOTATION_JOBS) == 1


@pytest.mark.asyncio
async def test_client_route_post_eval_attribution_and_low_confidence() -> None:
    mq = InMemoryMQ()
    client = DispatchClient(mq)
    result = await client.route_post_eval(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_quality",
        score=0.4,  # below 0.6
        confidence=0.3,
        label="bad",
        is_bad_case=True,
    )
    assert result["attribution_published"] is True
    assert result["annotation_published"] is True
    assert await mq.depth(Topic.ATTRIBUTION_JOBS) == 1
    assert await mq.depth(Topic.ANNOTATION_JOBS) == 1
    attr = (await mq.drain(Topic.ATTRIBUTION_JOBS, 1))[0]
    assert attr["attribution_agent"] == "attributor_default"
    assert attr["priority"] == "P1"


@pytest.mark.asyncio
async def test_client_good_case_skips_attribution() -> None:
    mq = InMemoryMQ()
    client = DispatchClient(mq)
    result = await client.route_post_eval(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_quality",
        score=0.95,
        label="good",
        is_bad_case=False,
    )
    assert result["attribution_published"] is False
    assert await mq.depth(Topic.ATTRIBUTION_JOBS) == 0


@pytest.mark.asyncio
async def test_client_is_bad_case_false_overrides_low_score() -> None:
    """Explicit good-case flag must skip attribution even if score looks low."""
    mq = InMemoryMQ()
    client = DispatchClient(mq)
    result = await client.route_post_eval(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_quality",
        score=0.1,
        is_bad_case=False,
    )
    assert result["attribution_published"] is False
    assert await mq.depth(Topic.ATTRIBUTION_JOBS) == 0
