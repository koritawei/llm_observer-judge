"""STEP-12: LLM ReAct Policy + stub fallback."""

from __future__ import annotations

import json

import httpx
import pytest

from llmobserverjudge.attribution_worker.context import AttributionContext
from llmobserverjudge.attribution_worker.llm_policy import (
    LLMReActPolicy,
    react_policy_from_env,
)
from llmobserverjudge.attribution_worker.react import ReActAttributor, StubReActPolicy
from llmobserverjudge.contracts.models import AttributionJob, Priority
from llmobserverjudge.shared.openai_chat import OpenAIChatClient


def _job() -> AttributionJob:
    return AttributionJob(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_safety",
        priority=Priority.P0,
        idempotency_key="llm-react-1",
        score_snapshot={"score": 0.3, "label": "bad"},
        trace_summary={"status": "ERROR", "tags": ["safety"]},
    )


def _ctx() -> AttributionContext:
    j = _job()
    return AttributionContext(
        tenant_id=j.tenant_id,
        trace_id=j.trace_id,
        trace_summary=j.trace_summary,
        spans=[{"span_id": "aaaaaaaaaaaaaaaa", "status": "ERROR", "name": "llm"}],
        score_snapshot=j.score_snapshot,
        eval_agent_id=j.eval_agent_id,
    )


@pytest.mark.asyncio
async def test_llm_policy_drives_tools_then_submit() -> None:
    steps = [
        {
            "thought": "start",
            "action": "get_trace_summary",
            "args": {},
            "confidence": 0.5,
        },
        {
            "thought": "errors",
            "action": "get_error_spans",
            "args": {},
            "confidence": 0.7,
        },
        {
            "thought": "done",
            "action": "submit_attribution",
            "args": {
                "root_cause": "RUNTIME_ERROR",
                "confidence": 0.95,
                "evidence": [{"k": "v"}],
                "summary": "llm conclude",
            },
            "confidence": 0.95,
        },
    ]
    idx = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = steps[min(idx["i"], len(steps) - 1)]
        idx["i"] += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(body)}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = OpenAIChatClient(
        base_url="https://llm.example.com",
        client=client,
        enforce_ssrf=False,
    )
    policy = LLMReActPolicy(chat)
    attr = ReActAttributor(policy=policy)
    result = await attr.analyze(_job(), _ctx())
    assert result.status == "COMPLETED"
    assert result.root_cause == "RUNTIME_ERROR"
    assert "get_trace_summary" in result.tools_called
    assert "submit_attribution" in result.tools_called
    assert policy.last_source == "llm"
    await chat.aclose()


@pytest.mark.asyncio
async def test_llm_policy_falls_back_on_bad_action() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "thought": "bad",
                                    "action": "drop_database",
                                    "args": {},
                                    "confidence": 0.9,
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = OpenAIChatClient(
        base_url="https://llm.example.com",
        client=client,
        enforce_ssrf=False,
    )
    policy = LLMReActPolicy(chat)
    # one step via policy — should stub fallback to get_trace_summary
    from llmobserverjudge.attribution_worker.tools import BuiltinToolRuntime

    tools = BuiltinToolRuntime(_ctx())
    thought, action, args, conf = await policy.next_step(
        rounds=1, tools=tools, observations=[]
    )
    assert action == "get_trace_summary"
    assert policy.last_source == "stub_fallback"
    assert conf == 0.45
    _ = thought, args
    await chat.aclose()


@pytest.mark.asyncio
async def test_llm_http_error_falls_back_full_analyze() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat = OpenAIChatClient(
        base_url="https://llm.example.com",
        client=client,
        enforce_ssrf=False,
    )
    policy = LLMReActPolicy(chat)
    attr = ReActAttributor(policy=policy)
    result = await attr.analyze(_job(), _ctx())
    assert result.status == "COMPLETED"
    assert result.root_cause == "RUNTIME_ERROR"
    assert policy.last_source == "stub_fallback"
    await chat.aclose()


def test_react_policy_from_env_stub_without_url() -> None:
    p = react_policy_from_env(env={})
    assert isinstance(p, StubReActPolicy)


def test_react_policy_from_env_prefers_react_prefix() -> None:
    p = react_policy_from_env(
        env={
            "LLMJUDGE_REACT_BASE_URL": "https://react.example.com",
            "LLMJUDGE_JUDGE_BASE_URL": "https://judge.example.com",
        },
        enforce_ssrf=False,
    )
    assert isinstance(p, LLMReActPolicy)
    assert p.chat.base_url == "https://react.example.com"
