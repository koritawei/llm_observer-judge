"""STEP-09: Prompt render, HTTP Judge, parse, fallback."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from llmobserverjudge.contracts.models import EvalConfig, EvalJob, Priority
from llmobserverjudge.eval_worker.graph import run_eval_graph
from llmobserverjudge.eval_worker.http_judge import HttpJudgeAdapter, http_judge_from_env
from llmobserverjudge.eval_worker.judges import (
    FailingJudgeAdapter,
    JudgeModelRouter,
    RuleJudgeAdapter,
    build_judge_router,
)
from llmobserverjudge.eval_worker.parse import parse_judge_response
from llmobserverjudge.eval_worker.prompt import (
    InMemoryPromptService,
    seed_default_prompts,
)


def test_safe_template_render_no_code_exec() -> None:
    out = InMemoryPromptService.render_template(
        "Hello {{ name }} {{ missing }}",
        name="world",
    )
    assert out == "Hello world "
    # expression-like tokens are not evaluated — unknown keys empty
    evil = InMemoryPromptService.render_template("{{ __import__ }}", **{})
    assert evil == ""


@pytest.mark.asyncio
async def test_prompt_publish_and_resolve() -> None:
    svc = InMemoryPromptService()
    draft = await svc.create_prompt(
        prompt_id="p1",
        content="score {{ agent_id }} :: {{ output_text }}",
        variables=["agent_id", "output_text"],
    )
    await svc.publish("p1", draft.version_id)
    active = await svc.get_active_version("p1")
    assert active is not None
    assert active.status == "ACTIVE"
    rendered = InMemoryPromptService.render_template(
        active.content, agent_id="judge_q", output_text="hi"
    )
    assert rendered == "score judge_q :: hi"


def test_parse_judge_json_variants() -> None:
    cfg = {"scale_max": 10, "threshold": 0.5, "weight": 1.0}
    r = parse_judge_response(
        '{"score": 8, "label": "ok", "reasoning": "fine"}',
        agent_id="a",
        cfg=cfg,
    )
    assert r["score_status"] == "scored"
    assert r["score"] == 8
    assert r["normalized_score"] == 0.8

    r2 = parse_judge_response(
        'Here you go:\n```json\n{"score": 3, "label": "bad"}\n```',
        agent_id="a",
        cfg=cfg,
    )
    assert r2["score"] == 3

    r3 = parse_judge_response("not json at all", agent_id="a", cfg=cfg)
    assert r3["score_status"] == "model_error"


@pytest.mark.asyncio
async def test_http_judge_success_and_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/chat/completions")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"score": 9, "label": "great", "reasoning": "mock"}'
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    http_judge = HttpJudgeAdapter(
        base_url="https://vllm.example.com",
        model_name="qwen",
        client=client,
        enforce_ssrf=False,
    )
    result = await http_judge.judge(
        agent_id="judge_quality",
        cfg={"scale_max": 10, "threshold": 0.5, "resolved_prompt": "rate this"},
        trace={"prompt": "q", "response": "a"},
        upstream={},
    )
    assert result["score_status"] == "scored"
    assert result["score"] == 9
    await http_judge.aclose()


@pytest.mark.asyncio
async def test_http_5xx_falls_back_to_code_eval() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http_judge = HttpJudgeAdapter(
        base_url="https://vllm.example.com",
        client=client,
        enforce_ssrf=False,
    )
    router = JudgeModelRouter(
        registry={
            "vllm": http_judge,
            "code_eval": RuleJudgeAdapter(),
        },
        default_adapter=FailingJudgeAdapter(),
    )
    result = await router.invoke_with_fallback(
        agent_id="judge_quality",
        cfg={
            "model": "vllm",
            "fallback_models": [],
            "scale_max": 10,
            "resolved_prompt": "x",
        },
        trace={"status": "OK", "response": "long enough answer text"},
        upstream={},
    )
    assert result["score_status"] == "scored"
    assert result["model_used"] == "code_eval"
    await http_judge.aclose()


@pytest.mark.asyncio
async def test_run_eval_graph_with_prompt_id_and_http() -> None:
    svc = InMemoryPromptService()
    await seed_default_prompts(svc)
    draft = await svc.create_prompt(
        prompt_id="prompt_quality_v1",
        content="Judge {{ agent_id }} output={{ output_text }}",
    )
    await svc.publish("prompt_quality_v1", draft.version_id)

    seen_prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        text = body.decode()
        seen_prompts.append(text)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"score": 6, "label": "mid", "reasoning": "ok"}'}}
                ]
            },
        )

    http_judge = HttpJudgeAdapter(
        base_url="https://vllm.example.com",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        enforce_ssrf=False,
    )
    router = build_judge_router(http_adapters={"http-judge": http_judge})
    job = EvalJob(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        trace_summary={"prompt": "q?", "response": "answer body"},
        eval_configs=[
            EvalConfig(
                agent_id="judge_quality",
                model="http-judge",
                prompt_id="prompt_quality_v1",
            )
        ],
        priority=Priority.P2,
        idempotency_key="http-prompt-1",
        created_at=datetime.now(UTC),
    )
    state = await run_eval_graph(
        job, judge_router=router, prompt_service=svc, seed_prompts=False
    )
    assert state.final_score == 0.6
    assert any("answer body" in p for p in seen_prompts)
    await http_judge.aclose()


def test_http_judge_from_env_absent() -> None:
    assert http_judge_from_env(env={}) is None


def test_http_ssrf_blocks_metadata() -> None:
    with pytest.raises(ValueError, match="Blocked"):
        HttpJudgeAdapter(base_url="http://169.254.169.254", enforce_ssrf=True)
