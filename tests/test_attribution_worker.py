"""STEP-07: Attribution Worker, stub ReAct, checkpoint, SSRF."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmobserverjudge.attribution_worker.checkpoint import InMemoryCheckpointStore
from llmobserverjudge.attribution_worker.context import (
    AttributionContext,
    InMemoryAttributionContextStore,
)
from llmobserverjudge.attribution_worker.react import ReActAttributor
from llmobserverjudge.attribution_worker.ssrf import ToolEndpointValidator
from llmobserverjudge.attribution_worker.store import InMemoryAttributionStore
from llmobserverjudge.attribution_worker.worker import AttributionWorker
from llmobserverjudge.contracts.idempotency import attribution_key
from llmobserverjudge.contracts.models import EvalConfig, EvalJob, Priority
from llmobserverjudge.contracts.topics import Component, ProducerForbiddenError, Topic
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.eval_worker.graph import run_eval_graph
from llmobserverjudge.eval_worker.judges import JudgeModelRouter, ScriptedJudgeAdapter
from llmobserverjudge.shared.mq import InMemoryMQ


def test_ssrf_blocks_metadata_and_non_whitelist() -> None:
    v = ToolEndpointValidator(allowed_hosts=["biz-api.example.com"])
    with pytest.raises(ValueError, match="Blocked host"):
        v.validate("http://169.254.169.254/latest/meta-data")
    with pytest.raises(ValueError, match="whitelist"):
        v.validate("https://evil.example.org/x")


@pytest.mark.asyncio
async def test_react_completes_with_tools_and_root_cause() -> None:
    ckpt = InMemoryCheckpointStore()
    attr = ReActAttributor(checkpoint=ckpt)
    # Raise early-stop bar so we exercise checkpoint at round 3
    attr.EARLY_STOP_CONFIDENCE = 0.99

    job_key = attribution_key("0123456789abcdef0123456789abcdef", "judge_safety")
    from llmobserverjudge.contracts.models import AttributionJob

    job = AttributionJob(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_safety",
        priority=Priority.P0,
        idempotency_key=job_key,
        score_snapshot={"score": 0.3, "label": "SAFETY_RISK"},
        trace_summary={"status": "ERROR", "tags": ["safety"]},
    )
    ctx = AttributionContext(
        tenant_id="t1",
        trace_id=job.trace_id,
        trace_summary=job.trace_summary,
        spans=[{"span_id": "0123456789abcdef", "status": "ERROR", "name": "llm"}],
        score_snapshot=job.score_snapshot,
        eval_agent_id="judge_safety",
    )
    result = await attr.analyze(job, ctx)
    assert result.status == "COMPLETED"
    assert result.root_cause == "RUNTIME_ERROR"
    assert "get_trace_summary" in result.tools_called
    assert result.rounds >= 3
    saved = await ckpt.load(
        job.trace_id, tenant_id=job.tenant_id, project_id=job.project_id or "default"
    )
    assert saved is not None
    assert saved["round"] >= 3


@pytest.mark.asyncio
async def test_react_early_stop() -> None:
    attr = ReActAttributor()
    from llmobserverjudge.contracts.models import AttributionJob

    job = AttributionJob(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_quality",
        priority=Priority.P1,
        idempotency_key="k-early",
        score_snapshot={"score": 0.4, "label": "bad"},
        trace_summary={"status": "OK", "tags": [], "response": "x"},
    )
    ctx = AttributionContext(
        tenant_id="t1",
        trace_id=job.trace_id,
        trace_summary=job.trace_summary,
        score_snapshot=job.score_snapshot,
        eval_agent_id="judge_quality",
    )
    result = await attr.analyze(job, ctx)
    assert result.status == "COMPLETED"
    assert result.early_stop is True
    assert result.confidence >= 0.85
    assert result.rounds >= 3


@pytest.mark.asyncio
async def test_worker_idempotent() -> None:
    mq = InMemoryMQ()
    store = InMemoryAttributionStore()
    worker = AttributionWorker(mq, store=store)
    payload = {
        "trace_id": "0123456789abcdef0123456789abcdef",
        "tenant_id": "t1",
        "eval_agent_id": "judge_safety",
        "attribution_agent": "attributor_default",
        "priority": "P0",
        "score_snapshot": {"score": 0.2},
        "trace_summary": {"status": "ERROR"},
        "idempotency_key": "idem-attr-1",
    }
    await mq.publish(Component.DISPATCH, Topic.ATTRIBUTION_JOBS, payload)
    r1 = await worker.process_once()
    assert len(r1) == 1
    await mq.publish(Component.DISPATCH, Topic.ATTRIBUTION_JOBS, payload)
    r2 = await worker.process_once()
    assert r2 == [] or (len(r2) == 1 and r2[0].attribution_id == r1[0].attribution_id)
    # store only one logical result
    listed = await store.list_for_trace("t1", payload["trace_id"])
    assert len(listed) == 1


@pytest.mark.asyncio
async def test_eval_to_attribution_e2e() -> None:
    mq = InMemoryMQ()
    client = DispatchClient(mq)
    scripted = ScriptedJudgeAdapter(scores={"judge_safety": 2.0})
    router = JudgeModelRouter(
        registry={"scripted": scripted, "code_eval": scripted},
        default_adapter=scripted,
    )
    job = EvalJob(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        trace_summary={"status": "ERROR", "tags": ["safety"], "response": "boom"},
        eval_configs=[EvalConfig(agent_id="judge_safety", model="scripted")],
        priority=Priority.P0,
        idempotency_key="eval-attr-e2e",
        created_at=datetime.now(UTC),
    )
    state = await run_eval_graph(job, judge_router=router, dispatch_client=client)
    assert state.bad_case_detected is True
    assert await mq.depth(Topic.ATTRIBUTION_JOBS) == 1

    ctx_store = InMemoryAttributionContextStore()
    await ctx_store.put(
        AttributionContext(
            tenant_id="t1",
            trace_id=job.trace_id,
            trace_summary=job.trace_summary,
            spans=[{"span_id": "aaaaaaaaaaaaaaaa", "status": "ERROR", "name": "chat"}],
            eval_agent_id="judge_safety",
        )
    )
    worker = AttributionWorker(mq, context_store=ctx_store)
    results = await worker.process_once()
    assert len(results) == 1
    assert results[0].status == "COMPLETED"
    assert results[0].root_cause == "RUNTIME_ERROR"
    assert await mq.depth(Topic.ATTRIBUTION_JOBS) == 0


@pytest.mark.asyncio
async def test_attribution_worker_cannot_publish_eval_jobs() -> None:
    mq = InMemoryMQ()
    with pytest.raises(ProducerForbiddenError):
        await mq.publish(Component.ATTRIBUTION_WORKER, Topic.EVAL_JOBS, {"x": 1})


def test_attribution_source_has_no_eval_jobs_publish() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "llmobserverjudge"
        / "attribution_worker"
    )
    for path in root.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_publish = (
                isinstance(func, ast.Attribute) and func.attr == "publish"
            ) or (isinstance(func, ast.Name) and func.id == "publish")
            if not is_publish:
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Attribute) and arg.attr == "EVAL_JOBS":
                    pytest.fail(f"forbidden publish(EVAL_JOBS) in {path}")
