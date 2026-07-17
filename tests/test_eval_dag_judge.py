"""STEP-06: EvalDAG execute, Judge fallback, Score, DispatchClient post-eval."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llmobserverjudge.contracts.models import EvalConfig, EvalJob, Priority
from llmobserverjudge.contracts.topics import Component, Topic
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.eval_worker.conditions import evaluate_eval_condition
from llmobserverjudge.eval_worker.dag import EvalDAGExecutor
from llmobserverjudge.eval_worker.graph import run_eval_graph
from llmobserverjudge.eval_worker.judges import (
    FailingJudgeAdapter,
    JudgeModelRouter,
    RuleJudgeAdapter,
    ScriptedJudgeAdapter,
)
from llmobserverjudge.eval_worker.score import aggregate_scores
from llmobserverjudge.eval_worker.worker import EvalWorker
from llmobserverjudge.shared.mq import InMemoryMQ


def test_dag_unknown_dependency_raises() -> None:
    with pytest.raises(ValueError, match="unknown agent"):
        EvalDAGExecutor(
            [
                {"agent_id": "a", "depends_on": ["missing"]},
            ]
        )


def test_dag_cycle_raises() -> None:
    with pytest.raises(ValueError, match="Cycle"):
        EvalDAGExecutor(
            [
                {"agent_id": "a", "depends_on": ["b"]},
                {"agent_id": "b", "depends_on": ["a"]},
            ]
        )


def test_condition_ast_safe() -> None:
    assert evaluate_eval_condition(
        "judge_safety.normalized_score < 0.7",
        {"judge_safety": {"normalized_score": 0.3}},
    )
    assert not evaluate_eval_condition(
        "judge_safety.normalized_score < 0.7",
        {"judge_safety": {"normalized_score": 0.9}},
    )
    # disallowed call → False (fail closed)
    assert not evaluate_eval_condition("__import__('os').system('x')", {})


@pytest.mark.asyncio
async def test_dag_skips_on_condition_and_upstream_fail() -> None:
    executor = EvalDAGExecutor(
        [
            {"agent_id": "judge_safety", "model": "scripted", "scale_max": 10},
            {
                "agent_id": "judge_safety_deep",
                "model": "scripted",
                "depends_on": ["judge_safety"],
                "condition": "judge_safety.normalized_score < 0.7",
                "scale_max": 10,
            },
            {
                "agent_id": "judge_followup",
                "model": "scripted",
                "depends_on": ["judge_safety"],
                "scale_max": 10,
            },
        ]
    )
    # safety high → deep skipped; followup runs
    scripted = ScriptedJudgeAdapter(scores={"judge_safety": 9.0, "judge_followup": 8.0})
    router = JudgeModelRouter(
        registry={"scripted": scripted, "code_eval": scripted},
        default_adapter=scripted,
    )

    async def eval_fn(aid, cfg, up):
        return await router.invoke_with_fallback(
            agent_id=aid, cfg=cfg, trace={}, upstream=up
        )

    results = await executor.execute({}, eval_fn)
    by_id = {r["agent_id"]: r for r in results}
    assert by_id["judge_safety"]["score_status"] == "scored"
    assert by_id["judge_safety_deep"]["score_status"] == "skipped"
    assert "condition not met" in by_id["judge_safety_deep"]["skip_reason"]
    assert by_id["judge_followup"]["score_status"] == "scored"

    # upstream infra_error → followup skipped
    executor2 = EvalDAGExecutor(
        [
            {"agent_id": "judge_safety", "model": "bad"},
            {
                "agent_id": "judge_followup",
                "model": "scripted",
                "depends_on": ["judge_safety"],
            },
        ]
    )
    failing = FailingJudgeAdapter()
    router2 = JudgeModelRouter(
        registry={"bad": failing, "code_eval": failing, "scripted": scripted},
        default_adapter=failing,
    )

    async def eval_fn2(aid, cfg, up):
        return await router2.invoke_with_fallback(
            agent_id=aid, cfg=cfg, trace={}, upstream=up
        )

    results2 = await executor2.execute({}, eval_fn2)
    by_id2 = {r["agent_id"]: r for r in results2}
    assert by_id2["judge_safety"]["score_status"] == "infra_error"
    assert by_id2["judge_followup"]["score_status"] == "skipped"
    assert by_id2["judge_followup"]["skip_reason"] == "upstream_failed"


@pytest.mark.asyncio
async def test_model_error_does_not_block_downstream() -> None:
    """Soft model_error (bad-case-like) must not abort dependents."""
    executor = EvalDAGExecutor(
        [
            {"agent_id": "judge_a", "model": "soft"},
            {
                "agent_id": "judge_b",
                "model": "scripted",
                "depends_on": ["judge_a"],
            },
        ]
    )
    scripted = ScriptedJudgeAdapter({"judge_b": 8.0})

    async def eval_fn(aid, cfg, up):
        if aid == "judge_a":
            return {
                "agent_id": "judge_a",
                "score_status": "model_error",
                "error": "parse",
            }
        return await scripted.judge(agent_id=aid, cfg=cfg, trace={}, upstream=up)

    results = await executor.execute({}, eval_fn)
    by_id = {r["agent_id"]: r for r in results}
    assert by_id["judge_a"]["score_status"] == "model_error"
    assert by_id["judge_b"]["score_status"] == "scored"


@pytest.mark.asyncio
async def test_infra_error_aborts_subsequent_layers() -> None:
    """Hard infra_error stops later layers (fail-stop)."""
    executor = EvalDAGExecutor(
        [
            {"agent_id": "judge_a", "model": "bad"},
            {"agent_id": "judge_b", "model": "scripted"},  # same layer peer — may run
            {"agent_id": "judge_c", "model": "scripted", "depends_on": ["judge_b"]},
        ]
    )
    scripted = ScriptedJudgeAdapter({"judge_b": 9.0, "judge_c": 9.0})

    async def eval_fn(aid, cfg, up):
        if aid == "judge_a":
            return {
                "agent_id": "judge_a",
                "score_status": "infra_error",
                "error": "boom",
            }
        return await scripted.judge(agent_id=aid, cfg=cfg, trace={}, upstream=up)

    results = await executor.execute({}, eval_fn)
    by_id = {r["agent_id"]: r for r in results}
    assert by_id["judge_a"]["score_status"] == "infra_error"
    # Layer-1 must be aborted after hard fail in layer-0
    assert by_id["judge_c"]["score_status"] == "skipped"
    assert "aborted_after" in by_id["judge_c"]["skip_reason"]


@pytest.mark.asyncio
async def test_fallback_to_code_eval() -> None:
    router = JudgeModelRouter(
        registry={
            "primary": FailingJudgeAdapter(),
            "code_eval": RuleJudgeAdapter(),
        },
        default_adapter=FailingJudgeAdapter(),
    )
    result = await router.invoke_with_fallback(
        agent_id="judge_quality",
        cfg={"model": "primary", "fallback_models": [], "scale_max": 10},
        trace={"status": "OK", "response": "hello world answer"},
        upstream={},
    )
    assert result["score_status"] == "scored"
    assert result["model_used"] == "code_eval"


def test_aggregate_skips_infra_and_flags_model_error() -> None:
    score = aggregate_scores(
        tenant_id="t",
        trace_id="0123456789abcdef0123456789abcdef",
        results=[
            {
                "agent_id": "a",
                "score_status": "scored",
                "score": 8,
                "scale_max": 10,
                "weight": 1,
                "threshold": 0.5,
            },
            {"agent_id": "b", "score_status": "infra_error", "error": "x"},
            {"agent_id": "c", "score_status": "model_error"},
            {"agent_id": "d", "score_status": "skipped", "skip_reason": "cond"},
        ],
    )
    assert score.final_score == 0.8
    assert "c" in score.bad_case_evals
    assert "b" not in score.bad_case_evals
    statuses = {e.agent_id: e.score_status.value for e in score.eval_scores}
    assert statuses["b"] == "infra_error"
    assert statuses["d"] == "skipped"


@pytest.mark.asyncio
async def test_run_eval_graph_routes_post_eval() -> None:
    mq = InMemoryMQ()
    client = DispatchClient(mq)
    job = EvalJob(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        trace_summary={"status": "OK", "tags": [], "response": "ok"},
        eval_configs=[
            EvalConfig(agent_id="judge_safety", model="scripted", threshold=0.5),
        ],
        priority=Priority.P0,
        idempotency_key="k1",
        created_at=datetime.now(UTC),
    )
    # score 0.3 → attribution (threshold 0.7) + annotation always
    scripted = ScriptedJudgeAdapter(scores={"judge_safety": 3.0})
    router = JudgeModelRouter(
        registry={"scripted": scripted, "code_eval": scripted},
        default_adapter=scripted,
    )
    state = await run_eval_graph(job, judge_router=router, dispatch_client=client)
    assert state.final_score == 0.3
    assert state.bad_case_detected is True
    assert await mq.depth(Topic.ANNOTATION_JOBS) == 1
    assert await mq.depth(Topic.ATTRIBUTION_JOBS) == 1
    assert state.annotation_task_count == 1


@pytest.mark.asyncio
async def test_worker_idempotent_and_no_eval_jobs_publish() -> None:
    mq = InMemoryMQ()
    job = {
        "trace_id": "0123456789abcdef0123456789abcdef",
        "tenant_id": "t1",
        "trace_summary": {"status": "OK", "response": "long enough answer here"},
        "eval_configs": [
            {"agent_id": "judge_quality", "model": "default-judge"},
        ],
        "priority": "P2",
        "idempotency_key": "idem-1",
        "created_at": datetime.now(UTC).isoformat(),
    }
    await mq.publish(Component.DISPATCH, Topic.EVAL_JOBS, job)
    worker = EvalWorker(mq)
    s1 = await worker.process_once()
    assert len(s1) == 1
    assert s1[0].final_score > 0
    await mq.publish(Component.DISPATCH, Topic.EVAL_JOBS, job)
    s2 = await worker.process_once()
    assert s2 == []  # idempotent skip
