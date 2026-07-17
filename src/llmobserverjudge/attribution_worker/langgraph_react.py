"""LangGraph ReAct loop for attribution (ADR-0001 / ADR-0004).

Business tools + Stub/LLM policy stay local; the control loop and optional
MemorySaver checkpointing use LangGraph. Domain checkpoint store is still
written every CHECKPOINT_EVERY rounds for SQL persistence compatibility.
"""

from __future__ import annotations

import time
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from llmobserverjudge.attribution_worker.checkpoint import InMemoryCheckpointStore
from llmobserverjudge.attribution_worker.context import AttributionContext
from llmobserverjudge.attribution_worker.tools import BuiltinToolRuntime
from llmobserverjudge.contracts.models import AttributionJob, AttributionResult


class ReactLoopState(TypedDict, total=False):
    rounds: int
    confidence: float
    observations: list[dict[str, Any]]
    conclusions: list[str]
    root_cause: str
    early_stop: bool
    done: bool
    tools_called: list[str]
    submitted: dict[str, Any] | None
    error: str | None


def build_react_graph(
    *,
    policy: Any,
    tools: BuiltinToolRuntime,
    job: AttributionJob,
    ctx: AttributionContext,
    checkpoint_store: Any,
    project_id: str,
    max_rounds: int,
    early_stop_confidence: float,
    early_stop_min_rounds: int,
    checkpoint_every: int,
    estimate_confidence: Any,
    infer_root_cause: Any,
    collect_evidence: Any,
) -> Any:
    async def step_node(state: ReactLoopState) -> dict[str, Any]:
        rounds = int(state.get("rounds") or 0) + 1
        observations = list(state.get("observations") or [])
        conclusions = list(state.get("conclusions") or [])
        thought, action, args, llm_self = await policy.next_step(
            rounds=rounds, tools=tools, observations=observations
        )
        obs = await tools.invoke(action, args)
        observations.append(
            {"round": rounds, "thought": thought, "action": action, "observation": obs}
        )

        if action == "submit_attribution" and tools.submitted:
            root_cause = tools.submitted["root_cause"]
            conclusions.append(root_cause)
            confidence = estimate_confidence(
                llm_self=float(tools.submitted.get("confidence") or llm_self),
                conclusions=conclusions,
                tools_called=tools.tools_called,
                observations=observations,
            )
            early_stop = (
                confidence >= early_stop_confidence and rounds >= early_stop_min_rounds
            )
            return {
                "rounds": rounds,
                "observations": observations,
                "conclusions": conclusions,
                "root_cause": root_cause,
                "confidence": confidence,
                "early_stop": early_stop,
                "done": True,
                "tools_called": list(tools.tools_called),
                "submitted": dict(tools.submitted),
            }

        provisional = infer_root_cause(
            status=str(ctx.trace_summary.get("status") or "OK").upper(),
            tags=ctx.trace_summary.get("tags") or [],
            score=ctx.score_snapshot.get("score"),
            observations=observations,
        )
        conclusions.append(provisional)
        confidence = estimate_confidence(
            llm_self=llm_self,
            conclusions=conclusions,
            tools_called=tools.tools_called,
            observations=observations,
        )

        if confidence >= early_stop_confidence and rounds >= early_stop_min_rounds:
            evidence = collect_evidence(tools, observations)
            await tools.invoke(
                "submit_attribution",
                {
                    "root_cause": provisional,
                    "confidence": confidence,
                    "evidence": evidence,
                    "summary": "early_stop",
                },
            )
            return {
                "rounds": rounds,
                "observations": observations,
                "conclusions": conclusions,
                "root_cause": provisional,
                "confidence": confidence,
                "early_stop": True,
                "done": True,
                "tools_called": list(tools.tools_called),
                "submitted": dict(tools.submitted or {}),
            }

        if rounds % checkpoint_every == 0:
            await checkpoint_store.save(
                job.trace_id,
                {
                    "round": rounds,
                    "confidence": confidence,
                    "tools": tools.tools_called,
                },
                tenant_id=job.tenant_id,
                project_id=project_id,
            )

        if rounds >= max_rounds:
            root_cause = conclusions[-1] if conclusions else "MAX_ROUNDS"
            await tools.invoke(
                "submit_attribution",
                {
                    "root_cause": root_cause,
                    "confidence": confidence,
                    "evidence": collect_evidence(tools, observations),
                    "summary": "max_rounds",
                },
            )
            return {
                "rounds": rounds,
                "observations": observations,
                "conclusions": conclusions,
                "root_cause": root_cause,
                "confidence": confidence,
                "early_stop": False,
                "done": True,
                "tools_called": list(tools.tools_called),
                "submitted": dict(tools.submitted or {}),
            }

        return {
            "rounds": rounds,
            "observations": observations,
            "conclusions": conclusions,
            "confidence": confidence,
            "early_stop": False,
            "done": False,
            "tools_called": list(tools.tools_called),
            "root_cause": provisional,
        }

    def route(state: ReactLoopState) -> Literal["step", "__end__"]:
        if state.get("done"):
            return "__end__"
        return "step"

    g = StateGraph(ReactLoopState)
    g.add_node("step", step_node)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", route, {"step": "step", "__end__": END})
    return g.compile(checkpointer=MemorySaver())


async def run_react_langgraph(
    *,
    attributor: Any,
    job: AttributionJob,
    ctx: AttributionContext,
) -> AttributionResult:
    started = time.perf_counter()
    tools = BuiltinToolRuntime(ctx)
    project_id = getattr(job, "project_id", None) or "default"
    checkpoint = attributor.checkpoint or InMemoryCheckpointStore()

    rounds0 = 0
    confidence0 = 0.0
    saved = await checkpoint.load(
        job.trace_id, tenant_id=job.tenant_id, project_id=project_id
    )
    if saved:
        rounds0 = int(saved.get("round", 0))
        confidence0 = float(saved.get("confidence", 0.0))

    from llmobserverjudge.attribution_worker.react import (
        _collect_evidence,
        _infer_root_cause,
    )

    graph = build_react_graph(
        policy=attributor.policy,
        tools=tools,
        job=job,
        ctx=ctx,
        checkpoint_store=checkpoint,
        project_id=project_id,
        max_rounds=attributor.max_rounds,
        early_stop_confidence=attributor.EARLY_STOP_CONFIDENCE,
        early_stop_min_rounds=attributor.EARLY_STOP_MIN_ROUNDS,
        checkpoint_every=attributor.CHECKPOINT_EVERY,
        estimate_confidence=attributor._estimate_confidence,
        infer_root_cause=_infer_root_cause,
        collect_evidence=_collect_evidence,
    )
    thread_id = f"{job.tenant_id}:{project_id}:{job.trace_id}:{job.eval_agent_id}"
    try:
        final = await graph.ainvoke(
            {
                "rounds": rounds0,
                "confidence": confidence0,
                "observations": [],
                "conclusions": [],
                "root_cause": "UNKNOWN",
                "early_stop": False,
                "done": False,
                "tools_called": [],
                "submitted": None,
            },
            {"configurable": {"thread_id": thread_id}},
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        return AttributionResult(
            attribution_id=str(uuid4()),
            trace_id=job.trace_id,
            tenant_id=job.tenant_id,
            project_id=project_id,
            eval_agent_id=job.eval_agent_id,
            attribution_agent=job.attribution_agent,
            root_cause=str(final.get("root_cause") or "UNKNOWN"),
            confidence=round(float(final.get("confidence") or 0.0), 4),
            evidence=_collect_evidence(tools, list(final.get("observations") or [])),
            tools_called=list(final.get("tools_called") or tools.tools_called),
            rounds=int(final.get("rounds") or 0),
            early_stop=bool(final.get("early_stop")),
            status="COMPLETED",
            duration_ms=duration_ms,
        )
    except Exception as e:
        return AttributionResult(
            attribution_id=str(uuid4()),
            trace_id=job.trace_id,
            tenant_id=job.tenant_id,
            project_id=project_id,
            eval_agent_id=job.eval_agent_id,
            attribution_agent=job.attribution_agent,
            root_cause="ATTRIBUTION_FAILED",
            confidence=0.0,
            evidence=[{"error": str(e), "error_type": type(e).__name__}],
            tools_called=list(tools.tools_called),
            rounds=0,
            early_stop=False,
            status="FAILED",
            error=str(e),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
