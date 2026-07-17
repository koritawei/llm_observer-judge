"""LangGraph orchestration for eval: judges → score → post_eval (ADR-0001 / ADR-0006).

Judge-layer DAG semantics (conditions, infra fail-stop) stay in EvalDAGExecutor;
this module is the production StateGraph wiring previously deferred in graph.py.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from llmobserverjudge.contracts.models import EvalJob, GraphState, ScoreStatus
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.eval_worker.dag import EvalDAGExecutor
from llmobserverjudge.eval_worker.execution_graph import build_execution_graph
from llmobserverjudge.eval_worker.judges import JudgeModelRouter, default_judge_router
from llmobserverjudge.eval_worker.prompt import InMemoryPromptService, seed_default_prompts
from llmobserverjudge.eval_worker.score import aggregate_scores


class EvalPipelineState(TypedDict, total=False):
    job: dict[str, Any]
    configs: list[dict[str, Any]]
    layers: list[list[str]]
    results: list[dict[str, Any]]
    score: dict[str, Any]
    annotation_ids: list[str]
    attribution_count: int
    project_id: str
    resolved_prompt_version: str | None
    graph_state: dict[str, Any]


def _build_eval_fn(
    *,
    job: EvalJob,
    router: JudgeModelRouter,
    svc: InMemoryPromptService | None,
) -> Any:
    from llmobserverjudge.eval_worker.graph import resolve_prompt_for_cfg

    trace = job.trace_summary

    async def eval_fn(
        agent_id: str, cfg: dict[str, Any], upstream: dict[str, Any]
    ) -> dict[str, Any]:
        resolved = await resolve_prompt_for_cfg(
            cfg,
            agent_id=agent_id,
            trace=trace,
            upstream=upstream,
            prompt_service=svc,
            trace_id=job.trace_id,
        )
        out = await router.invoke_with_fallback(
            agent_id=agent_id, cfg=resolved, trace=trace, upstream=upstream
        )
        if resolved.get("prompt_version_id") and "prompt_version_id" not in out:
            out["prompt_version_id"] = resolved["prompt_version_id"]
        return out

    return eval_fn


def build_eval_state_graph(
    *,
    judge_router: JudgeModelRouter | None = None,
    dispatch_client: DispatchClient | None = None,
    prompt_service: InMemoryPromptService | None = None,
    max_concurrent_per_layer: int = 10,
    seed_prompts: bool = True,
) -> Any:
    router = judge_router or default_judge_router()

    async def judges_node(state: EvalPipelineState) -> dict[str, Any]:
        job = EvalJob.model_validate(state["job"])
        svc = prompt_service
        if svc is None and seed_prompts:
            svc = InMemoryPromptService()
            await seed_default_prompts(svc)
        configs = [c.model_dump() for c in job.eval_configs]
        executor = EvalDAGExecutor(
            configs, max_concurrent_per_layer=max_concurrent_per_layer
        )
        layers = executor.topological_layers()
        results = await executor.execute(job.trace_summary, _build_eval_fn(job=job, router=router, svc=svc))
        for r in results:
            if r.get("score_status") != "scored":
                continue
            if "is_bad_case" not in r:
                norm = float(r.get("normalized_score") or 0)
                thr = float(r.get("threshold", 0.5))
                r["is_bad_case"] = norm < thr
        resolved_prompt_version = job.prompt_version_id
        for r in results:
            if r.get("prompt_version_id"):
                resolved_prompt_version = str(r["prompt_version_id"])
                break
        project_id = getattr(job, "project_id", None) or str(
            (job.trace_summary or {}).get("project_id") or "default"
        )
        return {
            "configs": configs,
            "layers": layers,
            "results": results,
            "resolved_prompt_version": resolved_prompt_version,
            "project_id": project_id,
            "annotation_ids": [],
            "attribution_count": 0,
        }

    async def score_node(state: EvalPipelineState) -> dict[str, Any]:
        job = EvalJob.model_validate(state["job"])
        score = aggregate_scores(
            tenant_id=job.tenant_id,
            trace_id=job.trace_id,
            results=list(state.get("results") or []),
            prompt_version_id=state.get("resolved_prompt_version"),
        )
        return {"score": score.model_dump(mode="json")}

    async def post_eval_node(state: EvalPipelineState) -> dict[str, Any]:
        job = EvalJob.model_validate(state["job"])
        score_raw = state.get("score") or {}
        annotation_ids: list[str] = []
        attribution_count = 0
        project_id = state.get("project_id") or "default"
        if dispatch_client is not None:
            for entry in score_raw.get("eval_scores") or []:
                if entry.get("score_status") != ScoreStatus.SCORED.value and entry.get(
                    "score_status"
                ) != "scored":
                    continue
                routed = await dispatch_client.route_post_eval(
                    trace_id=job.trace_id,
                    tenant_id=job.tenant_id,
                    project_id=project_id,
                    eval_agent_id=str(entry.get("agent_id") or ""),
                    score=entry.get("normalized_score"),
                    confidence=None,
                    label=entry.get("label"),
                    conflict_detected=bool(entry.get("conflict_detected")),
                    is_bad_case=bool(entry.get("is_bad_case")),
                    trace_summary=job.trace_summary,
                )
                if routed.get("annotation_published"):
                    annotation_ids.append(str(entry.get("agent_id") or ""))
                if routed.get("attribution_published"):
                    attribution_count += 1
        return {
            "annotation_ids": annotation_ids,
            "attribution_count": attribution_count,
        }

    async def assemble_node(state: EvalPipelineState) -> dict[str, Any]:
        job = EvalJob.model_validate(state["job"])
        results = list(state.get("results") or [])
        layers = list(state.get("layers") or [])
        score_raw = dict(state.get("score") or {})
        if not score_raw or "final_score" not in score_raw:
            rebuilt = aggregate_scores(
                tenant_id=job.tenant_id,
                trace_id=job.trace_id,
                results=results,
                prompt_version_id=state.get("resolved_prompt_version"),
            )
            score_raw = rebuilt.model_dump(mode="json")
        annotation_ids = list(state.get("annotation_ids") or [])
        attribution_count = int(state.get("attribution_count") or 0)
        project_id = state.get("project_id") or "default"

        node_errors: dict[str, str] = {}
        optional_skipped: list[str] = []
        for r in results:
            aid = str(r.get("agent_id") or "")
            st = r.get("score_status")
            if st == "infra_error":
                node_errors[aid] = str(r.get("error") or "infra_error")
            elif st == "skipped":
                optional_skipped.append(aid)

        execution_graph = build_execution_graph(
            layers=layers,
            results=results,
            bad_case_detected=bool(score_raw.get("bad_case_detected")),
            attribution_triggered=attribution_count > 0,
            annotation_triggered=len(annotation_ids) > 0,
        )
        gs = GraphState(
            trace_id=job.trace_id,
            tenant_id=job.tenant_id,
            project_id=project_id,
            trace_summary=job.trace_summary,
            prompt_version_id=state.get("resolved_prompt_version"),
            eval_configs=list(state.get("configs") or []),
            eval_results=results,
            eval_scores=list(score_raw.get("eval_scores") or []),
            eval_execution_layers=layers,
            final_score=float(score_raw.get("final_score") or 0.0),
            bad_case_detected=bool(score_raw.get("bad_case_detected")),
            bad_case_evals=list(score_raw.get("bad_case_evals") or []),
            annotation_task_ids=annotation_ids,
            annotation_task_count=len(annotation_ids),
            node_errors=node_errors,
            optional_skipped=optional_skipped,
            execution_graph=execution_graph,
        )
        return {"graph_state": gs.model_dump(mode="json")}

    g = StateGraph(EvalPipelineState)
    g.add_node("judges", judges_node)
    g.add_node("score", score_node)
    g.add_node("post_eval", post_eval_node)
    g.add_node("assemble", assemble_node)
    g.add_edge(START, "judges")
    g.add_edge("judges", "score")
    g.add_edge("score", "post_eval")
    g.add_edge("post_eval", "assemble")
    g.add_edge("assemble", END)
    return g.compile()


async def run_eval_langgraph(
    job: EvalJob,
    *,
    judge_router: JudgeModelRouter | None = None,
    dispatch_client: DispatchClient | None = None,
    prompt_service: InMemoryPromptService | None = None,
    max_concurrent_per_layer: int = 10,
    seed_prompts: bool = True,
) -> GraphState:
    app = build_eval_state_graph(
        judge_router=judge_router,
        dispatch_client=dispatch_client,
        prompt_service=prompt_service,
        max_concurrent_per_layer=max_concurrent_per_layer,
        seed_prompts=seed_prompts,
    )
    out = await app.ainvoke({"job": job.model_dump(mode="json")})
    return GraphState.model_validate(out["graph_state"])
