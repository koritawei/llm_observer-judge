"""Eval graph pipeline: Judge → Score → post-eval DispatchClient (SDD-EV-20).

LangGraph StateGraph wiring is deferred; this module is the logical production path
(ADR-0006: JudgeNode + EvalDAGExecutor).
"""

from __future__ import annotations

from typing import Any

from llmobserverjudge.contracts.models import EvalJob, GraphState, ScoreStatus
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.eval_worker.dag import EvalDAGExecutor
from llmobserverjudge.eval_worker.execution_graph import build_execution_graph
from llmobserverjudge.eval_worker.judges import JudgeModelRouter, default_judge_router
from llmobserverjudge.eval_worker.prompt import (
    InMemoryPromptService,
    build_prompt_variables,
    seed_default_prompts,
)
from llmobserverjudge.eval_worker.score import aggregate_scores


async def resolve_prompt_for_cfg(
    cfg: dict[str, Any],
    *,
    agent_id: str,
    trace: dict[str, Any],
    upstream: dict[str, Any],
    prompt_service: InMemoryPromptService | None,
    trace_id: str,
) -> dict[str, Any]:
    """Attach resolved_prompt when prompt_id is set (or default seed)."""
    out = dict(cfg)
    if out.get("resolved_prompt"):
        return out
    svc = prompt_service
    prompt_id = out.get("prompt_id")
    if svc is None:
        return out
    if not prompt_id:
        return out
    version = await svc.resolve_for_trace(str(prompt_id), trace_id)
    if version is None:
        return out
    vars_ = build_prompt_variables(
        agent_id=agent_id, cfg=out, trace=trace, upstream=upstream
    )
    out["resolved_prompt"] = InMemoryPromptService.render_template(
        version.content, **vars_
    )
    out["prompt_version_id"] = version.version_id
    return out


async def run_eval_graph(
    job: EvalJob,
    *,
    judge_router: JudgeModelRouter | None = None,
    dispatch_client: DispatchClient | None = None,
    prompt_service: InMemoryPromptService | None = None,
    max_concurrent_per_layer: int = 10,
    seed_prompts: bool = True,
) -> GraphState:
    router = judge_router or default_judge_router()
    svc = prompt_service
    if svc is None and seed_prompts:
        svc = InMemoryPromptService()
        await seed_default_prompts(svc)

    configs = [c.model_dump() for c in job.eval_configs]
    executor = EvalDAGExecutor(configs, max_concurrent_per_layer=max_concurrent_per_layer)
    layers = executor.topological_layers()
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

    results = await executor.execute(trace, eval_fn)
    # Enrich scored rows with is_bad_case for attribution gate + graph
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

    score = aggregate_scores(
        tenant_id=job.tenant_id,
        trace_id=job.trace_id,
        results=results,
        prompt_version_id=resolved_prompt_version,
    )

    annotation_ids: list[str] = []
    attribution_count = 0
    if dispatch_client is not None:
        for entry in score.eval_scores:
            if entry.score_status != ScoreStatus.SCORED:
                continue
            routed = await dispatch_client.route_post_eval(
                trace_id=job.trace_id,
                tenant_id=job.tenant_id,
                eval_agent_id=entry.agent_id,
                score=entry.normalized_score,
                confidence=None,
                label=entry.label,
                conflict_detected=bool(entry.conflict_detected),
                is_bad_case=bool(entry.is_bad_case),
                trace_summary=job.trace_summary,
            )
            if routed.get("annotation_published"):
                annotation_ids.append(entry.agent_id)
            if routed.get("attribution_published"):
                attribution_count += 1

    node_errors: dict[str, str] = {}
    optional_skipped: list[str] = []
    for r in results:
        aid = str(r.get("agent_id") or "")
        st = r.get("score_status")
        if st == "infra_error":
            node_errors[aid] = str(r.get("error") or "infra_error")
        elif st == "skipped":
            optional_skipped.append(aid)
        # model_error / bad_case: soft — recorded but do not block flow

    execution_graph = build_execution_graph(
        layers=layers,
        results=results,
        bad_case_detected=score.bad_case_detected,
        attribution_triggered=attribution_count > 0,
        annotation_triggered=len(annotation_ids) > 0,
    )

    return GraphState(
        trace_id=job.trace_id,
        tenant_id=job.tenant_id,
        trace_summary=job.trace_summary,
        prompt_version_id=resolved_prompt_version,
        eval_configs=configs,
        eval_results=results,
        eval_scores=[e.model_dump() for e in score.eval_scores],
        eval_execution_layers=layers,
        final_score=score.final_score,
        bad_case_detected=score.bad_case_detected,
        bad_case_evals=score.bad_case_evals,
        annotation_task_ids=annotation_ids,
        annotation_task_count=len(annotation_ids),
        node_errors=node_errors,
        optional_skipped=optional_skipped,
        execution_graph=execution_graph,
    )
