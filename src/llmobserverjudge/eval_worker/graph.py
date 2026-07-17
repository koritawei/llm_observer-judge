"""Eval graph pipeline: Judge → Score → post-eval DispatchClient (SDD-EV-20).

Production path is LangGraph StateGraph (ADR-0001); EvalDAGExecutor remains the
multi-judge layer engine (ADR-0006).
"""

from __future__ import annotations

from typing import Any

from llmobserverjudge.contracts.models import EvalJob, GraphState
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.eval_worker.judges import JudgeModelRouter
from llmobserverjudge.eval_worker.langgraph_pipeline import run_eval_langgraph
from llmobserverjudge.eval_worker.prompt import (
    InMemoryPromptService,
    build_prompt_variables,
)


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
    # Optional Langfuse prompt override
    try:
        from llmobserverjudge.shared.langfuse_bridge import try_get_prompt_content

        lf = await try_get_prompt_content(str(prompt_id))
        if lf:
            vars_ = build_prompt_variables(
                agent_id=agent_id, cfg=out, trace=trace, upstream=upstream
            )
            out["resolved_prompt"] = InMemoryPromptService.render_template(lf, **vars_)
            out["prompt_version_id"] = f"langfuse:{prompt_id}"
            return out
    except Exception:
        pass
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
    return await run_eval_langgraph(
        job,
        judge_router=judge_router,
        dispatch_client=dispatch_client,
        prompt_service=prompt_service,
        max_concurrent_per_layer=max_concurrent_per_layer,
        seed_prompts=seed_prompts,
    )
