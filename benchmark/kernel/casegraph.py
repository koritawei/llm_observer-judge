"""Run one worker phase on the evaluator LangGraph.

The topology is the same closed set the evaluator runner compiles:
build context, resolve the profile, collect evidence, run checkers, plan
judges, run judges, collect metrics, aggregate, cluster, and emit the envelope.
Scheduling and leases stay in the worker. This graph only calls the frozen
plugins and returns their outcome.
"""

from __future__ import annotations

from typing import Any, TypedDict

from benchmark.kernel.casebook import trace_from_report
from benchmark.kernel.types import Completed
from benchmark.pipeline.cluster import cluster_dimension_results
from benchmark.pipeline.runner import EvaluationPipelineHooks, build_evaluation_graph, skipped


class PhaseGraphState(TypedDict, total=False):
    context: dict[str, Any]
    profile: dict[str, Any]
    request: Any
    execution_ref: str
    outcome: Any
    error: str
    error_code: str
    stage_report: dict[str, Any]
    envelope: Any
    _exc: BaseException


async def _call(plugin: Any, method: str, *args: Any) -> Any:
    async_name = "a" + method
    target = getattr(plugin, async_name, None)
    if target is not None:
        return await target(*args)
    target = getattr(plugin, method)
    result = target(*args)
    if hasattr(result, "__await__"):
        return await result
    return result


async def execute_phase(host: Any, ctx: Any) -> Any:
    async def build_context(_state: PhaseGraphState) -> dict[str, Any]:
        return {"context": {"run_id": ctx.run_id, "item_id": ctx.item_id}}

    async def resolve_profile(_state: PhaseGraphState) -> dict[str, Any]:
        phase = ctx.phase or {}
        return {"profile": {"phase_id": phase.get("phase_id"), "role": phase.get("role")}}

    async def collect_evidence(_state: PhaseGraphState) -> dict[str, Any]:
        builder = host.load(ctx.phase["request_builder"])
        ctx.request = await _call(builder, "build", ctx)
        return {"request": ctx.request}

    async def run_checkers(_state: PhaseGraphState) -> dict[str, Any]:
        return skipped("phase has no checker pack")

    async def plan_judges(_state: PhaseGraphState) -> dict[str, Any]:
        execution = ctx.phase.get("execution") or {}
        return {"execution_ref": str(execution.get("class_ref") or "")}

    async def run_judges(_state: PhaseGraphState) -> dict[str, Any]:
        try:
            execution = host.load(ctx.phase["execution"])
            return {"outcome": await _call(execution, "execute", ctx)}
        except Exception as exc:
            return {"error": str(exc), "_exc": exc}

    async def collect_metrics(_state: PhaseGraphState) -> dict[str, Any]:
        return {}

    async def aggregate_results(state: PhaseGraphState) -> dict[str, Any]:
        return {"envelope": state.get("outcome")}

    async def cluster_results(state: PhaseGraphState) -> dict[str, Any]:
        outcome = state.get("outcome")
        payload = getattr(outcome, "result", outcome)
        rows: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            raw = payload.get("rows")
            rows = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else [payload]
        phase = ctx.phase or {}
        expected = list(phase.get("expected_dimensions") or [])
        return {"stage_report": {"cluster": cluster_dimension_results(rows, expected_dimensions=expected)}}

    async def emit_envelope(state: PhaseGraphState) -> dict[str, Any]:
        return {"outcome": state.get("outcome")}

    hooks = EvaluationPipelineHooks(
        build_context=build_context,
        resolve_profile=resolve_profile,
        collect_evidence=collect_evidence,
        run_checkers=run_checkers,
        plan_judges=plan_judges,
        run_judges=run_judges,
        collect_metrics=collect_metrics,
        aggregate_results=aggregate_results,
        cluster_results=cluster_results,
        emit_envelope=emit_envelope,
    )
    graph = build_evaluation_graph(PhaseGraphState, hooks)
    final = await graph.ainvoke({})
    outcome = final.get("outcome")
    if isinstance(outcome, Completed) and isinstance(outcome.result, dict):
        outcome.result["_graph"] = trace_from_report(final.get("stage_report") or {})
    exc = final.get("_exc")
    if exc is not None:
        raise exc
    return outcome
