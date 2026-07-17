"""§10.2 statistics aggregations over in-memory stores."""

from __future__ import annotations

from collections import Counter
from typing import Any

from llmobserverjudge.dashboard.store import DashboardContext


async def stats_overview(
    ctx: DashboardContext, *, tenant_id: str | None = None, project_id: str | None = None
) -> dict[str, Any]:
    traces = await ctx.span_store.list_traces(
        tenant_id=tenant_id, project_id=project_id, limit=500
    )
    evals = await ctx.eval_results.list_all(
        tenant_id=tenant_id, project_id=project_id, limit=500
    )
    scores = [e.final_score for e in evals]
    bad = sum(1 for e in evals if e.bad_case_detected)
    return {
        "trace_count": len(traces),
        "eval_count": len(evals),
        "bad_case_rate": (bad / len(evals)) if evals else 0.0,
        "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
        "bad_case_count": bad,
    }


async def stats_platform(ctx: DashboardContext) -> dict[str, Any]:
    depths: dict[str, int] = {}
    if ctx.mq is not None:
        from llmobserverjudge.contracts.topics import Topic

        depths = {
            "spans": await ctx.mq.depth(Topic.SPANS),
            "eval_jobs": await ctx.mq.depth(Topic.EVAL_JOBS),
            "attribution_jobs": await ctx.mq.depth(Topic.ATTRIBUTION_JOBS),
            "annotation_jobs": await ctx.mq.depth(Topic.ANNOTATION_JOBS),
        }
    telemetry: dict[str, Any] = {}
    if ctx.telemetry is not None and hasattr(ctx.telemetry, "by_tenant"):
        telemetry = {
            tid: p.model_dump() if hasattr(p, "model_dump") else dict(p.__dict__)
            for tid, p in ctx.telemetry.by_tenant.items()
        }
    stack = await ctx.stack_info() if callable(ctx.stack_info) else {}
    return {
        "mq_depths": depths,
        "telemetry_by_tenant": telemetry,
        "stack": stack,
        "sampling": ctx.sampling.snapshot(),
    }


async def stats_score_histogram(
    ctx: DashboardContext,
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
    buckets: int = 10,
) -> dict[str, Any]:
    evals = await ctx.eval_results.list_all(
        tenant_id=tenant_id, project_id=project_id, limit=2000
    )
    edges = [i / buckets for i in range(buckets + 1)]
    counts = [0] * buckets
    for e in evals:
        s = max(0.0, min(0.999999, float(e.final_score)))
        idx = min(buckets - 1, int(s * buckets))
        counts[idx] += 1
    return {
        "buckets": [
            {"lo": edges[i], "hi": edges[i + 1], "count": counts[i]} for i in range(buckets)
        ],
        "total": len(evals),
    }


async def stats_per_eval(
    ctx: DashboardContext,
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    evals = await ctx.eval_results.list_all(
        tenant_id=tenant_id, project_id=project_id, limit=2000
    )
    by_agent: dict[str, list[float]] = {}
    for e in evals:
        for es in e.eval_scores:
            aid = str(es.get("agent_id") or "unknown")
            ns = es.get("normalized_score")
            if ns is None:
                continue
            by_agent.setdefault(aid, []).append(float(ns))
    agents = []
    for aid, vals in sorted(by_agent.items()):
        agents.append(
            {
                "agent_id": aid,
                "count": len(vals),
                "mean_score": sum(vals) / len(vals) if vals else 0.0,
                "scores": vals[:200],
            }
        )
    return {"agents": agents}


async def stats_sampling(ctx: DashboardContext) -> dict[str, Any]:
    return ctx.sampling.snapshot()


async def stats_attribution(
    ctx: DashboardContext,
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    if ctx.attributions is None:
        return {"categories": [], "total": 0}
    rows: list[Any] = []
    if hasattr(ctx.attributions, "list_recent"):
        rows = await ctx.attributions.list_recent(limit=500)
    elif hasattr(ctx.attributions, "_by_id"):
        rows = list(ctx.attributions._by_id.values())  # noqa: SLF001

    cats: Counter[str] = Counter()
    for r in rows:
        if tenant_id and getattr(r, "tenant_id", None) != tenant_id:
            continue
        if project_id and (getattr(r, "project_id", None) or "default") != project_id:
            continue
        root = getattr(r, "root_cause", None) or (
            r.get("root_cause") if isinstance(r, dict) else None
        )
        cats[str(root or "unknown")] += 1
    return {
        "categories": [{"name": k, "count": v} for k, v in cats.most_common()],
        "total": sum(cats.values()),
    }


async def stats_annotation_accuracy(
    ctx: DashboardContext,
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    if ctx.annotations is None:
        return {"compared": 0, "score_agree_rate": None, "items": []}
    rows = await ctx.annotations.list(
        tenant_id=tenant_id, project_id=project_id, status=None, limit=500
    )
    compared = 0
    agree = 0
    items = []
    for r in rows:
        auto = r.auto_score
        human = r.human_score
        if auto is None or human is None:
            continue
        compared += 1
        # agree if within 0.15 on normalized-ish scale (scores may be 0-1 or 1-5)
        diff = abs(float(auto) - float(human))
        scale = 1.0 if max(float(auto), float(human)) <= 1.5 else 5.0
        ok = diff <= 0.15 * scale
        if ok:
            agree += 1
        items.append(
            {
                "annotation_id": r.annotation_id,
                "trace_id": r.trace_id,
                "auto_score": auto,
                "human_score": human,
                "agree": ok,
            }
        )
    return {
        "compared": compared,
        "score_agree_rate": (agree / compared) if compared else None,
        "items": items[:100],
    }


async def stats_benchmark(
    ctx: DashboardContext,
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    if ctx.benchmark_store is None:
        return {"reports": [], "count": 0}
    reports = []
    store = ctx.benchmark_store
    if hasattr(store, "reports"):
        vals = list(store.reports.values())
    elif hasattr(store, "list_reports"):
        vals = await store.list_reports(limit=50)
    else:
        vals = []
    for r in vals:
        if tenant_id and getattr(r, "tenant_id", None) != tenant_id:
            continue
        if project_id and (getattr(r, "project_id", None) or "default") != project_id:
            continue
        reports.append(r.model_dump(mode="json") if hasattr(r, "model_dump") else r)
    reports.sort(key=lambda x: str(x.get("created_at") or x.get("date") or ""), reverse=True)
    return {"reports": reports[:50], "count": len(reports)}


async def stats_prompts(
    ctx: DashboardContext,
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Prompt usage from eval results (prompt_version_id counts)."""
    evals = await ctx.eval_results.list_all(
        tenant_id=tenant_id, project_id=project_id, limit=2000
    )
    counts: Counter[str] = Counter()
    for e in evals:
        pid = e.prompt_version_id or "default"
        counts[pid] += 1
    return {
        "versions": [{"prompt_version_id": k, "usage": v} for k, v in counts.most_common()],
        "total_evals": len(evals),
    }


def _arm_stats(evals: list[Any], version_id: str) -> dict[str, Any]:
    rows = [e for e in evals if (e.prompt_version_id or "") == version_id]
    n = len(rows)
    if not n:
        return {
            "prompt_version_id": version_id,
            "count": 0,
            "mean_score": None,
            "bad_case_rate": None,
            "bad_case_count": 0,
        }
    scores = [float(e.final_score) for e in rows]
    bad = sum(1 for e in rows if e.bad_case_detected)
    return {
        "prompt_version_id": version_id,
        "count": n,
        "mean_score": sum(scores) / n,
        "bad_case_rate": bad / n,
        "bad_case_count": bad,
    }


async def stats_ab_contrast(
    ctx: DashboardContext,
    *,
    experiment_id: str | None = None,
    control_version_id: str | None = None,
    treatment_version_id: str | None = None,
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Control vs treatment contrast from eval_results.prompt_version_id."""
    experiment = None
    if experiment_id and ctx.prompt_service is not None:
        experiment = await ctx.prompt_service.get_experiment(experiment_id)
        if experiment is None:
            return {"error": "experiment not found"}
        control_version_id = experiment.control_version_id
        treatment_version_id = experiment.treatment_version_id
    if not control_version_id or not treatment_version_id:
        return {"error": "control_version_id and treatment_version_id required"}

    evals = await ctx.eval_results.list_all(
        tenant_id=tenant_id, project_id=project_id, limit=5000
    )
    control = _arm_stats(evals, control_version_id)
    treatment = _arm_stats(evals, treatment_version_id)
    delta = None
    if control["mean_score"] is not None and treatment["mean_score"] is not None:
        delta = float(treatment["mean_score"]) - float(control["mean_score"])

    return {
        "experiment": experiment.to_dict() if experiment else None,
        "control": control,
        "treatment": treatment,
        "delta_mean_score": delta,
        "sample_total": control["count"] + treatment["count"],
    }


async def build_node_timeline(ctx: DashboardContext, trace_id: str) -> list[dict[str, Any]]:
    """Synthesize pipeline node timeline for Trace detail (poll + WS)."""
    nodes: list[dict[str, Any]] = []
    if ctx.event_bus is not None:
        for ev in ctx.event_bus.history(trace_id):
            nodes.append(ev.to_dict())
    if nodes:
        return nodes
    # Fallback from stores
    spans = await ctx.span_store.list_spans_by_trace(trace_id)
    if spans:
        nodes.append(
            {
                "trace_id": trace_id,
                "tenant_id": spans[0].get("tenant_id"),
                "node": "WriteNode",
                "status": "completed",
                "duration_ms": None,
                "detail": {"span_count": len(spans)},
            }
        )
    ev = await ctx.eval_results.get(trace_id)
    if ev:
        for layer_i, layer in enumerate(ev.eval_execution_layers or [["Judge"]]):
            for name in layer:
                st = "failed" if name in (ev.node_errors or {}) else "completed"
                if name in (ev.optional_skipped or []):
                    st = "skipped"
                nodes.append(
                    {
                        "trace_id": trace_id,
                        "tenant_id": ev.tenant_id,
                        "node": name,
                        "status": st,
                        "duration_ms": None,
                        "detail": {"layer": layer_i},
                    }
                )
        nodes.append(
            {
                "trace_id": trace_id,
                "tenant_id": ev.tenant_id,
                "node": "ScoreNode",
                "status": "completed",
                "duration_ms": None,
                "detail": {"final_score": ev.final_score},
            }
        )
    return nodes
