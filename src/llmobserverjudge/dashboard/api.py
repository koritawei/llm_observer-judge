"""Dashboard BFF — Trace query, stats, WebSocket (design §10)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect

from llmobserverjudge.dashboard import analytics
from llmobserverjudge.dashboard.events import PipelineEventBus
from llmobserverjudge.dashboard.store import DashboardContext


def create_dashboard_app(ctx: DashboardContext) -> FastAPI:
    app = FastAPI(title="LLMJudge Dashboard API", version="0.1.0")
    app.state.dashboard_ctx = ctx
    if ctx.event_bus is None:
        ctx.event_bus = PipelineEventBus()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/traces")
    async def list_traces(
        tenant_id: str | None = None,
        project_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        items = await ctx.span_store.list_traces(
            tenant_id=tenant_id, project_id=project_id, limit=limit
        )
        # enrich with eval summary (Langfuse-style score chips)
        for it in items:
            ev = await ctx.eval_results.get(
                it["trace_id"],
                tenant_id=it.get("tenant_id") or tenant_id,
                project_id=it.get("project_id") or project_id,
            )
            if ev:
                it["final_score"] = ev.final_score
                it["bad_case_detected"] = ev.bad_case_detected
                scores = []
                for s in ev.eval_scores or []:
                    if not isinstance(s, dict):
                        continue
                    scores.append(
                        {
                            "name": s.get("agent_id") or s.get("name") or "score",
                            "value": s.get("normalized_score", s.get("raw_score")),
                            "label": s.get("label"),
                            "status": s.get("score_status"),
                        }
                    )
                it["scores"] = scores
            else:
                it.setdefault("scores", [])
        return {"items": items, "count": len(items)}

    @app.get("/api/v1/traces/{trace_id}/summary")
    async def trace_summary(
        trace_id: str,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        spans = await ctx.span_store.list_spans_by_trace(
            trace_id, tenant_id=tenant_id, project_id=project_id
        )
        if not spans and not await ctx.eval_results.get(
            trace_id, tenant_id=tenant_id, project_id=project_id
        ):
            raise HTTPException(404, "trace not found")
        ten = tenant_id or (spans[0]["tenant_id"] if spans else None)
        pid = project_id or (
            str(spans[0].get("project_id") or "default") if spans else None
        )
        ev = await ctx.eval_results.get(trace_id, tenant_id=ten, project_id=pid)
        attributions: list[Any] = []
        annotations: list[Any] = []
        if ctx.attributions is not None and ten:
            attributions = [
                a.model_dump(mode="json") if hasattr(a, "model_dump") else a
                for a in await ctx.attributions.list_for_trace(
                    ten, trace_id, project_id=pid
                )
            ]
        if ctx.annotations is not None:
            all_ann = await ctx.annotations.list(
                tenant_id=ten, project_id=pid, limit=200
            )
            annotations = [
                a.model_dump(mode="json")
                for a in all_ann
                if a.trace_id == trace_id
                and (
                    pid is None
                    or getattr(a, "project_id", "default") == pid
                )
            ]
        nodes = await analytics.build_node_timeline(ctx, trace_id)
        # Always rebuild from current eval + side results so Attribution/Annotation
        # status stays in sync with Observations (stored graph can be stale).
        execution_graph: list[Any] = []
        if ev:
            from llmobserverjudge.eval_worker.execution_graph import (
                build_execution_graph,
            )

            results = list(ev.eval_results or [])
            if not results and ev.eval_scores:
                # Fallback: reconstruct minimal result rows from scores
                results = [
                    {
                        "agent_id": s.get("agent_id"),
                        "score_status": s.get("score_status") or "scored",
                        "normalized_score": s.get("normalized_score"),
                        "score": s.get("raw_score"),
                        "is_bad_case": s.get("is_bad_case"),
                        "error": s.get("error"),
                        "skip_reason": s.get("skip_reason"),
                    }
                    for s in ev.eval_scores
                    if isinstance(s, dict)
                ]
            execution_graph = build_execution_graph(
                layers=list(ev.eval_execution_layers or []),
                results=results,
                pipeline_events=nodes,
                bad_case_detected=ev.bad_case_detected,
                attribution_triggered=len(attributions) > 0,
                annotation_triggered=len(annotations) > 0,
                annotations=annotations,
            )
        from llmobserverjudge.eval_worker.execution_graph import group_execution_stages

        execution_stages = group_execution_stages(execution_graph)
        return {
            "trace_id": trace_id,
            "tenant_id": ten,
            "project_id": pid,
            "spans": spans,
            "eval": ev.to_dict() if ev else None,
            "attributions": attributions,
            "annotations": annotations,
            "nodes": nodes,
            "eval_execution_layers": ev.eval_execution_layers if ev else [],
            "execution_graph": execution_graph,
            "execution_stages": execution_stages,
        }

    @app.get("/api/v1/spans/{span_id}")
    async def get_span(span_id: str, tenant_id: str = Query(...)) -> dict[str, Any]:
        row = await ctx.span_store.get_span(tenant_id, span_id)
        if row is None:
            raise HTTPException(404, "span not found")
        return row

    @app.get("/api/v1/stats/overview")
    async def overview(
        tenant_id: str | None = None, project_id: str | None = None
    ) -> dict[str, Any]:
        return await analytics.stats_overview(
            ctx, tenant_id=tenant_id, project_id=project_id
        )

    @app.get("/api/v1/stats/platform")
    async def platform() -> dict[str, Any]:
        return await analytics.stats_platform(ctx)

    @app.get("/api/v1/stats/score_histogram")
    async def score_histogram(
        tenant_id: str | None = None, project_id: str | None = None
    ) -> dict[str, Any]:
        return await analytics.stats_score_histogram(
            ctx, tenant_id=tenant_id, project_id=project_id
        )

    @app.get("/api/v1/stats/per_eval")
    async def per_eval(
        tenant_id: str | None = None, project_id: str | None = None
    ) -> dict[str, Any]:
        return await analytics.stats_per_eval(
            ctx, tenant_id=tenant_id, project_id=project_id
        )

    @app.get("/api/v1/stats/sampling")
    async def sampling() -> dict[str, Any]:
        return await analytics.stats_sampling(ctx)

    @app.get("/api/v1/stats/attribution")
    async def attribution(
        tenant_id: str | None = None, project_id: str | None = None
    ) -> dict[str, Any]:
        return await analytics.stats_attribution(
            ctx, tenant_id=tenant_id, project_id=project_id
        )

    @app.get("/api/v1/stats/annotation_accuracy")
    async def annotation_accuracy(
        tenant_id: str | None = None, project_id: str | None = None
    ) -> dict[str, Any]:
        return await analytics.stats_annotation_accuracy(
            ctx, tenant_id=tenant_id, project_id=project_id
        )

    @app.get("/api/v1/stats/benchmark")
    async def benchmark(
        tenant_id: str | None = None, project_id: str | None = None
    ) -> dict[str, Any]:
        return await analytics.stats_benchmark(
            ctx, tenant_id=tenant_id, project_id=project_id
        )

    @app.get("/api/v1/stats/prompts")
    async def prompts(
        tenant_id: str | None = None, project_id: str | None = None
    ) -> dict[str, Any]:
        return await analytics.stats_prompts(
            ctx, tenant_id=tenant_id, project_id=project_id
        )

    @app.get("/api/v1/stats/ab_contrast")
    async def ab_contrast(
        experiment_id: str | None = None,
        control_version_id: str | None = None,
        treatment_version_id: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return await analytics.stats_ab_contrast(
            ctx,
            experiment_id=experiment_id,
            control_version_id=control_version_id,
            treatment_version_id=treatment_version_id,
            tenant_id=tenant_id,
            project_id=project_id,
        )

    @app.websocket("/ws/traces/{trace_id}")
    async def ws_trace(websocket: WebSocket, trace_id: str) -> None:
        await websocket.accept()
        bus: PipelineEventBus = ctx.event_bus
        try:
            await websocket.send_json({"type": "hello", "trace_id": trace_id})
            async for ev in bus.subscribe(trace_id=trace_id):
                await websocket.send_json({"type": "node", **ev.to_dict()})
        except WebSocketDisconnect:
            return
        except Exception:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.websocket("/ws/events")
    async def ws_global(websocket: WebSocket) -> None:
        await websocket.accept()
        bus: PipelineEventBus = ctx.event_bus
        try:
            await websocket.send_json({"type": "hello", "scope": "global"})
            async for ev in bus.subscribe(trace_id=None):
                await websocket.send_json({"type": "node", **ev.to_dict()})
        except WebSocketDisconnect:
            return
        except Exception:
            try:
                await websocket.close()
            except Exception:
                pass

    return app
