"""STEP-17/18/19: Dashboard BFF — traces, stats, events."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from llmobserverjudge.dashboard.events import PipelineEvent, PipelineEventBus
from llmobserverjudge.dashboard.store import EvalResultRecord, EvalResultStore
from llmobserverjudge.runtime.stack import DevStack
from llmobserverjudge.shared.persistence import create_persistence


@pytest.mark.asyncio
async def test_dashboard_trace_and_stats_after_pipeline() -> None:
    stack = await DevStack.create_from_env(
        api_keys={"k": "t1"},
        force_sample=True,
        persistence=await create_persistence("memory"),
    )
    assert stack.app is not None
    transport = ASGITransport(app=stack.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/ingest",
            headers={"X-API-Key": "k"},
            json={
                "spans": [
                    {
                        "tenant_id": "ignored",
                        "trace_id": "0123456789abcdef0123456789abcdef",
                        "span_id": "0123456789abcdef",
                        "name": "chat",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "status": "OK",
                        "kind": "LLM",
                        "response": "a sufficiently long answer for quality judge",
                    }
                ]
            },
        )
        assert r.status_code == 200
        await stack.pump(max_rounds=40)

        dh = await client.get("/dashboard/healthz")
        assert dh.status_code == 200

        traces = await client.get("/dashboard/api/v1/traces", params={"tenant_id": "t1"})
        assert traces.status_code == 200
        body = traces.json()
        assert body["count"] >= 1
        assert body["items"][0]["trace_id"] == "0123456789abcdef0123456789abcdef"

        summary = await client.get(
            "/dashboard/api/v1/traces/0123456789abcdef0123456789abcdef/summary"
        )
        assert summary.status_code == 200
        s = summary.json()
        assert len(s["spans"]) >= 1
        assert s["eval"] is not None
        assert s["eval"]["final_score"] > 0
        assert s["nodes"]

        ov = await client.get("/dashboard/api/v1/stats/overview", params={"tenant_id": "t1"})
        assert ov.status_code == 200
        assert ov.json()["trace_count"] >= 1
        assert ov.json()["eval_count"] >= 1

        for path in (
            "/dashboard/api/v1/stats/platform",
            "/dashboard/api/v1/stats/score_histogram",
            "/dashboard/api/v1/stats/per_eval",
            "/dashboard/api/v1/stats/sampling",
            "/dashboard/api/v1/stats/attribution",
            "/dashboard/api/v1/stats/annotation_accuracy",
            "/dashboard/api/v1/stats/benchmark",
            "/dashboard/api/v1/stats/prompts",
        ):
            resp = await client.get(path)
            assert resp.status_code == 200, path


@pytest.mark.asyncio
async def test_event_bus_history() -> None:
    bus = PipelineEventBus()
    await bus.publish(
        PipelineEvent(trace_id="t", tenant_id="ten", node="A", status="completed")
    )
    assert len(bus.history("t")) == 1


@pytest.mark.asyncio
async def test_eval_result_store() -> None:
    store = EvalResultStore()
    await store.save(
        EvalResultRecord(
            trace_id="tr",
            tenant_id="t1",
            final_score=0.8,
            bad_case_detected=False,
            eval_scores=[{"agent_id": "j", "normalized_score": 0.8}],
            eval_execution_layers=[["j"]],
            eval_results=[],
            node_errors={},
            optional_skipped=[],
        )
    )
    assert (await store.get("tr")).final_score == 0.8
