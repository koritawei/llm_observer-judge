"""STEP-15: DevStack E2E — Gateway → Ingest → Dispatch → Eval (+ side workers)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.runtime.stack import DevStack, parse_api_keys
from llmobserverjudge.shared.persistence import create_persistence


def test_parse_api_keys() -> None:
    assert parse_api_keys("") == {"dev-key": "tenant-dev"}
    assert parse_api_keys("k1:t1,k2:t2") == {"k1": "t1", "k2": "t2"}
    with pytest.raises(ValueError):
        parse_api_keys("nocolon")


@pytest.mark.asyncio
async def test_dev_stack_ingest_to_eval() -> None:
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
                        "tags": ["safety"],
                    }
                ]
            },
        )
        assert r.status_code == 200
        assert r.json()["count"] == 1

        totals = await stack.pump(max_rounds=50)
        assert totals["ingest"] >= 1
        assert totals["dispatch"] >= 1
        assert totals["eval"] >= 1

        sz = await client.get("/stackz")
        assert sz.status_code == 200
        body = sz.json()
        assert body["persistence"] == "memory"
        assert body["mq_depths"]["spans"] == 0
        assert body["mq_depths"]["eval_jobs"] == 0

        # mounted APIs
        ah = await client.get("/annotation/healthz")
        assert ah.status_code == 200
        bh = await client.get("/benchmark/healthz")
        assert bh.status_code == 200

    # safety tag → may enqueue annotation/attribution depending on scores
    assert await stack.mq.depth(Topic.SPANS) == 0


@pytest.mark.asyncio
async def test_dev_stack_background_workers() -> None:
    """Background loops (same as ASGI lifespan on_startup) drain the MQ."""
    import asyncio

    stack = await DevStack.create_from_env(
        api_keys={"k": "t1"},
        force_sample=True,
        persistence=await create_persistence("memory"),
        poll_interval=0.01,
    )
    assert stack.app is not None
    await stack.start_workers()
    try:
        transport = ASGITransport(app=stack.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/ingest",
                headers={"X-API-Key": "k"},
                json={
                    "spans": [
                        {
                            "tenant_id": "ignored",
                            "trace_id": "1123456789abcdef0123456789abcdef",
                            "span_id": "1123456789abcdef",
                            "name": "chat",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "status": "OK",
                            "kind": "LLM",
                            "response": "another long enough quality answer here",
                        }
                    ]
                },
            )
            assert r.status_code == 200

            for _ in range(100):
                sz = await client.get("/stackz")
                depths = sz.json()["mq_depths"]
                if depths["spans"] == 0 and depths["eval_jobs"] == 0:
                    unpublished = await stack.persistence.outbox.fetch_unpublished()  # type: ignore[union-attr]
                    if not unpublished:
                        break
                await asyncio.sleep(0.02)
            else:
                pytest.fail("stack did not drain spans/eval_jobs in time")

            assert sz.json()["started"] is True
    finally:
        await stack.stop_workers()
