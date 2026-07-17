"""Gateway telemetry, rate limit, readyz MQ probe."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.gateway.app import create_app
from llmobserverjudge.gateway.auth import ApiKeyAuthenticator
from llmobserverjudge.gateway.rate_limit import RateLimiter
from llmobserverjudge.shared.mq import InMemoryMQ


def _span_payload() -> dict:
    return {
        "spans": [
            {
                "tenant_id": "x",
                "trace_id": "0123456789abcdef0123456789abcdef",
                "span_id": "0123456789abcdef",
                "name": "n",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    }


@pytest.mark.asyncio
async def test_telemetry_ok_and_rate_limit() -> None:
    limiter = RateLimiter(telemetry_rate_per_sec=1 / 30)
    app = create_app(
        authenticator=ApiKeyAuthenticator({"k": "t1"}),
        rate_limiter=limiter,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post(
            "/api/v1/telemetry",
            json={"sent_spans_total": 1, "dropped_spans_total": 0},
            headers={"X-API-Key": "k"},
        )
        assert r1.status_code == 200
        r2 = await client.post(
            "/api/v1/telemetry",
            json={"sent_spans_total": 2, "dropped_spans_total": 0},
            headers={"X-API-Key": "k"},
        )
        assert r2.status_code == 429
        assert r2.headers.get("Retry-After") == "30"


@pytest.mark.asyncio
async def test_ingest_rate_limited() -> None:
    limiter = RateLimiter(default_rate_per_sec=0.0, default_burst=0.0)
    app = create_app(
        authenticator=ApiKeyAuthenticator({"k": "t1"}),
        rate_limiter=limiter,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/ingest",
            json=_span_payload(),
            headers={"X-API-Key": "k"},
        )
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == "10"


@pytest.mark.asyncio
async def test_readyz_degraded_then_503() -> None:
    mq = InMemoryMQ(fail_health=True)
    app = create_app(mq=mq)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.get("/readyz")
        assert r1.status_code == 200
        assert r1.json()["status"] == "degraded"
        r2 = await client.get("/readyz")
        assert r2.json()["mq_failures"] == 2
        r3 = await client.get("/readyz")
        assert r3.status_code == 503


@pytest.mark.asyncio
async def test_readyz_ready_when_mq_ok() -> None:
    app = create_app(mq=InMemoryMQ(fail_health=False))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_ingest_still_publishes() -> None:
    mq = InMemoryMQ()
    app = create_app(mq=mq, authenticator=ApiKeyAuthenticator({"k": "t1"}))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/ingest",
            json=_span_payload(),
            headers={"X-API-Key": "k"},
        )
    assert r.status_code == 200
    assert await mq.depth(Topic.SPANS) == 1
