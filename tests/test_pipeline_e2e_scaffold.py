"""Minimal path: Gateway → Ingest → Outbox → Dispatch → Eval (noop)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from llmobserverjudge.dispatch.publisher import DispatchPublisher
from llmobserverjudge.eval_worker.worker import EvalWorker
from llmobserverjudge.gateway.app import create_app
from llmobserverjudge.gateway.auth import ApiKeyAuthenticator
from llmobserverjudge.ingest.outbox import InMemoryOutbox
from llmobserverjudge.ingest.worker import IngestWorker
from llmobserverjudge.shared.mq import InMemoryMQ


@pytest.mark.asyncio
async def test_scaffold_e2e_path() -> None:
    mq = InMemoryMQ()
    outbox = InMemoryOutbox()
    app = create_app(mq=mq, authenticator=ApiKeyAuthenticator({"k": "t1"}))

    payload = {
        "spans": [
            {
                "tenant_id": "ignored",
                "trace_id": "0123456789abcdef0123456789abcdef",
                "span_id": "0123456789abcdef",
                "name": "chat",
                "timestamp": datetime.now(UTC).isoformat(),
                "status": "OK",
                "kind": "LLM",
                "sampled": True,
            }
        ]
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/ingest",
            json=payload,
            headers={"X-API-Key": "k"},
        )
    assert r.status_code == 200
    assert r.json()["count"] == 1

    ingest = IngestWorker(mq, outbox=outbox, force_sample=True)
    assert await ingest.process_once() == 1
    assert await outbox.fetch_unpublished()

    dispatch = DispatchPublisher(mq, outbox)
    assert await dispatch.publish_pending() == 1
    assert await outbox.fetch_unpublished() == []

    eval_worker = EvalWorker(mq)
    states = await eval_worker.process_once()
    assert len(states) == 1
    assert states[0].tenant_id == "t1"
    assert states[0].final_score > 0


@pytest.mark.asyncio
async def test_gateway_requires_api_key() -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/ingest",
            json={
                "spans": [
                    {
                        "tenant_id": "t",
                        "trace_id": "0123456789abcdef0123456789abcdef",
                        "span_id": "0123456789abcdef",
                        "name": "x",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                ]
            },
        )
    assert r.status_code == 401
