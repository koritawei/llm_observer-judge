"""Distributed deployment fixes: outbox ordering, idempotency, readyz."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from llmobserverjudge.contracts.idempotency import span_key
from llmobserverjudge.contracts.models import EvalConfig, EvalJob, Priority
from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.dispatch.publisher import DispatchPublisher
from llmobserverjudge.eval_worker.worker import EvalWorker
from llmobserverjudge.gateway.app import create_app
from llmobserverjudge.ingest.idempotency import InMemoryIdempotencyStore
from llmobserverjudge.ingest.outbox import InMemoryOutbox
from llmobserverjudge.ingest.worker import IngestWorker
from llmobserverjudge.shared.mq import InMemoryMQ


@pytest.mark.asyncio
async def test_outbox_publish_before_mark_survives_crash_window() -> None:
    """Unpublished rows stay retryable until mark_published after MQ publish."""
    outbox = InMemoryOutbox()
    await outbox.insert_trace_closed(
        tenant_id="t1",
        trace_id="0123456789abcdef0123456789abcdef",
        trace_summary={"status": "OK"},
    )
    mq = InMemoryMQ()
    pub = DispatchPublisher(mq, outbox)
    assert await pub.publish_pending() == 1
    assert await outbox.fetch_unpublished() == []
    assert await mq.depth(Topic.EVAL_JOBS) == 1


@pytest.mark.asyncio
async def test_outbox_claim_lease_hides_row_from_other_dispatchers() -> None:
    outbox = InMemoryOutbox()
    await outbox.insert_trace_closed(
        tenant_id="t1",
        trace_id="0123456789abcdef0123456789abcdef",
        trace_summary={"status": "OK"},
    )
    rows = await outbox.claim_unpublished()
    assert rows and rows[0]["published"] is False
    assert rows[0].get("claimed_at") is not None
    assert await outbox.fetch_unpublished() == []


@pytest.mark.asyncio
async def test_outbox_publish_failure_leaves_row_unpublished() -> None:
    outbox = InMemoryOutbox()
    await outbox.insert_trace_closed(
        tenant_id="t1",
        trace_id="0123456789abcdef0123456789abcdef",
        trace_summary={"status": "OK"},
    )
    mq = InMemoryMQ()
    mq.publish = AsyncMock(side_effect=RuntimeError("mq down"))  # type: ignore[method-assign]
    pub = DispatchPublisher(mq, outbox)
    assert await pub.publish_pending() == 0
    rows = await outbox.fetch_unpublished()
    assert len(rows) == 1
    assert rows[0]["attempt_count"] == 1


@pytest.mark.asyncio
async def test_ingest_retries_after_failure_without_idempotency_block() -> None:
    mq = InMemoryMQ()
    idem = InMemoryIdempotencyStore()
    worker = IngestWorker(
        mq,
        idempotency=idem,
        force_sample=True,
    )
    span = {
        "tenant_id": "t1",
        "trace_id": "0123456789abcdef0123456789abcdef",
        "span_id": "0123456789abcdef",
        "name": "n",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    await mq.publish("gateway", Topic.SPANS, span)

    calls = {"n": 0}
    orig = worker.pipeline.run

    async def flaky_run(state):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return await orig(state)

    worker.pipeline.run = flaky_run  # type: ignore[method-assign]
    assert await worker.process_once() == 0
    key = span_key("t1", "0123456789abcdef")
    assert not await idem.exists(key)
    # InMemoryMQ drops unacked messages on drain; re-publish simulates Redis reclaim.
    await mq.publish("gateway", Topic.SPANS, span)
    assert await worker.process_once() == 1
    assert await idem.exists(key)


@pytest.mark.asyncio
async def test_eval_marks_idempotency_only_after_success() -> None:
    mq = InMemoryMQ()
    idem = InMemoryIdempotencyStore()
    worker = EvalWorker(mq, idempotency=idem)
    job = EvalJob(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        trace_summary={"status": "OK"},
        eval_configs=[EvalConfig(agent_id="judge_quality", model="scripted")],
        priority=Priority.P2,
        idempotency_key="eval-retry-key",
        created_at=datetime.now(UTC),
    )
    await mq.publish("dispatch", Topic.EVAL_JOBS, job.model_dump(mode="json"))

    calls = {"n": 0}
    from llmobserverjudge.eval_worker import worker as eval_mod

    orig_graph = eval_mod.run_eval_graph

    async def flaky_graph(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("judge timeout")
        return await orig_graph(*args, **kwargs)

    eval_mod.run_eval_graph = flaky_graph
    try:
        assert await worker.process_once() == []
        assert not await idem.exists("eval:eval-retry-key")
        await mq.publish("dispatch", Topic.EVAL_JOBS, job.model_dump(mode="json"))
        assert await worker.process_once()
        assert await idem.exists("eval:eval-retry-key")
    finally:
        eval_mod.run_eval_graph = orig_graph


@pytest.mark.asyncio
async def test_readyz_db_probe_degraded_then_503() -> None:
    mq = InMemoryMQ()

    async def db_check() -> None:
        raise OSError("db down")

    app = create_app(mq=mq, db_health_check=db_check)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.get("/readyz")
        assert r1.status_code == 200
        assert r1.json()["status"] == "degraded"
        assert "db_failures" in r1.json()
        await client.get("/readyz")
        r3 = await client.get("/readyz")
        assert r3.status_code == 503


@pytest.mark.asyncio
async def test_ingest_outbox_failure_does_not_mark_idempotency() -> None:
    mq = InMemoryMQ()
    idem = InMemoryIdempotencyStore()
    outbox = InMemoryOutbox()
    worker = IngestWorker(mq, outbox=outbox, idempotency=idem, force_sample=True)
    span = {
        "tenant_id": "t1",
        "trace_id": "0123456789abcdef0123456789abcdef",
        "span_id": "0123456789abcdef",
        "name": "n",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    await mq.publish("gateway", Topic.SPANS, span)

    async def fail_outbox(**kwargs):  # noqa: ANN003
        raise RuntimeError("outbox down")

    outbox.insert_trace_closed = fail_outbox  # type: ignore[method-assign]
    assert await worker.process_once() == 0
    key = span_key("t1", "0123456789abcdef")
    assert not await idem.exists(key)


def test_region_redis_key_prefix() -> None:
    import os

    from llmobserverjudge.shared.deployment import redis_key_prefix, resolve_region

    os.environ["LLMJUDGE_REGION"] = "cn-east-1"
    try:
        assert resolve_region() == "cn-east-1"
        assert redis_key_prefix() == "llmj:cn-east-1:"
    finally:
        os.environ.pop("LLMJUDGE_REGION", None)
        assert redis_key_prefix() == "llmj:"
