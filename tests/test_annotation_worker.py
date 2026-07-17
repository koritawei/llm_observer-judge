"""STEP-08: Annotation platform consumer, claim/complete, TTL expire."""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from llmobserverjudge.annotation_worker.api import create_annotation_app
from llmobserverjudge.annotation_worker.store import (
    BACKLOG_ALERT_THRESHOLD,
    InMemoryAnnotationStore,
    backlog_alert,
)
from llmobserverjudge.annotation_worker.worker import AnnotationWorker
from llmobserverjudge.contracts.models import (
    AnnotationStatus,
    EvalConfig,
    EvalJob,
    HumanAnnotation,
    Priority,
)
from llmobserverjudge.contracts.topics import Component, ProducerForbiddenError, Topic
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.eval_worker.graph import run_eval_graph
from llmobserverjudge.eval_worker.judges import JudgeModelRouter, ScriptedJudgeAdapter
from llmobserverjudge.shared.mq import InMemoryMQ


@pytest.mark.asyncio
async def test_worker_ingests_pending_idempotent() -> None:
    mq = InMemoryMQ()
    store = InMemoryAnnotationStore()
    worker = AnnotationWorker(mq, store=store)
    payload = {
        "annotation_id": str(uuid4()),
        "trace_id": "0123456789abcdef0123456789abcdef",
        "tenant_id": "t1",
        "eval_agent_id": "judge_safety",
        "trigger_reason": "always",
        "auto_score": 0.3,
        "auto_label": "bad",
        "idempotency_key": "annot-key-1",
    }
    await mq.publish(Component.DISPATCH, Topic.ANNOTATION_JOBS, payload)
    rows = await worker.process_once()
    assert len(rows) == 1
    assert rows[0].status == AnnotationStatus.PENDING
    assert rows[0].expires_at is not None

    await mq.publish(Component.DISPATCH, Topic.ANNOTATION_JOBS, payload)
    rows2 = await worker.process_once()
    assert rows2 == [] or rows2[0].annotation_id == rows[0].annotation_id
    listed = await store.list(tenant_id="t1")
    assert len(listed) == 1


@pytest.mark.asyncio
async def test_claim_complete_api() -> None:
    store = InMemoryAnnotationStore()
    await store.upsert_pending(
        HumanAnnotation(
            annotation_id="a1",
            trace_id="0123456789abcdef0123456789abcdef",
            tenant_id="t1",
            eval_agent_id="judge_quality",
            trigger_reason="low_confidence",
            auto_score=0.4,
            auto_label="bad",
            idempotency_key="k-a1",
        )
    )
    app = create_annotation_app(store)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/annotations/a1/claim", json={"annotator_id": "u1"}
        )
        assert r.status_code == 200
        assert r.json()["status"] == "IN_PROGRESS"

        r2 = await client.post(
            "/api/v1/annotations/a1/complete",
            json={
                "annotator_id": "u1",
                "human_score": 0.2,
                "human_label": "bad",
                "human_root_cause": "PROMPT_MISMATCH",
            },
        )
        assert r2.status_code == 200
        body = r2.json()
        assert body["status"] == "COMPLETED"
        assert body["score_delta"] == pytest.approx(-0.2)
        assert body["label_correct"] is True

        # other annotator cannot claim completed
        r3 = await client.post(
            "/api/v1/annotations/a1/claim", json={"annotator_id": "u2"}
        )
        assert r3.status_code == 409


@pytest.mark.asyncio
async def test_expire_pending_ttl() -> None:
    store = InMemoryAnnotationStore(pending_ttl=timedelta(days=7))
    old = datetime.now(UTC) - timedelta(days=8)
    await store.upsert_pending(
        HumanAnnotation(
            annotation_id="old1",
            trace_id="0123456789abcdef0123456789abcdef",
            tenant_id="t1",
            eval_agent_id="judge_quality",
            trigger_reason="always",
            created_at=old,
            expires_at=old + timedelta(days=7),
            idempotency_key="old-key",
        )
    )
    n = await store.expire_pending(now=datetime.now(UTC))
    assert n == 1
    row = await store.get("old1")
    assert row is not None
    assert row.status == AnnotationStatus.EXPIRED


def test_backlog_alert_threshold() -> None:
    assert not backlog_alert(BACKLOG_ALERT_THRESHOLD)
    assert backlog_alert(BACKLOG_ALERT_THRESHOLD + 1)


@pytest.mark.asyncio
async def test_eval_to_annotation_e2e() -> None:
    mq = InMemoryMQ()
    client = DispatchClient(mq)
    scripted = ScriptedJudgeAdapter(scores={"judge_safety": 2.0})
    router = JudgeModelRouter(
        registry={"scripted": scripted, "code_eval": scripted},
        default_adapter=scripted,
    )
    job = EvalJob(
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        trace_summary={"status": "OK", "tags": ["safety"]},
        eval_configs=[EvalConfig(agent_id="judge_safety", model="scripted")],
        priority=Priority.P0,
        idempotency_key="eval-annot-e2e",
        created_at=datetime.now(UTC),
    )
    await run_eval_graph(job, judge_router=router, dispatch_client=client)
    assert await mq.depth(Topic.ANNOTATION_JOBS) == 1

    store = InMemoryAnnotationStore()
    worker = AnnotationWorker(mq, store=store)
    rows = await worker.process_once()
    assert len(rows) == 1
    assert rows[0].trigger_reason == "always"

    app = create_annotation_app(store)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        aid = rows[0].annotation_id
        await http.post(f"/api/v1/annotations/{aid}/claim", json={"annotator_id": "rev"})
        done = await http.post(
            f"/api/v1/annotations/{aid}/complete",
            json={"annotator_id": "rev", "human_label": "unsafe", "human_score": 0.1},
        )
        assert done.status_code == 200
        backlog = await http.get("/api/v1/tenants/t1/annotation-backlog")
        assert backlog.json()["pending"] == 0


@pytest.mark.asyncio
async def test_annotation_platform_cannot_publish_eval_jobs() -> None:
    mq = InMemoryMQ()
    with pytest.raises(ProducerForbiddenError):
        await mq.publish(Component.ANNOTATION_PLATFORM, Topic.EVAL_JOBS, {})


def test_annotation_source_has_no_eval_jobs_publish() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "llmobserverjudge"
        / "annotation_worker"
    )
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_publish = (
                isinstance(func, ast.Attribute) and func.attr == "publish"
            ) or (isinstance(func, ast.Name) and func.id == "publish")
            if not is_publish:
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Attribute) and arg.attr == "EVAL_JOBS":
                    pytest.fail(f"forbidden publish(EVAL_JOBS) in {path}")
