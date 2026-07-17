"""STEP-11: SQL persistence (SQLite StaticPool) for Outbox/Annotation/etc."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from llmobserverjudge.attribution_worker.store import InMemoryAttributionStore
from llmobserverjudge.contracts.models import (
    AnnotationStatus,
    AttributionResult,
    BenchmarkReport,
    HumanAnnotation,
)
from llmobserverjudge.dispatch.publisher import DispatchPublisher
from llmobserverjudge.shared.mq import InMemoryMQ
from llmobserverjudge.shared.persistence import create_persistence
from llmobserverjudge.shared.sql_attribution import SqlAttributionStore


@pytest.fixture
async def sql_bundle():
    bundle = await create_persistence("sql")
    yield bundle
    if bundle.engine is not None:
        await bundle.engine.dispose()


@pytest.mark.asyncio
async def test_sql_outbox_merge_and_dispatch(sql_bundle) -> None:
    outbox = sql_bundle.outbox
    await outbox.insert_trace_closed(
        tenant_id="t1",
        trace_id="0123456789abcdef0123456789abcdef",
        trace_summary={"tags": ["a"], "status": "OK"},
    )
    await outbox.insert_trace_closed(
        tenant_id="t1",
        trace_id="0123456789abcdef0123456789abcdef",
        trace_summary={"tags": ["b"], "status": "OK"},
    )
    rows = await outbox.fetch_unpublished()
    assert len(rows) == 1
    assert set(rows[0]["trace_summary"]["tags"]) == {"a", "b"}

    mq = InMemoryMQ()
    pub = DispatchPublisher(mq, outbox)
    assert await pub.publish_pending() == 1
    assert await outbox.fetch_unpublished() == []


@pytest.mark.asyncio
async def test_sql_annotation_claim_complete_expire(sql_bundle) -> None:
    store = sql_bundle.annotations
    rec = HumanAnnotation(
        annotation_id=str(uuid4()),
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_quality",
        trigger_reason="always",
        auto_score=0.4,
        auto_label="bad",
        idempotency_key="sql-annot-1",
    )
    saved = await store.upsert_pending(rec)
    again = await store.upsert_pending(rec)
    assert again.annotation_id == saved.annotation_id

    claimed = await store.claim(saved.annotation_id, "u1")
    assert claimed.status == AnnotationStatus.IN_PROGRESS
    done = await store.complete(
        saved.annotation_id,
        annotator_id="u1",
        human_score=0.2,
        human_label="bad",
    )
    assert done.status == AnnotationStatus.COMPLETED
    assert done.score_delta == pytest.approx(-0.2)

    old = HumanAnnotation(
        annotation_id=str(uuid4()),
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_quality",
        trigger_reason="always",
        idempotency_key="sql-annot-old",
        created_at=datetime.now(UTC) - timedelta(days=8),
        expires_at=datetime.now(UTC) - timedelta(days=1),
    )
    await store.upsert_pending(old)
    n = await store.expire_pending(now=datetime.now(UTC))
    assert n >= 1


@pytest.mark.asyncio
async def test_sql_attribution_and_prompt(sql_bundle) -> None:
    attrs: SqlAttributionStore = sql_bundle.attributions
    result = AttributionResult(
        attribution_id=str(uuid4()),
        trace_id="0123456789abcdef0123456789abcdef",
        tenant_id="t1",
        eval_agent_id="judge_safety",
        attribution_agent="attributor_default",
        root_cause="RUNTIME_ERROR",
        confidence=0.9,
        evidence=[],
        tools_called=["get_trace_summary"],
        rounds=3,
        early_stop=True,
        status="COMPLETED",
    )
    await attrs.save(result, idempotency_key="attr-sql-1")
    await attrs.save(result, idempotency_key="attr-sql-1")  # idempotent
    got = await attrs.get_by_idempotency("attr-sql-1")
    assert got is not None
    assert got.root_cause == "RUNTIME_ERROR"
    listed = await attrs.list_for_trace("t1", result.trace_id)
    assert len(listed) == 1

    prompts = sql_bundle.prompts
    draft = await prompts.create_prompt(
        prompt_id="sql_prompt_1", content="Hello {{ name }}", variables=["name"]
    )
    await prompts.publish("sql_prompt_1", draft.version_id)
    active = await prompts.get_active_version("sql_prompt_1")
    assert active is not None
    assert active.status == "ACTIVE"
    text = prompts.render_template(active.content, name="world")
    assert text == "Hello world"


@pytest.mark.asyncio
async def test_sql_benchmark_report(sql_bundle) -> None:
    store = sql_bundle.benchmark_reports
    report = BenchmarkReport(
        report_id=str(uuid4()),
        tenant_id="t1",
        batch_job_id="batch-1",
        date="2026-07-16",
        summary={"mean_score": 0.8},
        sampling_meta={"total": 10, "sampled": 10},
        created_at=datetime.now(UTC),
    )
    await store.save_report(report)
    got = await store.get_report(report.report_id)
    assert got is not None
    assert got.summary["mean_score"] == 0.8
    by_batch = await store.get_report_by_batch("batch-1")
    assert by_batch is not None


@pytest.mark.asyncio
async def test_memory_persistence_default() -> None:
    bundle = await create_persistence("memory")
    assert bundle.driver == "memory"
    assert isinstance(bundle.attributions, InMemoryAttributionStore)
