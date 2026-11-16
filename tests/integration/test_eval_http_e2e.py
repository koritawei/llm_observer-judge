"""HTTP end-to-end integration — real lifespan + scoring → triage → the full persistence chain."""
# ruff: noqa: ANN001, PLR2004

import asyncio
import json
from unittest.mock import AsyncMock

import httpx  # noqa: TID251 — test transport layer (ASGITransport), not a business httpx call
import pytest
from benchmark.settings import Settings
from benchmark.records.models.trace_record import TraceRecord
from benchmark.service.main import create_app
from benchmark.attribution.badcase.publisher import NotificationConfig
from benchmark.attribution.service.service import TriageAssembly, TriageAttributionService
from benchmark.attribution.classifiers_registry import triage_classifiers
from fastapi.testclient import TestClient
from tests.integration.helpers import (
    FAKE_ATTRIBUTION,
    FakeLLM,
    async_lifespan,
    patch_attribution_call,
)


def _app(tmp_path):
    return create_app(settings=Settings(database_url=f"sqlite:///{tmp_path}/eval.db"))


@pytest.mark.asyncio
async def test_lifespan_wires_triage_service(async_app) -> None:
    """After a real lifespan, app.state holds triage_service, the evaluator holds a reference, and classifiers include example."""

    app, _client = async_app
    assert hasattr(app.state, "triage_service")

    before_hook = app.state.triage_service._before_hook
    assert "example" in before_hook._classifiers

    evaluator_service = app.state.evaluator_service
    assert evaluator_service._triage_service is app.state.triage_service
    assert not hasattr(evaluator_service, "_background_tasks")


def test_eval_endpoint_end_to_end_with_fake_llm(tmp_path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        agent = app.state.runtime.agent_manager.get("judge_example")
        agent._llm = FakeLLM(reply='{"score": 7, "reason": "ok"}')

        content = json.dumps(
            {
                "input": {"request_body": {"query": "你是谁"}},
                "output": {"component": {"text": "我是高德助手"}},
                "metadata": {"trace_id": "t-77"},
            }
        )
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "example", "messages": [{"role": "user", "content": content}]},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    verdict = json.loads(body["choices"][0]["message"]["content"])
    assert verdict["score"] == 7
    assert any("你是谁" in p for p in agent._llm.prompts)


def test_unknown_model_end_to_end_returns_400(tmp_path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "ghost", "messages": [{"role": "user", "content": "x"}]},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "model_not_found"


@pytest.mark.asyncio
async def test_eval_endpoint_triggers_triage_4_db_writes(async_app) -> None:
    """HTTP request → judge scoring → synchronous triage → four-segment TraceRecord write."""

    app, client = async_app
    judge = app.state.runtime.agent_manager.get("judge_example")
    judge._llm = FakeLLM(reply='{"score": 1.5, "reason": "严重偏差"}')

    attribution = app.state.runtime.agent_manager.get("attribution_example")
    patch_attribution_call(attribution, FAKE_ATTRIBUTION)

    evaluator_service = app.state.evaluator_service
    notification_cfg = NotificationConfig(
        enabled=True,
        dry_run=True,
        a1_bin="a1",
        project="agent-eval",
        category="bug",
        workitem_type="bug",
        title_prefix="[badcase]",
        timeout_seconds=30,
    )
    evaluator_service._triage_service = TriageAttributionService.with_defaults(
        assembly=TriageAssembly(
            agent_manager=app.state.runtime.agent_manager,
            classifiers=triage_classifiers(),
            triage_config=app.state.triage_service._before_hook._config,
            notification_config=notification_cfg,
            trace_record_service=app.state.trace_record_service,
        ),
    )

    content = json.dumps(
        {
            "input": {"request_body": {"query": "导航到机场"}},
            "output": {"component": {"text": "好的,正在为你规划路线"}},
            "metadata": {"trace_id": "t-77"},
            "agent_name": "amap-agent",
            "project_id": "proj-77",
        }
    )
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "example", "messages": [{"role": "user", "content": content}]},
    )
    assert resp.status_code == 200
    # triage runs synchronously inside evaluate(); DB writes are already done.

    rec = await TraceRecord.filter(
        trace_id="t-77",
        project_id="proj-77",
        agent_name="amap-agent",
        evaluator_name="example",
    ).first()
    assert rec is not None
    assert rec.triage_level == "severe_bad"
    assert rec.judge_json is not None
    assert rec.judge_json.get("triage_level") == "severe_bad"
    assert rec.root_cause == "missing_context"
    assert rec.attribution_detail_json is not None
    assert rec.attribution_detail_json.get("root_cause") == "missing_context"
    assert rec.notification_dispatch_json is not None
    assert rec.notification_dispatch_json.get("notification_dispatched_at") is not None
    assert rec.notification_job_json is not None
    assert rec.notification_job_json.get("notification_job_idempotency_key") is not None
    assert rec.notification_job_json.get("notification_job_status") == "pending"


@pytest.mark.asyncio
async def test_eval_good_score_skips_triage_attribution(async_app) -> None:
    """High score → triage skips attribution: only judge_json is written, and the attribution agent is not called."""

    app, client = async_app
    judge = app.state.runtime.agent_manager.get("judge_example")
    judge._llm = FakeLLM(reply='{"score": 9.0, "reason": "ok"}')

    attribution = app.state.runtime.agent_manager.get("attribution_example")
    attribution.call = AsyncMock()

    content = json.dumps(
        {
            "input": {"request_body": {"query": "导航到机场"}},
            "output": {"component": {"text": "好的,正在为你规划路线"}},
            "metadata": {"trace_id": "t-good"},
            "agent_name": "amap-agent",
            "project_id": "proj-good",
        }
    )
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "example", "messages": [{"role": "user", "content": content}]},
    )
    assert resp.status_code == 200

    rec = await TraceRecord.filter(
        trace_id="t-good",
        project_id="proj-good",
        agent_name="amap-agent",
        evaluator_name="example",
    ).first()
    # goodcase: level is persisted (a GOOD classification is written directly), but attribution + dispatch are skipped
    assert rec.status == "scored"
    assert rec.judge_json.get("score") == "9.0"
    assert rec.judge_json.get("reason") == "ok"
    assert rec.judge_json.get("triage_level") == "good"
    assert rec.triage_level == "good"
    assert rec.root_cause is None
    assert rec.attribution_detail_json is None
    assert rec.notification_dispatch_json is None
    attribution.call.assert_not_called()


@pytest.mark.asyncio
async def test_lifespan_eval_completes_triage_before_response(tmp_path) -> None:
    """Synchronous attribution: triage has finished by the time HTTP 200 returns, so shutdown does not wait for background tasks."""
    app = create_app(settings=Settings(database_url=f"sqlite:///{tmp_path}/shutdown.db"))
    async with async_lifespan(app):
        triage_svc = app.state.evaluator_service._triage_service
        original_run = triage_svc.run
        run_finished = asyncio.Event()

        async def tracked_run(**_kwargs: object) -> object:
            outcome = await original_run(**_kwargs)
            run_finished.set()
            return outcome

        triage_svc.run = tracked_run  # type: ignore[method-assign]

        judge = app.state.runtime.agent_manager.get("judge_example")
        judge._llm = FakeLLM(reply='{"score": 7, "reason": "ok"}')
        attribution = app.state.runtime.agent_manager.get("attribution_example")
        patch_attribution_call(attribution, FAKE_ATTRIBUTION)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            content = json.dumps(
                {
                    "input": {"request_body": {"query": "x"}},
                    "output": {"component": {"text": "y"}},
                    "metadata": {"trace_id": "t-shutdown"},
                    "agent_name": "amap-agent",
                    "project_id": "proj-shutdown",
                }
            )
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "example", "messages": [{"role": "user", "content": content}]},
            )
            assert resp.status_code == 200
            assert run_finished.is_set()

    assert not hasattr(app.state.evaluator_service, "_background_tasks")
