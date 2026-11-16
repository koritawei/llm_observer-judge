"""Badcase end-to-end integration — a low judge score → triage → attribution → aone, with all four segments persisted."""
# ruff: noqa: ANN001, PLR2004

import json

import httpx  # noqa: TID251 — test transport layer (ASGITransport), not a business httpx call
import pytest
from benchmark.settings import Settings
from benchmark.records.models.trace_record import TraceRecord
from benchmark.service.main import create_app
from benchmark.attribution.badcase.publisher import NotificationConfig
from benchmark.attribution.service.service import TriageAssembly, TriageAttributionService
from benchmark.attribution.classifiers_registry import triage_classifiers
from tests.integration.helpers import (
    FAKE_ATTRIBUTION,
    FakeLLM,
    async_lifespan,
    patch_attribution_call,
)

# badcase envelope: fabricated POI info, so the judge scores 0.
BADCASE_ENVELOPE = {
    "input": {"request_body": {"query": "帮我找一家附近评分最高的川菜馆"}},
    "output": {
        "component": {
            "text": (
                "为您推荐『麻辣人生·天河店』，地址天河路88号，评分4.9分，"
                "营业时间10:00-22:00，人均消费65元，欢迎前往。"
            )
        }
    },
    "metadata": {"trace_id": "t-badcase-int"},
    "agent_name": "amap-agent",
    "project_id": "proj-int",
}


# Attribution result lives in tests/integration/helpers.py: FAKE_ATTRIBUTION.
# root_cause = "missing_context", category = "prompt", confidence = 0.9


def _build_request_body(envelope: dict) -> dict:
    return {
        "model": "example",
        "messages": [{"role": "user", "content": json.dumps(envelope, ensure_ascii=False)}],
    }


@pytest.mark.asyncio
async def test_badcase_end_to_end_persists_full_triage_chain(tmp_path) -> None:
    """Contract that the badcase chain persists all four segments: judge → triage → attribution → aone."""

    app = create_app(settings=Settings(database_url=f"sqlite:///{tmp_path}/eval.db"))
    async with async_lifespan(app):
        # judge LLM: returns a score of 0.
        judge = app.state.runtime.agent_manager.get("judge_example")
        judge._llm = FakeLLM(
            reply=json.dumps(
                {"score": 0, "reason": "Agent 编造 POI 信息，无工具数据支撑。"},
                ensure_ascii=False,
            )
        )
        # patch attribution agent.call: skip the real attribution LLM and return a controlled AttributionResult.
        attribution = app.state.runtime.agent_manager.get("attribution_example")
        patch_attribution_call(attribution, FAKE_ATTRIBUTION)

        # aone dry_run: do not actually create a ticket, but still run the dispatch + enqueue persistence path.
        evaluator_service = app.state.evaluator_service
        evaluator_service._triage_service = TriageAttributionService.with_defaults(
            assembly=TriageAssembly(
                agent_manager=app.state.runtime.agent_manager,
                classifiers=triage_classifiers(),
                triage_config=app.state.triage_service._before_hook._config,
                notification_config=NotificationConfig(
                    enabled=True,
                    dry_run=True,
                    a1_bin="a1",
                    project="agent-eval",
                    category="bug",
                    workitem_type="bug",
                    title_prefix="[badcase]",
                    timeout_seconds=30,
                ),
                trace_record_service=app.state.trace_record_service,
            ),
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions", json=_build_request_body(BADCASE_ENVELOPE)
            )

        # synchronous-path assertions
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        verdict = json.loads(body["choices"][0]["message"]["content"])
        assert verdict["score"] == 0

        # triage/attribution already finished synchronously inside evaluate

        rec = await TraceRecord.filter(
            trace_id="t-badcase-int",
            project_id="proj-int",
            agent_name="amap-agent",
            evaluator_name="example",
        ).first()
        assert rec is not None

        # 1) judge segment (written synchronously by after_eval)
        assert rec.status == "scored"
        assert rec.judge_json is not None
        # score is serialized to a string via EvaluatorResult(float)
        assert float(rec.judge_json["score"]) == 0.0
        assert "编造" in rec.judge_json["reason"]

        # 2) triage segment (written by before_hook)
        assert rec.triage_level == "severe_bad"
        assert rec.judge_json.get("triage_level") == "severe_bad"

        # 3) attribution segment (written by the agent; values come from FAKE_ATTRIBUTION)
        assert rec.root_cause == "missing_context"
        assert rec.attribution_detail_json is not None
        assert rec.attribution_detail_json.get("root_cause") == "missing_context"
        assert rec.attribution_detail_json.get("confidence") == 0.9
        assert rec.attribution_detail_json.get("owner_domain") == "prompt"

        # 4) aone segment (written by after_hook)
        assert rec.notification_dispatch_json is not None
        assert rec.notification_dispatch_json.get("notification_dispatched_at") is not None

        assert rec.notification_job_json is not None
        assert rec.notification_job_json.get("notification_job_id", "").startswith("notification_job_")
        assert rec.notification_job_json.get("notification_job_status") == "pending"
        assert rec.notification_job_json.get("notification_job_idempotency_key") == "aone_t-badcase-int"
        assert rec.notification_job_json.get("notification_job_payload") is not None
