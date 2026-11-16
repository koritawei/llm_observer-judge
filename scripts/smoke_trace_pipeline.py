#!/usr/bin/env python3
"""Smoke: judge → triage → Langfuse resolve → attribution for one trace_id.

Usage (from services/agent-eval):
  .venv/bin/python scripts/smoke_trace_pipeline.py \\
    --trace-id 5305db51de164f24b30c409da11127fa \\
    --project-id vehicle

Loads ``services/agent-eval/.env`` automatically (same as ``./run-local.sh``).

DB behaviour:
- 默认复用 ``AGENT_EVAL_DATABASE_URL``(日常/生产为 PostgreSQL),结果**持久化**到
  ``trace_pipeline_records``;PG 库不会自动建表(假设 schema 已存在),本地 SQLite
  兜底时仍调 ``generate_schemas``。
- 同一 ``(trace_id, project_id, agent_name, evaluator_name)`` 重复跑会触发 upsert,
  覆盖上一轮记录(唯一键约束,设计如此)。

Aone behaviour:
- 强制 ``dry_run=True``,无论 yaml 配置如何都**不会**真创建工单,避免污染 Aone。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tortoise import Tortoise

# service root on sys.path when invoked as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Same as ./run-local.sh: load .env into os.environ before pydantic Settings().
_ENV_FILE = _ROOT / ".env"
if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE, override=False)
else:
    print(
        f"WARN: {_ENV_FILE} not found — set AGENT_EVAL_LANGFUSE_INSTANCES in the shell",
        file=sys.stderr,
    )

# 冒烟脚本聚焦 trace→评测流水线；默认关闭 MetaQ 消费，避免 lifespan 连 RocketMQ 阻塞。
import os as _os

_os.environ.setdefault("AGENT_EVAL_MQ_ENABLED", "false")

from benchmark.settings import Settings, load_trace_instances
from benchmark.records.models.trace_record import TraceRecord
from benchmark.service.main import create_app
from benchmark.attribution.badcase.publisher import NotificationConfig
from tests.integration.helpers import FakeLLM


async def _load_trace_payload(
    settings: Settings,
    trace_id: str,
    project_id: str,
) -> tuple[object, object]:
    from benchmark.trace.http_backend import TraceHttpBackend
    from benchmark.trace.schema import TraceEndpoint

    instances = load_trace_instances(settings.trace_instances)
    if not instances:
        raise RuntimeError(
            "No Langfuse instances loaded. Ensure services/agent-eval/.env exists and "
            "AGENT_EVAL_LANGFUSE_INSTANCES is set (or run: source .env before the script)."
        )
    if project_id not in instances:
        raise RuntimeError(
            f"project_id={project_id!r} not in Langfuse config. "
            f"Loaded keys: {sorted(instances)}. "
            "Add projectId to AGENT_EVAL_LANGFUSE_INSTANCES matching extra.project_id."
        )
    instance: TraceEndpoint = instances[project_id]
    async with httpx.AsyncClient(timeout=30.0) as http:
        client = TraceHttpBackend(instance, http)
        trace = await client.fetch(trace_id)
    if trace is None:
        raise RuntimeError(
            f"LangfuseClient.fetch returned None for trace_id={trace_id} project_id={project_id}. "
            "Check Langfuse credentials and metadata.trace_id on the trace."
        )
    payload = trace.model_dump(by_alias=True)
    return payload.get("input"), payload.get("output")


async def run_smoke(
    *,
    trace_id: str,
    project_id: str,
    agent_name: str,
    judge_score: float,
    use_real_judge: bool,
) -> None:
    settings = Settings()
    input_data, output_data = await _load_trace_payload(settings, trace_id, project_id)
    print(f"[1/5] Langfuse trace loaded trace_id={trace_id} project_id={project_id}")

    # Use the DB configured in .env (AGET_EVAL_DATABASE_URL). Smoke now persists
    # results to the real Postgres so you can inspect them via SQL/BI tools.
    is_postgres = settings.database_url.startswith(("postgres", "postgresql"))
    print(
        f"      DB: {settings.database_url.split('@')[-1] if is_postgres else settings.database_url}"
    )

    app = create_app(settings=settings)

    async with app.router.lifespan_context(app):
        if not is_postgres:
            await Tortoise.generate_schemas(safe=True)

        judge = app.state.runtime.agent_manager.get("judge_example")
        if not use_real_judge:
            judge._llm = FakeLLM(
                reply=json.dumps(
                    {"score": judge_score, "reason": "smoke: forced badcase for pipeline test"},
                    ensure_ascii=False,
                )
            )
            print(f"[2/5] Judge: FakeLLM score={judge_score} (triggers attribution)")
        else:
            print("[2/5] Judge: real LLM")

        # dry_run Aone so smoke does not create tickets
        triage_svc = app.state.triage_service
        triage_svc._after_hook._config = NotificationConfig(
            enabled=True,
            dry_run=True,
            a1_bin="a1",
            project="agent-eval",
            category="bug",
            workitem_type="bug",
            title_prefix="[badcase-smoke]",
            timeout_seconds=30,
        )

        provider = app.state.runtime.tool_manager.get_provider("trace")
        assert provider is not None, (
            "trace ToolProvider not registered "
            "(check tools.providers + AGENT_EVAL_LANGFUSE_INSTANCES)"
        )

        envelope = {
            "input": input_data,
            "output": output_data,
            "metadata": {"trace_id": trace_id},
            "agent_name": agent_name,
            "project_id": project_id,
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "example",
                    "messages": [
                        {"role": "user", "content": json.dumps(envelope, ensure_ascii=False)}
                    ],
                },
                timeout=300.0,
            )
        print(f"[3/5] HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(resp.text[:1000])
            raise SystemExit(1)

        verdict = json.loads(resp.json()["choices"][0]["message"]["content"])
        print(f"      judge verdict: score={verdict.get('score')} reason={verdict.get('reason')!r}")

        print("[4/5] triage/attribution completed synchronously with evaluate()")

        rec = await TraceRecord.filter(
            trace_id=trace_id,
            project_id=project_id,
            agent_name=agent_name,
            evaluator_name="example",
        ).first()

        print("[5/5] TraceRecord:")
        if rec is None:
            print("  NOT FOUND — check trace_id/project_id/agent_name")
            raise SystemExit(1)

        print(f"  status={rec.status}")
        print(f"  triage_level={rec.triage_level}")
        print(f"  root_cause={rec.root_cause}")
        print(f"  owner_group={rec.owner_group}")
        if rec.judge_json:
            print(f"  judge_json.score={rec.judge_json.get('score')}")
        if rec.attribution_detail_json:
            detail = rec.attribution_detail_json
            print(f"  attribution.confidence={detail.get('confidence')}")
            print(f"  attribution.reasoning={str(detail.get('reasoning'))[:200]}")
        else:
            print("  attribution_detail_json=None (triage may have skipped attribution)")
        if rec.trace_meta_json:
            digest = rec.trace_meta_json.get("digest", {})
            print(f"  trace_meta.langfuse_id={digest.get('langfuse_id')}")
            print(f"  trace_meta.observation_count={digest.get('observation_count')}")
        else:
            # 零预拉架构下不再写入 digest；None 不代表 Langfuse 未解析成功。
            print(
                "  trace_meta_json=None (legacy digest field; on-demand tools no longer write it)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke full eval pipeline for one trace_id")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--project-id", default="vehicle")
    parser.add_argument("--agent-name", default="vehicle-agent")
    parser.add_argument(
        "--judge-score",
        type=float,
        default=1.5,
        help="FakeLLM score when not using --real-judge (default 1.5 → severe_bad)",
    )
    parser.add_argument(
        "--real-judge",
        action="store_true",
        help="Use real judge LLM instead of FakeLLM",
    )
    args = parser.parse_args()
    asyncio.run(
        run_smoke(
            trace_id=args.trace_id,
            project_id=args.project_id,
            agent_name=args.agent_name,
            judge_score=args.judge_score,
            use_real_judge=args.real_judge,
        )
    )


if __name__ == "__main__":
    main()
