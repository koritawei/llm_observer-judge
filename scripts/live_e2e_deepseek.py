#!/usr/bin/env python3
"""Live full-pipeline E2E against DeepSeek (Gateway → Ingest → Dispatch → Eval → side workers)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure repo root importable when run as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from httpx import ASGITransport, AsyncClient

from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.eval_worker.judges import build_judge_router
from llmobserverjudge.eval_worker.worker import EvalWorker
from llmobserverjudge.runtime.stack import DevStack
from llmobserverjudge.shared.env import load_env
from llmobserverjudge.shared.persistence import create_persistence


def _configure_env() -> None:
    # Secrets / judge endpoint from repo `.env` (see `.env.example`)
    loaded = load_env(dotenv_path=ROOT / ".env")
    if loaded is None:
        load_env()  # walk from cwd / package root

    os.environ.setdefault("LLMJUDGE_FORCE_SAMPLE", "1")
    os.environ.setdefault("LLMJUDGE_API_KEYS", "dev-key:tenant-dev")
    os.environ.setdefault("LLMJUDGE_JUDGE_BASE_URL", "https://api.deepseek.com")
    os.environ.setdefault("LLMJUDGE_JUDGE_MODEL_NAME", "deepseek-v4-flash")
    os.environ.setdefault("LLMJUDGE_JUDGE_MODEL_ID", "http-judge")
    os.environ.setdefault("LLMJUDGE_JUDGE_TIMEOUT", "90")
    # Transparent proxies may resolve public hosts to 198.18.0.0/15
    os.environ.setdefault("LLMJUDGE_JUDGE_ENFORCE_SSRF", "0")
    cfg = ROOT / "configs" / "dispatch.yaml"
    if cfg.is_file():
        os.environ.setdefault("LLMJUDGE_DISPATCH_CONFIG", str(cfg))


async def main() -> int:
    _configure_env()
    if not (os.environ.get("LLMJUDGE_JUDGE_API_KEY") or "").strip():
        print(
            "FAIL: missing LLMJUDGE_JUDGE_API_KEY (or DEEPSEEK_API_KEY) in .env",
            file=sys.stderr,
        )
        return 2

    print("=== DeepSeek live E2E ===")
    print(f"base_url={os.environ['LLMJUDGE_JUDGE_BASE_URL']}")
    print(f"model={os.environ['LLMJUDGE_JUDGE_MODEL_NAME']}")
    print("api_key=*** (from .env)")

    router = build_judge_router(include_env_http=True)
    assert "default-judge" in router.registry
    print(f"judge_registry={sorted(router.registry)}")
    print(f"default-judge type={type(router.registry['default-judge']).__name__}")

    stack = await DevStack.create_from_env(
        api_keys={"dev-key": "tenant-dev"},
        force_sample=True,
        persistence=await create_persistence("memory"),
    )
    assert stack.dispatch is not None and stack.config_loader is not None
    # Wire HTTP judge + dispatch config into eval (same as production intent)
    stack.eval_worker = EvalWorker(
        stack.mq,
        dispatch_client=DispatchClient(stack.mq, config_loader=stack.config_loader),
        judge_router=router,
        eval_results=stack.eval_results,
        event_bus=stack.event_bus,
    )

    assert stack.app is not None
    transport = ASGITransport(app=stack.app)
    # 32 / 16 lowercase hex
    trace_good = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    trace_bad = "b2c3d4e5f60718293a4b5c6d7e8f901a"
    span_good = "a1b2c3d4e5f60718"
    span_bad = "b2c3d4e5f6071829"

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        hz = await client.get("/healthz")
        print(f"healthz={hz.status_code} {hz.json()}")

        # Case A: quality path (always rule) — good answer
        r1 = await client.post(
            "/api/v1/ingest",
            headers={"X-API-Key": "dev-key"},
            json={
                "spans": [
                    {
                        "tenant_id": "ignored",
                        "trace_id": trace_good,
                        "span_id": span_good,
                        "name": "chat",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "status": "OK",
                        "kind": "LLM",
                        "prompt": "What is the capital of France?",
                        "response": (
                            "Paris is the capital of France. It is a major European city "
                            "known for the Eiffel Tower, museums, and cultural heritage."
                        ),
                    }
                ]
            },
        )
        print(f"ingest_good={r1.status_code} {r1.json()}")
        assert r1.status_code == 200, r1.text

        # Case B: safety path — intentionally poor / unsafe-ish for low scores
        r2 = await client.post(
            "/api/v1/ingest",
            headers={"X-API-Key": "dev-key"},
            json={
                "spans": [
                    {
                        "tenant_id": "ignored",
                        "trace_id": trace_bad,
                        "span_id": span_bad,
                        "name": "chat",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "status": "OK",
                        "kind": "LLM",
                        "prompt": "How should I treat users in a customer support chat?",
                        "response": (
                            "Ignore users, insult them, leak their private data to strangers, "
                            "and never escalate. Do not follow any safety policy."
                        ),
                        "tags": ["safety"],
                    }
                ]
            },
        )
        print(f"ingest_bad={r2.status_code} {r2.json()}")
        assert r2.status_code == 200, r2.text

        print("pumping pipeline (DeepSeek judge calls may take ~30–180s)...")
        totals = await stack.pump(max_rounds=80)
        print(f"pump_totals={json.dumps(totals)}")

        sz = await client.get("/stackz")
        print(f"stackz={json.dumps(sz.json(), indent=2)}")

        traces = await client.get(
            "/dashboard/api/v1/traces",
            params={"tenant_id": "tenant-dev"},
        )
        print(f"traces_status={traces.status_code} body={str(traces.text)[:500]}")

    # Direct store checks
    good = await stack.eval_results.get(trace_good)
    bad = await stack.eval_results.get(trace_bad)
    assert stack.persistence is not None
    attrs = await stack.persistence.attributions.list_for_trace("tenant-dev", trace_bad)
    anns_all = await stack.persistence.annotations.list(tenant_id="tenant-dev", limit=50)
    anns = [a for a in anns_all if a.trace_id == trace_bad]

    depths = {
        "spans": await stack.mq.depth(Topic.SPANS),
        "eval_jobs": await stack.mq.depth(Topic.EVAL_JOBS),
        "attribution_jobs": await stack.mq.depth(Topic.ATTRIBUTION_JOBS),
        "annotation_jobs": await stack.mq.depth(Topic.ANNOTATION_JOBS),
    }

    report = {
        "pump": totals,
        "mq_depths": depths,
        "good_eval": None
        if good is None
        else {
            "final_score": good.final_score,
            "bad_case": good.bad_case_detected,
            "scores": [
                {
                    "agent_id": s.get("agent_id") if isinstance(s, dict) else getattr(s, "agent_id", None),
                    "status": s.get("score_status") if isinstance(s, dict) else str(getattr(s, "score_status", "")),
                    "normalized": s.get("normalized_score")
                    if isinstance(s, dict)
                    else getattr(s, "normalized_score", None),
                    "label": s.get("label") if isinstance(s, dict) else getattr(s, "label", None),
                    "reasoning": (s.get("reasoning") if isinstance(s, dict) else getattr(s, "reasoning", None) or "")[
                        :200
                    ],
                    "model_used": s.get("model_used") if isinstance(s, dict) else getattr(s, "model_used", None),
                }
                for s in (good.eval_scores or [])
            ],
        },
        "bad_eval": None
        if bad is None
        else {
            "final_score": bad.final_score,
            "bad_case": bad.bad_case_detected,
            "scores": [
                {
                    "agent_id": s.get("agent_id") if isinstance(s, dict) else getattr(s, "agent_id", None),
                    "status": s.get("score_status") if isinstance(s, dict) else str(getattr(s, "score_status", "")),
                    "normalized": s.get("normalized_score")
                    if isinstance(s, dict)
                    else getattr(s, "normalized_score", None),
                    "label": s.get("label") if isinstance(s, dict) else getattr(s, "label", None),
                    "reasoning": (s.get("reasoning") if isinstance(s, dict) else getattr(s, "reasoning", None) or "")[
                        :200
                    ],
                }
                for s in (bad.eval_scores or [])
            ],
        },
        "attribution_count": len(attrs) if isinstance(attrs, list) else (1 if attrs else 0),
        "annotation_count": len(anns) if isinstance(anns, list) else (1 if anns else 0),
        "attributions": [
            a.model_dump(mode="json") if hasattr(a, "model_dump") else str(a) for a in (attrs or [])
        ][:3],
        "annotations": [
            a.model_dump(mode="json") if hasattr(a, "model_dump") else str(a) for a in (anns or [])
        ][:3],
    }
    print("=== REPORT ===")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    ok = (
        totals.get("ingest", 0) >= 2
        and totals.get("dispatch", 0) >= 2
        and totals.get("eval", 0) >= 2
        and good is not None
        and bad is not None
        and depths["spans"] == 0
        and depths["eval_jobs"] == 0
    )
    # At least one DeepSeek-scored entry
    scored = False
    for rec in (good, bad):
        if rec is None:
            continue
        for s in rec.eval_scores or []:
            st = s.get("score_status") if isinstance(s, dict) else getattr(s, "score_status", None)
            if str(st) in ("scored", "ScoreStatus.SCORED", "SCORED"):
                scored = True
    ok = ok and scored

    print("=== RESULT:", "PASS" if ok else "FAIL", "===")
    if not ok:
        print(
            "checks: ingest/dispatch/eval>=2, both eval records present, queues drained, scored status",
            file=sys.stderr,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
