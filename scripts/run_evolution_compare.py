#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline evaluation, self-evolution, apply the sealed patch, evaluate again, and compare scores.

Code under test is written only under ``out/evolution_example/workspace``. The kernel, gates, and hidden-set labels never enter the patch.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("EVALBENCH_SECRET_DATABASE", "eval")
os.environ.setdefault("EVALBENCH_SECRET_PLUGIN_TRUST", "test-trust")
os.environ.setdefault("EVALBENCH_SECRET_CALLBACK", "test-callback")
os.environ.setdefault("EVALBENCH_ENV", "local")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "evolution_example"
WORKSPACE = OUT / "workspace"
SUBJECT = WORKSPACE / "subject.py"
CONF = ROOT / "conf"

BEFORE = (
    "def answer(query: str) -> str:\n"
    "    \"\"\"回复用户。问句被错误清空，精确匹配因此失败。\"\"\"\n"
    "    text = _known(query)\n"
    "    if query.endswith(\"？\") or query.endswith(\"?\"):\n"
    "        return \"\"\n"
    "    return text\n"
    "\n"
    "\n"
    "def _known(query: str) -> str:\n"
    "    if \"沸点\" in query:\n"
    "        return \"100摄氏度\"\n"
    "    if \"退\" in query:\n"
    "        return \"七天内可退\"\n"
    "    if \"红眼\" in query:\n"
    "        return \"用修复笔点掉红眼\"\n"
    "    if query == \"你好\":\n"
    "        return \"你好\"\n"
    "    return \"不知道\"\n"
)

AFTER = (
    "def answer(query: str) -> str:\n"
    "    \"\"\"回复用户。问句保留已知答案。\"\"\"\n"
    "    return _known(query)\n"
    "\n"
    "\n"
    "def _known(query: str) -> str:\n"
    "    if \"沸点\" in query:\n"
    "        return \"100摄氏度\"\n"
    "    if \"退\" in query:\n"
    "        return \"七天内可退\"\n"
    "    if \"红眼\" in query:\n"
    "        return \"用修复笔点掉红眼\"\n"
    "    if query == \"你好\":\n"
    "        return \"你好\"\n"
    "    return \"不知道\"\n"
)

CASES = [
    {"item_id": "hello", "role": "good", "query": "你好", "expected": "你好"},
    {"item_id": "boil", "role": "bad", "query": "水的沸点是多少？", "expected": "100摄氏度"},
    {"item_id": "refund", "role": "bad", "query": "这件衣服七天内能退吗？", "expected": "七天内可退"},
    {"item_id": "redeye", "role": "heldout", "query": "怎么把照片里的红眼去掉？", "expected": "用修复笔点掉红眼"},
]


def _load_answer(source: str):
    namespace: dict[str, object] = {}
    exec(source, namespace)
    answer = namespace["answer"]
    if not callable(answer):
        raise RuntimeError("subject has no answer function")
    return answer


def _cases_from(source: str, cases: list[dict[str, str]]) -> list[dict[str, object]]:
    answer = _load_answer(source)
    rows = []
    for case in cases:
        rows.append({
            "item_id": case["item_id"],
            "input": {"text": case["query"]},
            "output": {"text": answer(case["query"])},
            "expected": {"text": case["expected"]},
        })
    return rows


def _diff() -> str:
    return "".join(difflib.unified_diff(
        BEFORE.splitlines(keepends=True),
        AFTER.splitlines(keepends=True),
        fromfile="a/subject.py",
        tofile="b/subject.py",
    ))


def _apply(diff_text: str) -> None:
    SUBJECT.write_text(BEFORE, encoding="utf-8")
    subprocess.run(
        ["patch", "-p1"],
        input=diff_text,
        text=True,
        cwd=WORKSPACE,
        check=True,
        capture_output=True,
    )


async def _benchmark(kernel, name: str, source: str) -> dict[str, object]:
    from tests.conftest import wait_terminal

    created = await kernel.publish_run({
        "benchmark_name": name,
        "profile_id": "benchmark_sync",
        "business": "default",
        "cases": _cases_from(source, CASES),
    })
    finished = await wait_terminal(kernel, created["run_id"], timeout=30)
    items = await kernel.store.list_items(created["run_id"])
    by_id = {row["item_id"]: row for row in items}
    return {
        "run_id": created["run_id"],
        "status": finished["status"],
        "final_score": finished["final_score"],
        "items": [
            {
                "item_id": case["item_id"],
                "role": case["role"],
                "score": by_id[case["item_id"]]["score"],
                "output": by_id[case["item_id"]]["output"],
            }
            for case in CASES
        ],
    }


async def main() -> None:
    from benchmark.evolution.graph import advance_result, build_evolution_graph
    from benchmark.evolution.service import EvolutionService
    from benchmark.kernel.contract import freeze_policy
    from benchmark.kernel.runtime import open_kernel
    from tests.conftest import wait_terminal

    OUT.mkdir(parents=True, exist_ok=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SUBJECT.write_text(BEFORE, encoding="utf-8")
    (OUT / "before.py").write_text(BEFORE, encoding="utf-8")
    (OUT / "after.py").write_text(AFTER, encoding="utf-8")
    diff_text = _diff()
    (OUT / "candidate.diff").write_text(diff_text, encoding="utf-8")

    development = [case for case in CASES if case["role"] != "heldout"]
    policy = freeze_policy({
        "enabled_authorize_edges": ["manual_authorize"],
        "isolation_requirement": "standard",
        "required_check_ids": [case["item_id"] for case in development],
        "original_good_ids": ["hello"],
        "target_bad_ids": ["boil", "refund"],
        "security_violations": 0,
        "primary_metric_id": "hello",
        "evolution_capabilities": ["evolution-kernel"],
        "gate_receipt_trust": {"host_build_hash": "evolution-example"},
    })
    fixed_outputs = {
        row["item_id"]: row["output"]
        for row in _cases_from(AFTER, development)
    }
    body = {
        "gate_policy": policy,
        "heldout_ids": ["redeye"],
        "subject_hash": "subject-before",
        "repairer_hash": "repairer-example",
        "evaluator_hash": "exact_match_judge",
        "diff_text": diff_text,
        "allowed_prefixes": ["subject.py"],
        "proposed_outputs": fixed_outputs,
    }
    kernel = open_kernel(CONF)
    await kernel.setup()
    kernel.start_workers()
    try:
        before = await _benchmark(kernel, "evolution-before", BEFORE)
        service = EvolutionService(kernel)
        opened = await service.admit({
            "tenant_id": "default",
            "run_id": before["run_id"],
            "report_revision": 1,
        })
        if opened["status"] != "admitted":
            raise RuntimeError(f"admission rejected: {opened}")
        final = await build_evolution_graph(service).ainvoke({
            "campaign_id": opened["campaign_id"],
            "status": "admitted",
            "requested_step": "advance",
            "visited": [],
            "body": body,
        })
        advanced = advance_result(opened["campaign_id"], final)
        if advanced["status"] != "trial_running":
            raise RuntimeError(f"evolution stopped: {advanced}")
        trial = await wait_terminal(kernel, advanced["benchmark_run_id"], timeout=30)
        decided_state = await build_evolution_graph(service).ainvoke({
            "campaign_id": opened["campaign_id"],
            "status": "trial_running",
            "requested_step": "advance",
            "visited": [],
            "body": body,
        })
        decided = advance_result(opened["campaign_id"], decided_state)
        if decided["reason"] != "heldout_not_collected":
            raise RuntimeError(f"heldout confirmation was not withheld: {decided}")
        async with await kernel.store.connect() as conn:
            cur = await conn.execute(
                """
                SELECT artifact_hash, diff_text
                FROM evolution_candidates
                WHERE campaign_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (opened["campaign_id"],),
            )
            candidate = await cur.fetchone()
        if candidate["diff_text"] != diff_text:
            raise RuntimeError("sealed diff does not match the candidate")
        _apply(candidate["diff_text"])
        optimized = SUBJECT.read_text(encoding="utf-8")
        if optimized != AFTER:
            raise RuntimeError("patched subject does not match the candidate")
        after = await _benchmark(kernel, "evolution-after", optimized)
    finally:
        await kernel.close()

    report = {
        "before": before,
        "trial": {
            "run_id": advanced["benchmark_run_id"],
            "status": trial["status"],
            "final_score": trial["final_score"],
        },
        "evolution": {
            "campaign_id": opened["campaign_id"],
            "visited": final["visited"],
            "status": decided["status"],
            "stage_outcome": decided["stage_outcome"],
            "release_authorization": decided["release_authorization"],
            "reason": decided["reason"],
        },
        "after": after,
        "score_before": before["final_score"],
        "score_after": after["final_score"],
        "score_delta": after["final_score"] - before["final_score"],
    }
    (OUT / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({
        "score_before": report["score_before"],
        "score_trial": report["trial"]["final_score"],
        "score_after": report["score_after"],
        "score_delta": report["score_delta"],
        "visited": report["evolution"]["visited"],
        "evolution_status": report["evolution"]["status"],
        "release_authorization": report["evolution"]["release_authorization"],
        "items_before": before["items"],
        "items_after": after["items"],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
