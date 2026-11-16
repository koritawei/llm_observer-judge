#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full evaluation and self-evolution demo for the store customer-service agent.

Each question goes through intent recognition, tool selection, tool execution, and result composition.
After the dataset enters the business agent it runs through benchmark. Failures edit the skill, tools, or code from the trace.
After the patch is sealed, benchmark is rerun and a summary with each dimension is written.
"""

from __future__ import annotations

import asyncio
import difflib
import html
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("EVALBENCH_SECRET_DATABASE", "eval")
os.environ.setdefault("EVALBENCH_SECRET_PLUGIN_TRUST", "test-trust")
os.environ.setdefault("EVALBENCH_SECRET_CALLBACK", "test-callback")
os.environ.setdefault("EVALBENCH_ENV", "local")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "business_shop"
OUT = ROOT / "out" / "business_evolution"
WORKSPACE = OUT / "workspace"
CONF = ROOT / "conf"
FILES = ("skill.md", "tools.py", "agent.py")


def _read_tree(root: Path) -> dict[str, str]:
    return {name: (root / name).read_text(encoding="utf-8") for name in FILES}


def _write_tree(root: Path, files: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")


def _load_answer(workspace: Path):
    stamp = time.time_ns()
    spec = importlib.util.spec_from_file_location(
        f"business_shop_agent_{stamp}",
        workspace / "agent.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("business agent could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    answer = module.answer
    if not callable(answer):
        raise RuntimeError("business agent has no answer function")
    return answer


def _dataset() -> list[dict[str, str]]:
    rows = json.loads((SOURCE / "dataset.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("dataset is empty")
    return rows


def _run_agent(workspace: Path, cases: list[dict[str, str]]) -> list[dict[str, object]]:
    answer = _load_answer(workspace)
    produced = []
    for case in cases:
        trace = answer(case["query"])
        produced.append({
            "item_id": case["item_id"],
            "role": case["role"],
            "query": case["query"],
            "expected": case["expected"],
            "output": trace["text"],
            "intent_keyword": trace["intent_keyword"],
            "intent": trace["intent"],
            "selected_tool": trace["selected_tool"],
            "tool_result": trace["tool_result"],
            "catalog": trace["catalog"],
            "layer": trace["layer"],
            "action": trace["action"],
        })
    return produced


def _cases_for_kernel(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "item_id": row["item_id"],
            "input": {"text": row["query"]},
            "output": {"text": row["output"]},
            "expected": {"text": row["expected"]},
        }
        for row in rows
    ]


def _catalog(row: dict[str, object]) -> dict[str, str]:
    catalog = row["catalog"]
    if not isinstance(catalog, dict):
        raise RuntimeError(f"{row['item_id']} catalog is missing")
    return {str(name): str(value) for name, value in catalog.items()}


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"{label} is missing: {old}")
    return source.replace(old, new, 1)


def _tool_matching(catalog: dict[str, str], expected: str) -> str:
    matches = [name for name, value in catalog.items() if value == expected]
    if len(matches) != 1:
        raise RuntimeError(f"catalog has no single tool for {expected}: {matches}")
    return matches[0]


def _clear_code_discard(source: str) -> str:
    old = '    if intent == "知识" and "多少" in query:\n        del tool_result\n        return ""\n'
    new = "    del query, intent, tool_result\n"
    source = _replace_once(source, old, new, "code discard block")
    return source.replace(
        "知识问句里出现「多少」时丢掉工具结果。",
        "保留工具结果。",
        1,
    )


def _repair(
    files: dict[str, str],
    rows: list[dict[str, object]],
) -> tuple[dict[str, str], list[str]]:
    """From traces that failed on the dev set, edit intent, routes, the tool table, or result filtering."""

    skill = files["skill.md"]
    tools = files["tools.py"]
    agent = files["agent.py"]
    notes: list[str] = []
    for row in rows:
        if str(row["role"]) == "heldout" or float(row["score"]) >= 1:
            continue
        action = str(row["action"])
        expected = str(row["expected"])
        catalog = _catalog(row)
        item_id = row["item_id"]
        if action == "wrong_intent":
            keyword = str(row["intent_keyword"])
            intent = str(row["intent"])
            skill = _replace_once(
                skill,
                f"{keyword}: {intent}",
                f"{keyword}: 退货",
                "intent rule",
            )
            notes.append(
                f"{item_id}：意图识别把「{keyword}」标成「{intent}」，"
                "结果话术盖过了工具。skill.md 的意图改为「退货」。"
            )
            policy_value = catalog.get("policy", "")
            if policy_value not in {"", "不知道", expected}:
                tools = _replace_once(
                    tools,
                    f'"{policy_value}"',
                    f'"{expected}"',
                    "policy value",
                )
                notes.append(
                    f"{item_id}：政策工具返回「{policy_value}」。"
                    f"tools.py 改为「{expected}」。"
                )
        elif action == "wrong_route":
            tool = _tool_matching(catalog, expected)
            skill = _replace_once(
                skill,
                f"{row['intent']}: {row['selected_tool']}",
                f"{row['intent']}: {tool}",
                "route rule",
            )
            notes.append(
                f"{item_id}：意图「{row['intent']}」选了工具 {row['selected_tool']}，"
                f"该工具结果用不上。目录里 {tool} 已有「{expected}」。"
                "skill.md 的路由改到这个工具。"
            )
        elif action == "use_result" and str(row["tool_result"]) != expected:
            tools = _replace_once(
                tools,
                f'"{row["tool_result"]}"',
                f'"{expected}"',
                "tool value",
            )
            notes.append(
                f"{item_id}：工具 {row['selected_tool']} 返回「{row['tool_result']}」。"
                f"tools.py 改为「{expected}」。"
            )
        elif action == "drop_result":
            agent = _clear_code_discard(agent)
            notes.append(
                f"{item_id}：知识工具已返回「{row['tool_result']}」，"
                "结果层因问句含「多少」把答案清空。删除 agent.py 的这段过滤。"
            )
        else:
            raise RuntimeError(f"{item_id} has no repair for action {action}")
    changed = {"skill.md": skill, "tools.py": tools, "agent.py": agent}
    if changed == files:
        raise RuntimeError("failed report rows produced no edit")
    return changed, notes


def _diff(before: dict[str, str], after: dict[str, str]) -> str:
    chunks = []
    for name in FILES:
        chunks.append("".join(difflib.unified_diff(
            before[name].splitlines(keepends=True),
            after[name].splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
        )))
    return "".join(chunks)


def _apply(diff_text: str, before: dict[str, str]) -> None:
    _write_tree(WORKSPACE, before)
    subprocess.run(
        ["patch", "-p1"],
        input=diff_text,
        text=True,
        cwd=WORKSPACE,
        check=True,
        capture_output=True,
    )


async def _wait(kernel, run_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 30
    last = None
    while time.monotonic() < deadline:
        last = await kernel.store.get_run(run_id)
        if last and last["status"] in {"completed", "failed", "stopped"}:
            return last
        await asyncio.sleep(0.05)
    raise RuntimeError(f"run did not finish: {last}")


async def _benchmark(kernel, name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    created = await kernel.publish_run({
        "benchmark_name": name,
        "profile_id": "benchmark_sync",
        "business": "default",
        "cases": _cases_for_kernel(rows),
    })
    finished = await _wait(kernel, created["run_id"])
    stored = {item["item_id"]: item for item in await kernel.store.list_items(created["run_id"])}
    scored = []
    for row in rows:
        item = stored[str(row["item_id"])]
        scored.append({**row, "score": item["score"], "reason": item.get("reason")})
    meta = finished.get("report_meta_json") or {}
    return {
        "run_id": created["run_id"],
        "status": finished["status"],
        "final_score": finished["final_score"],
        "kernel_report": meta.get("path"),
        "items": scored,
    }


def _report_md(title: str, report: dict[str, object]) -> str:
    lines = [
        f"# {title}",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- status: `{report['status']}`",
        f"- final_score: {report['final_score']}",
        f"- kernel_report: `{report['kernel_report']}`",
        "",
        "| 用例 | 角色 | 意图 | 工具 | 执行结果 | 动作 | 输出 | 期望 | 分数 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["items"]:
        intent = row["intent"]
        if row["intent_keyword"]:
            intent = f"{row['intent_keyword']}→{row['intent']}"
        lines.append(
            "| {item_id} | {role} | {intent} | {tool} | {tool_result} | {action} | "
            "{output} | {expected} | {score} |".format(
                item_id=row["item_id"],
                role=row["role"],
                intent=intent,
                tool=row["selected_tool"],
                tool_result=row["tool_result"] or "空",
                action=row["action"],
                output=row["output"] or "空",
                expected=row["expected"],
                score=row["score"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _change_md(notes: list[str], diff_text: str, evolution: dict[str, object]) -> str:
    body = "\n".join(f"- {note}" for note in notes)
    return "\n".join([
        "# 修改报告",
        "",
        "修改来自开发集里分数为 0 的轨迹：意图、路由、工具返回值、结果过滤。",
        "",
        body,
        "",
        f"- evolution_status: `{evolution['status']}`",
        f"- stage_outcome: `{evolution['stage_outcome']}`",
        f"- release_authorization: `{evolution['release_authorization']}`",
        f"- reason: `{evolution['reason']}`",
        f"- trial_score: {evolution['trial_score']}",
        "",
        "```diff",
        diff_text.rstrip("\n"),
        "```",
        "",
    ])


def _expect_scores(
    report: dict[str, object],
    *,
    passed: set[str],
    failed: set[str],
) -> None:
    seen = {str(row["item_id"]): float(row["score"]) for row in report["items"]}
    for item_id in passed:
        if seen.get(item_id, 0) < 1:
            raise RuntimeError(f"{item_id} should pass, score={seen.get(item_id)}")
    for item_id in failed:
        if seen.get(item_id) != 0:
            raise RuntimeError(f"{item_id} should score 0, score={seen.get(item_id)}")


def _cell(value: object) -> str:
    return html.escape(str(value))


def _intent_label(row: dict[str, object]) -> str:
    if row["intent_keyword"]:
        return f"{row['intent_keyword']}→{row['intent']}"
    return str(row["intent"])


def _table(report: dict[str, object]) -> str:
    rows = []
    for row in report["items"]:
        rows.append(
            "<tr>"
            f"<td>{_cell(row['item_id'])}</td>"
            f"<td>{_cell(row['role'])}</td>"
            f"<td>{_cell(_intent_label(row))}</td>"
            f"<td>{_cell(row['selected_tool'])}</td>"
            f"<td>{_cell(row['tool_result'] or '空')}</td>"
            f"<td>{_cell(row['action'])}</td>"
            f"<td>{_cell(row['output'] or '空')}</td>"
            f"<td>{_cell(row['expected'])}</td>"
            f"<td>{_cell(row['score'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>用例</th><th>角色</th><th>意图</th><th>工具</th><th>执行结果</th>"
        "<th>动作</th><th>输出</th><th>期望</th><th>分数</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _dimension_rows(before: dict[str, object], after: dict[str, object]) -> str:
    labels = {
        "wrong_intent": "意图识别",
        "wrong_route": "工具选择",
        "no_tool": "工具选择",
        "use_result": "工具结果",
        "drop_result": "结果合成",
    }
    after_score = {str(row["item_id"]): float(row["score"]) for row in after["items"]}
    grouped: dict[str, list[str]] = {}
    for row in before["items"]:
        if float(row["score"]) >= 1:
            continue
        label = labels.get(str(row["action"]), str(row["layer"]))
        grouped.setdefault(label, []).append(str(row["item_id"]))
    body = []
    for label in ("意图识别", "工具选择", "工具结果", "结果合成"):
        item_ids = grouped.get(label, [])
        if not item_ids:
            continue
        later = "、".join(
            f"{item_id} {'通过' if after_score[item_id] >= 1 else '仍为 0'}"
            for item_id in item_ids
        )
        body.append(
            "<tr>"
            f"<td>{_cell(label)}</td>"
            f"<td>{_cell('、'.join(item_ids))}</td>"
            f"<td>{_cell(later)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>维度</th><th>基线未通过</th><th>进化后</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _summary_html(
    before: dict[str, object],
    after: dict[str, object],
    change: str,
) -> str:
    before_score = float(before["final_score"])
    after_score = float(after["final_score"])
    delta = after_score - before_score
    heldout = [
        str(row["item_id"])
        for row in after["items"]
        if str(row["role"]) == "heldout"
    ]
    dimensions = _dimension_rows(before, after)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>门店客服 agent 自进化总结</title>
<style>
body {{ font-family: sans-serif; margin: 32px; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
pre {{ white-space: pre-wrap; background: #f6f6f6; padding: 12px; }}
</style>
</head>
<body>
<h1>门店客服 agent 自进化总结</h1>
<p>流程是意图识别、工具选择、工具执行、结果合成。</p>
<p>基线 {_cell(before_score)}，进化后 {_cell(after_score)}，提升 {_cell(delta)}。</p>
<p>隐藏集 {_cell("、".join(heldout))} 的分数仍是 0。发布授权保持 denied，原因是 heldout_not_collected。</p>
<h2>维度</h2>
{dimensions}
<h2>代码修改报告</h2>
<pre>{_cell(change)}</pre>
<h2>基线 report</h2>
<p>run {_cell(before["run_id"])}，分数 {_cell(before_score)}</p>
{_table(before)}
<h2>进化后 report</h2>
<p>run {_cell(after["run_id"])}，分数 {_cell(after_score)}</p>
{_table(after)}
</body>
</html>
"""


async def main() -> None:
    from benchmark.evolution.graph import advance_result, build_evolution_graph
    from benchmark.evolution.service import EvolutionService
    from benchmark.kernel.contract import freeze_policy
    from benchmark.kernel.runtime import open_kernel

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SOURCE, WORKSPACE, ignore=shutil.ignore_patterns("dataset.json"))
    before_files = _read_tree(WORKSPACE)
    cases = _dataset()
    development = [case for case in cases if case["role"] != "heldout"]
    baseline_rows = _run_agent(WORKSPACE, cases)

    kernel = open_kernel(CONF)
    await kernel.setup()
    kernel.start_workers()
    try:
        before = await _benchmark(kernel, "shop-before", baseline_rows)
        failed = [
            row for row in before["items"]
            if str(row["role"]) != "heldout" and float(row["score"]) < 1
        ]
        repaired, notes = _repair(before_files, failed)
        diff_text = _diff(before_files, repaired)
        policy = freeze_policy({
            "enabled_authorize_edges": ["manual_authorize"],
            "isolation_requirement": "standard",
            "required_check_ids": [case["item_id"] for case in development],
            "original_good_ids": ["greet"],
            "target_bad_ids": ["refund", "shipping", "price", "boil"],
            "security_violations": 0,
            "primary_metric_id": "hello",
            "evolution_capabilities": ["evolution-kernel"],
            "gate_receipt_trust": {"host_build_hash": "business-shop-demo"},
        })
        _write_tree(WORKSPACE, repaired)
        fresh = {
            row["item_id"]: {"text": row["output"]}
            for row in _run_agent(WORKSPACE, development)
        }
        _write_tree(WORKSPACE, before_files)
        body = {
            "gate_policy": policy,
            "heldout_ids": ["photo"],
            "subject_hash": "business-shop-before",
            "repairer_hash": "business-shop-repairer",
            "evaluator_hash": "exact_match_judge",
            "diff_text": diff_text,
            "allowed_prefixes": list(FILES),
            "proposed_outputs": fresh,
        }
        service = EvolutionService(kernel)
        opened = await service.admit({
            "tenant_id": "default",
            "run_id": before["run_id"],
            "report_revision": 1,
        })
        if opened["status"] != "admitted":
            raise RuntimeError(f"admission rejected: {opened}")
        campaign_id = opened["campaign_id"]
        final = await build_evolution_graph(service).ainvoke({
            "campaign_id": campaign_id,
            "status": "admitted",
            "requested_step": "advance",
            "visited": [],
            "body": body,
        })
        advanced = advance_result(campaign_id, final)
        if advanced["status"] != "trial_running":
            raise RuntimeError(f"evolution stopped: {advanced}")
        trial = await _wait(kernel, advanced["benchmark_run_id"])
        decided_state = await build_evolution_graph(service).ainvoke({
            "campaign_id": campaign_id,
            "status": "trial_running",
            "requested_step": "advance",
            "visited": [],
            "body": body,
        })
        decided = advance_result(campaign_id, decided_state)
        if decided["reason"] != "heldout_not_collected":
            raise RuntimeError(f"heldout confirmation was not withheld: {decided}")
        async with await kernel.store.connect() as conn:
            cur = await conn.execute(
                """
                SELECT diff_text FROM evolution_candidates
                WHERE campaign_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (campaign_id,),
            )
            candidate = await cur.fetchone()
        if candidate["diff_text"] != diff_text:
            raise RuntimeError("sealed diff does not match the repair")
        _apply(candidate["diff_text"], before_files)
        after_rows = _run_agent(WORKSPACE, cases)
        after = await _benchmark(kernel, "shop-after", after_rows)
        _expect_scores(before, passed={"greet"}, failed={"refund", "shipping", "price", "boil", "photo"})
        _expect_scores(after, passed={"greet", "refund", "shipping", "price", "boil"}, failed={"photo"})
    finally:
        await kernel.close()

    evolution = {
        "status": decided["status"],
        "stage_outcome": decided["stage_outcome"],
        "release_authorization": decided["release_authorization"],
        "reason": decided["reason"],
        "trial_score": trial["final_score"],
        "visited": final["visited"],
    }
    change = _change_md(notes, diff_text, evolution)
    (OUT / "report_before.md").write_text(_report_md("基线 report", before), encoding="utf-8")
    (OUT / "report_after.md").write_text(_report_md("进化后 report", after), encoding="utf-8")
    (OUT / "modification.md").write_text(change, encoding="utf-8")
    (OUT / "candidate.diff").write_text(diff_text, encoding="utf-8")
    summary = OUT / "summary.html"
    summary.write_text(_summary_html(before, after, change), encoding="utf-8")
    print(json.dumps({
        "score_before": before["final_score"],
        "score_trial": trial["final_score"],
        "score_after": after["final_score"],
        "score_delta": float(after["final_score"]) - float(before["final_score"]),
        "evolution_status": evolution["status"],
        "reason": evolution["reason"],
        "summary": str(summary),
        "items_before": [
            {"item_id": row["item_id"], "score": row["score"], "output": row["output"]}
            for row in before["items"]
        ],
        "items_after": [
            {"item_id": row["item_id"], "score": row["score"], "output": row["output"]}
            for row in after["items"]
        ],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
