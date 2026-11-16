"""Group a run's sealed inputs and criterion rows into a case book.

The browser uses this shape for the reading view and for version comparison.
It does not change scores or run status.
"""

from __future__ import annotations

from typing import Any

from benchmark.pipeline.runner import STAGE_NODES

_CHECKER_SKIP = "phase has no checker pack"


def assemble_case_book(
    run: dict[str, Any],
    manifests: list[dict[str, Any]],
    items: list[dict[str, Any]],
    report: dict[str, Any] | None,
    cursors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_item: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in manifests:
        item_id = str(row.get("item_id") or "")
        if not item_id or item_id in by_item:
            continue
        payload = row.get("payload") or {}
        order.append(item_id)
        by_item[item_id] = {
            "item_id": item_id,
            "input": payload.get("input"),
            "output": payload.get("output"),
            "expected": payload.get("expected"),
            "criteria": [],
        }
    for item in items:
        item_id = str(item.get("item_id") or "")
        if not item_id:
            continue
        if item_id not in by_item:
            order.append(item_id)
            by_item[item_id] = {
                "item_id": item_id,
                "input": item.get("input"),
                "output": item.get("output"),
                "expected": None,
                "criteria": [],
            }
        entry = by_item[item_id]
        if entry["input"] is None and item.get("input") is not None:
            entry["input"] = item.get("input")
        if entry["output"] is None and item.get("output") is not None:
            entry["output"] = item.get("output")
        entry["criteria"].append({
            "criterion_id": item.get("criterion_id"),
            "status": item.get("status"),
            "score": _number(item.get("score")),
            "reason": item.get("reason") or item.get("error_message") or "",
            "judge": item.get("judge"),
        })
    cases = []
    for item_id in order:
        case = by_item[item_id]
        scores = [row["score"] for row in case["criteria"] if isinstance(row["score"], float)]
        case["score"] = round(sum(scores) / len(scores), 4) if scores else None
        case["graphs"] = _graphs_for(item_id, cursors or [], has_result=bool(case["criteria"]))
        cases.append(case)
    metrics = run.get("metrics") or {}
    return {
        "run": run,
        "report": report or {"metrics": metrics, "report": {"summary": metrics}},
        "cases": cases,
    }


def trace_from_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Turn a runner stage_report into the node graph the case page draws."""

    report = report or {}
    nodes = []
    for name in STAGE_NODES:
        record = report.get(name)
        if not isinstance(record, dict):
            nodes.append({"id": name, "status": "not_reached", "duration_ms": None, "reason": ""})
            continue
        nodes.append({
            "id": name,
            "status": str(record.get("status") or "executed"),
            "duration_ms": record.get("duration_ms"),
            "reason": str(record.get("reason") or ""),
        })
    edges = []
    for index, name in enumerate(STAGE_NODES[:-1]):
        current = nodes[index]
        nxt = nodes[index + 1]
        taken = current["status"] in {"executed", "skipped"} and nxt["status"] != "not_reached"
        edges.append({"source": name, "target": STAGE_NODES[index + 1], "taken": taken})
    return {"nodes": nodes, "edges": edges, "recorded": True}


def fallback_trace(has_result: bool) -> dict[str, Any]:
    """Rebuild the fixed case graph when an older run did not store node timings."""

    if not has_result:
        trace = trace_from_report({})
        trace["recorded"] = False
        return trace
    report = {
        name: (
            {"status": "skipped", "reason": _CHECKER_SKIP, "duration_ms": None}
            if name == "run_checkers"
            else {"status": "executed", "duration_ms": None, "reason": ""}
        )
        for name in STAGE_NODES
    }
    trace = trace_from_report(report)
    trace["recorded"] = False
    return trace


def _graphs_for(item_id: str, cursors: list[dict[str, Any]], *, has_result: bool) -> list[dict[str, Any]]:
    found = []
    for row in cursors:
        if str(row.get("item_id") or "") != item_id:
            continue
        output = row.get("output") or {}
        graph = output.get("_graph") if isinstance(output, dict) else None
        if not isinstance(graph, dict) or not graph.get("nodes"):
            continue
        found.append({
            "phase_id": str(row.get("phase_id") or ""),
            "nodes": list(graph.get("nodes") or []),
            "edges": list(graph.get("edges") or []),
            "recorded": bool(graph.get("recorded", True)),
        })
    if found:
        return found
    return [{"phase_id": "", **fallback_trace(has_result)}]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
