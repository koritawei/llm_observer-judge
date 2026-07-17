"""Build Langfuse-style pipeline execution graph for Trace detail."""

from __future__ import annotations

from typing import Any


def map_score_status_to_ui(
    score_status: str | None,
    *,
    is_bad_case: bool | None = None,
) -> str:
    """Map eval ``score_status`` to execution-graph / Observations UI status."""
    st = str(score_status or "pending")
    if st == "scored":
        return "bad_case" if is_bad_case else "completed"
    if st == "skipped":
        return "skipped"
    if st == "infra_error":
        return "failed"
    if st == "model_error":
        return "model_error"
    return st


def resolve_attribution_status(
    *,
    bad_case_detected: bool,
    attribution_triggered: bool,
) -> str:
    """Main-path attribution: skipped on good case; pending until results land."""
    if attribution_triggered:
        return "completed"
    if bad_case_detected:
        return "pending"
    return "skipped"


def resolve_annotation_status(
    *,
    annotation_triggered: bool = False,
    annotations: list[dict[str, Any]] | None = None,
) -> str:
    """Side-path annotation: pending while open; completed only when done."""
    rows = list(annotations or [])
    if rows:
        statuses = {str(a.get("status") or "").upper() for a in rows}
        if "COMPLETED" in statuses:
            return "completed"
        if statuses & {"PENDING", "CLAIMED", "IN_PROGRESS"}:
            return "pending"
        return "pending"
    if annotation_triggered:
        return "pending"
    return "skipped"


def build_execution_graph(
    *,
    layers: list[list[str]],
    results: list[dict[str, Any]],
    pipeline_events: list[dict[str, Any]] | None = None,
    bad_case_detected: bool = False,
    attribution_triggered: bool = False,
    annotation_triggered: bool = False,
    annotations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Ordered node list the trace walked; aborted nodes marked skipped.

    Soft outcomes (scored bad-case, model_error) do not abort the graph.
    Hard ``infra_error`` stops subsequent nodes.

    Topology after ScoreNode:
    - Attribution — **main** serial path (pipeline)
    - Annotation — **side** path (旁路), separate stage

    Each node includes:
    - ``mode``: ``serial`` | ``parallel``
    - ``stage``: stage index (serial steps / parallel groups)
    - ``stage_size``: nodes in the same stage (``>1`` ⇒ parallel group)
    """
    by_id = {str(r.get("agent_id")): r for r in results if r.get("agent_id")}
    stages: list[dict[str, Any]] = []
    aborted = False
    abort_after: str | None = None

    def _append_serial(node: dict[str, Any]) -> None:
        stages.append({"mode": "serial", "nodes": [node]})

    def _append_parallel(nodes: list[dict[str, Any]]) -> None:
        if not nodes:
            return
        if len(nodes) == 1:
            stages.append({"mode": "serial", "nodes": nodes})
        else:
            stages.append({"mode": "parallel", "nodes": nodes})

    prefix_names = [
        "SamplingNode",
        "IndexNode",
        "WriteNode",
        "DispatchPublish",
        "EvalRouteNode",
    ]
    latest_ev: dict[str, dict[str, Any]] = {}
    for ev in pipeline_events or []:
        name = str(ev.get("node") or "")
        if name:
            latest_ev[name] = ev

    for name in prefix_names:
        ev = latest_ev.get(name)
        if ev is None and name not in ("IndexNode", "DispatchPublish"):
            if not results and not pipeline_events:
                continue
        status = "pending"
        if ev:
            status = str(ev.get("status") or "completed")
        elif results or pipeline_events:
            status = "completed"
        if name == "EvalRouteNode" and results and status in ("running", "pending"):
            status = "completed"
        _append_serial(
            {
                "id": name,
                "label": name,
                "kind": "pipeline",
                "status": status,
                "error": None,
                "skip_reason": None,
            }
        )

    for layer_i, layer in enumerate(layers):
        layer_nodes: list[dict[str, Any]] = []
        for aid in layer:
            if aborted:
                layer_nodes.append(
                    {
                        "id": aid,
                        "label": aid,
                        "kind": "eval",
                        "status": "skipped",
                        "layer": layer_i,
                        "error": None,
                        "skip_reason": f"aborted_after:{abort_after}",
                    }
                )
                continue
            r = by_id.get(aid) or {}
            st = str(r.get("score_status") or "pending")
            ui_status = map_score_status_to_ui(st, is_bad_case=bool(r.get("is_bad_case")))
            layer_nodes.append(
                {
                    "id": aid,
                    "label": aid,
                    "kind": "eval",
                    "status": ui_status,
                    "layer": layer_i,
                    "error": r.get("error"),
                    "skip_reason": r.get("skip_reason"),
                    "score": r.get("normalized_score", r.get("score")),
                    "is_bad_case": r.get("is_bad_case"),
                }
            )
            if st == "infra_error":
                aborted = True
                abort_after = aid
        _append_parallel(layer_nodes)

    if results or layers:
        score_status = "completed"
        if any(
            str(r.get("score_status")) in ("scored", "model_error", "skipped")
            for r in results
        ):
            score_status = "completed"
        elif aborted:
            score_status = "skipped"
        if latest_ev.get("ScoreNode"):
            score_status = str(latest_ev["ScoreNode"].get("status") or score_status)
        _append_serial(
            {
                "id": "ScoreNode",
                "label": "ScoreNode",
                "kind": "pipeline",
                "status": score_status,
                "error": None,
                "skip_reason": None,
                "detail": {"bad_case_detected": bad_case_detected},
            }
        )

    # Attribution — ALWAYS main path (never side / 旁路)
    attr_status = resolve_attribution_status(
        bad_case_detected=bad_case_detected,
        attribution_triggered=attribution_triggered,
    )
    _append_serial(
        {
            "id": "Attribution",
            "label": "Attribution",
            "kind": "pipeline",  # main path only
            "status": attr_status,
            "error": None,
            "skip_reason": "good_case" if attr_status == "skipped" else None,
            "path": "main",
        }
    )

    # Annotation — side path only (旁路)
    ann_status = resolve_annotation_status(
        annotation_triggered=annotation_triggered,
        annotations=annotations,
    )
    _append_serial(
        {
            "id": "Annotation",
            "label": "Annotation",
            "kind": "side",
            "status": ann_status,
            "error": None,
            "skip_reason": None if ann_status != "skipped" else "not_triggered",
            "path": "side",
        }
    )

    return flatten_stages(stages)


def flatten_stages(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach stage / mode metadata and flatten for API + UI grouping."""
    out: list[dict[str, Any]] = []
    order = 0
    for stage_i, stage in enumerate(stages):
        mode = str(stage.get("mode") or "serial")
        nodes = list(stage.get("nodes") or [])
        size = len(nodes)
        for n in nodes:
            item = dict(n)
            item["mode"] = mode if size > 1 else "serial"
            item["stage"] = stage_i
            item["stage_size"] = size
            item["order"] = order
            out.append(item)
            order += 1
    return out


def group_execution_stages(graph: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group flat graph nodes into serial/parallel stages for rendering."""
    if not graph:
        return []
    stages: list[dict[str, Any]] = []
    by_stage: dict[int, list[dict[str, Any]]] = {}
    order_keys: list[int] = []
    for n in graph:
        sid = int(n.get("stage", n.get("order", 0)))
        if sid not in by_stage:
            by_stage[sid] = []
            order_keys.append(sid)
        by_stage[sid].append(n)
    for sid in order_keys:
        nodes = by_stage[sid]
        mode = "parallel" if len(nodes) > 1 else str(nodes[0].get("mode") or "serial")
        if len(nodes) > 1:
            mode = "parallel"
        stages.append({"id": sid, "mode": mode, "nodes": nodes})
    return stages
