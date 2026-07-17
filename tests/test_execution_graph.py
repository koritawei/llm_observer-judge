"""execution_graph stage / parallel metadata."""

from __future__ import annotations

from llmobserverjudge.eval_worker.execution_graph import (
    build_execution_graph,
    group_execution_stages,
    map_score_status_to_ui,
)


def test_eval_route_not_stuck_running_when_results_exist() -> None:
    graph = build_execution_graph(
        layers=[["judge_quality"]],
        results=[
            {
                "agent_id": "judge_quality",
                "score_status": "scored",
                "normalized_score": 0.9,
                "is_bad_case": False,
            }
        ],
        pipeline_events=[
            {"node": "EvalRouteNode", "status": "running"},
            {"node": "SamplingNode", "status": "completed"},
        ],
        bad_case_detected=False,
    )
    route = next(n for n in graph if n["id"] == "EvalRouteNode")
    assert route["status"] == "completed"


def test_map_score_status_to_ui() -> None:
    assert map_score_status_to_ui("scored") == "completed"
    assert map_score_status_to_ui("scored", is_bad_case=True) == "bad_case"
    assert map_score_status_to_ui("infra_error") == "failed"
    assert map_score_status_to_ui("model_error") == "model_error"
    assert map_score_status_to_ui("skipped") == "skipped"


def test_attribution_main_annotation_side_and_status() -> None:
    graph = build_execution_graph(
        layers=[["judge_safety"]],
        results=[
            {
                "agent_id": "judge_safety",
                "score_status": "scored",
                "normalized_score": 0.1,
                "is_bad_case": True,
            }
        ],
        bad_case_detected=True,
        attribution_triggered=True,
        annotation_triggered=True,
        annotations=[{"status": "PENDING"}],
    )
    attr = next(n for n in graph if n["id"] == "Attribution")
    ann = next(n for n in graph if n["id"] == "Annotation")
    assert attr["kind"] == "pipeline"
    assert attr.get("path") == "main"
    assert attr["kind"] != "side"
    assert attr["status"] == "completed"
    assert attr["mode"] == "serial"
    assert ann["kind"] == "side"
    assert ann.get("path") == "side"
    assert ann["status"] == "pending"  # not completed until human finishes
    assert ann["mode"] == "serial"
    # not in the same parallel stage
    assert attr["stage"] != ann["stage"]

    done = build_execution_graph(
        layers=[["judge_safety"]],
        results=[
            {
                "agent_id": "judge_safety",
                "score_status": "scored",
                "normalized_score": 0.1,
                "is_bad_case": True,
            }
        ],
        bad_case_detected=True,
        attribution_triggered=True,
        annotations=[{"status": "COMPLETED"}],
    )
    assert next(n for n in done if n["id"] == "Annotation")["status"] == "completed"


def test_parallel_eval_layer_and_serial_prefix() -> None:
    graph = build_execution_graph(
        layers=[["judge_safety", "judge_code_eval"]],
        results=[
            {
                "agent_id": "judge_safety",
                "score_status": "scored",
                "normalized_score": 0.2,
                "is_bad_case": True,
            },
            {
                "agent_id": "judge_code_eval",
                "score_status": "scored",
                "normalized_score": 0.1,
                "is_bad_case": True,
            },
        ],
        pipeline_events=[{"node": "SamplingNode", "status": "completed"}],
        bad_case_detected=True,
        attribution_triggered=True,
        annotation_triggered=True,
        annotations=[{"status": "PENDING"}],
    )
    stages = group_execution_stages(graph)
    parallel = [s for s in stages if s["mode"] == "parallel"]
    assert any(
        {n["id"] for n in s["nodes"]} == {"judge_safety", "judge_code_eval"}
        for s in parallel
    )
    # Attribution / Annotation are serial main+side, not a parallel pair
    assert not any(
        {n["id"] for n in s["nodes"]} == {"Attribution", "Annotation"}
        for s in parallel
    )
    sampling = next(n for n in graph if n["id"] == "SamplingNode")
    assert sampling["mode"] == "serial"
    assert sampling["stage_size"] == 1
    safety = next(n for n in graph if n["id"] == "judge_safety")
    assert safety["mode"] == "parallel"
    assert safety["stage_size"] == 2
    attr = next(n for n in graph if n["id"] == "Attribution")
    ann = next(n for n in graph if n["id"] == "Annotation")
    assert attr["kind"] == "pipeline"
    assert ann["kind"] == "side"