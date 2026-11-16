from benchmark.kernel.casebook import assemble_case_book


def test_assemble_case_book_groups_manifest_and_scores():
    book = assemble_case_book(
        {"run_id": "run-1", "metrics": {"final_score": 0.5}},
        [
            {
                "item_id": "b",
                "payload": {"input": {"text": "q2"}, "output": {"text": "a2"}, "expected": {"text": "e2"}},
            },
            {
                "item_id": "a",
                "payload": {"input": {"text": "q1"}, "output": {"text": "a1"}, "expected": {"text": "e1"}},
            },
        ],
        [
            {"item_id": "a", "criterion_id": "exact", "status": "success", "score": 1, "reason": "match"},
            {"item_id": "a", "criterion_id": "overlap", "status": "success", "score": 0, "reason": "miss"},
            {"item_id": "b", "criterion_id": "exact", "status": "failed", "score": None, "error_message": "down"},
        ],
        None,
    )
    assert [case["item_id"] for case in book["cases"]] == ["b", "a"]
    assert book["cases"][1]["score"] == 0.5
    assert book["cases"][1]["expected"] == {"text": "e1"}
    assert book["cases"][0]["criteria"][0]["reason"] == "down"
    assert book["cases"][0]["score"] is None
    assert book["cases"][0]["graphs"][0]["recorded"] is False
    assert book["cases"][0]["graphs"][0]["nodes"][3]["status"] == "skipped"
    assert book["cases"][1]["graphs"][0]["nodes"][5]["id"] == "run_judges"
    assert book["cases"][1]["graphs"][0]["nodes"][5]["status"] == "executed"
    assert book["report"]["metrics"]["final_score"] == 0.5


def test_stored_graph_replaces_the_fallback():
    stored = {
        "nodes": [{"id": "build_context", "status": "executed"}, {"id": "run_judges", "status": "failed"}],
        "edges": [{"source": "build_context", "target": "run_judges", "taken": False}],
        "recorded": True,
    }
    book = assemble_case_book(
        {"run_id": "run-1", "metrics": {}},
        [{"item_id": "a", "payload": {"input": "q"}}],
        [{"item_id": "a", "criterion_id": "exact", "status": "success", "score": 1}],
        None,
        [{"item_id": "a", "phase_id": "judge", "output": {"_graph": stored}}],
    )
    graph = book["cases"][0]["graphs"][0]
    assert graph["recorded"] is True
    assert graph["phase_id"] == "judge"
    assert graph["nodes"][1]["status"] == "failed"
