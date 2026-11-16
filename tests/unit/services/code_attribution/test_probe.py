"""Probe report: latency fields and result issues, without calling a real aoned."""

import json

import pytest
from benchmark.attribution.code.models import ResultPack
from benchmark.attribution.code.probe import ProbeStats, assess, format_report, load_task


def test_report_includes_timing_and_flags_line_patch(tmp_path) -> None:
    result = ResultPack(
        status="ok",
        label_id="tool.execution_fault",
        module_root_cause="见 nodes/tool.py:12",
        evidence_chain="读到执行策略后定 tool.execution_fault，不归生成层。",
        fix_plan="```diff\n+ x\n```",
        commit_sha="abc",
    )
    issues = assess(result)
    assert any("行号" in item for item in issues)
    assert any("补丁" in item for item in issues)
    report = json.loads(
        format_report(
            result,
            ProbeStats(wall_sec=12.5, session_sec=1.2, poll_count=4, attempts=1),
        )
    )
    assert report["wall_sec"] == 12.5
    assert report["poll_count"] == 4
    assert report["result"]["commit_sha"] == "abc"


def test_load_task_requires_fields(tmp_path) -> None:
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"case_id": "c1", "query": "q"}))
    with pytest.raises(ValueError, match="missing"):
        load_task(path)
