"""Contract, parsing, and prompts."""

from __future__ import annotations

import re
from pathlib import Path

from benchmark.attribution.code.models import LABEL_IDS, TaskPack, label_catalog
from benchmark.attribution.code.parse import parse_result
from benchmark.attribution.code.prompts import render_system, render_user

_OK = {
    "label_id": "tool.execution_fault",
    "tool_name": "plan_route",
    "module_root_cause": "tool_execution_policy 未对超时做降级",
    "evidence_chain": (
        "症状是路线失败。读到 tool_execution_policy 直接把超时当空结果，"
        "所以定 tool.execution_fault。"
    ),
    "fix_plan": "1. 在执行策略里加超时降级\n2. 风险是误降级\n3. 验收看超时仍有可读结果",
    "gate": None,
    "commit_sha": "model-sha",
}


def _task() -> TaskPack:
    return TaskPack(
        case_id="case-1",
        query="附近加油站",
        symptom_layer="tool",
        symptom_category="tool_unavailable",
        symptom_root_cause="缺工具",
        symptom_chain="查询加油站但没有对应工具",
        symptom_suggestion="补工具",
        trace_summary="plan_route 超时",
    )


def test_closed_set_has_formal_and_gate_labels() -> None:
    assert len(LABEL_IDS) == 14
    assert "gate.eval_misjudge" in LABEL_IDS
    assert "gate.non_agent" in LABEL_IDS
    catalog = label_catalog()
    assert "tool.execution_fault" in catalog
    assert "ieval/core" not in catalog


def test_parse_strips_fence_and_keeps_model_commit() -> None:
    text = "```json\n" + __import__("json").dumps(_OK, ensure_ascii=False) + "\n```"
    result = parse_result(text)
    assert result.status == "ok"
    assert result.commit_sha == "model-sha"
    assert result.tool_name == "plan_route"
    assert result.gate is None


def test_parse_rejects_unknown_label_and_bad_json() -> None:
    bad_label = dict(_OK, label_id="tool.unknown")
    result = parse_result(__import__("json").dumps(bad_label))
    assert result.status == "failed"
    assert result.commit_sha == "model-sha"

    broken = parse_result("not-json")
    assert broken.status == "failed"
    assert broken.error_message.startswith("json:")


def test_non_tool_label_clears_tool_name() -> None:
    payload = dict(
        _OK,
        label_id="gate.eval_misjudge",
        tool_name="plan_route",
        gate="gate.eval_misjudge",
    )
    result = parse_result(__import__("json").dumps(payload))
    assert result.status == "ok"
    assert result.tool_name == ""
    assert result.gate == "gate.eval_misjudge"


def test_prompt_fills_case_fields() -> None:
    system = render_system(repo_root="workspace/amap-ai-agent", repo="demo-repo")
    user = render_user(_task(), repo_root="workspace/amap-ai-agent")
    assert "你是 demo-repo 的代码归因助手" in system
    assert "workspace/amap-ai-agent" in system
    assert "tool.capability_gap" in system
    assert "module_root_cause、evidence_chain、fix_plan 必须用中文" in system
    assert "上游症状" in user
    assert "case-1" in user
    assert "附近加油站" in user
    assert "plan_route 超时" in user


def test_package_does_not_mention_ieval() -> None:
    root = Path(__file__).resolve().parents[4] / "app" / "services" / "code_attribution"
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "import core" not in text
        assert "from core" not in text
        assert "benchmark.routes" not in text
        assert re.search(r"\bieval", text.lower()) is None
