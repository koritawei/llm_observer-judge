"""Display replacement: on success rewrite the markdown and the Chinese category; on failure keep the input arguments."""

from benchmark.attribution.code.document import (
    TaskSource,
    apply_code_attribution,
    build_task,
    merge_evidence,
)


def _attribution() -> dict[str, str]:
    return {
        "primary_issue_layer": "tool",
        "primary_issue_category": "tool_unavailable",
        "primary_issue_detail": "- 用户要的：加油站\n- 实际做的：没调工具",
        "root_cause": "缺工具",
        "root_cause_chain": "查询加油站但没有对应工具",
        "improvement_suggestion": "补工具",
        "severity": "major",
        "improvement_priority": "P1",
        "affected_dims": ["D2"],
        "attribution_md": "旧稿",
    }


def _ok() -> dict[str, str]:
    return {
        "status": "ok",
        "label_id": "tool.execution_fault",
        "module_root_cause": "执行策略把超时当空结果",
        "evidence_chain": "症状是路线失败。读到 tool_execution_policy 直接把超时当空结果。",
        "fix_plan": "1. 在执行策略里加超时降级",
    }


def test_apply_rewrites_markdown_and_category() -> None:
    attribution = _attribution()
    assert apply_code_attribution(attribution, _ok()) is True
    assert attribution["primary_issue_category"] == "执行失败/超时/5xx"
    assert attribution["root_cause"] == "缺工具"
    assert "执行策略把超时当空结果" in attribution["attribution_md"]
    assert "tool.execution_fault" in attribution["attribution_md"]


def test_apply_keeps_document_on_timeout() -> None:
    attribution = _attribution()
    assert apply_code_attribution(attribution, {"status": "timeout"}) is False
    assert attribution["attribution_md"] == "旧稿"
    assert attribution["primary_issue_category"] == "tool_unavailable"


def test_apply_rejects_root_not_shorter_than_chain() -> None:
    attribution = _attribution()
    result = _ok()
    result["module_root_cause"] = result["evidence_chain"] + "更长"
    assert apply_code_attribution(attribution, result) is False
    assert attribution["primary_issue_category"] == "tool_unavailable"


def test_apply_covers_already_rewritten_category() -> None:
    attribution = _attribution()
    attribution["primary_issue_category"] = "缺工具/可见性"
    attribution["attribution_md"] = "上游已经替换过的稿"
    assert apply_code_attribution(attribution, _ok()) is True
    assert attribution["primary_issue_category"] == "执行失败/超时/5xx"
    assert "上游已经替换过的稿" not in attribution["attribution_md"]


def test_apply_failure_keeps_rewritten_input() -> None:
    attribution = _attribution()
    attribution["primary_issue_category"] = "缺工具/可见性"
    attribution["attribution_md"] = "上游已经替换过的稿"
    assert apply_code_attribution(attribution, {"status": "failed"}) is False
    assert attribution["primary_issue_category"] == "缺工具/可见性"
    assert attribution["attribution_md"] == "上游已经替换过的稿"


def test_merge_evidence_reads_callback_evidence_object() -> None:
    detail = merge_evidence(
        {"primary_issue_layer": "generation", "raw_json": {"primary_issue_layer": "generation"}},
        {
            "evidence": {
                "full_llm_calls_json": '[{"reasoning": "规划"}]',
                "full_tool_calls_json": "[]",
                "full_last_generation_input": "{}",
            }
        },
    )
    assert detail["llm_calls_json"] == '[{"reasoning": "规划"}]'
    assert detail["tool_calls_json"] == "[]"
    assert detail["last_generation_input"] == "{}"
    assert detail["raw_json"]["llm_calls_json"] == '[{"reasoning": "规划"}]'


def test_merge_evidence_reads_metadata_evidence() -> None:
    detail = merge_evidence(
        {"primary_issue_layer": "generation", "raw_json": {"primary_issue_layer": "generation"}},
        {
            "metadata": {
                "evidence": {
                    "full_llm_calls_json": '[{"reasoning": "规划"}]',
                    "full_tool_calls_json": '[{"name": "plan_route"}]',
                    "full_last_generation_input": '[{"role": "user", "content": "去天朗园"}]',
                }
            }
        },
    )
    assert detail["llm_calls_json"] == '[{"reasoning": "规划"}]'
    assert detail["tool_calls_json"] == '[{"name": "plan_route"}]'
    assert detail["last_generation_input"] == '[{"role": "user", "content": "去天朗园"}]'
    assert detail["raw_json"]["tool_calls_json"] == '[{"name": "plan_route"}]'


def test_build_task_reads_raw_json_symptoms() -> None:
    document = {
        "root_cause": "被链路盖住的文本",
        "raw_json": {
            **_attribution(),
            "tool_calls_json": "[]",
            "llm_calls_json": '[{"reasoning": "先查路线", "tools_called": []}]',
            "last_generation_input": '[{"role": "user", "content": "去天朗园"}]',
        },
    }
    task = build_task(
        document,
        TaskSource(
            case_id="run:item",
            query="附近加油站",
            tool_calls_json="",
            response_text="地铁换乘写错了",
        ),
    )
    assert task.case_id == "run:item"
    assert task.symptom_root_cause == "缺工具"
    assert task.symptom_category == "tool_unavailable"
    assert "### 工具调用记录" in task.trace_summary
    assert "(无工具调用)" in task.trace_summary
    assert "先查路线" in task.trace_summary
    assert "去天朗园" in task.trace_summary
    assert "地铁换乘写错了" in task.trace_summary
