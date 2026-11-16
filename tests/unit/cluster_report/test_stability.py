"""Stability L1 evidence-model tests."""

import pytest
from benchmark.reporting.cluster.schema import BatchCaseRecord, CaseKey
from benchmark.reporting.cluster.stats.stability import build_stability_groups, build_stability_overview


def _record(
    *,
    trace_id: str,
    score: float,
    query: str = "导航回家",
    request_query: str = "导航回家",
    env: dict | None = None,
    text: str = "已为你规划回家路线",
) -> BatchCaseRecord:
    return BatchCaseRecord(
        trace_id=trace_id,
        score=score,
        profile_id="vehicle-agent.standard",
        profile_version="v1",
        case_key=CaseKey(tag="navigation.direct", query=query),
        trace_meta={
            "input": {
                "request_body": {
                    "query": request_query,
                    "scene_id": "navigation",
                    "history": [{"content": "不得进入 ContextDiff"}],
                    "debug": {
                        "eval_config": {
                            "case_meta": {"suite": "standard"},
                            "multi_turn": {"turn_index": "1"},
                        }
                    },
                }
            },
            "output": {
                "env": env or {"model": "v1", "temperature": 0},
                "code": 0,
                "component": {"type": "text", "text": text},
                "content_desc": {"category": "navigation", "used_tool": ["poi_search"]},
                "dto": {"secret": "must-not-leak"},
                "tool_calls": [{"arguments": {"secret": "must-not-leak"}}],
            },
        },
    )


def _groups_by_query(records: list[BatchCaseRecord]) -> dict[str, dict]:
    return {group["query"]: group for group in build_stability_groups(records)}


def test_same_context_with_output_and_score_change_is_nondeterminism() -> None:
    group = _groups_by_query(
        [
            _record(trace_id="good", score=2, text="路线 A"),
            _record(trace_id="bad", score=0, text="路线 B"),
        ]
    )["导航回家"]

    assert group["judge_score_distribution"] == {"0": 1, "1": 0, "2": 1}
    assert group["span"] == 2
    assert group["evaluation_comparability"] == {
        "status": "comparable",
        "reason": "same_known_version",
        "known_versions": group["evaluation_comparability"]["known_versions"],
    }
    assert group["agent_execution_observation"] == "nondeterminism"
    assert group["evidence_strength"] == "strong"
    assert group["priority"] == "P0"
    assert group["governance"] == {
        "priority": "P0",
        "reason": "相同输入与依赖下，执行输出或评测结果出现不一致。",
        "domain": "Agent 执行",
        "action": "排查模型参数、编排分支、缓存和调用链路。",
        "is_agent_defect": True,
    }


def test_context_diff_precedes_score_variance_and_does_not_leak_outputs() -> None:
    group = _groups_by_query(
        [
            _record(trace_id="first", score=2, request_query="导航回家"),
            _record(trace_id="second", score=0, request_query="导航公司"),
        ]
    )["导航回家"]

    assert group["agent_execution_observation"] == "context_diff"
    run = group["representative_runs"][0]
    request = run["context"]["request"]
    assert request["query"] in {"导航回家", "导航公司"}
    assert request["scene_id"] == "navigation"
    assert request["tag"] == "navigation.direct"
    assert request["case_meta_fingerprint"]
    assert request["turn_index"] == "1"
    assert run["context"]["env"]["paths"] == ["model", "temperature"]
    assert "v1" not in str(run["context"]["env"])
    assert "history" not in str(run["context"])
    assert "must-not-leak" not in str(run["output_evidence"])
    assert run["output_evidence"]["used_tools"] == ["poi_search"]


def test_equal_output_with_score_change_is_evaluation_inconsistency() -> None:
    group = _groups_by_query(
        [
            _record(trace_id="first", score=2, text="一致回复"),
            _record(trace_id="second", score=0, text="一致回复"),
        ]
    )["导航回家"]

    assert group["agent_execution_observation"] == "evaluation_inconsistency"
    assert group["evidence_strength"] == "strong"
    assert group["priority"] == "P0"
    assert group["governance"]["domain"] == "评测规则"
    assert group["governance"]["is_agent_defect"] is False


def test_overview_counts_stability_and_governance_domains() -> None:
    groups = build_stability_groups(
        [
            _record(trace_id="stable-1", score=2, query="导航回家", text="路线 A"),
            _record(trace_id="stable-2", score=2, query="导航回家", text="路线 A"),
            _record(trace_id="risk-1", score=2, query="导航公司", text="路线 A"),
            _record(trace_id="risk-2", score=0, query="导航公司", text="路线 B"),
        ]
    )

    overview = build_stability_overview(groups)

    assert overview["total_queries"] == 2
    assert overview["stable_queries"] == 1
    assert overview["stable_rate"] == 0.5
    assert overview["high_risk_queries"] == 1
    assert overview["governance_count"] == 1
    assert overview["classification_distribution"] == {"nondeterminism": 1, "stable": 1}
    assert overview["governance_domain_distribution"] == {"Agent 执行": 1}


@pytest.mark.parametrize("missing_field", ["both", "profile_id", "profile_version"])
def test_mixed_known_and_unknown_versions_are_not_comparable(missing_field: str) -> None:
    known = _record(trace_id="known", score=2.0, text="目的地 A")
    unknown = _record(trace_id="unknown", score=0.0, text="目的地 B")
    if missing_field in ("both", "profile_id"):
        unknown.profile_id = ""
    if missing_field in ("both", "profile_version"):
        unknown.profile_version = ""
    group = build_stability_groups([known, unknown])[0]
    assert group["evaluation_comparability"]["status"] == "risk"
    assert group["evaluation_comparability"]["reason"] == "evaluation_version_unknown"
    assert group["agent_execution_observation"] == "evaluation_comparability_risk"
    assert group["evidence_strength"] == "weak"
    assert group["governance"]["is_agent_defect"] is False


@pytest.mark.parametrize(
    "second_score,expected",
    [
        (2.0, "expected_diversity"),
        (0.0, "nondeterminism"),
    ],
)
def test_long_common_prefix_does_not_hide_response_difference(
    second_score: float, expected: str
) -> None:
    from benchmark.reporting.cluster.stats.stability import _output_evidence

    prefix = "相同前缀" * 80
    first = _record(trace_id="A", score=2.0, text=prefix + "目的地 A")
    second = _record(trace_id="B", score=second_score, text=prefix + "目的地 B")
    first_output, second_output = _output_evidence(first), _output_evidence(second)
    assert first_output["response_fingerprint"] != second_output["response_fingerprint"]
    assert len(first_output["response_preview"]) <= 240
    assert first_output["response_preview"] == second_output["response_preview"]
    group = build_stability_groups([first, second])[0]
    assert group["agent_execution_observation"] == expected
