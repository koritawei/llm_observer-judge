"""MR6 review reproduction: candidate isolation, scoring evidence, ambiguous references, and the L2 boundary."""

import json

import pytest
from benchmark.reporting.cluster.agent import (
    ClusterReportAgent,
    ClusterReportAgentOptions,
    _build_user_prompt,
    _parse_summary,
)


class _LLM:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return json.dumps(payload)


def _agent(llm: _LLM) -> ClusterReportAgent:
    return ClusterReportAgent(
        ClusterReportAgentOptions("review", "cluster_summary", 8),
        llm=llm,
        prompt_manager=None,
    )


async def test_mixed_root_fallback_excludes_candidate_traces() -> None:
    candidate = {"case_id": "c1", "trace_id": "t1"}
    stats = {
        "attribution": {
            "root_cause_cases": {
                "unclassified": {
                    "cases": [
                        candidate,
                        {"case_id": "c2", "trace_id": "t2"},
                    ]
                }
            },
            "other_candidate_stats": {
                "candidates": [
                    {
                        "agent_scope": "vehicle",
                        "candidate_key": "other:missing",
                        "cases": [candidate],
                    }
                ]
            },
        }
    }
    summary = await _agent(
        _LLM(
            [
                {
                    "clusters": [{"root_cause_codes": ["unclassified"], "related_cases": []}],
                    "aone_suggestions": [
                        {"root_cause_codes": ["unclassified"], "related_cases": []}
                    ],
                }
            ]
        )
    ).summarize(stats)
    assert summary["clusters"][0]["related_cases"] == ["t2"]
    assert summary["clusters"][0]["affected_count"] == 1
    assert summary["aone_suggestions"][0]["related_cases"] == ["t2"]


def test_vui_prompt_keeps_original_judge_evidence_only() -> None:
    case = {
        "case_id": "low",
        "trace_id": "private-trace",
        "judge_score": 0.0,
        "reason": "D4 未完成",
        "dimension_deductions": [{"dimension": "D4", "score": 0.0}],
        "attribution": {"root_cause": "private-root"},
    }
    prompt = _build_user_prompt(
        {"report_job_type": "vui_low_score", "evaluation_low_score_analysis": {"cases": [case]}},
        max_clusters=8,
    )
    payload = json.loads(prompt.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert payload["evaluation_low_score_analysis"]["cases"] == [
        {key: case[key] for key in ("case_id", "judge_score", "reason", "dimension_deductions")}
    ]
    assert "private-trace" not in prompt
    assert "private-root" not in prompt


@pytest.mark.parametrize("candidate", [False, True])
async def test_duplicate_case_id_never_selects_last_trace(candidate: bool) -> None:
    cases = [
        {"suite": "A", "case_id": "1", "turn_index": "0", "trace_id": "trace-A"},
        {"suite": "B", "case_id": "1", "turn_index": "1", "trace_id": "trace-B"},
    ]
    if candidate:
        stats = {
            "attribution": {
                "other_candidate_stats": {
                    "candidates": [
                        {
                            "agent_scope": "vehicle",
                            "candidate_key": "other:test",
                            "cases": cases,
                        }
                    ]
                }
            }
        }
        key = "candidate_clusters"
        item = {"agent_scope": "vehicle", "candidate_key": "other:test", "related_cases": ["1"]}
    else:
        stats = {"attribution": {"root_cause_cases": {"failure": {"cases": cases}}}}
        key = "clusters"
        item = {"root_cause_codes": ["failure"], "related_cases": ["1"]}
    summary = await _agent(_LLM([{key: [item]}])).summarize(stats)
    assert summary[key][0]["related_cases"] == []


async def test_stability_finding_uses_own_group_traces() -> None:
    groups = [
        {
            "scene_tag": tag,
            "query": tag,
            "agent_execution_observation": "nondeterminism",
            "evidence_strength": "strong",
            "representative_runs": [{"trace_id": trace}],
        }
        for tag, trace in [("A", "trace-A"), ("B", "trace-B")]
    ]
    stats = {"report_job_type": "stability", "observations": {"stability": groups}}
    summary = await _agent(
        _LLM(
            [
                {
                    "findings": [
                        {
                            "scene_tag": "A",
                            "query": "A",
                            "representative_traces": ["trace-B"],
                        }
                    ]
                }
            ]
        )
    ).summarize(stats)
    assert summary["findings"][0]["representative_traces"] == ["trace-A"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("caveats", 1),
        ("caveats", "warning"),
        ("caveats", [{}]),
        ("clusters", {}),
        ("clusters", [1]),
        ("findings", ["bad"]),
        ("clusters", [{"related_cases": "case"}]),
    ],
)
def test_wrong_summary_field_types_are_parse_errors(field: str, value: object) -> None:
    assert _parse_summary(json.dumps({field: value})).get("parse_error")


async def test_wrong_summary_types_trigger_existing_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr("benchmark.reporting.cluster.agent.asyncio.sleep", no_sleep)
    llm = _LLM([{"caveats": 1}, {"headline": "valid", "caveats": ["evidence limited"]}])
    summary = await _agent(llm).summarize({})
    assert llm.calls == 2
    assert summary["caveats"] == ["evidence limited"]


@pytest.mark.parametrize("refs,expected", [(["1"], []), (["1", "2"], ["formal-C"])])
async def test_candidate_isolation_cannot_enable_fallback_for_ambiguous_refs(
    refs: list[str], expected: list[str]
) -> None:
    candidate = {"suite": "B", "case_id": "1", "trace_id": "candidate-B"}
    stats = {
        "attribution": {
            "root_cause_cases": {
                "unclassified": {
                    "cases": [
                        {"suite": "A", "case_id": "1", "trace_id": "formal-A"},
                        candidate,
                        {"suite": "C", "case_id": "2", "trace_id": "formal-C"},
                    ]
                }
            },
            "other_candidate_stats": {
                "candidates": [
                    {
                        "agent_scope": "vehicle",
                        "candidate_key": "other:missing",
                        "cases": [candidate],
                    }
                ]
            },
        }
    }
    summary = await _agent(
        _LLM(
            [
                {
                    "clusters": [
                        {
                            "root_cause_codes": ["unclassified"],
                            "related_cases": refs,
                        }
                    ]
                }
            ]
        )
    ).summarize(stats)
    assert summary["clusters"][0]["related_cases"] == expected
    assert summary["clusters"][0]["affected_count"] == len(expected)
