"""T30/T31 unit tests: the L2 clustering agent and soft numeric checks.

L2 only does semantic grouping. The tests do not focus on whether the LLM "said it right" (that is model capability). They focus on:
1. Boundary protection: a missing LLM, a failed call, or illegal JSON must not affect the report body
2. Soft numeric checks: fabricated numbers are warned; legitimate phrasing is not falsely rejected (T31)
3. Input trimming: per-case detail does not enter the prompt (volume control)
"""

import json

import benchmark.reporting.cluster.agent as agent_module
import pytest
from benchmark.libs.agenthost.models.errors import ProviderError
from benchmark.reporting.cluster.agent import (
    ClusterReportAgent,
    ClusterReportAgentOptions,
    ClusterReportConfigSchema,
    ClusterReportL2Error,
    _LazyChatModel,
    guard_llm_numbers,
)
from benchmark.reporting.cluster.constants import (
    CLUSTER_PROMPT_CONF_DIR,
    DEFAULT_CLUSTER_PROMPT_NAME,
    PROMPT_FILE_SUFFIX,
)

_L1_STATS = {
    "effectiveness": {
        "total": 100,
        "pass_rate": 0.62,
        "score_distribution": {"0": 12, "1": 26, "2": 62},
        "dimension_fail_rate": {"D4": {"judged": 80, "failed": 24, "fail_rate": 0.3}},
    },
    "attribution": {
        "attributed": 21,
        "attribution_coverage": 0.9130,
        "root_cause_top": [
            {"code": "cmd_tag_missing", "count": 9, "share": 0.4286},
        ],
        "overturn": {
            "overturn_count": 4,
            "overturn_rate": 0.1905,
            "by_type": {"eval_rule_issue": 3, "judge_false_positive": 1},
            "cases": [{"trace_id": "t-1", "case_id": "c-1"}],
        },
        "human_review": {"count": 2, "cases": [{"trace_id": "t-2", "case_id": "c-2"}]},
        "root_cause_cases": {
            "cmd_tag_missing": {
                "count": 2,
                "cases": [
                    {"trace_id": "t-3", "case_id": "c-3", "suite": "车机-Standard"},
                    {"trace_id": "t-4", "case_id": "c-4", "suite": "车机-Standard"},
                ],
            },
        },
    },
    "performance": {"e2e_ms": {"avg": 3200.5, "p95": 8800.0}},
}


class _FakeLLM:
    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _agent(llm: object | None) -> ClusterReportAgent:
    return ClusterReportAgent(
        options=ClusterReportAgentOptions(
            name="cluster_report_agent", prompt_name=DEFAULT_CLUSTER_PROMPT_NAME, max_clusters=8
        ),
        llm=llm,
        prompt_manager=None,
    )


class TestPromptLoading:
    def test_conf_prompt_file_exists(self) -> None:
        """The prompt file must exist at the agreed path (same pattern as the attribution side)."""
        from pathlib import Path

        import benchmark.reporting.cluster.agent as agent_module

        path = (
            Path(agent_module.__file__)
            .resolve()
            .parents[2]
            .joinpath(
                *CLUSTER_PROMPT_CONF_DIR, f"{DEFAULT_CLUSTER_PROMPT_NAME}{PROMPT_FILE_SUFFIX}"
            )
        )
        assert path.is_file(), f"prompt not found: {path}"
        content = path.read_text(encoding="utf-8")
        # The three hard constraints must be written into the prompt
        assert "禁止重新计数" in content
        assert "禁止改动分组" in content
        assert "禁止自评置信度" in content

    def test_loads_conf_prompt_when_manager_absent(self) -> None:
        prompt = _agent(None)._load_prompt()
        assert "车机评测聚类分析专家" in prompt

    def test_falls_back_when_prompt_name_missing(self) -> None:
        agent = ClusterReportAgent(
            options=ClusterReportAgentOptions(
                name="a", prompt_name="no_such_prompt", max_clusters=8
            ),
            llm=None,
            prompt_manager=None,
        )
        # Fall back to the built-in default prompt, and do not raise
        assert "禁止重新计数" in agent._load_prompt()

    def test_prompt_manager_takes_priority(self) -> None:
        class _Manager:
            def get(self, name: str) -> str:
                return "MANAGED PROMPT"

        agent = ClusterReportAgent(
            options=ClusterReportAgentOptions(
                name="a", prompt_name=DEFAULT_CLUSTER_PROMPT_NAME, max_clusters=8
            ),
            llm=None,
            prompt_manager=_Manager(),
        )
        assert agent._load_prompt() == "MANAGED PROMPT"

    def test_prompt_manager_failure_falls_back_to_file(self) -> None:
        class _BadManager:
            def get(self, name: str) -> str:
                raise RuntimeError("boom")

        agent = ClusterReportAgent(
            options=ClusterReportAgentOptions(
                name="a", prompt_name=DEFAULT_CLUSTER_PROMPT_NAME, max_clusters=8
            ),
            llm=None,
            prompt_manager=_BadManager(),
        )
        assert "车机评测聚类分析专家" in agent._load_prompt()


class TestSummarizeDegradation:
    async def test_no_model_fails_l2_required_report(self) -> None:
        with pytest.raises(ClusterReportL2Error, match="not configured"):
            await _agent(None).summarize(_L1_STATS)

    async def test_llm_failure_retries_with_jitter_then_fails(self, monkeypatch) -> None:
        sleeps: list[float] = []

        async def _sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(agent_module.asyncio, "sleep", _sleep)
        monkeypatch.setattr(agent_module.random, "uniform", lambda _low, _high: 0.25)
        llm = _FakeLLM(RuntimeError("timeout"))

        with pytest.raises(ClusterReportL2Error, match="after 3 attempts"):
            await _agent(llm).summarize(_L1_STATS)

        assert len(llm.calls) == 3
        assert sleeps == [1.25, 2.25]

    async def test_invalid_json_retries_until_valid_summary(self, monkeypatch) -> None:
        class _SequenceLLM:
            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, system_prompt: str, user_prompt: str) -> str:
                del system_prompt, user_prompt
                self.calls += 1
                return "not-json" if self.calls < 3 else '{"clusters": []}'

        sleeps: list[float] = []

        async def _sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(agent_module.asyncio, "sleep", _sleep)
        monkeypatch.setattr(agent_module.random, "uniform", lambda _low, _high: 0.0)
        llm = _SequenceLLM()

        summary = await _agent(llm).summarize(_L1_STATS)

        assert summary["clusters"] == []
        assert llm.calls == 3
        assert sleeps == [1.0, 2.0]

    async def test_invalid_json_fails_after_retry_budget(self, monkeypatch) -> None:
        async def _sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(agent_module.asyncio, "sleep", _sleep)
        with pytest.raises(ClusterReportL2Error, match="invalid_json"):
            await _agent(_FakeLLM("这不是 JSON")).summarize(_L1_STATS)

    async def test_non_object_json_fails_after_retry_budget(self, monkeypatch) -> None:
        async def _sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(agent_module.asyncio, "sleep", _sleep)
        with pytest.raises(ClusterReportL2Error, match="not_an_object"):
            await _agent(_FakeLLM("[1, 2, 3]")).summarize(_L1_STATS)

    async def test_json_fence_stripped(self) -> None:
        fenced = '```json\n{"headline": "ok", "clusters": []}\n```'
        summary = await _agent(_FakeLLM(fenced)).summarize(_L1_STATS)
        assert summary["headline"] == "ok"

    async def test_clusters_key_always_present(self) -> None:
        summary = await _agent(_FakeLLM('{"headline": "no clusters key"}')).summarize(_L1_STATS)
        assert summary["clusters"] == []

    async def test_stability_uses_dedicated_prompt_and_filters_llm_output(self) -> None:
        stats = {
            "report_job_type": "stability",
            "observations": {
                "stability": [
                    {
                        "scene_tag": "navigation.direct",
                        "query": "导航回家",
                        "agent_execution_observation": "nondeterminism",
                        "evidence_strength": "strong",
                        "run_count": 2,
                        "judge_score_distribution": {"0": 1, "1": 0, "2": 1},
                        "span": 2,
                        "representative_runs": [
                            {"trace_id": "run-good"},
                            {"trace_id": "run-bad"},
                        ],
                    }
                ]
            },
        }
        llm = _FakeLLM(
            json.dumps(
                {
                    "headline": "存在编排波动",
                    "clusters": [{"title": "must be removed"}],
                    "aone_suggestions": [{"title": "must be removed"}],
                    "findings": [
                        {
                            "classification": "nondeterminism",
                            "scene_tag": "navigation.direct",
                            "query": "导航回家",
                            "summary": "相同上下文的输出不同",
                            "representative_traces": ["run-good", "run-bad", "invented"],
                            "governance_action": "investigate",
                        },
                        {"classification": "invalid", "summary": "must be removed"},
                    ],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(stats)

        assert "执行稳定性分析专家" in llm.calls[0][0]
        assert "不生成 aone_suggestions" in llm.calls[0][1]
        assert "clusters" not in summary
        assert "aone_suggestions" not in summary
        assert len(summary["findings"]) == 1
        finding = summary["findings"][0]
        assert finding["classification"] == "nondeterminism"
        assert finding["scene_tag"] == "navigation.direct"
        assert finding["query"] == "导航回家"
        assert finding["evidence_strength"] == "strong"
        assert finding["priority"] == "P0"
        assert finding["summary"] == "相同上下文的输出不同"
        assert finding["representative_traces"] == ["run-good", "run-bad"]
        assert finding["governance"]["domain"] == "Agent 执行"
        assert summary["governance_queue"] == [finding]

    def test_stability_governance_queue_covers_all_nonstable_l1_groups(self) -> None:
        classifications = [
            "context_diff",
            "expected_diversity",
            "tool_data_variance",
            "nondeterminism",
            "quality_regression",
            "evaluation_inconsistency",
            "evaluation_comparability_risk",
            "insufficient_evidence",
        ]
        groups = [
            {
                "scene_tag": "navigation.direct",
                "query": classification,
                "agent_execution_observation": classification,
                "evidence_strength": "strong",
                "run_count": 2,
                "judge_score_distribution": {"0": 1, "1": 0, "2": 1},
                "span": 2,
                "representative_runs": [{"trace_id": f"run-{classification}"}],
            }
            for classification in classifications
        ]
        summary = agent_module._normalize_stability_summary(
            {
                "findings": [
                    {
                        "classification": "expected_diversity",
                        "scene_tag": "navigation.direct",
                        "query": classification,
                        "representative_traces": [f"run-{classification}"],
                        "governance_action": "investigate",
                    }
                    for classification in classifications
                ]
            },
            {"observations": {"stability": groups}},
        )

        assert [item["classification"] for item in summary["governance_queue"]] == [
            "evaluation_inconsistency",
            "nondeterminism",
            "quality_regression",
            "tool_data_variance",
            "context_diff",
            "evaluation_comparability_risk",
            "expected_diversity",
            "insufficient_evidence",
        ]
        assert len(summary["findings"]) == 5
        context_item = next(
            item for item in summary["governance_queue"] if item["classification"] == "context_diff"
        )
        assert context_item["governance"]["domain"] == "评测输入与数据"
        assert context_item["governance_action"] == "observe"

    def test_stability_l2_cannot_override_l1_observation_or_strength(self) -> None:
        summary = agent_module._normalize_stability_summary(
            {
                "findings": [
                    {
                        "classification": "quality_regression",
                        "scene_tag": "navigation.direct",
                        "query": "导航回家",
                        "evidence_strength": "weak",
                        "summary": "LLM 解释",
                    }
                ]
            },
            {
                "observations": {
                    "stability": [
                        {
                            "scene_tag": "navigation.direct",
                            "query": "导航回家",
                            "agent_execution_observation": "nondeterminism",
                            "evidence_strength": "strong",
                            "representative_runs": [{"trace_id": "run-a"}],
                        }
                    ]
                }
            },
        )

        assert summary["findings"][0]["classification"] == "nondeterminism"
        assert summary["findings"][0]["evidence_strength"] == "strong"
        assert summary["findings"][0]["priority"] == "P0"


class TestPromptTrimming:
    async def test_case_details_reduced_to_case_id_only(self) -> None:
        """Per-case detail is trimmed to case_id only (volume control + hide trace_id, so the LLM cannot fabricate it),
        but the distribution, Top-N, and case_id (the related_cases reference key) must remain."""
        llm = _FakeLLM('{"clusters": []}')
        await _agent(llm).summarize(_L1_STATS)

        _, user_prompt = llm.calls[0]

        assert "cmd_tag_missing" in user_prompt  # Top-N is kept
        assert "eval_rule_issue" in user_prompt  # Overturned classifications are kept
        assert "overturn_count" in user_prompt
        assert "c-1" in user_prompt  # case_id is kept, so related_cases can reference it
        assert "c-2" in user_prompt
        assert "c-3" in user_prompt  # case_id on root_cause_cases is kept the same way
        assert "t-1" not in user_prompt  # trace_id is not passed through, so the LLM cannot fabricate it directly
        assert "t-2" not in user_prompt
        assert "t-3" not in user_prompt  # trace_id on root_cause_cases is likewise not passed through
        assert "t-4" not in user_prompt

    async def test_max_clusters_in_prompt(self) -> None:
        llm = _FakeLLM('{"clusters": []}')
        agent = ClusterReportAgent(
            options=ClusterReportAgentOptions(
                name="a", prompt_name=DEFAULT_CLUSTER_PROMPT_NAME, max_clusters=3
            ),
            llm=llm,
            prompt_manager=None,
        )
        await agent.summarize(_L1_STATS)
        assert "至多 3 个问题簇" in llm.calls[0][1]


class TestRelatedCaseResolution:
    """The LLM may cite only case_id (see TestPromptTrimming). trace_id is filled back on the program side by
    looking up case_id; anything not found is dropped, so a fabricated identifier never reaches the report."""

    async def test_related_cases_resolved_from_case_id_to_trace_id(self) -> None:
        llm = _FakeLLM(
            json.dumps(
                {
                    "clusters": [
                        {
                            "title": "CMD 缺失簇",
                            "related_cases": ["c-1"],
                        }
                    ],
                    "aone_suggestions": [
                        {
                            "title": "修复 CMD 缺失",
                            "priority": "P1",
                            "owner_group": "prompt-team",
                            "acceptance": "归零",
                            "related_cases": ["c-1", "c-2"],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(_L1_STATS)

        assert summary["clusters"][0]["related_cases"] == ["t-1"]
        assert summary["aone_suggestions"][0]["related_cases"] == ["t-1", "t-2"]

    async def test_candidate_clusters_are_scope_isolated_and_trace_resolved(self) -> None:
        """A candidate cluster may cite only cases in L1 with the same scope/key, and fabricated candidates are not allowed."""
        l1_stats = json.loads(json.dumps(_L1_STATS))
        l1_stats["attribution"]["other_candidate_stats"] = {
            "candidates": [
                {
                    "agent_scope": "vehicle",
                    "candidate_key": "vehicle:tool_data_missing",
                    "count": 2,
                    "cases": [
                        {"case_id": "c-5", "trace_id": "t-5"},
                        {"case_id": "c-6", "trace_id": "t-6"},
                    ],
                }
            ]
        }
        llm = _FakeLLM(
            json.dumps(
                {
                    "candidate_clusters": [
                        {
                            "title": "工具数据缺失",
                            "agent_scope": "vehicle",
                            "candidate_key": "vehicle:tool_data_missing",
                            "affected_count": 2,
                            "related_cases": ["c-5", "not-in-candidate"],
                        },
                        {
                            "title": "跨 scope 编造",
                            "agent_scope": "vui",
                            "candidate_key": "vehicle:tool_data_missing",
                            "related_cases": ["c-5"],
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(l1_stats)

        assert len(summary["candidate_clusters"]) == 1
        candidate = summary["candidate_clusters"][0]
        assert candidate["related_cases"] == ["t-5"]
        assert candidate["affected_count"] == 1

    async def test_candidate_cases_are_removed_from_formal_clusters_and_notification_suggestions(
        self,
    ) -> None:
        """The program side isolates candidate cases, so the LLM cannot dispatch a candidate issue by bypassing the prompt constraint."""
        l1_stats = json.loads(json.dumps(_L1_STATS))
        l1_stats["attribution"]["root_cause_cases"]["unclassified"] = {
            "count": 1,
            "cases": [{"case_id": "c-5", "trace_id": "t-5"}],
        }
        l1_stats["attribution"]["other_candidate_stats"] = {
            "candidates": [
                {
                    "agent_scope": "vehicle",
                    "candidate_key": "vehicle:tool_data_missing",
                    "cases": [{"case_id": "c-5", "trace_id": "t-5"}],
                }
            ]
        }
        llm = _FakeLLM(
            json.dumps(
                {
                    "clusters": [
                        {
                            "title": "正式簇",
                            "root_cause_codes": ["cmd_tag_missing"],
                            "related_cases": ["c-3"],
                        },
                        {
                            "title": "伪正式候选",
                            "root_cause_codes": ["unclassified"],
                            "related_cases": ["c-5"],
                        },
                    ],
                    "candidate_clusters": [
                        {
                            "title": "候选工具问题",
                            "agent_scope": "vehicle",
                            "candidate_key": "vehicle:tool_data_missing",
                            "related_cases": ["c-5"],
                        }
                    ],
                    "aone_suggestions": [
                        {"title": "正式建议", "related_cases": ["c-3"]},
                        {"title": "候选建议", "related_cases": ["c-5"]},
                    ],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(l1_stats)

        assert [item["title"] for item in summary["clusters"]] == ["正式簇"]
        assert summary["clusters"][0]["related_cases"] == ["t-3"]
        assert [item["title"] for item in summary["aone_suggestions"]] == ["正式建议"]
        assert summary["aone_suggestions"][0]["related_cases"] == ["t-3"]
        assert summary["candidate_clusters"][0]["related_cases"] == ["t-5"]

    async def test_unknown_case_id_dropped_not_hallucinated(self) -> None:
        """A fabricated or misspelled case_id has no mapping and is dropped rather than kept as a forged value."""
        llm = _FakeLLM(
            json.dumps(
                {
                    "clusters": [{"title": "x", "related_cases": ["c-1", "made-up-id"]}],
                    "aone_suggestions": [
                        {"title": "x", "related_cases": ["c-1", "made-up-id"]},
                    ],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(_L1_STATS)

        assert summary["clusters"][0]["related_cases"] == ["t-1"]
        assert summary["aone_suggestions"][0]["related_cases"] == ["t-1"]

    async def test_missing_related_cases_key_is_safe(self) -> None:
        llm = _FakeLLM(
            json.dumps({"clusters": [], "aone_suggestions": [{"title": "x"}]}, ensure_ascii=False)
        )
        summary = await _agent(llm).summarize(_L1_STATS)
        assert summary["aone_suggestions"][0]["related_cases"] == []

    async def test_related_cases_auto_filled_from_root_cause_codes_when_llm_omits(self) -> None:
        """When the LLM omits related_cases, the program side fills traces back automatically from root_cause_codes."""
        llm = _FakeLLM(
            json.dumps(
                {
                    "clusters": [
                        {
                            "title": "CMD 缺失簇",
                            "root_cause_codes": ["cmd_tag_missing"],
                            "related_cases": [],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(_L1_STATS)

        assert set(summary["clusters"][0]["related_cases"]) == {"t-3", "t-4"}

    async def test_explicit_related_cases_take_priority_over_root_cause_codes(self) -> None:
        """When the LLM already emitted valid related_cases, the root_cause_codes auto-fill does not run."""
        llm = _FakeLLM(
            json.dumps(
                {
                    "clusters": [
                        {
                            "title": "CMD 缺失簇",
                            "root_cause_codes": ["cmd_tag_missing"],
                            "related_cases": ["c-1"],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(_L1_STATS)

        assert summary["clusters"][0]["related_cases"] == ["t-1"]

    async def test_affected_count_synced_to_related_cases_length(self) -> None:
        """The title count ``affected_count`` must match the actual length of ``related_cases``.

        The LLM may fabricate ``affected_count`` (for example writing 2 when 4 traces actually parsed).
        After the program side fills traces back, ``affected_count`` must be synced to the real count.
        """
        llm = _FakeLLM(
            json.dumps(
                {
                    "clusters": [
                        {
                            "title": "CMD 缺失簇",
                            "root_cause_codes": ["cmd_tag_missing"],
                            "affected_count": 999,  # a value fabricated by the LLM
                            "related_cases": [],  # triggers the fallback auto-fill
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(_L1_STATS)

        related = summary["clusters"][0]["related_cases"]
        assert summary["clusters"][0]["affected_count"] == len(related)
        assert len(related) > 0  # the fallback did fill the trace back in

    async def test_affected_count_synced_when_related_cases_explicit(self) -> None:
        """When the LLM emits valid related_cases, affected_count is likewise synced to the actual length."""
        llm = _FakeLLM(
            json.dumps(
                {
                    "clusters": [
                        {
                            "title": "CMD 缺失簇",
                            "affected_count": 5,  # does not match the actual related_cases
                            "related_cases": ["c-1", "c-2"],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(_L1_STATS)

        assert summary["clusters"][0]["affected_count"] == 2  # c-1→t-1, c-2→t-2

    async def test_fallback_truncated_by_affected_count(self) -> None:
        """The fallback path truncates to the LLM's affected_count, so a root cause does not over-include every trace.

        L1 has 2 cases for cmd_tag_missing (c-3/t-3, c-4/t-4), but the LLM thinks the cluster has only 1.
        The fallback must return 1 trace, not all 2.
        """
        llm = _FakeLLM(
            json.dumps(
                {
                    "clusters": [
                        {
                            "title": "CMD 缺失簇",
                            "root_cause_codes": ["cmd_tag_missing"],
                            "affected_count": 1,  # the LLM thinks there is only 1
                            "related_cases": [],  # triggers the fallback
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(_L1_STATS)

        related = summary["clusters"][0]["related_cases"]
        assert len(related) == 1  # truncated to affected_count=1, not the full 2
        assert summary["clusters"][0]["affected_count"] == 1


class TestNumberGuard:
    def test_exact_l1_numbers_pass(self) -> None:
        summary = {
            "headline": "通过率 0.62，D4 失败率 0.3",
            "clusters": [{"summary": "cmd_tag_missing 命中 9 次"}],
        }
        assert guard_llm_numbers(summary, _L1_STATS) == []

    def test_percent_form_of_ratio_passes(self) -> None:
        """L1 stores 0.62 and the LLM writes 62% — two writings of the same fact, and this must not be a false reject."""
        summary = {"headline": "通过率 62%"}
        assert guard_llm_numbers(summary, _L1_STATS) == []

    def test_thousands_separator_passes(self) -> None:
        """Thousands-separator form (3,200.5) is equivalent to the L1 value 3200.5."""
        summary = {"headline": "端到端平均耗时 3,200.5 ms"}
        assert guard_llm_numbers(summary, _L1_STATS) == []

    def test_rounding_within_tolerance_passes(self) -> None:
        """Rounding while the LLM restates (0.9130 → 91.3%) is inside tolerance."""
        summary = {"headline": "归因覆盖率 91.3%"}
        assert guard_llm_numbers(summary, _L1_STATS) == []

    def test_fabricated_number_flagged(self) -> None:
        summary = {"headline": "通过率 88.7%，共 4321 个 case"}
        warnings = guard_llm_numbers(summary, _L1_STATS)
        assert "4321" in warnings

    def test_count_inflated_100x_flagged(self) -> None:
        """Regression anchor (found in review): a count metric scaled by 100 must be caught.

        An early implementation registered ``value*100`` for every number (meant to equate ratios and percents),
        so ``attributed=21`` made 2100 count as "present in L1" and the guard did nothing.
        After the fix, percent expansion applies only to ratios in the 0–1 range.
        """
        # In _L1_STATS, attributed=21 / overturn_count=4 / total=100
        assert "2100" in guard_llm_numbers({"headline": "归因 2100 条"}, _L1_STATS)
        assert "400" in guard_llm_numbers({"headline": "推翻 400 条"}, _L1_STATS)

    def test_legitimate_ratio_percent_still_passes_after_fix(self) -> None:
        """The fix must not reject the percent spelling of a ratio (0.62 ↔ 62%)."""
        assert guard_llm_numbers({"headline": "通过率 62%，覆盖率 91.3%"}, _L1_STATS) == []

    def test_small_ordinals_not_flagged(self) -> None:
        """Ordinals and cardinals such as Top 3 / D0-D6 are not checked (avoid false reports)."""
        summary = {
            "headline": "Top 3 问题簇",
            "clusters": [{"summary": "D0-D6 共 7 个维度，优先级 P1"}],
        }
        assert guard_llm_numbers(summary, _L1_STATS) == []

    def test_number_warnings_field_not_self_checked(self) -> None:
        """An existing number_warnings field is not checked again (prevent a self-reference loop)."""
        summary = {"headline": "通过率 0.62", "number_warnings": ["99999"]}
        assert guard_llm_numbers(summary, _L1_STATS) == []

    def test_nested_structures_scanned(self) -> None:
        summary = {
            "aone_suggestions": [
                {"title": "修复", "acceptance": "把失败率从 0.3 降到 0.05"},
            ],
        }
        warnings = guard_llm_numbers(summary, _L1_STATS)
        # 0.3 exists in L1; 0.05 does not → warn only on the latter
        assert warnings == ["0.05"]

    def test_empty_l1_skips_guard(self) -> None:
        """When L1 is empty (no baseline), skip the check, so every number is not judged fabricated."""
        assert guard_llm_numbers({"headline": "123456"}, {}) == []

    async def test_warnings_attached_to_summary(self) -> None:
        """On a hit, attach number_warnings but do not delete the wording (soft check)."""
        llm = _FakeLLM(
            json.dumps(
                {
                    "headline": "通过率 99.9%",
                    "clusters": [{"title": "簇", "affected_count": 777}],
                },
                ensure_ascii=False,
            )
        )

        summary = await _agent(llm).summarize(_L1_STATS)

        assert "number_warnings" in summary
        assert summary["headline"] == "通过率 99.9%"  # original text is kept
        assert summary["clusters"][0]["title"] == "簇"  # structure is kept

    async def test_clean_summary_has_no_warnings_key(self) -> None:
        llm = _FakeLLM(json.dumps({"headline": "通过率 0.62", "clusters": []}, ensure_ascii=False))
        summary = await _agent(llm).summarize(_L1_STATS)
        assert "number_warnings" not in summary


class TestAgentCallInterface:
    async def test_call_roundtrip(self) -> None:
        """call() follows the AgentRequest/AgentResponse contract (the AgentManager call path)."""
        from benchmark.libs.agenthost.agents.request import AgentRequest, ContentPart

        llm = _FakeLLM('{"headline": "ok", "clusters": []}')
        request = AgentRequest(
            traceid="t-req",
            session_id="s-1",
            content_parts=[ContentPart(type="text", data=json.dumps(_L1_STATS))],
        )

        response = await _agent(llm).call(request)

        payload = json.loads(response.content_parts[0].data)
        assert payload["headline"] == "ok"
        assert response.traceid == "t-req"

    async def test_run_delegates_to_call(self) -> None:
        """``run`` is an abstract method of BaseAgent and must be usable (the AgentManager entry)."""
        from benchmark.libs.agenthost.agents.request import AgentRequest, ContentPart

        llm = _FakeLLM('{"headline": "via run", "clusters": []}')
        request = AgentRequest(
            traceid="t-run",
            session_id="s-1",
            content_parts=[ContentPart(type="text", data=json.dumps(_L1_STATS))],
        )

        response = await _agent(llm).run(request)

        assert json.loads(response.content_parts[0].data)["headline"] == "via run"

    async def test_stats_from_metadata_channel(self) -> None:
        """The orchestration layer may pass metadata.l1_stats directly (no JSON serialization round trip)."""
        from benchmark.libs.agenthost.agents.request import AgentRequest

        llm = _FakeLLM('{"clusters": []}')
        request = AgentRequest(traceid="t-meta", session_id="s-1", metadata={"l1_stats": _L1_STATS})

        await _agent(llm).call(request)

        assert "cmd_tag_missing" in llm.calls[0][1]

    async def test_call_with_unparsable_content_still_responds(self) -> None:
        from benchmark.libs.agenthost.agents.request import AgentRequest, ContentPart

        llm = _FakeLLM('{"clusters": []}')
        request = AgentRequest(
            traceid="t-bad",
            session_id="s-1",
            content_parts=[ContentPart(type="text", data="not json")],
        )

        response = await _agent(llm).call(request)

        assert response.content_parts[0].data


class TestConfigSchema:
    def test_defaults(self) -> None:
        cfg = ClusterReportConfigSchema()
        assert cfg.prompt_name == DEFAULT_CLUSTER_PROMPT_NAME
        assert cfg.max_clusters == 8
        assert cfg.default_model is None

    def test_registered_as_agent_component(self) -> None:
        """Registered alongside attribution_vehicle_agent, and name is passed through correctly."""
        assert _agent(None).name == "cluster_report_agent"


class TestLazyChatModel:
    """The real runtime call path: streaming accumulation avoids idle gateway disconnects; transient errors are retried uniformly by the outer Agent."""

    @staticmethod
    def _streaming_model(texts: list[str], fail_times: int = 0) -> object:
        class _Chunk:
            def __init__(self, content: str) -> None:
                self.content = content

        class _Model:
            def __init__(self) -> None:
                self.calls = 0

            async def astream(self, messages: object) -> object:
                self.calls += 1
                if fail_times and self.calls <= fail_times:
                    raise ProviderError("Connection error.")
                for text in texts:
                    yield _Chunk(text)

        return _Model()

    async def test_stream_chunks_concatenated(self) -> None:
        model = self._streaming_model(['{"a":', " 1}"])
        result = await _LazyChatModel(lambda: model).complete("sys", "user")
        assert result == '{"a": 1}'
        assert model.calls == 1

    async def test_transient_failure_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(agent_module, "_L2_RETRY_DELAYS_S", (0, 0))
        monkeypatch.setattr(agent_module.random, "uniform", lambda _low, _high: 0)
        model = self._streaming_model(['{"ok": true}'], fail_times=1)
        result = await _agent(_LazyChatModel(lambda: model)).summarize({})
        assert result["ok"] is True
        assert model.calls == 2

    async def test_exhausted_retries_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(agent_module, "_L2_RETRY_DELAYS_S", (0, 0))
        monkeypatch.setattr(agent_module.random, "uniform", lambda _low, _high: 0)
        model = self._streaming_model([], fail_times=99)
        with pytest.raises(ClusterReportL2Error, match="after 3 attempts"):
            await _agent(_LazyChatModel(lambda: model)).summarize({})
        assert model.calls == agent_module._L2_MAX_ATTEMPTS

    async def test_non_streaming_model_falls_back_to_call_model(self) -> None:
        """A client with no astream (an older shape other than a callable) falls back to the non-streaming path."""

        class _Model:
            async def ainvoke(self, messages: object) -> object:
                return type("R", (), {"content": "fallback"})()

        result = await _LazyChatModel(_Model).complete("sys", "user")
        assert result.content == "fallback"


class TestRegistryWiring:
    """Regression anchor (found in review): without an eager import of the registry, AgentManager can never
    discover cluster_report_agent — @agent_component takes effect only when the module is imported.
    """

    def test_registry_module_imports_agent(self) -> None:
        from benchmark.reporting.cluster import registry

        assert registry.ClusterReportAgent.__name__ == "ClusterReportAgent"

    def test_main_eager_imports_registry(self) -> None:
        """main.py must eager-import the registry (same mechanism as evaluators/triage)."""
        from pathlib import Path

        import app

        main_src = (Path(app.__file__).parent / "main.py").read_text(encoding="utf-8")
        assert "cluster_report" in main_src, (
            "app/main.py 缺 cluster_report registry 的 eager import，"
            "AgentManager 将发现不到 cluster_report_agent"
        )
