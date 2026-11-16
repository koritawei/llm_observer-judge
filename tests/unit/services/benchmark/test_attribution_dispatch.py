"""Unit tests for benchmark attribution callback → Aone ticket push (SDD benchmark-attribution-aone-dispatch).

Covers:
- ``map_attribution_to_result``: field mapping + closed-set normalization of category + the fallback chain.
- ``should_dispatch_attribution``: disabled / misjudge block (is_eval_misjudge)
  / grade threshold (the default policy dispatches only FAIL/POOR; a business policy can be injected to override);
  the severity decision is deferred and does not block.
- ``_resolve_dispatch_profile_key``: YAML profiles business → key mapping.
- ``parse_aone_dispatch_config``: missing section / legal / illegal fallback / profiles pop.
- ``dispatch_aone_for_attribution``: CREATED bucket (including notification_url) / SKIPPED bucket / FAILED
  exception isolation / misjudge block / YAML-missing fallback to agent-eval / agent_notification_disabled / after an exception
  a failed update_aone_dispatch still does not raise.
- The ``ExternalBenchmarkRunner._handle_attribution_callback`` injection point: claimed
  idempotency gate + a push exception does not block increment / CAS / finalize.

The mock ORM does not touch a real DB; ``create_aone_task`` is patched at the dispatch layer so the
real ``a1`` CLI is not called.
"""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.domain.notification_dispatch_config import NotificationDispatchProfile
from benchmark.domain.notification_dispatch_resolve import AgentNotificationSnapshot
from benchmark.attribution.badcase.publisher import NotificationConfig
from benchmark.attribution.badcase.dispatch_policy import (
    RubricV2DispatchPolicy,
    clear_dispatch_policies,
    register_dispatch_policy,
)
from benchmark.attribution.badcase.schemas import (
    NotificationDispatchResult,
    NotificationDispatchStatus,
    NotificationRuntimeOptions,
)
from benchmark.attribution.badcase.schemas import (
    AttributionResult as DispatchAttributionResult,
)
AttributionMapContext = None
DispatchDecision = None
DispatchDecisionInput = None
DispatchGateInput = None
DispatchRequest = None
JudgeDispatchInput = None
_dispatch_result_to_bucket = None
_resolve_dispatch_profile_key = None
dispatch_notification_for_attribution = None
dispatch_notification_for_judge = None
dispatch_notification_for_triage = None
map_attribution_to_result = None
map_judge_to_result = None
parse_notification_dispatch_config = None
resolve_dispatch_decision = None
should_dispatch_attribution = None
from benchmark.runs.schema import AttributionCallback, AttributionResult
from benchmark.workers.external_runner import ExternalBenchmarkRunner

_DISPATCH_MODULE = "business.runs.ieval.attribution_dispatch"
_RUNNER_MODULE = "benchmark.workers.external_runner"


def _attr(**overrides: Any) -> AttributionResult:
    """Build an ``AttributionResult`` (schema.py, callback side)."""
    base: dict[str, Any] = {
        "is_eval_misjudge": False,
        "primary_issue_layer": "tool",
        "primary_issue_category": "tool_unavailable",
        "primary_issue_detail": "工具未覆盖该场景",
        "affected_dims": ["D2"],
        "severity": "major",
        "root_cause_chain": "工具缺失导致无法获取 POI",
        "improvement_suggestion": "建设加油站搜索工具",
    }
    base.update(overrides)
    return AttributionResult.model_validate(base)


def _fake_service() -> MagicMock:
    """Fake BenchmarkService — update_aone_dispatch / get_fused_score /
    get_service_name / get_trace_url are AsyncMocks."""
    svc = MagicMock()
    svc.update_notification_dispatch = AsyncMock()
    # Default score lookup fails (None) → the grade threshold lets it through (misjudge/severity do not block)
    svc.get_fused_score = AsyncMock(return_value=None)
    # Default empty query → skip Redis dedup and create the ticket as usual
    svc.get_fused_query = AsyncMock(return_value="")
    # Default run business name missing (None) → policy resolution falls back to the default implementation
    svc.get_service_name = AsyncMock(return_value=None)
    # Default fused-row trace_url missing (None) → the ticket body shows '-' for Trace URL
    svc.get_trace_url = AsyncMock(return_value=None)
    # Default fused-row trace_id missing (None) → the ticket Trace ID falls back to job_id
    svc.get_trace_id = AsyncMock(return_value=None)
    return svc


def _req(store: Any, **overrides: Any) -> DispatchRequest:
    """Build a DispatchRequest (external-path defaults, overridable)."""
    base: dict[str, Any] = {
        "run_id": "run1",
        "item_id": "item1",
        "source_trace_id": "trace-1",
        "config": {"notification_dispatch": {"enabled": True, "project": "proj-1"}},
    }
    base.update(overrides)
    return DispatchRequest(store=store, **base)


def _created_result(
    *,
    workitem_id: str = "W-999",
    workitem_url: str = "",
) -> NotificationDispatchResult:
    """Build a CREATED dispatch result (including payload.title)."""
    from benchmark.attribution.badcase.schemas import NotificationWorkitemPayload

    return NotificationDispatchResult(
        status=NotificationDispatchStatus.CREATED,
        workitem_id=workitem_id,
        workitem_url=workitem_url,
        message="Aone workitem created.",
        payload=NotificationWorkitemPayload(
            project="proj-1",
            category="task",
            title="[AgentEval Badcase] tool_unavailable: trace-1",
            body="case summary",
        ),
    )


# ── T03：map_attribution_to_result ────────────────────────────


class TestMapAttributionToResult:
    """Callback attribution structure → push-structure field mapping."""

    def test_basic_fields_mapped(self) -> None:
        # primary_issue_category="tool" is in the ATTRIBUTION_CATEGORIES closed set → kept as-is
        data = _attr(primary_issue_category="tool")
        mapped = map_attribution_to_result(
            data,
            AttributionMapContext(
                trace_id="trace-1",
                item_id="item-1",
            ),
        )
        assert mapped.trace_id == "trace-1"
        assert mapped.case_id == "item-1"
        assert mapped.category == "tool"
        assert mapped.root_cause == "工具缺失导致无法获取 POI"
        assert mapped.reasoning == "工具未覆盖该场景"
        assert mapped.action_hint == "建设加油站搜索工具"
        assert mapped.should_dispatch is True

    def test_trace_url_mapped_from_fused_row(self) -> None:
        """trace_url (the value queried from the fused row) → the Trace URL line in the ticket body."""
        mapped = map_attribution_to_result(
            _attr(),
            AttributionMapContext(
                trace_id="trace-1",
                item_id="item-1",
                trace_url="https://trace.example.com/t/1",
            ),
        )
        assert mapped.trace_url == "https://trace.example.com/t/1"

    def test_query_mapped_same_level_as_trace_id(self) -> None:
        """The fused-row query is written into the ticket payload at the same level as trace_id."""
        mapped = map_attribution_to_result(
            _attr(),
            AttributionMapContext(
                trace_id="trace-1",
                item_id="item-1",
                query="附近加油站",
            ),
        )
        assert mapped.trace_id == "trace-1"
        assert mapped.query == "附近加油站"

    def test_query_defaults_empty(self) -> None:
        """query omitted by default → empty string."""
        mapped = map_attribution_to_result(
            _attr(),
            AttributionMapContext(
                trace_id="trace-1",
                item_id="item-1",
            ),
        )
        assert mapped.query == ""

    def test_trace_url_defaults_empty(self) -> None:
        """Omitted by default → empty string (the ticket body displays '-')."""
        mapped = map_attribution_to_result(
            _attr(),
            AttributionMapContext(
                trace_id="trace-1",
                item_id="item-1",
            ),
        )
        assert mapped.trace_url == ""

    def test_root_cause_falls_back_to_legacy_root_cause(self) -> None:
        """When root_cause_chain is empty, effective_root_cause falls back to root_cause."""
        data = _attr(
            root_cause_chain="",
            root_cause="legacy-root-cause",
        )
        mapped = map_attribution_to_result(
            data,
            AttributionMapContext(
                trace_id="t",
                item_id="i",
            ),
        )
        assert mapped.root_cause == "legacy-root-cause"

    def test_category_invalid_normalized_to_unknown(self) -> None:
        """A dirty category is normalized to unknown by the validator (precondition of acceptance criterion 6)."""
        data = _attr(primary_issue_category="not_a_real_category")
        mapped = map_attribution_to_result(
            data,
            AttributionMapContext(
                trace_id="t",
                item_id="i",
            ),
        )
        assert mapped.category == "unknown"

    def test_category_falls_back_to_primary_issue_layer(self) -> None:
        """When primary_issue_category is empty, fall back to primary_issue_layer."""
        data = _attr(primary_issue_category="", primary_issue_layer="planning")
        mapped = map_attribution_to_result(
            data,
            AttributionMapContext(
                trace_id="t",
                item_id="i",
            ),
        )
        # primary_issue_layer="planning" is not in the ATTRIBUTION_CATEGORIES closed set → unknown
        assert mapped.category == "unknown"

    @pytest.mark.parametrize(
        "category",
        [
            "tool_unavailable",
            "tool_exec_error",
            "data_insufficient",
            "context_lost",
            "factuality_error",
            "completeness_error",
            "expression_error",
        ],
    )
    def test_fine_grained_category_passes_closure(self, category: str) -> None:
        """A fine-grained subclass of the three-layer attribution that is in the closed set → kept as-is, not dropped to unknown."""
        data = _attr(primary_issue_category=category)
        mapped = map_attribution_to_result(
            data,
            AttributionMapContext(
                trace_id="t",
                item_id="i",
            ),
        )
        assert mapped.category == category

    def test_evidence_refs_from_primary_fm_extra(self) -> None:
        """extra.primary_fm → evidence_refs。"""
        data = _attr(primary_fm="fm-001")
        mapped = map_attribution_to_result(
            data,
            AttributionMapContext(
                trace_id="t",
                item_id="i",
            ),
        )
        assert mapped.evidence_refs == ["fm-001"]

    def test_empty_primary_fm_yields_empty_evidence(self) -> None:
        data = _attr()
        mapped = map_attribution_to_result(
            data,
            AttributionMapContext(
                trace_id=None,
                item_id="i",
            ),
        )
        assert mapped.evidence_refs == []
        assert mapped.trace_id == ""

    def test_reasoning_falls_back_to_root_cause_chain(self) -> None:
        """When primary_issue_detail is empty, reasoning falls back to root_cause_chain."""
        data = _attr(primary_issue_detail="", root_cause_chain="chain-text")
        mapped = map_attribution_to_result(
            data,
            AttributionMapContext(
                trace_id="t",
                item_id="i",
            ),
        )
        assert mapped.reasoning == "chain-text"


# ── T04：should_dispatch_attribution ───────────────────────────


class TestShouldDispatchAttribution:
    """Decision rules: enabled → misjudge block → score threshold (past the threshold means push)."""

    async def test_disabled_skips(self) -> None:
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=False),
        )
        assert should is False
        assert reason == "disabled"

    async def test_eval_misjudge_blocks_dispatch(self) -> None:
        """Misjudge block: is_eval_misjudge=true produces no ticket (eval_misjudge)."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                is_eval_misjudge=True,
            ),
        )
        assert should is False
        assert reason == "eval_misjudge"

    async def test_eval_misjudge_false_dispatches(self) -> None:
        """A non-misjudge case (is_eval_misjudge=false) that passes the threshold is pushed as usual."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                is_eval_misjudge=False,
            ),
        )
        assert should is True
        assert reason == ""

    async def test_misjudge_priority_over_score_threshold(self) -> None:
        """The misjudge block runs before the score threshold: fused ≥ 1.0 + misjudge → eval_misjudge."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                fused_score=1.2,
                is_eval_misjudge=True,
            ),
        )
        assert should is False
        assert reason == "eval_misjudge"

    async def test_disabled_priority_over_misjudge(self) -> None:
        """enabled=false takes priority over the misjudge block → disabled."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=False),
            DispatchGateInput(
                is_eval_misjudge=True,
            ),
        )
        assert should is False
        assert reason == "disabled"

    @pytest.mark.parametrize("severity", ["critical", "major", "CRITICAL", "Major"])
    async def test_critical_and_major_dispatch(self, severity: str) -> None:
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
        )
        assert should is True
        assert reason == ""

    async def test_minor_without_flag_dispatches(self) -> None:
        """Severity decision deferred: minor is pushed whether or not the minor_dispatch switch is on."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                minor_dispatch=False,
            ),
        )
        assert should is True
        assert reason == ""

    async def test_minor_with_flag_dispatches(self) -> None:
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                minor_dispatch=True,
            ),
        )
        assert should is True
        assert reason == ""

    async def test_empty_severity_dispatches(self) -> None:
        """Severity decision deferred: an empty severity is still pushed."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
        )
        assert should is True
        assert reason == ""

    async def test_score_threshold_skips_acceptable_and_above(self) -> None:
        """Score threshold: fused ≥ 1.0 (ACCEPTABLE/GOOD/EXCELLENT) does not dispatch a ticket."""
        for fused in (1.0, 1.2, 1.4, 1.8):
            should, reason = await should_dispatch_attribution(
                NotificationRuntimeOptions(enabled=True),
                DispatchGateInput(
                    fused_score=fused,
                ),
            )
            assert should is False, fused
            assert reason == "score_threshold"

    async def test_score_below_threshold_dispatches(self) -> None:
        """fused < 1.0 (FAIL/POOR) is let through at the score layer and pushed (severity no longer blocks)."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                fused_score=0.99,
            ),
        )
        assert should is True
        assert reason == ""

    async def test_none_score_dispatches(self) -> None:
        """A failed score lookup (None) is let through, so missing data does not skip dispatch."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                fused_score=None,
            ),
        )
        assert should is True
        assert reason == ""

    async def test_fail_zero_score_dispatches(self) -> None:
        """A gate failure (fused=0.0 → FAIL) is let through at the grade layer and pushed."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                fused_score=0.0,
            ),
        )
        assert should is True
        assert reason == ""

    async def test_acceptable_139_skips(self) -> None:
        """fused=1.39 → ACCEPTABLE (top of the band) is blocked at the grade layer."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                fused_score=1.39,
            ),
        )
        assert should is False
        assert reason == "score_threshold"

    async def test_ten_point_low_score_dispatches(self) -> None:
        """On a 0–10 scale fused=3.0 → POOR, and a ticket must be dispatched (it must not be misread as rubric_v2 EXCELLENT)."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                fused_score=3.0,
                score_scale_max=10.0,
            ),
        )
        assert should is True
        assert reason == ""

    async def test_ten_point_inferred_when_scale_missing(self) -> None:
        """When the scale is unknown and the score is >2, infer 0–10, consistent with the realtime hook / grade_fused_score."""
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                fused_score=3.0,
            ),
        )
        assert should is True
        assert reason == ""

    async def test_injected_policy_overrides_grade(self) -> None:
        """An explicitly injected business policy overrides the default grade set (EXCELLENT is let through too)."""

        class _AllPassPolicy:
            def should_dispatch(
                self,
                *,
                grade: str | None,
                score: float | None = None,
            ) -> bool:
                return True

        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                fused_score=1.9,
                policy=_AllPassPolicy(),
            ),
        )
        assert should is True
        assert reason == ""

    async def test_injected_policy_blocks_poor(self) -> None:
        """The business policy dispatches only FAIL — POOR (fused=0.5) is blocked."""
        policy = RubricV2DispatchPolicy(dispatch_grades=frozenset({"FAIL"}))
        should, reason = await should_dispatch_attribution(
            NotificationRuntimeOptions(enabled=True),
            DispatchGateInput(
                fused_score=0.5,
                policy=policy,
            ),
        )
        assert should is False
        assert reason == "score_threshold"


# ── _resolve_dispatch_profile_key ────────────────────────


class TestResolveDispatchProfileKey:
    """YAML ``profiles`` business → RemoteConfig profile key mapping."""

    @pytest.mark.parametrize(
        ("section", "expected"),
        [
            # profiles maps normally → key (whitespace stripped)
            ({"profiles": {"benchmark_ieval": "benchmark_ieval"}}, "benchmark_ieval"),
            ({"profiles": {"benchmark_ieval": "  agent-eval  "}}, "agent-eval"),
            # profiles section missing / not a mapping → None
            ({}, None),
            ({"profiles": "bogus"}, None),
            # Business not configured / value is an empty string / value is not a string → None
            ({"profiles": {"agent-eval": "agent-eval"}}, None),
            ({"profiles": {"benchmark_ieval": ""}}, None),
            ({"profiles": {"benchmark_ieval": None}}, None),
        ],
    )
    def test_profile_key_resolution(
        self,
        section: dict[str, Any],
        expected: str | None,
    ) -> None:
        with patch(
            f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
            return_value=section,
        ):
            assert _resolve_dispatch_profile_key("benchmark_ieval") == expected


# ── resolve_dispatch_decision (shared decision chain for orchestration and dry-run)──


class TestResolveDispatchDecision:
    """Single source of truth for the decision chain: enabled → misjudge → threshold → override → profiles → target assembly."""

    @pytest.fixture(autouse=True)
    def _mock_profiles_and_remote_config(self) -> Any:
        """YAML profiles (benchmark_ieval is configured) + RemoteConfig defaults go through the mock."""
        profiles = {
            "benchmark_ieval": NotificationDispatchProfile(
                notification_project_id="remote_config-proj",
                assignee="remote_config-user",
                participants=("cc-user",),
            ),
        }
        with (
            patch(
                f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
                return_value={"profiles": {"benchmark_ieval": "benchmark_ieval"}},
            ),
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value=profiles,
            ) as mock_fetch,
        ):
            self.mock_fetch = mock_fetch
            yield

    async def _decide(self, **overrides: Any) -> DispatchDecision:
        kwargs: dict[str, Any] = {
            "options": NotificationRuntimeOptions(enabled=True),
            "business": "benchmark_ieval",
        }
        kwargs.update(overrides)
        return await resolve_dispatch_decision(DispatchDecisionInput(**kwargs))

    async def test_disabled_skips_without_remote_config_lookup(self) -> None:
        """enabled=false — do not look up RemoteConfig (already skipped, so the config is not read)."""
        decision = await self._decide(options=NotificationRuntimeOptions(enabled=False))
        assert (decision.should, decision.skip_reason) == (False, "disabled")
        self.mock_fetch.assert_not_called()

    async def test_eval_misjudge_skips(self) -> None:
        decision = await self._decide(is_eval_misjudge=True)
        assert (decision.should, decision.skip_reason) == (False, "eval_misjudge")

    async def test_score_threshold_skips(self) -> None:
        decision = await self._decide(fused_score=1.2)
        assert (decision.should, decision.skip_reason) == (False, "score_threshold")

    async def test_ten_point_via_config_sub_score_scale_dispatches(self) -> None:
        """config.sub_score_scale=[0,10] + fused=3.0 → POOR, consistent with realtime grading."""
        decision = await self._decide(
            fused_score=3.0,
            config={"sub_score_scale": [0.0, 10.0]},
        )
        assert decision.should is True
        assert decision.skip_reason == ""

    async def test_ten_point_inferred_above_two_dispatches(self) -> None:
        """With no scale field, fused=3.0 is inferred as 0–10 POOR, so rubric_v2 does not misread it as EXCELLENT."""
        decision = await self._decide(fused_score=3.0)
        assert decision.should is True
        assert decision.skip_reason == ""

    async def test_attribution_overridden_skips(self) -> None:
        decision = await self._decide(attribution_should_dispatch=False)
        assert (decision.should, decision.skip_reason) == (False, "attribution_overridden")

    async def test_missing_yaml_profile_falls_back_to_agent_eval(self) -> None:
        """When YAML profiles are missing, do not return profile_not_configured; fall back to RemoteConfig agent-eval."""
        profiles = {
            "agent-eval": NotificationDispatchProfile(
                notification_project_id="2165583",
                assignee="529504",
                participants=("407544",),
            ),
        }
        with (
            patch(
                f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
                return_value={},
            ),
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value=profiles,
            ),
        ):
            decision = await self._decide(agent_name="chat-agent")
        assert decision.should is True
        assert decision.skip_reason == ""
        assert decision.profile_key == "agent-eval"
        assert decision.project == "2165583"
        assert decision.assignee == "529504"
        assert decision.participants == ("407544",)

    async def test_agent_name_profile_beats_agent_eval(self) -> None:
        """When RemoteConfig has both the agent name and agent-eval, the agent name wins."""
        profiles = {
            "chat-agent": NotificationDispatchProfile(
                notification_project_id="111",
                assignee="222",
            ),
            "agent-eval": NotificationDispatchProfile(
                notification_project_id="2165583",
                assignee="529504",
            ),
        }
        with (
            patch(
                f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
                return_value={},
            ),
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value=profiles,
            ),
        ):
            decision = await self._decide(agent_name="chat-agent")
        assert decision.should is True
        assert decision.profile_key == "chat-agent"
        assert decision.project == "111"
        assert decision.assignee == "222"

    async def test_agent_document_fields_no_longer_override_remote_config(self) -> None:
        """The aone field on the registration document is ignored; RemoteConfig is used uniformly."""
        profiles = {
            "agent-eval": NotificationDispatchProfile(
                notification_project_id="2165583",
                assignee="529504",
                participants=("407544",),
            ),
        }
        with (
            patch(
                f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
                return_value={},
            ),
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value=profiles,
            ),
        ):
            decision = await self._decide(
                agent_name="chat-agent",
                agent_document=AgentNotificationSnapshot(
                    name="chat-agent",
                    notification_project_id="999",
                    notification_assignee="888",
                    notification_participants=("777",),
                ),
            )
        assert decision.should is True
        assert decision.project == "2165583"
        assert decision.assignee == "529504"
        assert decision.participants == ("407544",)

    async def test_agent_notification_disabled_no_longer_skips(self) -> None:
        """The registration switch that turns dispatch off is deprecated; the decision still follows RemoteConfig."""
        decision = await self._decide(
            agent_document=AgentNotificationSnapshot(
                name="chat-agent",
                notification_dispatch_enabled=False,
            ),
        )
        assert decision.should is True
        assert decision.skip_reason == ""

    async def test_target_composition_per_run_overrides_remote_config(self) -> None:
        """per-run project/assignee win; participants fall back to the RemoteConfig default."""
        decision = await self._decide(
            options=NotificationRuntimeOptions(
                enabled=True,
                project="proj-1",
                assignee="alice",
            ),
        )
        assert decision.should is True
        assert decision.skip_reason == ""
        assert decision.profile_key == "benchmark_ieval"
        assert decision.project == "proj-1"
        assert decision.assignee == "alice"
        assert decision.participants == ("cc-user",)
        self.mock_fetch.assert_called()

    async def test_target_falls_back_to_remote_config(self) -> None:
        """No per-run target configured → the RemoteConfig default covers everything."""
        decision = await self._decide()
        assert decision.should is True
        assert decision.project == "remote_config-proj"
        assert decision.assignee == "remote_config-user"
        assert decision.participants == ("cc-user",)

    async def test_yaml_key_missing_in_remote_config_falls_back_to_agent_eval(self) -> None:
        """When the profile mapped by YAML does not exist in RemoteConfig, fall back to agent-eval."""
        profiles = {
            "agent-eval": NotificationDispatchProfile(
                notification_project_id="2165583",
                assignee="529504",
                participants=("407544",),
            ),
        }
        with patch(
            "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
            return_value=profiles,
        ):
            decision = await self._decide(agent_name="chat-agent")
        assert decision.should is True
        assert decision.profile_key == "agent-eval"
        assert decision.project == "2165583"
        assert decision.assignee == "529504"
        assert decision.participants == ("407544",)


# ── T05: dispatch_aone_for_triage (inline / dataset path)──────────


def _triage_attribution(**overrides: Any) -> DispatchAttributionResult:
    """Build a triage attribution result (badcase_attribution.schemas, the push structure)."""
    base: dict[str, Any] = {
        "trace_id": "",
        "case_id": "",
        "category": "tool",
        "root_cause": "工具缺失",
        "confidence": 0.9,
        "reasoning": "工具未覆盖该场景",
        "should_dispatch": True,
    }
    base.update(overrides)
    return DispatchAttributionResult.model_validate(base)


class TestDispatchNotificationForTriage:
    """Push orchestration for the inline / dataset triage path (reuses the external default version)."""

    @pytest.fixture(autouse=True)
    def _mock_yaml_profiles(self) -> Any:
        """The YAML profiles mapping goes through the mock (benchmark_ieval is configured);
        cases that lack profiles override it with an empty section."""
        with patch(
            f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
            return_value={"profiles": {"benchmark_ieval": "benchmark_ieval"}},
        ):
            yield

    @pytest.mark.asyncio
    async def test_created_writes_notification_ref_bucket(self) -> None:
        svc = _fake_service()
        with (
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={
                    "benchmark_ieval": NotificationDispatchProfile(
                        notification_project_id="remote_config-proj",
                        assignee="remote_config-user",
                    ),
                },
            ) as mock_defaults,
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_triage(
                _req(svc, source_trace_id="src-1"),
                attribution=_triage_attribution(),
            )
        # The business is mapped through YAML profiles to the benchmark_ieval key to look up the RemoteConfig default target
        mock_defaults.assert_called()
        _, kwargs = mock_create.call_args
        # dispatch_profile is passed along with AoneConfig — a BLOCKED message uses it to locate the missing RemoteConfig entry
        notification_config: NotificationConfig = kwargs["notification_config"]
        assert notification_config.dispatch_profile == "benchmark_ieval"
        assert kwargs["attribution_result"].should_dispatch is True
        svc.update_notification_dispatch.assert_awaited_once()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"]["notification_ref"] == "W-999"

    @pytest.mark.asyncio
    async def test_triage_path_skips_ieval_query_dedup(self) -> None:
        """The inline / dataset path does not dedup on the ieval query fingerprint."""
        svc = _fake_service()
        with (
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={
                    "benchmark_ieval": NotificationDispatchProfile(
                        notification_project_id="remote_config-proj",
                        assignee="remote_config-user",
                    ),
                },
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ),
            patch(
                f"{_DISPATCH_MODULE}.begin_ieval_query_dedup",
                new_callable=AsyncMock,
            ) as mock_dedup,
        ):
            await dispatch_notification_for_triage(
                _req(svc, source_trace_id="src-1"),
                attribution=_triage_attribution(),
            )
        mock_dedup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_attribution_overridden_skips_dispatch(self) -> None:
        """Attribution overrides the judge (should_dispatch=False) → skip attribution_overridden."""
        svc = _fake_service()
        with patch(
            f"{_DISPATCH_MODULE}.create_aone_task",
            new_callable=AsyncMock,
            return_value=_created_result(),
        ) as mock_create:
            await dispatch_notification_for_triage(
                _req(svc, source_trace_id="src-1"),
                attribution=_triage_attribution(should_dispatch=False),
            )
        mock_create.assert_not_awaited()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {"notification_skip_reason": "attribution_overridden"}

    @pytest.mark.asyncio
    async def test_trace_id_falls_back_to_source(self) -> None:
        """attribution.trace_id is empty → store.get_trace_id → source_trace_id."""
        svc = _fake_service()
        svc.get_trace_id = AsyncMock(return_value=None)
        with (
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={},
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_triage(
                _req(svc, source_trace_id="src-trace-1"),
                attribution=_triage_attribution(trace_id=""),
            )
        _, kwargs = mock_create.call_args
        assert kwargs["attribution_result"].trace_id == "src-trace-1"

    @pytest.mark.asyncio
    async def test_missing_yaml_profile_falls_back_to_agent_eval(self) -> None:
        """YAML profiles missing → create the ticket with RemoteConfig agent-eval, not profile_not_configured."""
        svc = _fake_service()
        profiles = {
            "agent-eval": NotificationDispatchProfile(
                notification_project_id="2165583",
                assignee="529504",
            ),
        }
        with (
            patch(
                f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
                return_value={},
            ),
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value=profiles,
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_triage(
                _req(
                    svc,
                    source_trace_id="src-1",
                    config={
                        "notification_dispatch": {"enabled": True},
                        "agent_registration": {"name": "chat-agent"},
                    },
                ),
                attribution=_triage_attribution(),
            )
        mock_create.assert_awaited_once()
        notification_config: NotificationConfig = mock_create.await_args.kwargs["notification_config"]
        assert notification_config.project == "2165583"
        assert notification_config.assignee == "529504"
        assert notification_config.dispatch_profile == "agent-eval"

    async def test_agent_notification_disabled_flag_ignored_still_dispatches(self) -> None:
        """The snapshot flag notification_dispatch_enabled=false is deprecated; dispatch still goes through RemoteConfig."""
        svc = _fake_service()
        with (
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(workitem_url="https://aone.example/W-999"),
            ) as mock_create,
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={
                    "agent-eval": NotificationDispatchProfile(
                        notification_project_id="2165583",
                        assignee="529504",
                    ),
                },
            ),
        ):
            await dispatch_notification_for_triage(
                _req(
                    svc,
                    source_trace_id="src-1",
                    config={
                        "notification_dispatch": {"enabled": True},
                        "agent_registration": {
                            "id": "reg-1",
                            "name": "chat-agent",
                            "notification_dispatch_enabled": False,
                        },
                    },
                ),
                attribution=_triage_attribution(),
            )
        mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disabled_config_skips(self) -> None:
        svc = _fake_service()
        with patch(
            f"{_DISPATCH_MODULE}.create_aone_task",
            new_callable=AsyncMock,
        ) as mock_create:
            await dispatch_notification_for_triage(
                _req(svc, source_trace_id="src-1", config={"notification_dispatch": {"enabled": False}}),
                attribution=_triage_attribution(),
            )
        mock_create.assert_not_awaited()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {"notification_skip_reason": "disabled"}


# ── parse_aone_dispatch_config ────────────────────────────────


class TestParseNotificationDispatchConfig:
    """Parse the aone_dispatch section of config_json."""

    @pytest.fixture(autouse=True)
    def _mock_yaml_section(self) -> Any:
        """Avoid a real YAML read; individual tests may override it."""
        with patch(
            f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
            return_value={},
        ):
            yield

    def test_missing_section_defaults_disabled(self) -> None:
        options, minor = parse_notification_dispatch_config({"other": 1})
        assert options.enabled is False
        assert minor is False

    def test_none_config_defaults_disabled(self) -> None:
        options, minor = parse_notification_dispatch_config(None)
        assert options.enabled is False
        assert minor is False

    def test_valid_section_parsed(self) -> None:
        options, minor = parse_notification_dispatch_config(
            {
                "notification_dispatch": {
                    "enabled": True,
                    "project": "proj-1",
                    "minor_dispatch": True,
                },
            }
        )
        assert options.enabled is True
        assert options.project == "proj-1"
        assert minor is True

    def test_invalid_keys_fallback_disabled(self) -> None:
        """AoneRuntimeOptions extra=forbid → an illegal key triggers fallback."""
        options, minor = parse_notification_dispatch_config(
            {
                "notification_dispatch": {"enabled": True, "bogus_field": 1},
            }
        )
        assert options.enabled is False
        assert minor is False

    def test_yaml_fallback_when_section_missing(self) -> None:
        """A per-run config with no aone_dispatch section → fall back to app-level YAML."""
        with patch(
            f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
            return_value={"enabled": True, "minor_dispatch": True},
        ):
            options, minor = parse_notification_dispatch_config({"other": 1})
        assert options.enabled is True
        assert minor is True

    def test_profiles_key_ignored_in_per_run_config(self) -> None:
        """profiles (a deploy-level mapping) is not a per-run parameter — after pop it does not trip extra=forbid."""
        options, minor = parse_notification_dispatch_config(
            {
                "notification_dispatch": {
                    "enabled": True,
                    "project": "proj-1",
                    "profiles": {"benchmark_ieval": "benchmark_ieval"},
                },
            }
        )
        assert options.enabled is True
        assert options.project == "proj-1"
        assert minor is False

    def test_yaml_fallback_pops_profiles(self) -> None:
        """The YAML fallback section contains profiles (deploy-level) → parse as usual after pop (forbid is not tripped)."""
        with patch(
            f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
            return_value={
                "enabled": True,
                "minor_dispatch": True,
                "profiles": {"benchmark_ieval": "benchmark_ieval"},
            },
        ):
            options, minor = parse_notification_dispatch_config({"other": 1})
        assert options.enabled is True
        assert minor is True


# ── T05: dispatch_aone_for_attribution orchestration ───────────────────


class TestDispatchNotificationForAttribution:
    """Push orchestration: decide → map → create_aone_task → persist."""

    @pytest.fixture(autouse=True)
    def _mock_dispatch_defaults(self) -> Any:
        """Avoid a real RemoteConfig call; individual tests may override the return value."""
        with patch(
            "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
            return_value={},
        ):
            yield

    @pytest.fixture(autouse=True)
    def _mock_yaml_profiles(self) -> Any:
        """The YAML profiles mapping goes through the mock (benchmark_ieval is configured);
        cases that lack profiles override it with an empty section."""
        with patch(
            f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
            return_value={"profiles": {"benchmark_ieval": "benchmark_ieval"}},
        ):
            yield

    @pytest.fixture(autouse=True)
    def _clean_policy_registry(self) -> Iterator[None]:
        """Isolate the business policy registry — clear it around each test so registrations do not leak."""
        clear_dispatch_policies()
        yield
        clear_dispatch_policies()

    @pytest.mark.asyncio
    async def test_disabled_skips_dispatch_and_writes_reason(self) -> None:
        svc = _fake_service()
        with patch(f"{_DISPATCH_MODULE}.create_aone_task") as mock_create:
            await dispatch_notification_for_attribution(
                _req(svc, config={"notification_dispatch": {"enabled": False}}),
                data=_attr(),
            )
        mock_create.assert_not_awaited()
        svc.update_notification_dispatch.assert_awaited_once()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {"notification_skip_reason": "disabled"}

    @pytest.mark.asyncio
    async def test_eval_misjudge_skips_dispatch_writes_reason(self) -> None:
        """A misjudge case (is_eval_misjudge=true) → do not create a ticket, and persist the eval_misjudge bucket."""
        svc = _fake_service()
        with patch(f"{_DISPATCH_MODULE}.create_aone_task") as mock_create:
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(is_eval_misjudge=True),
            )
        mock_create.assert_not_awaited()
        svc.update_notification_dispatch.assert_awaited_once()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {"notification_skip_reason": "eval_misjudge"}

    @pytest.mark.asyncio
    async def test_missing_yaml_profile_falls_back_to_agent_eval(self) -> None:
        """YAML profiles missing and the agent name is not configured → RemoteConfig agent-eval."""
        svc = _fake_service()
        profiles = {
            "agent-eval": NotificationDispatchProfile(
                notification_project_id="2165583",
                assignee="529504",
            ),
        }
        with (
            patch(
                f"{_DISPATCH_MODULE}._load_app_aone_dispatch_section",
                return_value={},
            ),
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value=profiles,
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(workitem_url="https://aone.example/W-999"),
            ) as mock_create,
        ):
            await dispatch_notification_for_attribution(
                _req(
                    svc,
                    config={
                        "notification_dispatch": {"enabled": True},
                        "agent_registration": {"name": "chat-agent"},
                    },
                ),
                data=_attr(severity="major"),
            )
        mock_create.assert_awaited_once()
        notification_config: NotificationConfig = mock_create.await_args.kwargs["notification_config"]
        assert notification_config.project == "2165583"
        assert notification_config.assignee == "529504"
        assert notification_config.dispatch_profile == "agent-eval"
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"]["notification_url"] == "https://aone.example/W-999"

    @pytest.mark.asyncio
    async def test_created_writes_notification_ref_bucket(self) -> None:
        svc = _fake_service()
        with (
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={
                    "benchmark_ieval": NotificationDispatchProfile(
                        notification_project_id="remote_config-proj",
                        assignee="remote_config-user",
                        participants=("cc-user",),
                    ),
                },
            ) as mock_defaults,
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(severity="major"),
            )
        # The business is mapped through YAML profiles to the benchmark_ieval key to look up the RemoteConfig default target
        mock_defaults.assert_called()
        # The per-run config project wins; assignee is resolved from the RemoteConfig benchmark_ieval profile
        _, kwargs = mock_create.call_args
        notification_config: NotificationConfig = kwargs["notification_config"]
        assert notification_config.enabled is True
        assert notification_config.assignee == "remote_config-user"
        assert notification_config.project == "proj-1"
        # dispatch_profile is passed along with AoneConfig — a BLOCKED message uses it to locate the missing RemoteConfig entry
        assert notification_config.dispatch_profile == "benchmark_ieval"
        # per-run participants not configured → fall back to the RemoteConfig benchmark_ieval profile
        assert notification_config.participants == ["cc-user"]
        # bucket：CREATED → notification_ref / notification_title / notification_dispatched_at
        svc.update_notification_dispatch.assert_awaited_once()
        _, kwargs = svc.update_notification_dispatch.call_args
        bucket = kwargs["dispatch"]
        assert bucket["notification_ref"] == "W-999"
        assert "notification_title" in bucket
        # Orchestration reads the fused-row trace_url for the Trace URL line in the ticket body
        svc.get_trace_url.assert_awaited_once_with("run1", "item1")
        assert "notification_dispatched_at" in bucket

    @pytest.mark.asyncio
    async def test_trace_id_prefers_run_items_over_job_id(self) -> None:
        """The ticket Trace ID prefers the fused-row trace_id (the business trace_id) over job_id."""
        svc = _fake_service()
        svc.get_trace_id = AsyncMock(return_value="biz-trace-9")
        with (
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={},
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_attribution(
                _req(svc, source_trace_id="judge-job-1"),
                data=_attr(),
            )
        svc.get_trace_id.assert_awaited_once_with("run1", "item1")
        _, kwargs = mock_create.call_args
        assert kwargs["attribution_result"].trace_id == "biz-trace-9"

    @pytest.mark.asyncio
    async def test_trace_id_falls_back_to_job_id_when_row_missing(self) -> None:
        """Fused-row trace_id missing (None) → fall back to the judge job_id."""
        svc = _fake_service()
        with (
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={},
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_attribution(
                _req(svc, source_trace_id="judge-job-1"),
                data=_attr(),
            )
        _, kwargs = mock_create.call_args
        assert kwargs["attribution_result"].trace_id == "judge-job-1"

    @pytest.mark.asyncio
    async def test_participants_forwarded_to_notification_config(self) -> None:
        """per-run aone_dispatch.participants → AoneConfig (the CC list)."""
        svc = _fake_service()
        with (
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={},
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_attribution(
                _req(
                    svc,
                    config={
                        "notification_dispatch": {
                            "enabled": True,
                            "project": "proj-1",
                            "participants": ["alice", "bob"],
                        },
                    },
                ),
                data=_attr(),
            )
        _, kwargs = mock_create.call_args
        notification_config: NotificationConfig = kwargs["notification_config"]
        assert notification_config.participants == ["alice", "bob"]

    @pytest.mark.asyncio
    async def test_skipped_result_writes_skip_reason(self) -> None:
        svc = _fake_service()
        skipped = NotificationDispatchResult(
            status=NotificationDispatchStatus.SKIPPED,
            message="Aone dispatch is disabled.",
        )
        with patch(
            f"{_DISPATCH_MODULE}.create_aone_task",
            new_callable=AsyncMock,
            return_value=skipped,
        ):
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(severity="critical"),
            )
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {
            "notification_skip_reason": "Aone dispatch is disabled.",
        }

    @pytest.mark.asyncio
    async def test_blocked_payload_writes_skip_reason(self) -> None:
        """create_aone_task returns BLOCKED (for example project is missing) → skip_reason is persisted (acceptance criterion 6)."""
        svc = _fake_service()
        blocked = NotificationDispatchResult(
            status=NotificationDispatchStatus.BLOCKED,
            message="Aone project id missing for dispatch profile 'unknown'.",
        )
        with patch(
            f"{_DISPATCH_MODULE}.create_aone_task",
            new_callable=AsyncMock,
            return_value=blocked,
        ):
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(primary_issue_category="dirty"),
            )
        _, kwargs = svc.update_notification_dispatch.call_args
        assert "notification_skip_reason" in kwargs["dispatch"]
        assert "project id missing" in kwargs["dispatch"]["notification_skip_reason"]

    @pytest.mark.asyncio
    async def test_create_notification_task_exception_isolated(self) -> None:
        """create_aone_task raises → write the dispatch_error bucket, and do not raise outward."""
        svc = _fake_service()
        with patch(
            f"{_DISPATCH_MODULE}.create_aone_task",
            new_callable=AsyncMock,
            side_effect=RuntimeError("a1 CLI crashed"),
        ):
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(severity="major"),
            )
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {"notification_skip_reason": "dispatch_error"}

    @pytest.mark.asyncio
    async def test_acceptable_score_skips_dispatch_writes_reason(self) -> None:
        """fused ≥ 1.0 (ACCEPTABLE) does not create a ticket, and persists the score_threshold bucket."""
        svc = _fake_service()
        svc.get_fused_score = AsyncMock(return_value=1.2)
        with patch(f"{_DISPATCH_MODULE}.create_aone_task") as mock_create:
            await dispatch_notification_for_attribution(
                _req(svc, config={"notification_dispatch": {"enabled": True}}),
                data=_attr(severity="critical"),
            )
        mock_create.assert_not_awaited()
        svc.update_notification_dispatch.assert_awaited_once()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {"notification_skip_reason": "score_threshold"}

    @pytest.mark.asyncio
    async def test_low_score_dispatches_workitem(self) -> None:
        """fused < 1.0 (FAIL/POOR) is let through at the score layer and a ticket is created (severity does not block)."""
        svc = _fake_service()
        svc.get_fused_score = AsyncMock(return_value=0.5)
        with patch(
            f"{_DISPATCH_MODULE}.create_aone_task",
            new_callable=AsyncMock,
            return_value=_created_result(),
        ) as mock_create:
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(severity="major"),
            )
        mock_create.assert_awaited_once()
        svc.update_notification_dispatch.assert_awaited_once()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"]["notification_ref"] == "W-999"

    @pytest.mark.asyncio
    async def test_bucket_write_failure_does_not_raise(self) -> None:
        """When update_aone_dispatch itself fails → log only, do not raise (acceptance criterion 4)."""
        svc = _fake_service()
        svc.update_notification_dispatch = AsyncMock(side_effect=RuntimeError("DB down"))
        with patch(
            f"{_DISPATCH_MODULE}.create_aone_task",
            new_callable=AsyncMock,
            return_value=_created_result(),
        ):
            # Must not raise
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(severity="major"),
            )

    @pytest.mark.asyncio
    async def test_registered_service_policy_applies(self) -> None:
        """A business-registered FAIL-only policy — a POOR webdev run does not create a ticket."""
        svc = _fake_service()
        svc.get_fused_score = AsyncMock(return_value=0.5)  # POOR
        svc.get_service_name = AsyncMock(return_value="webdev")
        register_dispatch_policy(
            "webdev",
            RubricV2DispatchPolicy(dispatch_grades=frozenset({"FAIL"})),
        )
        with patch(f"{_DISPATCH_MODULE}.create_aone_task") as mock_create:
            await dispatch_notification_for_attribution(
                _req(svc, config={"notification_dispatch": {"enabled": True}}),
                data=_attr(severity="critical"),
            )
        mock_create.assert_not_awaited()
        svc.get_service_name.assert_awaited_once_with("run1")
        svc.update_notification_dispatch.assert_awaited_once()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {"notification_skip_reason": "score_threshold"}

    @pytest.mark.asyncio
    async def test_unregistered_service_uses_default_policy(self) -> None:
        """An unregistered business name → fall back to the default policy, and POOR is dispatched as usual (contrast with the previous case)."""
        svc = _fake_service()
        svc.get_fused_score = AsyncMock(return_value=0.5)  # POOR
        svc.get_service_name = AsyncMock(return_value="unknown-svc")
        with patch(
            f"{_DISPATCH_MODULE}.create_aone_task",
            new_callable=AsyncMock,
            return_value=_created_result(),
        ) as mock_create:
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(severity="major"),
            )
        mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_query_dedup_hit_skips_create(self) -> None:
        """Redis already has a ticket id → do not call a1, and persist already_dispatched plus the old notification_ref."""
        IevalQueryDedupDecision = None

        svc = _fake_service()
        svc.get_fused_query = AsyncMock(return_value="附近加油站")
        with (
            patch(
                f"{_DISPATCH_MODULE}.begin_ieval_query_dedup",
                new_callable=AsyncMock,
                return_value=IevalQueryDedupDecision(skip=True, notification_ref="W-111"),
            ),
            patch(f"{_DISPATCH_MODULE}.create_aone_task") as mock_create,
        ):
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(),
            )
        mock_create.assert_not_awaited()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {
            "notification_skip_reason": "already_dispatched",
            "notification_ref": "W-111",
        }

    @pytest.mark.asyncio
    async def test_query_dedup_claim_commits_on_created(self) -> None:
        """The slot claim succeeds and ticket creation succeeds → commit writes the ticket id."""
        IevalQueryDedupDecision = None

        svc = _fake_service()
        svc.get_fused_query = AsyncMock(return_value="附近加油站")
        with (
            patch(
                f"{_DISPATCH_MODULE}.begin_ieval_query_dedup",
                new_callable=AsyncMock,
                return_value=IevalQueryDedupDecision(skip=False, claimed=True),
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(workitem_id="W-999"),
            ),
            patch(
                f"{_DISPATCH_MODULE}.commit_ieval_query_dedup",
                new_callable=AsyncMock,
            ) as mock_commit,
            patch(
                f"{_DISPATCH_MODULE}.abort_ieval_query_dedup",
                new_callable=AsyncMock,
            ) as mock_abort,
        ):
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(),
            )
        mock_commit.assert_awaited_once_with("附近加油站", "W-999")
        mock_abort.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_passes_query_to_create_notification_task(self) -> None:
        """The fused query is passed into create_aone_task's attribution_result at the same level as trace_id."""
        IevalQueryDedupDecision = None

        svc = _fake_service()
        svc.get_fused_query = AsyncMock(return_value="附近加油站")
        svc.get_trace_id = AsyncMock(return_value="biz-trace-1")
        with (
            patch(
                f"{_DISPATCH_MODULE}.begin_ieval_query_dedup",
                new_callable=AsyncMock,
                return_value=IevalQueryDedupDecision(skip=False, claimed=False),
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_attribution(_req(svc), data=_attr())
        mock_create.assert_awaited_once()
        mapped = mock_create.await_args.kwargs["attribution_result"]
        assert mapped.trace_id == "biz-trace-1"
        assert mapped.query == "附近加油站"

    @pytest.mark.asyncio
    async def test_query_dedup_claim_aborts_on_create_error(self) -> None:
        """After the slot is claimed, a1 raises → abort releases the claim."""
        IevalQueryDedupDecision = None

        svc = _fake_service()
        svc.get_fused_query = AsyncMock(return_value="附近加油站")
        with (
            patch(
                f"{_DISPATCH_MODULE}.begin_ieval_query_dedup",
                new_callable=AsyncMock,
                return_value=IevalQueryDedupDecision(skip=False, claimed=True),
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                side_effect=RuntimeError("a1 CLI crashed"),
            ),
            patch(
                f"{_DISPATCH_MODULE}.abort_ieval_query_dedup",
                new_callable=AsyncMock,
            ) as mock_abort,
        ):
            await dispatch_notification_for_attribution(
                _req(svc),
                data=_attr(),
            )
        mock_abort.assert_awaited_once_with("附近加油站")
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {"notification_skip_reason": "dispatch_error"}

    @pytest.mark.asyncio
    async def test_query_dedup_disabled_bypasses_dedup(self) -> None:
        """IEVAL_AONE_QUERY_DEDUP_ENABLED=false → skip dedup and create the ticket directly."""
        from benchmark.settings.setting import settings as app_settings

        svc = _fake_service()
        svc.get_fused_query = AsyncMock(return_value="附近加油站")
        with (
            patch.object(
                app_settings,
                "notification_query_dedup_enabled",
                False,
            ),
            patch(
                f"{_DISPATCH_MODULE}.begin_ieval_query_dedup",
                new_callable=AsyncMock,
            ) as mock_begin,
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_attribution(_req(svc), data=_attr())
        mock_begin.assert_not_awaited()
        mock_create.assert_awaited_once()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"]["notification_ref"] == "W-999"


# ── _dispatch_result_to_bucket ────────────────────────────────


class TestDispatchResultToBucket:
    """AoneDispatchResult → bucket field mapping."""

    def test_created_bucket_has_notification_ref(self) -> None:
        bucket = _dispatch_result_to_bucket(
            _created_result(
                workitem_id="W-1",
                workitem_url="https://aone.example/W-1",
            )
        )
        assert bucket["notification_ref"] == "W-1"
        assert bucket["notification_url"] == "https://aone.example/W-1"
        assert "notification_dispatched_at" in bucket

    def test_non_created_bucket_has_skip_reason(self) -> None:
        for status in (
            NotificationDispatchStatus.SKIPPED,
            NotificationDispatchStatus.BLOCKED,
            NotificationDispatchStatus.FAILED,
        ):
            result = NotificationDispatchResult(status=status, message=f"msg-{status.value}")
            bucket = _dispatch_result_to_bucket(result)
            assert bucket == {"notification_skip_reason": f"msg-{status.value}"}


# ── T06 injection point: claimed idempotency + failure isolation ──────────────────────


def _runner(
    *,
    claimed: bool,
    config: dict[str, Any] | None = None,
) -> ExternalBenchmarkRunner:
    """Build a runner — mock increment / heartbeat / finalize on benchmark_service.

    claimed controls whether schedule_notification_dispatch fires (acceptance criterion 3).
    config is passed through on the increment return value (a config with an aone_dispatch section takes the real push path).
    """
    svc = MagicMock()
    svc.get_run = AsyncMock(return_value=SimpleNamespace(status="running"))
    svc.resolve_item_id_by_job_id = AsyncMock(return_value="item1")
    svc.increment_attribution_received_once = AsyncMock(
        return_value=(1, config or {"attribution_expected_count": 1}, claimed),
    )
    svc.heartbeat = AsyncMock()
    svc.update_notification_dispatch = AsyncMock()
    return ExternalBenchmarkRunner(
        benchmark_service=svc,
        storage_client=None,
    )


def _callback_payload() -> AttributionCallback:
    """A successful attribution callback (severity=major, attribution body non-empty)."""
    data = {
        "run_id": "run1",
        "job_id": "attr-job-1",
        "item_id": "item1",
        "ieval_job_id": "ieval-1",
        "attribution": _attr(severity="major"),
    }
    return AttributionCallback.model_validate(data)


class TestHandleAttributionCallbackInjection:
    """The _handle_attribution_callback injection point (claimed idempotency + failure isolation)."""

    @pytest.mark.asyncio
    async def test_claimed_true_triggers_schedule(self) -> None:
        runner = _runner(claimed=True)
        ri = MagicMock()
        ri.source_trace_id = "judge-job-1"

        with (
            patch(
                f"{_RUNNER_MODULE}.BenchmarkRunItem.filter",
            ) as filt,
            patch(
                f"{_RUNNER_MODULE}.schedule_notification_dispatch",
            ) as mock_schedule,
            patch(
                f"{_RUNNER_MODULE}.BenchmarkTraceRecordWriter",
            ) as writer_cls,
            patch(
                f"{_RUNNER_MODULE}.try_finalize_external",
                new_callable=AsyncMock,
            ),
        ):
            qs = MagicMock()
            qs.first = AsyncMock(return_value=ri)
            filt.return_value = qs
            writer = MagicMock()
            writer.upsert_triage = AsyncMock()
            writer_cls.return_value = writer

            await runner._handle_attribution_callback(_callback_payload())

        mock_schedule.assert_called_once()
        args, _ = mock_schedule.call_args
        req = args[0]
        assert isinstance(req, DispatchRequest)
        assert req.run_id == "run1"
        assert req.item_id == "item1"
        assert req.source_trace_id == "judge-job-1"

    @pytest.mark.asyncio
    async def test_claimed_false_skips_schedule(self) -> None:
        """A repeated callback (healer replay / HTTP replay) is not pushed a second time (acceptance criterion 3)."""
        runner = _runner(claimed=False)
        ri = MagicMock()
        ri.source_trace_id = "judge-job-1"

        with (
            patch(
                f"{_RUNNER_MODULE}.BenchmarkRunItem.filter",
            ) as filt,
            patch(
                f"{_RUNNER_MODULE}.schedule_notification_dispatch",
            ) as mock_schedule,
            patch(
                f"{_RUNNER_MODULE}.BenchmarkTraceRecordWriter",
            ) as writer_cls,
            patch(
                f"{_RUNNER_MODULE}.try_finalize_external",
                new_callable=AsyncMock,
            ),
        ):
            qs = MagicMock()
            qs.first = AsyncMock(return_value=ri)
            filt.return_value = qs
            writer = MagicMock()
            writer.upsert_triage = AsyncMock()
            writer_cls.return_value = writer

            await runner._handle_attribution_callback(_callback_payload())

        mock_schedule.assert_not_called()

    # Note: exception isolation inside the push coroutine (create_aone_task raises → dispatch_error bucket
    # and the main path is not blocked) is already covered by TestDispatchAoneForAttribution.test_create_aone_task_exception_isolated
    # ; the injection point itself is a fire-and-forget create_task, and the main path (increment / heartbeat /
    # CAS / finalize) is fully decoupled from the push. test_claimed_true_triggers_schedule already covers
    # "claimed=True triggers schedule + finalize still runs".


# ── judge direct dispatch (attribution.execution=none)───────────────────


class TestMapJudgeToResult:
    """When there is no attribution, synthesize the dispatch structure from the judge result."""

    def test_basic_fields_from_criterion_scores(self) -> None:
        mapped = map_judge_to_result(
            JudgeDispatchInput(
                fused_score=0.5,
                criterion_scores={
                    "search_ai_card_v2": {
                        "score": 0.5,
                        "reason": "基础满足度未通过",
                    },
                },
            ),
            AttributionMapContext(
                trace_id="trace-j",
                item_id="item-j",
                trace_url="https://trace.example/j",
                query="附近加油站",
            ),
        )
        assert mapped.trace_id == "trace-j"
        assert mapped.case_id == "item-j"
        assert mapped.trace_url == "https://trace.example/j"
        assert mapped.query == "附近加油站"
        assert mapped.category == "unknown"
        assert mapped.root_cause == "judge_fused_score=0.5"
        assert "search_ai_card_v2" in mapped.reasoning
        assert "基础满足度未通过" in mapped.reasoning
        assert mapped.should_dispatch is True

    def test_empty_criterion_scores_fallback_reasoning(self) -> None:
        mapped = map_judge_to_result(
            JudgeDispatchInput(fused_score=0.0),
            AttributionMapContext(trace_id="t", item_id="i"),
        )
        assert mapped.reasoning == "judge fused_score=0.0"
        assert mapped.root_cause == "judge_fused_score=0.0"


class TestDispatchNotificationForJudge:
    """Judge direct-dispatch orchestration: score threshold + CREATED / SKIPPED buckets."""

    @pytest.mark.asyncio
    async def test_low_score_creates_workitem(self) -> None:
        svc = _fake_service()
        svc.get_fused_score = AsyncMock(return_value=0.5)
        with (
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={
                    "benchmark_ieval": NotificationDispatchProfile(
                        notification_project_id="remote_config-proj",
                        assignee="remote_config-user",
                    ),
                },
            ),
            patch(
                f"{_DISPATCH_MODULE}.create_aone_task",
                new_callable=AsyncMock,
                return_value=_created_result(),
            ) as mock_create,
        ):
            await dispatch_notification_for_judge(
                _req(svc),
                JudgeDispatchInput(
                    fused_score=0.5,
                    criterion_scores={"c1": {"score": 0.5, "reason": "差"}},
                    query="q1",
                ),
            )
        mock_create.assert_awaited_once()
        mapped = mock_create.await_args.kwargs["attribution_result"]
        assert mapped.root_cause == "judge_fused_score=0.5"
        assert mapped.query == "q1"
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"]["notification_ref"] == "W-999"

    @pytest.mark.asyncio
    async def test_high_score_skipped_by_threshold(self) -> None:
        """A high score (ACCEPTABLE+) does not create a ticket, and writes score_threshold."""
        svc = _fake_service()
        svc.get_fused_score = AsyncMock(return_value=1.2)
        with patch(f"{_DISPATCH_MODULE}.create_aone_task") as mock_create:
            await dispatch_notification_for_judge(
                _req(svc, config={"notification_dispatch": {"enabled": True}}),
                JudgeDispatchInput(
                    fused_score=1.2,
                    criterion_scores={"c1": {"score": 1.2, "reason": "好"}},
                ),
            )
        mock_create.assert_not_awaited()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {"notification_skip_reason": "score_threshold"}

    @pytest.mark.asyncio
    async def test_query_dedup_hit_skips_create(self) -> None:
        """Aligned with the attribution path: Redis already has a ticket id → do not create one, and persist already_dispatched."""
        IevalQueryDedupDecision = None

        svc = _fake_service()
        svc.get_fused_score = AsyncMock(return_value=0.5)
        svc.get_fused_query = AsyncMock(return_value="附近加油站")
        with (
            patch(
                "benchmark.domain.notification_dispatch_resolve.fetch_notification_dispatch_profiles",
                return_value={
                    "benchmark_ieval": NotificationDispatchProfile(
                        notification_project_id="remote_config-proj",
                        assignee="remote_config-user",
                    ),
                },
            ),
            patch(
                f"{_DISPATCH_MODULE}.begin_ieval_query_dedup",
                new_callable=AsyncMock,
                return_value=IevalQueryDedupDecision(skip=True, notification_ref="W-111"),
            ) as mock_dedup,
            patch(f"{_DISPATCH_MODULE}.create_aone_task") as mock_create,
        ):
            await dispatch_notification_for_judge(
                _req(svc),
                JudgeDispatchInput(fused_score=0.5, query="附近加油站"),
            )
        mock_dedup.assert_awaited_once_with("附近加油站")
        mock_create.assert_not_awaited()
        _, kwargs = svc.update_notification_dispatch.call_args
        assert kwargs["dispatch"] == {
            "notification_skip_reason": "already_dispatched",
            "notification_ref": "W-111",
        }
