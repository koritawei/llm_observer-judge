"""AttributionAgent tool-calling regression (framework-level trace; the business does not need an allowlist)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.libs.agenthost.agents.request import AgentRequest, ResponseStatus
from benchmark.libs.agenthost.tools.content import ToolResult

from benchmark.records.enums import TriageLevel
from benchmark.attribution.badcase.schemas import AttributionResult
from benchmark.trace import TraceSkeletonResult, TraceSpanSkeleton
from benchmark.trace.tools import TraceToolProvider, TraceRequestBound
from business.agents.example.attribution.agent import (
    AttributionAgent,
    AttributionAgentConfigSchema,
)


@pytest.mark.asyncio
async def test_example_config_accepts_react_runtime_options() -> None:
    payload = {
        "type": "attribution_example",
        "badcase_runtime_options": {"max_react_rounds": 2},
    }
    parsed = AttributionAgentConfigSchema.model_validate(payload)
    assert parsed.badcase_runtime_options.max_react_rounds == 2


class _FakeAttributionLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0
        self.evidence_prompts: list[str] = []

    async def complete(self, system_prompt: str, evidence_prompt: str) -> dict[str, Any]:
        self.evidence_prompts.append(evidence_prompt)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _agent(
    llm: _FakeAttributionLLM,
    *,
    provider: TraceToolProvider | None = None,
    max_rounds: int = 3,
    runtime: object | None = None,
) -> AttributionAgent:
    from benchmark.attribution.badcase.schemas import BadcaseAttributionRuntimeOptions

    return AttributionAgent(
        name="attribution_example",
        llm=llm,
        prompt_manager=None,
        prompt_name="badcase_attribution",
        max_react_rounds=max_rounds,
        runtime_options=BadcaseAttributionRuntimeOptions(
            max_react_rounds=max_rounds,
            confidence_threshold=0.5,
        ),
        runtime=runtime,
        tool_provider=provider,
    )


def _request() -> AgentRequest:
    return AgentRequest(
        traceid="tr-1",
        session_id="s-1",
        messages=[],
        metadata={
            "trace_id": "tr-1",
            "project_id": "proj-x",
            "evaluator_request": {"trace_id": "tr-1", "project_id": "proj-x", "output": "bad"},
            "evaluator_result": {"score": 1.0, "reason": "loop"},
        },
    )


def _tool_then_final() -> list[dict[str, Any]]:
    return [
        {
            "decision_type": "tool_calls",
            "rationale_summary": "need skeleton",
            "tool_calls": [{"name": "fetch_trace_skeleton", "arguments": {}}],
        },
        {
            "decision_type": "final",
            "rationale_summary": "done",
            "final": {
                "trace_id": "tr-1",
                "case_id": "tr-1",
                "category": "prompt",
                "root_cause": "tool_path_ok",
                "confidence": 0.9,
                "reasoning": "saw skeleton",
                "evidence_refs": ["obs-span-1"],
                "action_hint": "fix",
                "needs_human_review": False,
            },
        },
    ]


def _skeleton_provider(
    *,
    span_id: str = "obs-span-1",
) -> TraceToolProvider:
    resolver = MagicMock()
    resolver.fetch_skeleton = AsyncMock(
        return_value=TraceSkeletonResult(
            primary_id="lf-1",
            business_trace_id="tr-1",
            project_id="proj-x",
            spans=[TraceSpanSkeleton(id=span_id, name="agent")],
        )
    )
    return TraceToolProvider(resolver)


@pytest.mark.asyncio
async def test_attribution_agent_uses_trace_tools_then_finalizes() -> None:
    provider = _skeleton_provider()
    llm = _FakeAttributionLLM(_tool_then_final())
    agent = _agent(llm, provider=provider)
    response = await agent.call(_request())

    assert response.status == ResponseStatus.SUCCESS
    assert llm.calls == 2
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.root_cause == "tool_path_ok"
    assert result.needs_human_review is False


@pytest.mark.asyncio
async def test_evidence_prompt_carries_user_query() -> None:
    """evaluator_request.input.request_body.query → the User Query section of the evidence prompt."""
    llm = _FakeAttributionLLM(_tool_then_final()[1:])  # Go straight to final; no tool is needed
    agent = _agent(llm)
    request = AgentRequest(
        traceid="tr-1",
        session_id="s-1",
        messages=[],
        metadata={
            "trace_id": "tr-1",
            "project_id": "proj-x",
            "evaluator_request": {
                "trace_id": "tr-1",
                "project_id": "proj-x",
                "output": "bad",
                "input": {"request_body": {"query": "附近的咖啡店"}},
            },
            "evaluator_result": {"score": 1.0, "reason": "loop"},
        },
    )

    response = await agent.call(request)

    assert response.status == ResponseStatus.SUCCESS
    rendered = llm.evidence_prompts[0]
    assert "## User Query" in rendered
    assert "附近的咖啡店" in rendered


@pytest.mark.asyncio
async def test_evidence_prompt_user_query_missing_renders_dash() -> None:
    """No query from upstream → the User Query section renders "-" (the prompt says the model may fall back to a tool)."""
    llm = _FakeAttributionLLM(_tool_then_final()[1:])
    agent = _agent(llm)

    response = await agent.call(_request())

    assert response.status == ResponseStatus.SUCCESS
    assert "## User Query\n-" in llm.evidence_prompts[0]


@pytest.mark.asyncio
async def test_from_config_uses_framework_provider_without_agent_whitelist() -> None:
    """The business agent does not declare tool_providers; once the framework registers tools.providers, they are usable."""

    from benchmark.libs.agenthost.agents import AgentProviderContext
    from benchmark.libs.agenthost.tools.manager import ToolManager

    tool_manager = ToolManager()
    runtime = MagicMock()
    runtime.get_llm = MagicMock(side_effect=RuntimeError("no llm"))
    runtime.prompt_manager = None
    runtime.tool_manager = tool_manager

    cfg = AttributionAgentConfigSchema.model_validate(
        {
            "type": "attribution_example",
            "badcase_runtime_options": {"max_react_rounds": 3, "confidence_threshold": 0.5},
        }
    )
    ctx = AgentProviderContext(
        name="attribution_example",
        service_root=".",
        runtime=runtime,
        app_config=None,
    )
    agent = AttributionAgent.from_config(cfg, ctx)
    assert agent._resolve_trace_provider() is None

    resolver = MagicMock()
    resolver.fetch_skeleton = AsyncMock(
        return_value=TraceSkeletonResult(
            primary_id="lf-1",
            business_trace_id="tr-1",
            project_id="proj-x",
            spans=[TraceSpanSkeleton(id="obs-span-1", name="agent")],
        )
    )
    tool_manager.register_provider(TraceToolProvider(resolver))
    assert isinstance(agent._resolve_trace_provider(), TraceToolProvider)

    agent._llm = _FakeAttributionLLM(_tool_then_final())  # type: ignore[assignment]
    response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.root_cause == "tool_path_ok"
    assert result.needs_human_review is False
    resolver.fetch_skeleton.assert_awaited()


@pytest.mark.asyncio
async def test_attribution_agent_human_reviews_when_provider_missing_on_tool_call() -> None:
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "need skeleton",
                "tool_calls": [{"name": "fetch_trace_skeleton", "arguments": {}}],
            }
        ]
    )
    agent = _agent(llm, provider=None, max_rounds=1)
    response = await agent.call(_request())
    assert response.status == ResponseStatus.SUCCESS
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.needs_human_review is True
    assert "not registered" in result.root_cause


@pytest.mark.asyncio
async def test_attribution_agent_force_synthesizes_when_final_pass_requests_tools() -> None:
    """Final pass still asks for tools → force synthesis; root_cause stays business."""

    resolver = MagicMock()
    resolver.fetch_skeleton = AsyncMock(
        return_value=TraceSkeletonResult(
            primary_id="lf-1",
            business_trace_id="tr-1",
            project_id="proj-x",
            spans=[TraceSpanSkeleton(id="obs-span-1", name="agent")],
        )
    )
    provider = TraceToolProvider(resolver)
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "need skeleton",
                "tool_calls": [{"name": "fetch_trace_skeleton", "arguments": {}}],
            },
            {
                "decision_type": "tool_calls",
                "rationale_summary": "still digging",
                "tool_calls": [{"name": "fetch_trace_skeleton", "arguments": {}}],
            },
            {
                "decision_type": "tool_calls",
                "rationale_summary": "want more on final pass",
                "tool_calls": [{"name": "fetch_observation_detail", "arguments": {}}],
            },
            {
                "decision_type": "final",
                "rationale_summary": "forced conclude",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "prompt",
                    "root_cause": "agent looped without answering",
                    "confidence": 0.8,
                    "reasoning": "skeleton shows repeated agent spans",
                    "evidence_refs": ["obs-span-1"],
                    "action_hint": "tighten stop criteria",
                    "needs_human_review": False,
                },
            },
        ]
    )
    agent = _agent(llm, provider=provider, max_rounds=2)
    response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert "budget exhausted" not in result.root_cause
    assert result.root_cause == "agent looped without answering"
    assert result.category == "prompt"
    # haircut 0.1 from 0.8 → 0.7; threshold in _agent is 0.5 so no escalate
    assert result.confidence == pytest.approx(0.7)
    assert result.needs_human_review is False
    # 2 tool rounds + 1 final pass + 1 force synthesis
    assert llm.calls == 4


@pytest.mark.asyncio
async def test_attribution_agent_force_synthesis_falls_back_when_still_tool_calls() -> None:
    """Force synthesis still returns tool_calls → business human_review, no harness text."""

    resolver = MagicMock()
    resolver.fetch_skeleton = AsyncMock(
        return_value=TraceSkeletonResult(
            primary_id="lf-1",
            business_trace_id="tr-1",
            project_id="proj-x",
            spans=[],
        )
    )
    provider = TraceToolProvider(resolver)
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "keep digging",
                "tool_calls": [{"name": "fetch_trace_skeleton", "arguments": {}}],
            }
        ]
    )
    agent = _agent(llm, provider=provider, max_rounds=2)
    response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.needs_human_review is True
    assert "budget exhausted" not in result.root_cause
    assert "insufficient evidence" in result.root_cause
    # 2 tool rounds + 1 final + 1 force (all tool_calls via fake recycle)
    assert llm.calls == 4


@pytest.mark.asyncio
async def test_attribution_agent_human_review_decision() -> None:
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "human_review",
                "rationale_summary": "ambiguous",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "unknown",
                    "root_cause": "need human",
                    "confidence": 0.2,
                    "reasoning": "ambiguous",
                    "evidence_refs": [],
                    "action_hint": "manual_review",
                    "needs_human_review": True,
                },
            }
        ]
    )
    agent = _agent(llm, provider=None, max_rounds=1)
    response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.needs_human_review is True
    assert result.root_cause == "need human"


@pytest.mark.asyncio
async def test_attribution_agent_does_not_require_raw_trace_in_metadata() -> None:
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "final",
                "rationale_summary": "enough from envelope",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "prompt",
                    "root_cause": "no prefetch",
                    "confidence": 0.9,
                    "reasoning": "judge only",
                    "evidence_refs": [],
                    "action_hint": "fix",
                    "needs_human_review": False,
                },
            }
        ]
    )
    agent = _agent(llm, provider=None, max_rounds=1)
    req = _request()
    assert "raw_trace" not in (req.metadata or {})
    response = await agent.call(req)
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.root_cause == "no prefetch"
    assert result.needs_human_review is False
    assert result.raw_response is not None
    assert "no prefetch" in result.raw_response


@pytest.mark.asyncio
async def test_attribution_agent_truncates_tool_calls_per_round() -> None:
    resolver = MagicMock()
    resolver.fetch_skeleton = AsyncMock(
        return_value=TraceSkeletonResult(
            primary_id="lf-1",
            business_trace_id="tr-1",
            project_id="proj-x",
            spans=[TraceSpanSkeleton(id="obs-1", name="agent")],
        )
    )
    provider = TraceToolProvider(resolver)
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "flood",
                "tool_calls": [
                    {"name": "fetch_trace_skeleton", "arguments": {}},
                    {"name": "fetch_trace_skeleton", "arguments": {}},
                    {"name": "fetch_trace_skeleton", "arguments": {}},
                    {"name": "fetch_trace_skeleton", "arguments": {}},
                ],
            },
            {
                "decision_type": "final",
                "rationale_summary": "done",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "prompt",
                    "root_cause": "truncated_ok",
                    "confidence": 0.9,
                    "reasoning": "ok",
                    "evidence_refs": ["obs-1"],
                    "action_hint": "fix",
                    "needs_human_review": False,
                },
            },
        ]
    )
    # DEFAULT_MAX_TOOL_CALLS_PER_ROUND=3; 4 calls → truncate to 3.
    agent = _agent(llm, provider=provider, max_rounds=2)
    response = await agent.call(_request())
    assert response.status == ResponseStatus.SUCCESS
    assert resolver.fetch_skeleton.await_count == 3


# ── human_review escalation (bug #9) ──────────────────────────────────────────


def _agent_with_threshold(
    llm: _FakeAttributionLLM,
    *,
    threshold: float,
    provider: TraceToolProvider | None = None,
) -> AttributionAgent:
    """Variant of ``_agent`` that lets the test pin a specific
    ``confidence_threshold`` so we can drive the escalation branches.
    """

    from benchmark.attribution.badcase.schemas import BadcaseAttributionRuntimeOptions

    return AttributionAgent(
        name="attribution_example",
        llm=llm,
        prompt_manager=None,
        prompt_name="badcase_attribution",
        max_react_rounds=3,
        runtime_options=BadcaseAttributionRuntimeOptions(
            max_react_rounds=3,
            confidence_threshold=threshold,
        ),
        runtime=None,
        tool_provider=provider,
    )


@pytest.mark.asyncio
async def test_final_below_threshold_escalates_and_logs_reasons(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LLM emits a final with confidence below the threshold. The
    agent must (1) flip ``needs_human_review`` to True, and (2) emit
    a structured log line recording *why* — closing the silent
    OR-short-circuit that bug #9 reported.
    """
    import logging

    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "final",
                "rationale_summary": "confident",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "prompt",
                    "root_cause": "tool_path_ok",
                    "confidence": 0.40,  # below 0.95
                    "reasoning": "model low",
                    "evidence_refs": [],
                    "action_hint": "fix",
                    "needs_human_review": False,
                },
            }
        ]
    )
    agent = _agent_with_threshold(llm, threshold=0.95)
    with caplog.at_level(logging.INFO):
        response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)

    assert result.needs_human_review is True
    # The escalation log line must surface both the trigger reason
    # ("confidence_below_threshold(...)") and the numeric values, so
    # operators can see *why* an otherwise-final result was bumped.
    assert result.should_dispatch is False
    escalation_records = [
        rec for rec in caplog.records
        if rec.name == "business.agents.example.attribution.agent"
        and "attribution.escalated_to_human_review" in rec.getMessage()
    ]
    assert len(escalation_records) == 1, caplog.text
    msg = escalation_records[0].getMessage()
    assert "confidence_below_threshold" in msg
    assert "0.400" in msg
    assert "0.950" in msg


@pytest.mark.asyncio
async def test_corrected_triage_level_forces_should_dispatch_false() -> None:
    """Overturning the judge: keep corrected_triage_level, and force should_dispatch=False."""

    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "final",
                "rationale_summary": "judge false positive",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "unknown",
                    "root_cause": "Judge 误判：preview 截断。",
                    "confidence": 0.92,
                    "reasoning": "TOOL detail 已含完整数值",
                    "evidence_refs": [],
                    "action_hint": "improve_judge",
                    "needs_human_review": False,
                    "should_dispatch": True,
                    "corrected_triage_level": "good",
                },
            }
        ]
    )
    agent = _agent_with_threshold(llm, threshold=0.65)
    response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)

    assert result.needs_human_review is False
    assert result.should_dispatch is False
    assert result.corrected_triage_level == TriageLevel.GOOD
    assert result.root_cause == "Judge 误判：preview 截断。"


@pytest.mark.asyncio
async def test_escalation_clears_corrected_triage_level() -> None:
    """A low-confidence escalate drops corrected_triage_level, so a half-finished result is not written back onto triage."""

    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "final",
                "rationale_summary": "weak overturn",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "unknown",
                    "root_cause": "maybe judge wrong",
                    "confidence": 0.40,
                    "reasoning": "uncertain",
                    "evidence_refs": [],
                    "action_hint": "manual_review",
                    "needs_human_review": False,
                    "corrected_triage_level": "good",
                },
            }
        ]
    )
    agent = _agent_with_threshold(llm, threshold=0.95)
    response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)

    assert result.needs_human_review is True
    assert result.should_dispatch is False
    assert result.corrected_triage_level is None


@pytest.mark.asyncio
async def test_final_above_threshold_stays_final_without_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mirror of the escalation test: when the LLM's confidence is
    above threshold and it did not flag human review, the result must
    stay final AND the escalation log must NOT be emitted. This
    guards against a future regression where the log fires on every
    final.
    """
    import logging

    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "final",
                "rationale_summary": "ok",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "prompt",
                    "root_cause": "tool_path_ok",
                    "confidence": 0.97,
                    "reasoning": "high",
                    "evidence_refs": [],
                    "action_hint": "fix",
                    "needs_human_review": False,
                },
            }
        ]
    )
    agent = _agent_with_threshold(llm, threshold=0.95)
    with caplog.at_level(logging.INFO):
        response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)

    assert result.needs_human_review is False
    escalation_records = [
        rec for rec in caplog.records
        if rec.name == "business.agents.example.attribution.agent"
        and "attribution.escalated_to_human_review" in rec.getMessage()
    ]
    assert escalation_records == [], caplog.text


# ── evidence_refs quality gate (missing refs → human_review) ───────────────────


@pytest.mark.asyncio
async def test_fabricated_evidence_refs_escalate_to_human_review(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Model cites an observation_id never returned by any tool.

    Soft prompt alone is insufficient — the agent must re-raise
    ``needs_human_review`` (parity with the pre-hard-cut ``missing_refs`` gate).
    """
    import logging

    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "final",
                "rationale_summary": "fabricated",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "prompt",
                    "root_cause": "hallucinated_evidence",
                    "confidence": 0.99,
                    "reasoning": "made it up",
                    "evidence_refs": ["obs-does-not-exist"],
                    "action_hint": "fix",
                    "needs_human_review": False,
                },
            }
        ]
    )
    agent = _agent(llm, provider=None, max_rounds=1)
    with caplog.at_level(logging.INFO):
        response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)

    assert result.needs_human_review is True
    assert result.should_dispatch is False
    assert result.evidence_refs == ["obs-does-not-exist"]
    escalation_records = [
        rec
        for rec in caplog.records
        if rec.name == "business.agents.example.attribution.agent"
        and "attribution.escalated_to_human_review" in rec.getMessage()
    ]
    assert len(escalation_records) == 1, caplog.text
    assert "missing_evidence_refs" in escalation_records[0].getMessage()


@pytest.mark.asyncio
async def test_evidence_refs_seen_in_tool_output_stay_final() -> None:
    """Refs that appear as skeleton ``spans[].id`` must pass the gate."""

    provider = _skeleton_provider(span_id="obs-real-1")
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "need skeleton",
                "tool_calls": [{"name": "fetch_trace_skeleton", "arguments": {}}],
            },
            {
                "decision_type": "final",
                "rationale_summary": "done",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "prompt",
                    "root_cause": "tool_path_ok",
                    "confidence": 0.9,
                    "reasoning": "saw real obs",
                    "evidence_refs": ["obs-real-1"],
                    "action_hint": "fix",
                    "needs_human_review": False,
                },
            },
        ]
    )
    agent = _agent(llm, provider=provider)
    response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.needs_human_review is False
    assert result.evidence_refs == ["obs-real-1"]


@pytest.mark.asyncio
async def test_truncated_skeleton_emitted_refs_still_pass_gate() -> None:
    """Body char-cap drops trailing spans; remaining ids must still pass the gate."""

    resolver = MagicMock()
    spans = [
        TraceSpanSkeleton(
            id=f"obs-{i}",
            name=f"step-{i}",
            input_preview="Y" * 200,
            output_preview="Z" * 200,
        )
        for i in range(20)
    ]
    resolver.fetch_skeleton = AsyncMock(
        return_value=TraceSkeletonResult(
            primary_id="lf-1",
            business_trace_id="tr-1",
            project_id="proj-x",
            spans=spans,
        )
    )
    provider = TraceToolProvider(resolver, max_observation_chars=800)
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "need skeleton",
                "tool_calls": [{"name": "fetch_trace_skeleton", "arguments": {}}],
            },
            {
                "decision_type": "final",
                "rationale_summary": "cite first kept id",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "prompt",
                    "root_cause": "ok",
                    "confidence": 0.9,
                    "reasoning": "from truncated skeleton",
                    "evidence_refs": ["obs-0"],
                    "action_hint": "fix",
                    "needs_human_review": False,
                },
            },
        ]
    )
    agent = _agent(llm, provider=provider)
    response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.needs_human_review is False
    assert result.evidence_refs == ["obs-0"]


@pytest.mark.asyncio
async def test_nested_business_id_in_detail_is_not_a_valid_evidence_ref(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``poi.id`` inside observation input must not whitelist evidence_refs."""
    import logging

    bound = TraceRequestBound(trace_id="tr-1", project_id="proj-x")

    async def _call_tool(
        tool_name: str,
        arguments: dict[str, object],
        **kwargs: object,
    ) -> ToolResult:
        req_bound = kwargs.get("bound")
        if isinstance(req_bound, TraceRequestBound) and tool_name == "fetch_observation_detail":
            obs = arguments.get("observation_id")
            if isinstance(obs, str):
                req_bound.known_refs.add(obs)
        return ToolResult.from_text(
            json.dumps(
                {
                    "id": "obs-real-1",
                    "input": {"poi": {"id": "poi-123"}},
                    "output": {"ok": True},
                }
            )
        )

    provider = MagicMock(spec=TraceToolProvider)
    provider.bound_from_metadata = MagicMock(return_value=bound)
    provider.call_tool = AsyncMock(side_effect=_call_tool)
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "need detail",
                "tool_calls": [
                    {
                        "name": "fetch_observation_detail",
                        "arguments": {"observation_id": "obs-real-1"},
                    }
                ],
            },
            {
                "decision_type": "final",
                "rationale_summary": "cite nested poi",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "tool",
                    "root_cause": "bad_poi",
                    "confidence": 0.95,
                    "reasoning": "cited business id",
                    "evidence_refs": ["poi-123"],
                    "action_hint": "fix",
                    "needs_human_review": False,
                },
            },
        ]
    )
    agent = _agent(llm, provider=provider)
    with caplog.at_level(logging.INFO):
        response = await agent.call(_request())
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.needs_human_review is True
    assert "missing_evidence_refs" in caplog.text


# ── upstream contract guard (replaces test_node trace_id guard) ────────────────


@pytest.mark.asyncio
async def test_agent_returns_human_review_when_trace_id_missing() -> None:
    """A missing trace_id is an upstream contract error.

    The old ``AttributionNode._missing_trace_id_patch`` path has been replaced by an inline short-circuit inside ``AttributionAgent`` —
    do not spend more LLM / tool budget inside ``_run_tool_react``.
    Return human_review directly with ``action_hint="fix_upstream_trace_id_contract"``.
    """
    agent = _agent(_FakeAttributionLLM([]), provider=None, max_rounds=1)
    req = AgentRequest(
        traceid="",  # no trace id
        session_id="s-1",
        messages=[],
        metadata={"evaluator_request": {"trace_id": ""}},
    )
    response = await agent.call(req)
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.needs_human_review is True
    assert "trace_id is required" in result.root_cause
    assert result.action_hint == "fix_upstream_trace_id_contract"
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_agent_does_not_consult_llm_when_trace_id_missing() -> None:
    """A missing trace_id must not trigger an LLM call — the early-return path must be budget-free."""

    class _MustNotCallLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, system_prompt: str, evidence_prompt: str) -> object:
            self.calls += 1
            raise AssertionError("LLM must not be called when trace_id is missing")

    llm = _MustNotCallLLM()
    agent = _agent(llm, provider=None, max_rounds=2)
    req = AgentRequest(
        traceid="",
        session_id="s-1",
        messages=[],
        metadata={},
    )
    await agent.call(req)
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_missing_llm_returns_human_review_not_failed() -> None:
    """Missing default_model / _llm=None → human_review, and do not take the top-level attribution_failed."""

    agent = _agent(llm=_FakeAttributionLLM([]), provider=None, max_rounds=1)
    agent._llm = None  # type: ignore[assignment]
    response = await agent.call(_request())
    assert response.status == ResponseStatus.SUCCESS
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.needs_human_review is True
    assert result.should_dispatch is False
    assert "LLM is not configured" in result.root_cause


@pytest.mark.asyncio
async def test_tool_raise_feeds_observation_not_agent_failed() -> None:
    """A single tool raising is fed back as an observation, and the whole case can still continue to final."""

    resolver = MagicMock()
    resolver.fetch_skeleton = AsyncMock(side_effect=RuntimeError("langfuse down"))
    provider = TraceToolProvider(resolver)
    llm = _FakeAttributionLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "need skeleton",
                "tool_calls": [{"name": "fetch_trace_skeleton", "arguments": {}}],
            },
            {
                "decision_type": "final",
                "rationale_summary": "fallback",
                "final": {
                    "trace_id": "tr-1",
                    "case_id": "tr-1",
                    "category": "infra",
                    "root_cause": "tool_failed_but_agent_ok",
                    "confidence": 0.9,
                    "reasoning": "saw tool error",
                    "evidence_refs": [],
                    "action_hint": "retry",
                    "needs_human_review": False,
                },
            },
        ]
    )
    agent = _agent(llm, provider=provider, max_rounds=2)
    response = await agent.call(_request())
    assert response.status == ResponseStatus.SUCCESS
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.root_cause == "tool_failed_but_agent_ok"
    assert llm.calls == 2
