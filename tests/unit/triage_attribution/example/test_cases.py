"""example attribution standard cases and prompt injection. Does not connect to Langfuse."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.libs.agenthost.agents.request import AgentRequest, ResponseStatus
from benchmark.libs.agenthost.tools.content import ToolResult
from benchmark.records.enums import TriageLevel
from benchmark.attribution.badcase.schemas import (
    AttributionResult,
    BadcaseAttributionRuntimeOptions,
)
from benchmark.trace.tools import TraceRequestBound, TraceToolProvider
from business.agents.example.attribution.agent import AttributionAgent

_CASES = Path(__file__).parent / "cases"
_OBS_EMPTY = "obs-empty-tool-1"
_OBS_CLARIFY = "obs-clarify-1"
_CONF_PROMPT_MARKER = "你是 Agent Eval 的 badcase 归因 Agent"


class _ScriptedLLM:
    """Record system / evidence, and return ReAct JSON from the script."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0
        self.system_prompts: list[str] = []
        self.evidence_prompts: list[str] = []

    async def complete(self, system_prompt: str, evidence_prompt: str) -> dict[str, Any]:
        self.system_prompts.append(system_prompt)
        self.evidence_prompts.append(evidence_prompt)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class _RecordingPromptManager:
    """Implements only load_prompt. Call count and body are pinned by the test."""

    def __init__(self, body: str = "FILE_PROMPT", *, explode: bool = False) -> None:
        self.body = body
        self.explode = explode
        self.calls: list[str] = []

    async def load_prompt(self, name: str) -> str:
        self.calls.append(name)
        if self.explode:
            raise RuntimeError("prompt store down")
        return self.body


def _load_case(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the fixture. expected is left for assertions only and does not enter AgentRequest.metadata."""

    payload = json.loads((_CASES / name).read_text(encoding="utf-8"))
    expected = payload.pop("expected")
    metadata = dict(payload["metadata"])
    assert "expected" not in metadata
    visible = json.dumps(metadata, ensure_ascii=False)
    root_cause = expected.get("root_cause")
    if isinstance(root_cause, str):
        assert root_cause not in visible
    return metadata, expected


def _request(metadata: dict[str, Any]) -> AgentRequest:
    trace_id = str(metadata.get("trace_id") or "")
    return AgentRequest(
        traceid=trace_id,
        session_id="s-1",
        messages=[],
        metadata=metadata,
    )


def _agent(
    llm: _ScriptedLLM,
    *,
    provider: TraceToolProvider | None = None,
    prompt_manager: object | None = None,
    confidence_threshold: float = 0.5,
) -> AttributionAgent:
    return AttributionAgent(
        name="attribution_example",
        llm=llm,
        prompt_manager=prompt_manager,
        prompt_name="badcase_attribution",
        max_react_rounds=3,
        runtime_options=BadcaseAttributionRuntimeOptions(
            max_react_rounds=3,
            confidence_threshold=confidence_threshold,
        ),
        runtime=None,
        tool_provider=provider,
    )


def _provider(
    bound: TraceRequestBound,
    handler: Any,
) -> TraceToolProvider:
    provider = MagicMock(spec=TraceToolProvider)
    provider.bound_from_metadata = MagicMock(return_value=bound)
    provider.call_tool = AsyncMock(side_effect=handler)
    return provider


def _final(
    *,
    trace_id: str,
    category: str,
    root_cause: str,
    evidence_refs: list[str],
    corrected_triage_level: str | None = None,
    needs_human_review: bool = False,
) -> dict[str, Any]:
    final: dict[str, Any] = {
        "trace_id": trace_id,
        "case_id": trace_id,
        "category": category,
        "root_cause": root_cause,
        "confidence": 0.9,
        "reasoning": "scripted",
        "evidence_refs": evidence_refs,
        "action_hint": "fix",
        "needs_human_review": needs_human_review,
    }
    if corrected_triage_level is not None:
        final["corrected_triage_level"] = corrected_triage_level
    return {
        "decision_type": "final",
        "rationale_summary": "done",
        "final": final,
    }


def _assert_expected_hidden(llm: _ScriptedLLM, expected: dict[str, Any]) -> None:
    """The expected wording must not appear in the system / evidence visible to ReAct."""

    blobs = "\n".join([*llm.system_prompts, *llm.evidence_prompts])
    root_cause = expected.get("root_cause")
    if isinstance(root_cause, str):
        assert root_cause not in blobs


async def _run(agent: AttributionAgent, metadata: dict[str, Any]) -> AttributionResult:
    response = await agent.call(_request(metadata))
    assert response.status == ResponseStatus.SUCCESS
    return AttributionResult.model_validate_json(response.content_parts[0].data)


@pytest.mark.asyncio
async def test_empty_tool_fabricated_is_category_tool() -> None:
    """The TOOL body contains empty / not found, the script's final category=tool, and it cites the mock id."""

    metadata, expected = _load_case("empty_tool_fabricated.json")
    bound = TraceRequestBound(trace_id="tr-empty-tool", project_id="proj-example")

    async def _call_tool(tool_name: str, arguments: dict[str, object]) -> ToolResult:
        del arguments
        assert tool_name == "fetch_observation_detail"
        bound.known_refs.add(_OBS_EMPTY)
        return ToolResult.from_text(
            json.dumps(
                {
                    "id": _OBS_EMPTY,
                    "type": "TOOL",
                    "name": "get_order",
                    "output": "empty; not found",
                },
                ensure_ascii=False,
            )
        )

    llm = _ScriptedLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "read tool output",
                "tool_calls": [
                    {
                        "name": "fetch_observation_detail",
                        "arguments": {"observation_id": _OBS_EMPTY},
                    }
                ],
            },
            _final(
                trace_id="tr-empty-tool",
                category="tool",
                root_cause="tool result empty",
                evidence_refs=[_OBS_EMPTY],
            ),
        ]
    )
    result = await _run(_agent(llm, provider=_provider(bound, _call_tool)), metadata)

    assert result.category == expected["category"]
    assert result.evidence_refs == [_OBS_EMPTY]
    assert result.needs_human_review is False
    _assert_expected_hidden(llm, expected)
    assert "not found" in llm.evidence_prompts[-1]
    assert "empty" in llm.evidence_prompts[-1]


@pytest.mark.asyncio
async def test_clarification_false_kill_overrides_to_good() -> None:
    """Low-score clarification. confidence_threshold=0.5, final confidence=0.9, overturned to good."""

    metadata, expected = _load_case("clarification_false_kill.json")
    bound = TraceRequestBound(trace_id="tr-clarify", project_id="proj-example")

    async def _call_tool(tool_name: str, arguments: dict[str, object]) -> ToolResult:
        del arguments
        assert tool_name == "fetch_observation_detail"
        bound.known_refs.add(_OBS_CLARIFY)
        return ToolResult.from_text(
            json.dumps({"id": _OBS_CLARIFY, "output": "请提供订单号"}, ensure_ascii=False)
        )

    llm = _ScriptedLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "read clarify span",
                "tool_calls": [
                    {
                        "name": "fetch_observation_detail",
                        "arguments": {"observation_id": _OBS_CLARIFY},
                    }
                ],
            },
            _final(
                trace_id="tr-clarify",
                category="unknown",
                root_cause="judge false kill",
                evidence_refs=[_OBS_CLARIFY],
                corrected_triage_level="good",
                needs_human_review=False,
            ),
        ]
    )
    agent = _agent(
        llm,
        provider=_provider(bound, _call_tool),
        confidence_threshold=0.5,
    )
    result = await _run(agent, metadata)

    assert result.needs_human_review is False
    assert result.corrected_triage_level == TriageLevel.GOOD
    assert result.confidence == pytest.approx(0.9)
    assert result.category == "unknown"
    _assert_expected_hidden(llm, expected)


@pytest.mark.asyncio
async def test_no_tool_evidence_requests_human_review() -> None:
    """find_observations returns empty, the script returns human_review, and the result is needs_human_review."""

    metadata, expected = _load_case("no_tool_evidence.json")
    bound = TraceRequestBound(trace_id="tr-no-tool", project_id="proj-example")
    seen: list[str] = []

    async def _call_tool(tool_name: str, arguments: dict[str, object]) -> ToolResult:
        del arguments
        seen.append(tool_name)
        assert tool_name == "find_observations"
        return ToolResult.from_text("[]")

    llm = _ScriptedLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "list tools",
                "tool_calls": [{"name": "find_observations", "arguments": {"type": "TOOL"}}],
            },
            {
                "decision_type": "human_review",
                "rationale_summary": "no tool spans",
                "final": {
                    "trace_id": "tr-no-tool",
                    "case_id": "tr-no-tool",
                    "category": "unknown",
                    "root_cause": "no tool evidence",
                    "confidence": 0.0,
                    "reasoning": "find_observations empty",
                    "evidence_refs": [],
                    "action_hint": "manual_review",
                    "needs_human_review": True,
                },
            },
        ]
    )
    result = await _run(_agent(llm, provider=_provider(bound, _call_tool)), metadata)

    assert seen == ["find_observations"]
    assert result.needs_human_review is True
    assert expected["needs_human_review"] is True
    _assert_expected_hidden(llm, expected)


@pytest.mark.asyncio
async def test_attribution_prompt_text_overrides_and_still_calls_trace_tool() -> None:
    """evaluator_request.attribution_prompt_text is the system. load_prompt is not called, but fetch_trace_skeleton still is."""

    manager = _RecordingPromptManager(body="SHOULD_NOT_LOAD")
    bound = TraceRequestBound(trace_id="tr-prompt", project_id="proj-x")
    seen: list[str] = []

    async def _call_tool(tool_name: str, arguments: dict[str, object]) -> ToolResult:
        del arguments
        seen.append(tool_name)
        return ToolResult.from_text("{}")

    llm = _ScriptedLLM(
        [
            {
                "decision_type": "tool_calls",
                "rationale_summary": "need skeleton",
                "tool_calls": [{"name": "fetch_trace_skeleton", "arguments": {}}],
            },
            _final(
                trace_id="tr-prompt",
                category="prompt",
                root_cause="override ok",
                evidence_refs=[],
            ),
        ]
    )
    metadata = {
        "trace_id": "tr-prompt",
        "project_id": "proj-x",
        "evaluator_request": {
            "trace_id": "tr-prompt",
            "output": "ok",
            "attribution_prompt_text": "UPSTREAM_PROMPT_BODY",
            "judge_prompt_text": "JUDGE_SCORE_TEMPLATE",
        },
        "evaluator_result": {"score": 2.0, "reason": "low"},
    }
    await _run(
        _agent(llm, provider=_provider(bound, _call_tool), prompt_manager=manager),
        metadata,
    )

    assert manager.calls == []
    assert seen == ["fetch_trace_skeleton"]
    system = llm.system_prompts[0]
    assert "UPSTREAM_PROMPT_BODY" in system
    assert "JUDGE_SCORE_TEMPLATE" not in system
    assert "fetch_trace_skeleton" in system


@pytest.mark.asyncio
async def test_load_prompt_used_when_attribution_prompt_text_absent() -> None:
    """When the request does not carry attribution_prompt_text, system uses the load_prompt body and the tool description."""

    manager = _RecordingPromptManager(body="FILE_PROMPT")
    llm = _ScriptedLLM(
        [
            _final(
                trace_id="tr-file",
                category="prompt",
                root_cause="from file prompt",
                evidence_refs=[],
            )
        ]
    )
    metadata = {
        "trace_id": "tr-file",
        "project_id": "proj-x",
        "evaluator_request": {"trace_id": "tr-file", "output": "ok"},
        "evaluator_result": {"score": 2.0, "reason": "low"},
    }
    await _run(_agent(llm, prompt_manager=manager), metadata)

    assert manager.calls == ["badcase_attribution"]
    system = llm.system_prompts[0]
    assert "FILE_PROMPT" in system
    assert "fetch_trace_skeleton" in system


@pytest.mark.asyncio
async def test_judge_prompt_fields_do_not_leak_into_system() -> None:
    """judge_prompt_text / prompt_text are scoring templates; the attribution system does not use them."""

    manager = _RecordingPromptManager(body="FILE_PROMPT")
    llm = _ScriptedLLM(
        [
            _final(
                trace_id="tr-judge",
                category="prompt",
                root_cause="ignore judge template",
                evidence_refs=[],
            )
        ]
    )
    metadata = {
        "trace_id": "tr-judge",
        "project_id": "proj-x",
        "evaluator_request": {
            "trace_id": "tr-judge",
            "output": "ok",
            "prompt_text": "REMOTE_CONFIG_SCORE_TEMPLATE",
            "judge_prompt_text": "JUDGE_SCORE_TEMPLATE",
        },
        "evaluator_result": {"score": 2.0, "reason": "low"},
    }
    await _run(_agent(llm, prompt_manager=manager), metadata)

    system = llm.system_prompts[0]
    assert "FILE_PROMPT" in system
    assert "JUDGE_SCORE_TEMPLATE" not in system
    assert "REMOTE_CONFIG_SCORE_TEMPLATE" not in system
    assert "fetch_trace_skeleton" in system


@pytest.mark.asyncio
async def test_load_prompt_error_falls_back_to_conf_file() -> None:
    """After load_prompt raises, read the conf file, and still append the tool description."""

    manager = _RecordingPromptManager(explode=True)
    llm = _ScriptedLLM(
        [
            _final(
                trace_id="tr-fallback",
                category="prompt",
                root_cause="conf fallback",
                evidence_refs=[],
            )
        ]
    )
    metadata = {
        "trace_id": "tr-fallback",
        "project_id": "proj-x",
        "evaluator_request": {"trace_id": "tr-fallback", "output": "ok"},
        "evaluator_result": {"score": 2.0, "reason": "low"},
    }
    await _run(_agent(llm, prompt_manager=manager), metadata)

    assert manager.calls == ["badcase_attribution"]
    system = llm.system_prompts[0]
    assert _CONF_PROMPT_MARKER in system
    assert "fetch_trace_skeleton" in system


@pytest.mark.asyncio
async def test_judge_shaped_output_falls_back_to_human_review() -> None:
    """The model copies the judge's score/reason back (missing decision_type) → degrade to human_review, not FAILED.

    Live shape: content={"score": 1, "reasoning": "..."}, validation reports
    "1 validation error for BadcaseAttributionDecision", and attribution is therefore judged a failure.
    """

    llm = _ScriptedLLM([{"score": 1, "reasoning": "核心检查点未满足，符合 1 分判定。"}])
    metadata = {
        "trace_id": "tr-judge-shape",
        "project_id": "proj-x",
        "evaluator_request": {"trace_id": "tr-judge-shape", "output": "跑偏了"},
        "evaluator_result": {"score": 1.0, "reason": "核心检查点未满足"},
    }
    response = await _agent(llm, prompt_manager=_RecordingPromptManager()).call(_request(metadata))

    assert response.status == ResponseStatus.SUCCESS
    assert response.metadata["fallback"] == "schema_validation"
    result = AttributionResult.model_validate_json(response.content_parts[0].data)
    assert result.trace_id == "tr-judge-shape"
    assert result.needs_human_review is True
    assert result.should_dispatch is False
    assert "decision schema" in result.root_cause


@pytest.mark.asyncio
async def test_evidence_prompt_uses_text_judge_signal_and_output_contract() -> None:
    """The judge conclusion is injected as natural language, and the output contract is appended at the end, so it is not copied as a JSON template."""

    llm = _ScriptedLLM(
        [
            _final(
                trace_id="tr-contract",
                category="prompt",
                root_cause="contract case",
                evidence_refs=[],
            )
        ]
    )
    metadata = {
        "trace_id": "tr-contract",
        "project_id": "proj-x",
        "evaluator_request": {"trace_id": "tr-contract", "output": "ok"},
        "evaluator_result": {
            "score": 1.0,
            "reason": "核心检查点未满足",
            "evaluator": "example",
        },
    }
    await _run(_agent(llm, prompt_manager=_RecordingPromptManager()), metadata)

    evidence = llm.evidence_prompts[0]
    assert "## Judge Failure Signal" in evidence
    assert "不是你应输出的结构" in evidence
    assert "判分理由：核心检查点未满足" in evidence
    assert '"score":' not in evidence
    assert "## Output Contract" in evidence
    assert "`decision_type` is REQUIRED" in evidence
