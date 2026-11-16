"""Unit tests for the ExampleEvaluatorAgent (graph-backed evaluator).

The agent is built with an injected :class:`EvaluatorLLM` seam (the framework's
``runtime.get_llm`` in prod) and a real ``PromptManager`` over the bundled prompts;
``from_config`` is exercised against a fake ``ctx.runtime`` (get_llm + prompt_manager).
"""
# ruff: noqa: ANN001, ANN201, PLC0415, PLR2004  (test-only: literal asserts / fixture params / local imports)

from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmark.libs.agenthost.agents.request import AgentRequest, AgentResponse, ChatMessage, ResponseStatus
from benchmark.libs.agenthost.prompts import FilePromptLoader, PromptManager
from benchmark.evaluators.base import (
    EvaluatorRequest,
    EvaluatorResult,
    PromptBundleRef,
    SkillBundleRef,
    decode_eval_text,
    encode_eval_json,
)
from business.evaluators.example.agent import ExampleEvaluatorAgent, ExampleEvaluatorConfigSchema
from benchmark.paths import default_conf_dir
from benchmark.utils.exceptions import EvaluatorError

_PROMPTS_DIR = default_conf_dir() / "agents" / "judge_example" / "prompts"
_CASES_DIR = Path(__file__).parent / "cases"


def _prompt_manager() -> PromptManager:
    return PromptManager(loader=FilePromptLoader(_PROMPTS_DIR))


class _FakeLLM:
    def __init__(self, reply: str | None = None, *, boom: bool = False) -> None:
        self.reply = reply
        self.boom = boom
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.boom:
            raise RuntimeError("down")
        assert self.reply is not None
        return self.reply


def _agent(*, llm: _FakeLLM | None = None, prompt_name: str = "evaluator") -> ExampleEvaluatorAgent:
    from business.evaluators.example.agent import ExampleEvaluatorDeps

    return ExampleEvaluatorAgent(
        name="example",
        description=None,
        deps=ExampleEvaluatorDeps(
            llm=llm or _FakeLLM(reply='{"score": 7, "reason": "ok"}'),
            prompt_manager=_prompt_manager(),
            prompt_name=prompt_name,
        ),
    )


def _request(input_text: str = "你是谁") -> AgentRequest:
    # input/output are parsed JSON values (objects); the service then serializes the
    # whole EvaluatorRequest as message-content JSON to the agent.
    eval_request = EvaluatorRequest(
        input={"request_body": {"query": input_text}},
        output={"component": {"text": "我是助手"}},
        metadata={"trace_id": "t-1"},
        trace_id="t-1",
    )
    return AgentRequest(
        traceid="t-1",
        session_id="s-1",
        messages=[ChatMessage(role="user", content=encode_eval_json(eval_request))],
    )


def _load_case(name: str) -> EvaluatorRequest:
    """Load one fixture as an EvaluatorRequest. Cases do not reference each other."""

    return EvaluatorRequest.model_validate_json((_CASES_DIR / name).read_text(encoding="utf-8"))


def _verdict(resp: AgentResponse) -> EvaluatorResult:
    return EvaluatorResult.model_validate_json(decode_eval_text(resp.content_parts))


def _error(resp: AgentResponse) -> EvaluatorError:
    return EvaluatorError.from_payload(decode_eval_text(resp.content_parts))


@pytest.mark.asyncio
async def test_call_success_returns_score_and_reason() -> None:
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)

    resp = await agent.call(_request("你是谁"))

    assert resp.status == ResponseStatus.SUCCESS
    verdict = _verdict(resp)
    assert verdict.score == 7.0
    assert verdict.reason == "ok"
    # the rendered prompt (real evaluator.prompt) carried the user query to the LLM
    assert any("你是谁" in p for p in llm.prompts)


@pytest.mark.asyncio
async def test_call_without_trace_injects_unavailable_sentinel() -> None:
    """trace unavailable (no provider) → inject the "(trace 不可用)" sentinel, and do not leave a literal placeholder."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)

    resp = await agent.call(_request("你是谁"))

    assert resp.status == ResponseStatus.SUCCESS
    rendered = llm.prompts[0]
    assert "(trace 不可用)" in rendered
    assert "${trace_evidence}" not in rendered


@pytest.mark.asyncio
async def test_call_prompt_not_found_is_failed() -> None:
    agent = _agent(prompt_name="__missing__")

    resp = await agent.call(_request("x"))

    assert resp.status == ResponseStatus.FAILED
    assert _error(resp).error_type == "evaluator_prompt_not_found"


@pytest.mark.asyncio
async def test_call_llm_failure_is_failed() -> None:
    agent = _agent(llm=_FakeLLM(boom=True))
    resp = await agent.call(_request("x"))
    assert resp.status == ResponseStatus.FAILED
    assert _error(resp).error_type == "llm_failed"


@pytest.mark.asyncio
async def test_call_unparseable_reply_is_failed() -> None:
    agent = _agent(llm=_FakeLLM(reply="not json"))
    resp = await agent.call(_request("x"))
    assert resp.status == ResponseStatus.FAILED
    assert _error(resp).error_type == "llm_unparseable"


@pytest.mark.asyncio
async def test_run_delegates_to_call() -> None:
    agent = _agent(llm=_FakeLLM(reply='{"score": 5, "reason": "x"}'))
    resp = await agent.run(_request("hi"))
    assert resp.status == ResponseStatus.SUCCESS
    assert _verdict(resp).score == 5.0


def test_from_config_resolves_via_runtime() -> None:
    # _init_agent path: from_config(parsed ext_schema, ctx); the LLM is fetched
    # lazily through ctx.runtime.get_llm, prompts via ctx.runtime.prompt_manager.
    runtime = SimpleNamespace(
        get_llm=lambda key: (_ for _ in ()).throw(AssertionError("get_llm called eagerly")),
        prompt_manager=_prompt_manager(),
    )
    ctx = SimpleNamespace(name="example", service_root=default_conf_dir().parent, runtime=runtime)
    cfg = ExampleEvaluatorConfigSchema(default_model="example-judge", prompt="evaluator")

    agent = ExampleEvaluatorAgent.from_config(cfg, ctx)

    assert isinstance(agent, ExampleEvaluatorAgent)
    assert agent.name == "example"
    # lazy: building the agent did not touch get_llm
    assert agent._prompt_name == "evaluator"


def test_agent_is_registered_as_component() -> None:
    meta = getattr(ExampleEvaluatorAgent, "__agent_component__", None)
    assert meta is not None
    assert meta["name"] == "judge_example"


def test_get_agent_card_returns_card() -> None:
    from benchmark.libs.agenthost.agents.card import AgentCard

    agent = _agent()
    card = agent.get_agent_card()
    assert isinstance(card, AgentCard)
    assert card.name == "example"
    assert card.description


def test_from_config_passes_axes_and_scale_through() -> None:
    """from_config passes ext_schema sub_score_axes / sub_score_scale through onto the agent instance."""
    runtime = SimpleNamespace(
        get_llm=lambda key: (_ for _ in ()).throw(AssertionError("get_llm called eagerly")),
        prompt_manager=_prompt_manager(),
    )
    ctx = SimpleNamespace(name="example", service_root=default_conf_dir().parent, runtime=runtime)
    cfg = ExampleEvaluatorConfigSchema(
        default_model="example-judge",
        prompt="evaluator",
        sub_score_axes=["准确度", "相关性"],
        sub_score_scale=(0.0, 10.0),
        dimension_weights={"准确度": 0.6, "相关性": 0.4},
        attribution_threshold=6.0,
    )

    agent = ExampleEvaluatorAgent.from_config(cfg, ctx)

    assert agent._sub_score_axes == ["准确度", "相关性"]
    assert agent._sub_score_scale == (0.0, 10.0)
    assert agent._dimension_weights == {"准确度": 0.6, "相关性": 0.4}
    assert agent._attribution_threshold == 6.0


def test_from_config_default_axes_empty() -> None:
    """When axes are not declared, the old behavior: sub_score_axes=[] / sub_score_scale=None."""
    runtime = SimpleNamespace(
        get_llm=lambda key: (_ for _ in ()).throw(AssertionError("get_llm called eagerly")),
        prompt_manager=_prompt_manager(),
    )
    ctx = SimpleNamespace(name="example", service_root=default_conf_dir().parent, runtime=runtime)
    cfg = ExampleEvaluatorConfigSchema(default_model="example-judge", prompt="evaluator")

    agent = ExampleEvaluatorAgent.from_config(cfg, ctx)

    assert agent._sub_score_axes == []
    assert agent._sub_score_scale is None


@pytest.mark.asyncio
async def test_call_with_sub_scores_returns_them_in_result() -> None:
    """_FakeLLM returns sub_scores → they pass through on EvaluatorResult.sub_scores."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok", "sub_scores": {"准确度": 8, "相关性": 9}}')
    agent = _agent(llm=llm)

    resp = await agent.call(_request("你是谁"))

    assert resp.status == ResponseStatus.SUCCESS
    verdict = _verdict(resp)
    assert verdict.score == 7.0
    assert verdict.sub_scores == {"准确度": 8.0, "相关性": 9.0}


@pytest.mark.asyncio
async def test_call_fills_weights_and_threshold_from_config_when_llm_omits() -> None:
    """When the LLM does not write weights/thresholds back, do not persist model fields; recompute fused from declared_weights."""
    from business.evaluators.example.agent import ExampleEvaluatorDeps

    llm = _FakeLLM(reply='{"score": 7, "reason": "ok", "sub_scores": {"准确度": 8, "相关性": 6}}')
    agent = ExampleEvaluatorAgent(
        name="example",
        description=None,
        deps=ExampleEvaluatorDeps(
            llm=llm,
            prompt_manager=_prompt_manager(),
            prompt_name="evaluator",
            sub_score_axes=["准确度", "相关性"],
            dimension_weights={"准确度": 0.6, "相关性": 0.4},
            attribution_threshold=6.0,
        ),
    )
    resp = await agent.call(_request("你是谁"))
    assert resp.status == ResponseStatus.SUCCESS
    verdict = _verdict(resp)
    assert "weights" not in verdict.extra
    assert "attribution_threshold" not in verdict.extra
    # The default threshold registered on the rubric is not persisted; it is only an internal fallback for grading
    assert "default_attribution_threshold" not in verdict.extra
    assert "selected_attribution_threshold" not in verdict.extra
    assert verdict.extra["dynamic_rubric_fallback_reason"] == "invalid_dynamic_parameters"
    assert verdict.extra["dynamic_rubric_opt_in"] is True
    assert verdict.extra["declared_weights"] == {"准确度": 0.6, "相关性": 0.4}
    assert verdict.extra["computed_fused_score"] == 7.2
    # The framework grades computed 7.2 on the default 0–10 scale → GOOD
    assert verdict.extra["grade"] == "GOOD"


@pytest.mark.asyncio
async def test_call_recomputes_fused_score_instead_of_trusting_model_total() -> None:
    """When the model's self-reported total disagrees with the weighted result, extra uses the score computed on the server."""
    from business.evaluators.example.agent import ExampleEvaluatorDeps

    llm = _FakeLLM(
        reply=(
            '{"score": 99, "reason": "bad total", "sub_scores": {"准确度": 8, "相关性": 6},'
            ' "weights": {"准确度": 0.6, "相关性": 0.4}, "attribution_threshold": 6.0}'
        )
    )
    agent = ExampleEvaluatorAgent(
        name="example",
        description=None,
        deps=ExampleEvaluatorDeps(
            llm=llm,
            prompt_manager=_prompt_manager(),
            prompt_name="evaluator",
            sub_score_axes=["准确度", "相关性"],
            dimension_weights={"准确度": 0.6, "相关性": 0.4},
            attribution_threshold=6.0,
        ),
    )
    resp = await agent.call(_request("你是谁"))
    verdict = _verdict(resp)
    assert verdict.score == 7.2
    assert verdict.extra["model_score"] == 99.0
    assert verdict.extra["computed_fused_score"] == 7.2
    assert verdict.extra["score_mismatch"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "score_scale", "expected_grade"),
    [
        (2.9, (0.0, 10.0), "POOR"),
        (1.0, (0.0, 10.0), "POOR"),
        (2.9, None, "POOR"),
        (0.9, (0.0, 1.0), "EXCELLENT"),
    ],
)
async def test_call_ten_point_grade_without_attribution_threshold(
    score: float,
    score_scale: tuple[float, float] | None,
    expected_grade: str,
) -> None:
    """When no attribution threshold is configured, both an explicit scale and the default 10-point scale grade correctly."""
    from business.evaluators.example.agent import ExampleEvaluatorDeps

    llm = _FakeLLM(reply=f'{{"score": {score}, "reason": "low"}}')
    agent = ExampleEvaluatorAgent(
        name="example",
        description=None,
        deps=ExampleEvaluatorDeps(
            llm=llm,
            prompt_manager=_prompt_manager(),
            prompt_name="evaluator",
            sub_score_scale=score_scale,
        ),
    )

    resp = await agent.call(_request())

    verdict = _verdict(resp)
    assert verdict.extra["grade"] == expected_grade
    assert "attribution_threshold" not in verdict.extra


@pytest.mark.asyncio
async def test_default_prompt_renders_custom_rubric_configuration() -> None:
    """The real default prompt uses the axes, weights, scale, and thresholds from YAML, and does not hardcode sample values."""
    from business.evaluators.example.agent import ExampleEvaluatorDeps

    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = ExampleEvaluatorAgent(
        name="example",
        description=None,
        deps=ExampleEvaluatorDeps(
            llm=llm,
            prompt_manager=_prompt_manager(),
            prompt_name="evaluator",
            sub_score_axes=["事实性", "指令遵循"],
            sub_score_scale=(0.0, 10.0),
            dimension_weights={"事实性": 0.7, "指令遵循": 0.3},
            attribution_threshold=5.5,
        ),
    )

    await agent.call(_request())

    rendered = llm.prompts[0]
    assert "事实性 (weight=0.7)" in rendered
    assert "指令遵循 (weight=0.3)" in rendered
    assert "**5.5**" in rendered
    assert "准确度 (weight=0.4)" not in rendered
    assert "${axes}" not in rendered
    assert "${score_min}" not in rendered
    assert "${score_max}" not in rendered
    assert "${attribution_threshold}" not in rendered


@pytest.mark.asyncio
async def test_call_without_sub_scores_is_backward_compatible() -> None:
    """An old _FakeLLM that returns only {score, reason} → sub_scores=None, backward compatible."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)

    resp = await agent.call(_request("你是谁"))

    assert resp.status == ResponseStatus.SUCCESS
    verdict = _verdict(resp)
    assert verdict.sub_scores is None


@pytest.mark.asyncio
async def test_missing_order_id_hard_check_returns_zero_without_llm() -> None:
    """A mistaken get_order call: the hard check fails straight to score 0, with no LLM call and no trace fetch."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)

    async def _fetch_trace_evidence(request: EvaluatorRequest) -> str:
        raise AssertionError("must not fetch trace")

    agent._fetch_trace_evidence = _fetch_trace_evidence  # type: ignore[method-assign]
    request = _load_case("missing_order_id.json")
    assert request.tool_check == {"must_be_empty": True}

    result = await agent.evaluate(request)

    assert result.score == 0.0
    assert result.reason is not None
    assert result.reason.startswith("硬检失败：")
    assert "get_order" in result.reason
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_missing_order_id_clarify_passes_hard_check_and_calls_llm() -> None:
    """No order id and used_tool is empty: the hard check passes, then the LLM is called once."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)

    request = _load_case("missing_order_id_clarify.json")
    assert request.tool_check == {"must_be_empty": True}

    result = await agent.evaluate(request)

    assert len(llm.prompts) == 1
    assert result.score == 7.0
    assert "must_be_empty" not in llm.prompts[0]
    assert "请提供订单号" in llm.prompts[0]


@pytest.mark.asyncio
async def test_existing_order_passes_hard_check_and_calls_llm() -> None:
    """required get_order and the trace contains that name: the LLM is still called after the hard check passes."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)

    request = _load_case("existing_order.json")
    assert request.tool_check == {"required": ["get_order"]}

    result = await agent.evaluate(request)

    assert len(llm.prompts) == 1
    assert result.score == 7.0
    assert "202410220001" in llm.prompts[0]
    assert "get_order" in llm.prompts[0]


@pytest.mark.asyncio
async def test_unknown_order_name_check_passes() -> None:
    """A name that is in required passes. The LLM score is not asserted."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)

    request = _load_case("unknown_order.json")
    assert request.tool_check == {"required": ["get_order"]}

    await agent.evaluate(request)

    assert len(llm.prompts) == 1
    assert "get_order" in llm.prompts[0]


@pytest.mark.asyncio
async def test_empty_tool_check_still_calls_llm() -> None:
    """An empty dict has no constraint; after the hard check passes, enter the LLM."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "你是谁"}},
        output={"component": {"text": "我是助手"}, "content_desc": {"used_tool": []}},
        tool_check={},
    )
    assert request.tool_check == {}

    result = await agent.evaluate(request)

    assert len(llm.prompts) == 1
    assert result.score == 7.0


@pytest.mark.asyncio
async def test_invalid_tool_check_is_request_error_without_llm() -> None:
    """An illegal tool_check shape: EvaluatorError, and the LLM is not called."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "你是谁"}},
        output={"component": {"text": "我是助手"}, "content_desc": {"used_tool": []}},
        tool_check={"required": "get_order"},
    )
    assert request.tool_check == {"required": "get_order"}

    with pytest.raises(EvaluatorError) as exc_info:
        await agent.evaluate(request)

    assert exc_info.value.code == 400
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_judge_bundle_resolution_filters_attribution_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When shared fields are mixed with a stage, judge parsing must not consume the attribution skill or a body override."""
    judge_prompt_ref = PromptBundleRef(
        stage="judge",
        git_path="prompts/example",
        git_commit="abc123",
        bundle_sha256="sha256",
    )
    attribution_skill_ref = SkillBundleRef(
        stage="attribution",
        git_path="skills/example",
        git_commit="def456",
        bundle_sha256="sha256",
    )
    resolved_requests: list[EvaluatorRequest] = []

    async def _fake_resolve(request, prompt_manager, slot, default_name) -> str:
        resolved_requests.append(request)
        return "JUDGE_BUNDLE ${query}"

    monkeypatch.setattr("business.evaluators.example.agent.resolve_stage_prompt", _fake_resolve)
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "你是谁"}},
        output={"component": {"text": "我是助手"}},
        prompt_bundle_ref=judge_prompt_ref,
        skill_bundle_ref=attribution_skill_ref,
        judge_prompt_text="REQUEST_TEXT ${query}",
    )

    await agent.evaluate(request)

    assert len(resolved_requests) == 1
    assert resolved_requests[0].prompt_bundle_ref == judge_prompt_ref
    assert resolved_requests[0].skill_bundle_ref is None
    assert "JUDGE_BUNDLE 你是谁" in llm.prompts[0]
    assert "REQUEST_TEXT" not in llm.prompts[0]


@pytest.mark.asyncio
async def test_judge_prompt_text_replaces_file_template() -> None:
    """judge_prompt_text replaces the default scoring template; attribution_prompt_text does not enter rendering."""
    llm = _FakeLLM(reply='{"score": 8, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "你是谁"}},
        output={"component": {"text": "我是助手"}},
        judge_prompt_text="JUDGE_CUSTOM ${query}",
        attribution_prompt_text="ATTR_CUSTOM",
        prompt_text="REMOTE_CONFIG_CUSTOM ${query}",
    )

    result = await agent.evaluate(request)

    assert result.score == 8.0
    rendered = llm.prompts[0]
    assert "JUDGE_CUSTOM 你是谁" in rendered
    assert "REMOTE_CONFIG_CUSTOM" not in rendered
    assert "ATTR_CUSTOM" not in rendered
    assert "AI Agent 输出质量评审专家" not in rendered


@pytest.mark.asyncio
async def test_prompt_text_used_when_judge_prompt_text_absent() -> None:
    """When judge_prompt_text is absent, prompt_text still replaces the file template."""
    llm = _FakeLLM(reply='{"score": 6, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "你好"}},
        output={"component": {"text": "你好"}},
        prompt_text="REMOTE_CONFIG_CUSTOM ${query}",
    )

    await agent.evaluate(request)

    assert "REMOTE_CONFIG_CUSTOM 你好" in llm.prompts[0]


@pytest.mark.asyncio
async def test_expected_output_replaces_when_prompt_has_placeholder() -> None:
    """The request carries expected_output and the customized prompt contains ${expected_output} → substitute it as-is."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "附近咖啡"}},
        output={"component": {"text": "CHAT芝田"}},
        expected_output="应推荐附近咖啡馆",
        judge_prompt_text="Q=${query} E=${expected_output}",
    )

    await agent.evaluate(request)

    assert "Q=附近咖啡 E=应推荐附近咖啡馆" in llm.prompts[0]


@pytest.mark.asyncio
async def test_expected_output_dict_serialized_into_prompt() -> None:
    """When expected_output is a dict, substitute the JSON serialization (the structure is kept and no field is dropped)."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "q"}},
        output={"component": {"text": "a"}},
        expected_output={"reference": "应提醒减速"},
        judge_prompt_text="E=${expected_output}",
    )

    await agent.evaluate(request)

    assert (
        '{"reference": "应提醒减速"}' in llm.prompts[0]
        or '{"reference":"应提醒减速"}' in llm.prompts[0]
    )


@pytest.mark.asyncio
async def test_expected_output_injected_into_default_prompt() -> None:
    """The default file prompt contains ${expected_output}; when a value is present it is injected as the basis for a satisfaction evaluation."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "你是谁"}},
        output={"component": {"text": "我是助手"}},
        expected_output="应自我介绍并说明能力边界",
    )

    await agent.evaluate(request)

    assert "应自我介绍并说明能力边界" in llm.prompts[0]
    assert "${expected_output}" not in llm.prompts[0]


@pytest.mark.asyncio
async def test_expected_output_skipped_when_custom_prompt_has_no_placeholder() -> None:
    """When the customized prompt has no ${expected_output}, it is not forced to appear (Template ignores extra variables)."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "你是谁"}},
        output={"component": {"text": "我是助手"}},
        expected_output="不该出现在无占位符定制模板里",
        judge_prompt_text="Q=${query} A=${answer}",
    )

    await agent.evaluate(request)

    assert "不该出现在无占位符定制模板里" not in llm.prompts[0]
    assert "${expected_output}" not in llm.prompts[0]


@pytest.mark.asyncio
async def test_missing_expected_output_fills_sentinel_when_prompt_asks() -> None:
    """When the request has no expected_output, inject a sentinel so a literal ${expected_output} is not left for the LLM."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "q"}},
        output={"component": {"text": "a"}},
        judge_prompt_text="E=${expected_output}",
    )

    await agent.evaluate(request)

    assert "E=(未配置期望输出)" in llm.prompts[0]
    assert "${expected_output}" not in llm.prompts[0]


@pytest.mark.asyncio
async def test_prompt_macro_replaced_from_existing_input_fields() -> None:
    """Prompt macros are extensible: a field present on input (such as scene_id) automatically replaces the matching ${macro}."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={
            "request_body": {
                "query": "附近咖啡",
                "scene_id": "scene-42",
                "dialog_type": "search",
            }
        },
        output={"component": {"text": "CHAT芝田"}},
        judge_prompt_text="Q=${query} S=${scene_id} D=${dialog_type} M=${missing_field}",
    )

    await agent.evaluate(request)

    rendered = llm.prompts[0]
    assert "Q=附近咖啡 S=scene-42 D=search M=" in rendered
    assert "${missing_field}" not in rendered


@pytest.mark.asyncio
async def test_request_variables_override_case_bag() -> None:
    """variables extracted by variableMapping override a case field of the same name."""
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    agent = _agent(llm=llm)
    request = EvaluatorRequest(
        input={"request_body": {"query": "原始问题"}},
        output={"component": {"text": "a"}},
        variables={"query": "映射后的问题", "extra_tag": "from-mapping"},
        judge_prompt_text="Q=${query} T=${extra_tag}",
    )

    await agent.evaluate(request)

    assert "Q=映射后的问题 T=from-mapping" in llm.prompts[0]
