"""Unit tests for the example evaluator's graph nodes (build_prompt / evaluate / response)."""
# ruff: noqa: PLR2004  (test-only: literal asserts / fixture params / local imports)

import logging

import pytest
from business.evaluators.example.nodes.build_prompt import BuildPromptNode
from business.evaluators.example.nodes.evaluate import EvaluateNode
from business.evaluators.example.nodes.response import ResponseNode


class _FakePromptManager:
    def __init__(self, template: str | None = None, *, raise_missing: bool = False) -> None:
        self.template = template
        self.raise_missing = raise_missing

    async def load_prompt(self, name: str) -> str:
        if self.raise_missing:
            raise FileNotFoundError(f"Prompt {name!r} not found")
        assert self.template is not None
        return self.template


class _FakeLLM:
    def __init__(self, reply: str | None = None, *, boom: bool = False) -> None:
        self.reply = reply
        self.boom = boom
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.boom:
            raise RuntimeError("llm down")
        assert self.reply is not None
        return self.reply


@pytest.mark.asyncio
async def test_build_prompt_renders_prompt(caplog: pytest.LogCaptureFixture) -> None:
    node = BuildPromptNode(
        _FakePromptManager("Q=${query}"),
        "main",
        tool_prompt="fetch_trace_skeleton\n${used_tool}\n${trace_evidence}",
    )
    with caplog.at_level(logging.INFO, logger="agent_eval.evaluator.example"):
        patch = await node({"render_context": {"query": "你是谁"}})

    assert patch["prompt"].startswith("Q=你是谁\n")
    assert "fetch_trace_skeleton" in patch["prompt"]
    assert not patch.get("error")
    assert any("evaluator.example.prompt_rendered" in r.message for r in caplog.records)
    # Record only chars/source, and no longer dump the full prompt
    assert not any("Q=你是谁" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_build_prompt_not_found_sets_error_code() -> None:
    node = BuildPromptNode(_FakePromptManager(raise_missing=True), "missing")
    patch = await node({"render_context": {"input": "x"}})

    assert patch["error_code"] == "evaluator_prompt_not_found"
    assert patch["error"]


@pytest.mark.asyncio
async def test_evaluate_calls_and_stores_content() -> None:
    llm = _FakeLLM(reply='{"score": 7, "reason": "ok"}')
    node = EvaluateNode(llm=llm)

    patch = await node({"prompt": "render me"})

    assert llm.prompts == ["render me"]
    assert patch["llm_content"] == '{"score": 7, "reason": "ok"}'


@pytest.mark.asyncio
async def test_evaluate_failure_sets_error_code() -> None:
    node = EvaluateNode(llm=_FakeLLM(boom=True))
    patch = await node({"prompt": "x"})
    assert patch["error_code"] == "llm_failed"
    assert patch["error"]


@pytest.mark.asyncio
async def test_evaluate_skips_when_error_present() -> None:
    llm = _FakeLLM(reply="never")
    node = EvaluateNode(llm=llm)
    patch = await node({"error": "boom", "error_code": "evaluator_prompt_not_found"})
    assert llm.prompts == []
    assert patch == {}


@pytest.mark.asyncio
async def test_response_extracts_evaluation() -> None:
    node = ResponseNode()
    patch = await node({"llm_content": '{"score": 8, "reason": "good"}'})
    assert patch["result"].score == 8.0
    assert patch["result"].reason == "good"


@pytest.mark.asyncio
async def test_response_unparseable_sets_error_code() -> None:
    node = ResponseNode()
    patch = await node({"llm_content": "not json"})
    assert patch["error_code"] == "llm_unparseable"
    assert patch["error"]


@pytest.mark.asyncio
async def test_response_unparseable_logs_complete_llm_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    content = f"before\n{'x' * 5000}\nafter"
    node = ResponseNode()

    with caplog.at_level(logging.WARNING):
        await node({"llm_content": content})

    messages = [
        record.getMessage()
        for record in caplog.records
        if "evaluator.example.llm_unparseable" in record.getMessage()
    ]
    assert messages == [f"evaluator.example.llm_unparseable llm_content={content!r}"]


@pytest.mark.asyncio
async def test_response_skips_when_error_present() -> None:
    node = ResponseNode()
    patch = await node({"error": "boom", "error_code": "llm_failed"})
    assert patch == {}


@pytest.mark.asyncio
async def test_build_prompt_injects_axes_and_scale() -> None:
    """When sub_score_axes / sub_score_scale are non-empty, inject axes / score_min / score_max into render_context."""
    node = BuildPromptNode(
        _FakePromptManager("axes=${axes} min=${score_min} max=${score_max}"),
        "main",
        sub_score_axes=["准确度", "相关性"],
        sub_score_scale=(0.0, 10.0),
    )
    patch = await node({"render_context": {"query": "x"}})

    assert "axes=准确度\n相关性" in patch["prompt"]
    assert "min=0.0" in patch["prompt"]
    assert "max=10.0" in patch["prompt"]


@pytest.mark.asyncio
async def test_build_prompt_injects_weights_and_attribution_threshold() -> None:
    node = BuildPromptNode(
        _FakePromptManager("axes=${axes} thr=${attribution_threshold}"),
        "main",
        sub_score_axes=["准确度", "相关性"],
        dimension_weights={"准确度": 0.6, "相关性": 0.4},
        attribution_threshold=6.0,
    )
    patch = await node({"render_context": {}})
    assert "准确度 (weight=0.6)" in patch["prompt"]
    assert "相关性 (weight=0.4)" in patch["prompt"]
    assert "thr=6.0" in patch["prompt"]


@pytest.mark.asyncio
async def test_without_tool_prompt_returns_body_only() -> None:
    """When no tool description is passed in, do not concatenate one."""
    node = BuildPromptNode(_FakePromptManager("Q=${query}"), "main")
    patch = await node({"render_context": {"query": "hi"}})

    assert patch["prompt"] == "Q=hi"


@pytest.mark.asyncio
async def test_build_prompt_without_axes_omits_vars() -> None:
    """When axes are empty and the template has no per-item variables, an old template still renders as-is."""
    node = BuildPromptNode(
        _FakePromptManager("Q=${query}"),
        "main",
        sub_score_axes=[],
        sub_score_scale=None,
    )
    patch = await node({"render_context": {"query": "hi"}})

    assert patch["prompt"] == "Q=hi"
    assert "axes=" not in patch["prompt"]


@pytest.mark.asyncio
async def test_build_prompt_without_axes_fills_sentinel_defaults() -> None:
    """axes/scale unconfigured and the template contains per-item variables → inject sentinels/defaults, and do not leave a literal ${...} placeholder."""
    node = BuildPromptNode(
        _FakePromptManager("axes=${axes} min=${score_min} max=${score_max}"),
        "main",
        sub_score_axes=[],
        sub_score_scale=None,
    )
    patch = await node({"render_context": {}})

    assert "axes=(未配置分项评分)" in patch["prompt"]
    assert "min=0.0" in patch["prompt"]
    assert "max=10.0" in patch["prompt"]
    assert "${" not in patch["prompt"]


@pytest.mark.asyncio
async def test_build_prompt_fills_expected_output_sentinel() -> None:
    """expected_output was not injected and the template contains the placeholder → fall back to a sentinel."""
    node = BuildPromptNode(
        _FakePromptManager("E=${expected_output}"),
        "main",
    )
    patch = await node({"render_context": {}})

    assert patch["prompt"] == "E=(未配置期望输出)"


@pytest.mark.asyncio
async def test_build_prompt_override_fills_expected_output_sentinel() -> None:
    """The override path falls back for expected_output the same way."""
    node = BuildPromptNode(_FakePromptManager("unused"), "main")
    patch = await node(
        {
            "prompt_override": "E=${expected_output}",
            "render_context": {},
        }
    )

    assert patch["prompt"] == "E=(未配置期望输出)"


@pytest.mark.asyncio
async def test_build_prompt_axes_configured_scale_defaulted() -> None:
    """axes configured but scale not configured → axes use the real value, and score_min/max use the display default 0–10."""
    node = BuildPromptNode(
        _FakePromptManager("axes=${axes} min=${score_min} max=${score_max}"),
        "main",
        sub_score_axes=["准确度"],
        sub_score_scale=None,
    )
    patch = await node({"render_context": {}})

    assert "axes=准确度" in patch["prompt"]
    assert "min=0.0" in patch["prompt"]
    assert "max=10.0" in patch["prompt"]


@pytest.mark.asyncio
async def test_response_axes_extra_key_warns_but_passes() -> None:
    """sub_scores returned by the LLM has keys outside axes → warn and do not raise, and keep the original value."""
    node = ResponseNode(sub_score_axes=["准确度"])
    patch = await node({"llm_content": '{"score": 7, "sub_scores": {"准确度": 8, "彩蛋": 9}}'})
    assert "result" in patch
    assert patch["result"].sub_scores == {"准确度": 8.0, "彩蛋": 9.0}


@pytest.mark.asyncio
async def test_response_axes_missing_key_warns_but_passes() -> None:
    """The LLM omits a key that is in axes → warn and do not raise, and keep the original value."""
    node = ResponseNode(sub_score_axes=["准确度", "相关性"])
    patch = await node({"llm_content": '{"score": 7, "sub_scores": {"准确度": 8}}'})
    assert "result" in patch
    assert patch["result"].sub_scores == {"准确度": 8.0}


@pytest.mark.asyncio
async def test_response_axes_empty_skips_validation() -> None:
    """axes=[] → do not validate; the old behavior."""
    node = ResponseNode(sub_score_axes=[])
    patch = await node({"llm_content": '{"score": 7, "sub_scores": {"任意": 5}}'})
    assert "result" in patch
    assert patch["result"].sub_scores == {"任意": 5.0}
