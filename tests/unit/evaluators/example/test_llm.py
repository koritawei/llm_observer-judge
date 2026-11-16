"""Unit tests for the example evaluator's LLM seam.

The ChatModel is built by the agentgraph runtime (``runtime.get_llm``); this module
only adapts it to :class:`EvaluatorLLM`. Covers the lazy :class:`ChatModelLLM` seam.
"""
# ruff: noqa: ANN001, ANN201, ANN202, PLR2004  (test-only: fakes / monkeypatch)

import logging

import pytest
from business.evaluators.example.llm import ChatModelLLM


class _FakeReply:
    def __init__(
        self,
        content: str,
        *,
        response_metadata: dict[str, object] | None = None,
        usage_metadata: dict[str, object] | None = None,
    ) -> None:
        self.content = content
        self.response_metadata = response_metadata or {}
        self.usage_metadata = usage_metadata


class _FakeChatModel:
    """Stand-in for a LangChain BaseChatModel (only ``ainvoke`` is used)."""

    def __init__(
        self,
        reply: str,
        *,
        response_metadata: dict[str, object] | None = None,
        usage_metadata: dict[str, object] | None = None,
    ) -> None:
        self.reply = reply
        self.response_metadata = response_metadata
        self.usage_metadata = usage_metadata
        self.seen: list = []

    async def ainvoke(self, messages) -> _FakeReply:
        self.seen = messages
        return _FakeReply(
            self.reply,
            response_metadata=self.response_metadata,
            usage_metadata=self.usage_metadata,
        )


@pytest.mark.asyncio
async def test_chat_model_llm_completes_prompt() -> None:
    model = _FakeChatModel('{"score": 6, "reason": "ok"}')
    llm = ChatModelLLM(lambda: model)

    out = await llm.complete("hello prompt")

    assert out == '{"score": 6, "reason": "ok"}'
    assert model.seen[0].content == "hello prompt"  # sent as one human message


@pytest.mark.asyncio
async def test_chat_model_llm_logs_response_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _FakeChatModel(
        "",
        response_metadata={"finish_reason": "length"},
        usage_metadata={"input_tokens": 123, "output_tokens": 2048},
    )
    llm = ChatModelLLM(lambda: model)

    with caplog.at_level(logging.INFO, logger="agent_eval.evaluator.example"):
        out = await llm.complete("hello prompt")

    assert out == ""
    records = [
        record
        for record in caplog.records
        if record.name == "agent_eval.evaluator.example"
        and "evaluator.example.llm_response" in record.getMessage()
    ]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "content_len=0" in message
    assert "content_type=str" in message
    assert "finish_reason='length'" in message
    assert "usage_metadata={'input_tokens': 123, 'output_tokens': 2048}" in message
    assert any(
        "evaluator.example.llm_truncated_empty" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_chat_model_llm_coerces_list_content() -> None:
    class _ListContentModel:
        seen: list = []

        async def ainvoke(self, messages) -> object:
            self.seen = messages
            return _FakeReply([{"type": "text", "text": '{"score": 8}'}])

    model = _ListContentModel()
    llm = ChatModelLLM(lambda: model)
    out = await llm.complete("x")
    assert out == '{"score": 8}'


@pytest.mark.asyncio
async def test_chat_model_llm_builds_model_once() -> None:
    calls: list[int] = []

    def factory() -> _FakeChatModel:
        calls.append(1)
        return _FakeChatModel('{"score": 1}')

    llm = ChatModelLLM(factory)
    await llm.complete("a")
    await llm.complete("b")

    assert calls == [1]  # fetched lazily, exactly once
