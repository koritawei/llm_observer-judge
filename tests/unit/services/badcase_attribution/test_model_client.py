"""model_client — multi-shape calls and response parsing."""
# ruff: noqa: PLR2004

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from benchmark.attribution.badcase.model_client import (
    call_model,
    maybe_await,
    parse_model_response,
    response_text,
)
from pydantic import BaseModel, ValidationError


class _EchoModel(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_call_model_invokes_callable_client() -> None:
    async def client(**kwargs: object) -> str:
        msgs = kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        return "ok"

    out = await call_model(client, "sys", "user")
    assert out == "ok"


@pytest.mark.asyncio
async def test_call_model_invokes_ainvoke_method() -> None:
    client = SimpleNamespace(ainvoke=AsyncMock(return_value="from-ainvoke"))
    out = await call_model(client, "sys", "user")
    assert out == "from-ainvoke"
    client.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_await_handles_sync_and_async() -> None:
    assert await maybe_await(42) == 42

    async def _async_val() -> int:
        return 7

    assert await maybe_await(_async_val()) == 7


def test_response_text_extracts_content_and_text() -> None:
    assert response_text("plain") == "plain"
    assert response_text(SimpleNamespace(content="c")) == "c"
    assert response_text(SimpleNamespace(text="t")) == "t"


def test_parse_model_response_from_dict() -> None:
    parsed = parse_model_response({"answer": "yes"}, _EchoModel)
    assert parsed.answer == "yes"


def test_parse_model_response_from_basemodel_wrapper_extracts_content() -> None:
    """LangChain AIMessage-style wrapper: parse JSON from the content field."""

    from langchain_core.messages import AIMessage

    response = AIMessage(content='{"answer": "from-wrapper"}')
    parsed = parse_model_response(response, _EchoModel)
    assert parsed.answer == "from-wrapper"


def test_parse_model_response_strips_json_code_fence() -> None:
    content = '```json\n{"answer": "fenced"}\n```'
    parsed = parse_model_response(content, _EchoModel)
    assert parsed.answer == "fenced"


def test_parse_model_response_repairs_unescaped_quotes() -> None:
    response = SimpleNamespace(content='{"answer":"reported "missing poi""}')

    parsed = parse_model_response(response, _EchoModel)

    assert parsed.answer == 'reported "missing poi"'


def test_parse_model_response_preserves_validation_error_for_truncated_output() -> None:
    response = SimpleNamespace(content='{"answer":"')

    with pytest.raises(ValidationError):
        parse_model_response(response, _EchoModel)
