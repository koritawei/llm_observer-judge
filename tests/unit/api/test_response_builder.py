"""Unit tests for the OpenAI-compatible response/error builders."""

import json

from benchmark.routes.schema import build_chat_completion
from benchmark.utils.exceptions import APIError


def test_chat_completion_shape_is_openai_compatible() -> None:
    body = build_chat_completion(model="example", score=7.0, reasoning="ok")

    assert body.object == "chat.completion"
    assert body.model == "example"
    assert isinstance(body.id, str) and body.id
    assert isinstance(body.created, int)
    choice = body.choices[0]
    assert choice.index == 0
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"


def test_chat_completion_content_carries_score_and_reasoning() -> None:
    body = build_chat_completion(model="example", score=5.4, reasoning="why")
    content = json.loads(body.choices[0].message.content)
    assert content["score"] == 5.4
    assert content["reasoning"] == "why"


def test_chat_completion_reasoning_none_serialized_as_null() -> None:
    body = build_chat_completion(model="example", score=9.0, reasoning=None)
    content = json.loads(body.choices[0].message.content)
    assert content["score"] == 9.0
    assert content["reasoning"] is None


def test_error_body_shape() -> None:
    api_err = APIError.of(code=400, error_type="model_not_found", message="bad")
    api_err.param = "model"
    body = api_err.to_envelope()
    err = body["error"]
    assert err["message"] == "bad"
    assert err["code"] == 400
    assert err["param"] == "model"
    assert err["type"] == "model_not_found"


def test_error_body_custom_type() -> None:
    body = APIError.of(code=502, error_type="server_error", message="x").to_envelope()
    assert body["error"]["type"] == "server_error"
    assert body["error"]["code"] == 502
    assert body["error"]["param"] is None
