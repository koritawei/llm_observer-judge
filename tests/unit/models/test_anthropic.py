"""Anthropic ChatModel adapter tests."""

import pytest
from benchmark.libs.agenthost.models.schemas import ModelConfigSchema
from benchmark.models.anthropic import build_anthropic_chat_model
from langchain_anthropic import ChatAnthropic

_API_KEY_ENV = "AGENT_EVAL_JUDGE_AMAP_AGENT_MODEL_API_KEY"


def _model_config() -> ModelConfigSchema:
    return ModelConfigSchema(
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        base_url="https://ai-llm-gateway.amap.com/open_api/anthropic",
        api_key_env=_API_KEY_ENV,
        temperature=0.0,
        max_tokens=4096,
        timeout=120,
    )


def test_build_anthropic_chat_model_maps_standard_model_config(monkeypatch) -> None:
    monkeypatch.setenv(_API_KEY_ENV, "test-placeholder-not-used")

    model = build_anthropic_chat_model(_model_config())

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-4-6"
    assert model.anthropic_api_url == ("https://ai-llm-gateway.amap.com/open_api/anthropic")
    assert model.temperature == 0.0
    assert model.max_tokens == 4096
    assert model.default_request_timeout == 120
    assert model.max_retries == 0
    assert model.streaming is False


def test_build_anthropic_chat_model_rejects_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv(_API_KEY_ENV, raising=False)

    with pytest.raises(ValueError, match=_API_KEY_ENV):
        build_anthropic_chat_model(_model_config())
