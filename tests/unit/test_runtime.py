"""Agent Eval runtime model routing tests."""

from unittest.mock import MagicMock

from benchmark.libs.agenthost.app import AppRuntime
from benchmark.libs.agenthost.models.schemas import ModelConfigSchema
from benchmark.service.bootstrap import AgentEvalRuntime
from benchmark.utils.model_retry import RetryingChatModel


def _model_config(provider: str) -> ModelConfigSchema:
    return ModelConfigSchema(
        provider=provider,
        model_name="model-name",
        base_url="https://example.test",
        api_key_env="TEST_MODEL_API_KEY",
    )


def test_runtime_routes_anthropic_provider_to_service_adapter(monkeypatch) -> None:
    delegate = MagicMock()
    adapter = MagicMock(return_value=delegate)
    monkeypatch.setattr("benchmark.service.bootstrap.build_anthropic_chat_model", adapter)

    def unexpected_default_builder(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("default Agent Graph builder must not handle Anthropic")

    monkeypatch.setattr(
        AppRuntime,
        "_build_llm_from_model_cfg",
        unexpected_default_builder,
    )

    model = AgentEvalRuntime()._build_llm_from_model_cfg(
        "judge_amap_agent",
        _model_config("anthropic"),
        "agent-eval",
    )

    assert isinstance(model, RetryingChatModel)
    assert model._delegate is delegate
    adapter.assert_called_once()


def test_runtime_keeps_default_builder_for_other_providers(monkeypatch) -> None:
    delegate = MagicMock()
    default_builder = MagicMock(return_value=delegate)
    monkeypatch.setattr(AppRuntime, "_build_llm_from_model_cfg", default_builder)

    model = AgentEvalRuntime()._build_llm_from_model_cfg(
        "judge_example",
        _model_config("openai"),
        "agent-eval",
    )

    assert isinstance(model, RetryingChatModel)
    assert model._delegate is delegate
    default_builder.assert_called_once()
