"""Anthropic Messages adapter for Agent Eval named-model configuration."""

from benchmark.libs.agenthost.models.schemas import ModelConfigSchema
from langchain_anthropic import ChatAnthropic

from benchmark.settings.model_credentials import resolve_model_api_key


def build_anthropic_chat_model(model_config: ModelConfigSchema) -> ChatAnthropic:
    """Build a non-streaming Anthropic ChatModel from the standard model schema."""

    api_key = resolve_model_api_key(  # allow-secret -- env-derived, no literal credential
        api_key_env=model_config.api_key_env,
        default_env="ANTHROPIC_API_KEY",
        model_name=model_config.model_name,
    )

    return ChatAnthropic(
        model=model_config.model_name,
        api_key=api_key,
        base_url=model_config.base_url,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout=model_config.timeout,
        max_retries=0,
        streaming=False,
        model_kwargs=model_config.extra_params,
    )


__all__ = ["build_anthropic_chat_model"]
