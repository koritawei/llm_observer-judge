"""OpenAI-compatible chat model. Callers only need ``ainvoke``."""

from __future__ import annotations

import os
from typing import Any

import httpx


class ChatCompletion:
    def __init__(self, content: str, finish_reason: str = "stop", usage: dict | None = None):
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason}
        self.usage_metadata = usage or {}


class OpenAICompatibleChatModel:
    def __init__(self, model: str, api_key: str, base_url: str, temperature: float, max_tokens: int):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def ainvoke(self, messages: list[Any]) -> ChatCompletion:
        payload_messages = []
        for message in messages:
            content = getattr(message, "content", message)
            role = getattr(message, "type", None) or getattr(message, "role", "user")
            if role == "human":
                role = "user"
            payload_messages.append({"role": role, "content": content if isinstance(content, str) else str(content)})
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": payload_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()
            body = response.json()
        choice = body["choices"][0]
        return ChatCompletion(choice["message"].get("content") or "", choice.get("finish_reason") or "stop", body.get("usage"))


def build_chat_model(cfg: dict[str, Any], **overrides) -> OpenAICompatibleChatModel:
    models = cfg.get("models") or cfg.get("model") or []
    if isinstance(models, str):
        models = [models]
    model = overrides.get("model") or (models[0] if models else os.environ.get("BENCHMARK_MODEL", "gpt-4o-mini"))
    return OpenAICompatibleChatModel(
        model=model,
        api_key=overrides.get("api_key") or os.environ.get("BENCHMARK_MODEL_API_KEY", ""),
        base_url=overrides.get("base_url") or os.environ.get("BENCHMARK_MODEL_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(overrides.get("temperature", cfg.get("temperature", 0))),
        max_tokens=int(overrides.get("max_tokens", cfg.get("max_tokens", 1024))),
    )
