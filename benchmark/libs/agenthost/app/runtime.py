import os
from pathlib import Path
from typing import Any

import httpx

from benchmark.libs.agenthost.agents.config.context import AgentProviderContext
from benchmark.libs.agenthost.agents.manager import AgentManager
from benchmark.libs.agenthost.app.config import AppConfigManager
from benchmark.libs.agenthost.app.providers import ProviderRegistry
from benchmark.libs.agenthost.models.errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelTimeoutError,
    ProviderError,
)
from benchmark.libs.agenthost.models.schemas import ModelConfigSchema
from benchmark.libs.agenthost.prompts import FilePromptLoader, PromptManager
from benchmark.libs.agenthost.tools.manager import ToolManager


class _ChatReply:
    def __init__(self, content: str, finish_reason: str = "stop", usage: dict | None = None):
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason}
        self.usage_metadata = usage or {}


class _GatewayChatModel:
    """OpenAI 兼容 Chat Completions 客户端。"""

    def __init__(
        self,
        model: str,
        base_url: str = "",
        api_key: str = "",
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def ainvoke(self, *args: object, **kwargs: object) -> _ChatReply:
        messages = args[0] if args else kwargs.get("messages") or []
        payload_messages = []
        for message in messages if isinstance(messages, list) else [messages]:
            content = getattr(message, "content", message)
            role = getattr(message, "type", None) or getattr(message, "role", "user")
            if role in {"human", "HumanMessage"}:
                role = "user"
            elif role in {"ai", "AIMessage"}:
                role = "assistant"
            payload_messages.append(
                {
                    "role": str(role),
                    "content": content if isinstance(content, str) else str(content),
                }
            )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
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
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(str(exc), provider="openai") from exc
        if response.status_code in {401, 403}:
            raise AuthenticationError(
                "model authentication failed",
                provider="openai",
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            error_cls = InvalidRequestError if response.status_code < 500 else ProviderError
            raise error_cls(
                f"model request failed with status {response.status_code}",
                provider="openai",
                status_code=response.status_code,
            )
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or message.get("reasoning_content") or ""
        return _ChatReply(
            str(content),
            str(choice.get("finish_reason") or "stop"),
            body.get("usage") if isinstance(body.get("usage"), dict) else {},
        )

    def bind_tools(self, *args: object, **kwargs: object):
        del args, kwargs
        return self


class AppRuntime:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config_manager = AppConfigManager(config)
        self.app_config = self.config_manager
        self.providers = ProviderRegistry()
        self.agent_manager = AgentManager()
        self.tool_manager = ToolManager()
        self.prompt_manager = PromptManager()
        self.service_root = Path(".")
        self.bootstrapped = False

    @classmethod
    def from_app_config(cls, app_config: AppConfigManager, service_root: Path | None = None) -> "AppRuntime":
        runtime = cls(app_config.config)
        runtime.app_config = app_config
        runtime.config_manager = app_config
        runtime.service_root = Path(service_root or ".")
        runtime.prompt_manager = _prompt_manager(app_config, runtime.service_root)
        return runtime

    def get_llm(self, model_key: str):
        models = self.app_config.config.get("models") or {}
        raw = models.get(model_key) or {}
        if not isinstance(raw, dict):
            raw = {}
        payload = dict(raw)
        payload.setdefault("model_name", model_key)
        schema = ModelConfigSchema(**payload)
        return self._build_llm_from_model_cfg(schema.model_name or model_key, schema, model_key)

    def _build_llm_from_model_cfg(self, model_name: str, model_cfg: ModelConfigSchema, business_id: str):
        del business_id
        api_key = getattr(model_cfg, "api_key", "") or ""
        if not api_key:
            env_name = getattr(model_cfg, "api_key_env", "") or ""
            if env_name:
                api_key = os.environ.get(env_name, "")
        return _GatewayChatModel(
            model=getattr(model_cfg, "model_name", None) or model_name,
            base_url=getattr(model_cfg, "base_url", "") or "",
            api_key=api_key,
            temperature=float(getattr(model_cfg, "temperature", 0.0) or 0.0),
            max_tokens=int(getattr(model_cfg, "max_tokens", 1024) or 1024),
            timeout=float(getattr(model_cfg, "timeout", 120.0) or 120.0),
        )

    async def bootstrap(self) -> None:
        if not ProviderRegistry._discovered:
            ProviderRegistry.discover()
        agents = self.app_config.config.get("agents") or {}
        for name, raw in agents.items():
            if not isinstance(raw, dict):
                continue
            type_name = str(raw.get("type") or name)
            component = ProviderRegistry._components.get(("agent", type_name))
            if component is None:
                continue
            metadata = getattr(component, "component_metadata", {}) or {}
            schema = (metadata.get("kwargs") or {}).get("ext_schema")
            payload = {key: value for key, value in raw.items() if key != "type"}
            cfg = schema.model_validate(payload) if schema is not None and hasattr(schema, "model_validate") else payload
            ctx = AgentProviderContext(name=name, runtime=self)
            self.agent_manager.register(name, component.from_config(cfg, ctx))
        self.bootstrapped = True

    async def shutdown(self) -> None:
        self.bootstrapped = False


def _prompt_manager(app_config: AppConfigManager, service_root: Path) -> PromptManager:
    raw = ((app_config.config.get("prompts") or {}).get("template_dir")) or []
    dirs = [service_root / item for item in raw] if isinstance(raw, list) else []
    manager = PromptManager(loader=FilePromptLoader(dirs[0]) if dirs else None)
    manager._default = type("Loader", (), {"base_dirs": tuple(dirs)})()
    return manager
