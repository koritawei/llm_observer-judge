from typing import Any


class ModelConfigSchema:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "",
        model_name: str = "",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0,
        max_tokens: int = 1024,
        options: dict[str, Any] | None = None,
        **extra: Any,
    ):
        self.provider = provider
        self.model_name = model_name or model or str(extra.get("model") or "")
        self.model = self.model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.options = options or {}
        self.extra = extra
        for key, value in extra.items():
            setattr(self, key, value)
