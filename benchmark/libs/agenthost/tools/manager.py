from typing import Any


class ToolManager:
    def __init__(self, tools: list[Any] | None = None):
        self.tools = {getattr(tool, "name", str(index)): tool for index, tool in enumerate(tools or [])}
        self.providers: list[Any] = []

    def get(self, name: str):
        return self.tools[name]

    def register_provider(self, provider: Any) -> None:
        self.providers.append(provider)

    def get_provider(self, name: str) -> Any:
        for provider in self.providers:
            if getattr(provider, "name", None) == name:
                return provider
        raise KeyError(name)

    async def list_tools(self) -> list[Any]:
        listed: list[Any] = []
        for provider in self.providers:
            listed.extend(await provider.list_tools())
        return listed

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        for provider in self.providers:
            return await provider.call_tool(name, arguments or {}, **kwargs)
        raise KeyError(name)
