from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    success: bool = True

    @classmethod
    def from_text(cls, text: str, **extra: Any) -> "ToolResult":
        return cls(text=text, data=dict(extra) if extra else {}, success=True)

    @classmethod
    def from_error(cls, error: str, **extra: Any) -> "ToolResult":
        return cls(text=str(error), error=str(error), success=False, data=dict(extra) if extra else {})

    def to_text(self) -> str:
        return self.text


@dataclass
class ToolDefinition:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    input_schema: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, description: str = "", parameters=None, input_schema=None, **extra):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.input_schema = input_schema or {}
        self.metadata = extra.pop("metadata", {})
        self.extra = extra

    def __getitem__(self, key: str):
        if key in self.extra:
            return self.extra[key]
        if isinstance(key, str) and hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@dataclass
class ToolExecutionContext:
    run_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentTool:
    name = "tool"
    description = ""

    async def run(self, arguments: dict[str, Any], context: ToolExecutionContext | None = None) -> ToolResult:
        return ToolResult(text="", data=arguments)


class BaseToolProvider:
    def __init__(self, name: str = "", source_type: str = "", **_kwargs: Any):
        self.name = name
        self.source_type = source_type

    def tools(self) -> list[AgentTool]:
        return []

    async def list_tools(self) -> list[Any]:
        fetch = getattr(self, "_fetch_tools", None)
        if fetch is None:
            return []
        return await fetch()


def _build_tool_component_provider(*_args, **_kwargs) -> BaseToolProvider:
    return BaseToolProvider()
