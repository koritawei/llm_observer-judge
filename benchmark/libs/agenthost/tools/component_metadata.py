from dataclasses import dataclass, field
from typing import Any

from benchmark.libs.agenthost.tools.content import BaseToolProvider, _build_tool_component_provider


@dataclass
class ToolProviderContext:
    name: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    service_root: Any = None
    runtime: Any = None


def tool_component(*_args, **_kwargs):
    def decorator(cls):
        cls.component_metadata = {"kind": "tool", "args": _args, "kwargs": _kwargs}
        return cls

    return decorator


__all__ = ["ToolProviderContext", "_build_tool_component_provider", "tool_component", "BaseToolProvider"]
