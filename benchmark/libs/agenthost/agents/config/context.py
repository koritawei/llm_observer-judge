from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentProviderContext(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    name: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    providers: dict[str, Any] = Field(default_factory=dict)
    runtime: Any = None


class AgentConfigSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""
    model: str = ""
    prompt: str = ""
    options: dict[str, Any] = Field(default_factory=dict)
