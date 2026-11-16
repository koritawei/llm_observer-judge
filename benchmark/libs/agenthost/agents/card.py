from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentCard:
    name: str
    description: str = ""
    version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)
