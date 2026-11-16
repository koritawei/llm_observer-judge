from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class ContentPart:
    type: str = "text"
    text: str = ""
    data: Any = None

    def __init__(self, type="text", text="", data=None, **extra):
        self.type = type
        self.text = text
        self.data = data
        self.extra = extra


@dataclass
class ChatMessage:
    role: str = "user"
    content: str | list[Any] = ""

    def __init__(self, role="user", content="", **extra):
        self.role = role
        self.content = content
        self.extra = extra


@dataclass
class AgentRequest:
    messages: list[ChatMessage] = field(default_factory=list)
    input: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    traceid: str = ""
    request_id: str = ""

    def __init__(self, messages=None, input=None, metadata=None, traceid="", request_id="", **extra):
        self.messages = list(messages or [])
        self.input = dict(input or {})
        self.metadata = dict(metadata or {})
        self.traceid = traceid
        self.request_id = request_id
        self.extra = extra
        for key, value in extra.items():
            setattr(self, key, value)


@dataclass
class AgentResponse:
    status: ResponseStatus = ResponseStatus.OK
    content: list[ContentPart] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __init__(self, status=ResponseStatus.OK, content=None, output=None, error=None, **extra):
        self.status = status
        self.content = list(content or [])
        self.output = dict(output or {})
        self.error = error
        self.metadata = {}
        self.content_parts = list(content or [])
        self.extra = extra
        for key, value in extra.items():
            setattr(self, key, value)
