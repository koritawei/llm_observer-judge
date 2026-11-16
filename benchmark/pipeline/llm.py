"""Minimal LLM protocol for evaluators, plus a lazily constructed adapter."""

import asyncio
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

_MAX_RETRIES = 2
_RETRY_DELAY_S = 1.0
_RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError)


@runtime_checkable
class EvaluatorLLM(Protocol):
    async def complete(self, user_prompt: str, system_prompt: str = "") -> str: ...


class ChatModelLLM:
    """Adapt a LangChain ChatModel obtained lazily at runtime into an ``EvaluatorLLM``."""

    def __init__(self, model_factory: Callable[[], BaseChatModel]) -> None:
        self._factory = model_factory
        self._model: BaseChatModel | None = None

    def _model_or_build(self) -> BaseChatModel:
        if self._model is None:
            self._model = self._factory()
        return self._model

    async def complete(self, user_prompt: str, system_prompt: str = "") -> str:
        messages: list = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._model_or_build().ainvoke(messages)
                content = getattr(response, "content", response)
                return content if isinstance(content, str) else str(content)
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_DELAY_S)
        assert last_exc is not None  # the loop runs at least once, so an exception exists
        raise last_exc
