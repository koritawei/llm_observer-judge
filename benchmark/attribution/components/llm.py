"""Shared LangChain ChatModel adapter for the attribution ReAct Agent."""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from benchmark.attribution.badcase.llm_attribution import AttributionLLM
from benchmark.attribution.badcase.model_client import call_model, maybe_await


class ChatModelAttributionLLM:
    """Adapt a lazily loaded runtime ChatModel to the two-segment prompt attribution interface."""

    def __init__(self, model_factory: Callable[[], BaseChatModel]) -> None:
        self._factory = model_factory
        self._model: BaseChatModel | None = None

    def _model_or_build(self) -> BaseChatModel:
        if self._model is None:
            self._model = self._factory()
        return self._model

    async def complete(self, system_prompt: str, evidence_prompt: str) -> object:
        model = await maybe_await(self._model_or_build())
        return await call_model(model, system_prompt, evidence_prompt)


__all__ = ["AttributionLLM", "ChatModelAttributionLLM"]
