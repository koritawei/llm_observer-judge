import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from business.agents.example._internal.llm import (
    AttributionLLM,
    ChatModelAttributionLLM,
)


class _RecordingChatModel(FakeMessagesListChatModel):
    """Record the last messages list received, for assertions."""

    last_messages: list[dict] | None = None

    def __init__(self, responses: list[AIMessage]):
        super().__init__(responses=responses)

    async def _agenerate(self, messages, stop=None, **kwargs):  # type: ignore[override]
        # FakeMessagesListChatModel accepts list[BaseMessage]; BaseMessage.model_dump() yields a dict
        self.last_messages = [
            m.model_dump() if hasattr(m, "model_dump") else m for m in messages
        ]
        return await super()._agenerate(messages, stop, **kwargs)


@pytest.mark.asyncio
async def test_chat_model_attribution_llm_passes_system_and_evidence_to_call_model():
    """system + evidence are passed through to model_client.call_model — call_model owns the two-message structure."""

    model = _RecordingChatModel(responses=[AIMessage(content="ok")])
    factory_called = 0

    def factory():
        nonlocal factory_called
        factory_called += 1
        return model

    sut = ChatModelAttributionLLM(factory)
    result = await sut.complete("you are a judge.", "evidence: A")

    assert isinstance(sut, AttributionLLM)
    # The message the underlying model receives should contain two dicts, a system role and a user role
    assert model.last_messages is not None
    assert len(model.last_messages) == 2
    assert model.last_messages[0]["type"] == "system"
    assert model.last_messages[0]["content"] == "you are a judge."
    assert model.last_messages[1]["type"] == "human"
    assert model.last_messages[1]["content"] == "evidence: A"
    assert factory_called == 1


@pytest.mark.asyncio
async def test_chat_model_attribution_llm_defers_model_creation_until_first_call():
    factory_called = 0

    def factory():
        nonlocal factory_called
        factory_called += 1
        return _RecordingChatModel(responses=[AIMessage(content="x")])

    sut = ChatModelAttributionLLM(factory)
    assert factory_called == 0

    await sut.complete("s", "e")
    assert factory_called == 1

    await sut.complete("s", "e")
    assert factory_called == 1
