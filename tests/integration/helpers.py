"""Integration test helpers — shared by conftest and the cases."""
# ruff: noqa: PLC0415

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from benchmark.libs.agenthost.agents.request import AgentResponse, ContentPart, ResponseStatus
from benchmark.attribution.badcase.schemas import AttributionResult
from fastapi import FastAPI


@asynccontextmanager
async def async_lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with app.router.lifespan_context(app):
        from tortoise import Tortoise

        await Tortoise.generate_schemas(safe=True)
        yield


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


FAKE_ATTRIBUTION = AttributionResult(
    trace_id="t-77",
    case_id="case-77",
    category="prompt",
    root_cause="missing_context",
    confidence=0.9,
    reasoning="judge signal points to prompt issue",
    evidence_refs=[],
    action_hint="revise prompt template",
    needs_human_review=False,
)


def patch_attribution_call(
    attribution_agent: object, attribution_result: AttributionResult
) -> AsyncMock:
    response = AgentResponse(
        traceid=attribution_result.trace_id,
        status=ResponseStatus.SUCCESS,
        content_parts=[ContentPart(type="text", data=attribution_result.model_dump_json())],
        metadata={},
    )
    mock = AsyncMock(return_value=response)
    attribution_agent.call = mock  # type: ignore[method-assign]
    return mock
