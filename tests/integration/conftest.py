"""Integration test fixtures — real lifespan + httpx ASGI client."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from benchmark.settings import Settings
from benchmark.records.models.trace_record import TraceRecord
from benchmark.service.main import create_app

from tests.integration.helpers import async_lifespan


@pytest.fixture(autouse=True)
def _disable_live_mq(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Integration tests default to env=daily, which builds a real MetaqBackend; force consumption off so the suite does not connect to RocketMQ."""

    monkeypatch.setenv("AGENT_EVAL_MQ_ENABLED", "false")
    yield


@pytest_asyncio.fixture
async def async_app(tmp_path) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    app = create_app(settings=Settings(database_url=f"sqlite:///{tmp_path}/eval.db"))
    async with async_lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client


@pytest.fixture(autouse=True)
def _reset_observers() -> Iterator[None]:
    TraceRecord.clear_observer()
    yield
    TraceRecord.clear_observer()
