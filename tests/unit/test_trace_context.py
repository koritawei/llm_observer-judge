from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.http.middleware import trace_context
from fastapi import Request, Response


@pytest.mark.asyncio
async def test_middleware_uses_agent_eval_env_for_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "query_string": b"",
            "headers": [],
        }
    )
    init_context = MagicMock(return_value=object())
    monkeypatch.setattr(
        trace_context,
        "settings",
        SimpleNamespace(env="pre", environment="development"),
    )
    monkeypatch.setattr(
        trace_context,
        "init_request_context",
        init_context,
    )
    monkeypatch.setattr(
        trace_context,
        "clear_request_context",
        MagicMock(),
    )

    await trace_context.trace_context_middleware(
        request,
        AsyncMock(return_value=Response()),
    )

    init_context.assert_called_once_with(
        workflow_name="agent-eval.unknown",
        service="agent-eval",
        environment="pre",
    )
