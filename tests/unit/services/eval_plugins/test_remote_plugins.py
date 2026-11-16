"""Unit tests for local/remote plugin bindings and remote client."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_plugin_config import (
    EvalPluginsConfig,
    PluginFilter,
    resolve_plugin_binding,
)
from benchmark.domain.plugin_binding import PluginBinding
from benchmark.runs.common.parse_interceptor import DataParseContext
from benchmark.plugins.remote_client import RemotePluginClient, RemotePluginError
from benchmark.plugins.remote_parse import RemoteParseInterceptor


def test_plugin_binding_mode_aliases() -> None:
    assert PluginBinding(name="a", mode="platform").mode == "local"
    assert PluginBinding(name="a", mode="external_http").mode == "remote"
    assert PluginBinding(name="a", mode="remote", endpoint_url="https://x").is_remote_active()
    assert not PluginBinding(name="a", mode="remote").is_remote_active()


def test_resolve_binding_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmark.domain.eval_plugin_config as mod

    monkeypatch.setattr(
        mod,
        "fetch_eval_plugins",
        lambda **_k: EvalPluginsConfig(
            output_providers=PluginFilter(
                bindings=[
                    PluginBinding(
                        name="agent_chat",
                        mode="remote",
                        endpoint_url="https://biz.example/out",
                    ),
                ],
            ),
        ),
    )
    binding = resolve_plugin_binding("output_providers", "agent_chat")
    assert binding is not None
    assert binding.is_remote_active()


@pytest.mark.asyncio
async def test_remote_parse_interceptor() -> None:
    client = MagicMock(spec=RemotePluginClient)
    client.invoke = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "1",
                    "item_id": "1",
                    "input": {},
                    "output": {},
                    "metadata": {"csv_row": {}},
                },
            ],
        },
    )
    binding = PluginBinding(
        name="ref",
        mode="remote",
        endpoint_url="https://biz.example/parse",
    )
    interceptor = RemoteParseInterceptor(binding, client=client)
    items = await interceptor.aparse(
        DataParseContext(benchmark_name="search_eval", raw_items=[{"id": "1"}]),
    )
    assert len(items) == 1
    assert items[0]["item_id"] == "1"


@pytest.mark.asyncio
async def test_remote_client_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark.utils import http_transport

    class _Boom:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def post(self, *a: Any, **k: Any) -> Any:
            raise http_transport.HTTPError("boom")

    monkeypatch.setattr(http_transport, "AsyncClient", _Boom)
    client = RemotePluginClient()
    with pytest.raises(RemotePluginError, match="http error"):
        await client.invoke(
            binding=PluginBinding(
                name="x",
                mode="remote",
                endpoint_url="https://x",
            ),
            kind="output_provider",
            action="fetch_output",
            payload={},
        )
