"""Remote parse-interceptor adapter.

The protocol requires a synchronous ``parse(ctx)``; when ``anyio.from_thread.run`` is not
available outside a thread, the remote call falls back to an ``asyncio.run`` wrapper. When the submit pipeline calls from an async context,
prefer :meth:`aparse` (optionally detected by SubmitPipeline).
"""

import asyncio
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.plugin_binding import PluginBinding
from benchmark.runs.common.parse_interceptor import (
    DataParseContext,
    validate_business_item,
)
from benchmark.plugins.remote_client import RemotePluginClient, RemotePluginError

logger = get_logger(__name__)


class RemoteParseInterceptor:
    """DataParseInterceptor implementation for ``mode=remote``."""

    def __init__(
        self,
        binding: PluginBinding,
        *,
        client: RemotePluginClient | None = None,
    ) -> None:
        self._binding = binding
        self._client = client or RemotePluginClient()

    def parse(self, ctx: DataParseContext) -> list[dict[str, Any]]:
        """Sync entry: run the async call outside an existing event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aparse(ctx))
        # Already inside a running loop: cannot nest asyncio.run; the submit pipeline should call aparse.
        raise RuntimeError(
            "RemoteParseInterceptor.parse cannot be called from a running "
            "event loop; use aparse() instead",
        )

    async def aparse(self, ctx: DataParseContext) -> list[dict[str, Any]]:
        """Async parse (preferred path)."""
        payload = {
            "benchmark_name": ctx.benchmark_name,
            "rubric_ref": ctx.rubric_ref,
            "raw_items": ctx.raw_items,
            "profile_snapshot": ctx.profile_snapshot,
        }
        try:
            data = await self._client.invoke(
                binding=self._binding,
                kind="parse_interceptor",
                action="parse",
                payload=payload,
            )
        except RemotePluginError as exc:
            logger.warning(
                "remote_parse.failed name=%s err=%s",
                self._binding.name,
                exc,
            )
            raise
        items = data.get("items")
        if items is None and isinstance(data.get("result"), list):
            items = data["result"]
        if not isinstance(items, list):
            raise RemotePluginError(
                f"remote parse {self._binding.name!r} response missing items list",
            )
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                raise RemotePluginError("remote parse item must be dict")
            err = validate_business_item(item)
            if err:
                raise RemotePluginError(f"remote parse invalid item: {err}")
            out.append(item)
        return out


__all__ = ["RemoteParseInterceptor"]
