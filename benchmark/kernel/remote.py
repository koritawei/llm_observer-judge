"""Remote strategy adapter. Endpoints come from the frozen snapshot, not from the request."""

from __future__ import annotations

from typing import Any

import httpx

from benchmark.kernel.types import PhaseContext, Retryable, TerminalFailure


class RemoteAdapter:
    submits = True

    def __init__(self, frozen: dict[str, Any]):
        self.frozen = frozen

    async def execute(self, ctx: PhaseContext) -> Any:
        url = str(self.frozen.get("base_url") or "")
        envelope = {
            "plugin_kind": self.frozen.get("kind"),
            "plugin_name": self.frozen.get("plugin_id"),
            "action": "execute",
            "artifact_hash": self.frozen.get("artifact_hash"),
            "payload": {"item_id": ctx.item_id, "request": ctx.request or {}},
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json=envelope)
        except httpx.HTTPError as exc:
            if self.frozen.get("fallback_local"):
                return Retryable("remote_unreachable")
            return TerminalFailure(f"remote_unreachable:{exc.__class__.__name__}")
        if response.status_code >= 500:
            return Retryable(f"remote_status:{response.status_code}")
        if response.status_code >= 400:
            return TerminalFailure(f"remote_status:{response.status_code}")
        body = response.json()
        if body.get("artifact_hash") != self.frozen.get("artifact_hash"):
            return TerminalFailure("remote_artifact_mismatch")
        return body
