"""Example user plugin: record an attribution decision without touching the judge row."""

from __future__ import annotations

from typing import Any

from benchmark.kernel.types import PhaseContext


class RecordingAttribution:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def attribute(self, ctx: PhaseContext) -> dict[str, Any]:
        if (ctx.case.get("expected") or {}).get("attribution") == "fail":
            return {"status": "failed", "detail": {"reason": "unresolved", "judge_status_unchanged": True}}
        return {"status": "success", "detail": {"reason": "recorded"}}
