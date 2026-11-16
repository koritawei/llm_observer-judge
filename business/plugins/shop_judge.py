# coding: utf-8
"""Shop-business scoring plugin.

Score 1 when the output text contains the expected phrase.
"""

from __future__ import annotations

from typing import Any

from benchmark.kernel.types import Completed, PhaseContext


def _phrase(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or "")
    if value is None:
        return ""
    return str(value)


class ShopPhraseJudge:
    submits = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def execute(self, ctx: PhaseContext) -> Completed:
        expected = _phrase(ctx.case.get("expected"))
        output = _phrase(ctx.case.get("output"))
        matched = bool(expected) and expected in output
        score = 1.0 if matched else 0.0
        criteria = [
            item["id"]
            for item in ctx.snapshot.get("criteria_plan") or []
            if item.get("counts_toward_progress")
        ] or ["fused"]
        rows = [
            {
                "criterion_id": criterion_id,
                "score": score,
                "reason": "phrase" if matched else "missing_phrase",
                "status": "success",
            }
            for criterion_id in criteria
        ]
        return Completed({"rows": rows})
