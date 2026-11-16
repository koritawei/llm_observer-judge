"""Kernel default plugins. Business evaluators live under business/."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from benchmark.kernel.types import Completed, PhaseContext, SubmitRequest


def _texts_match(output: Any, expected: Any) -> bool:
    if isinstance(expected, dict) and "text" in expected and isinstance(output, dict):
        return output.get("text") == expected.get("text")
    return output == expected


class InlineSource:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}


class DefaultLimits:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}


class PassthroughBuilder:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def build(self, ctx: PhaseContext) -> dict[str, Any]:
        return {"item_id": ctx.item_id, "phase_id": ctx.phase["phase_id"]}


class OutputPassthrough:
    submits = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def execute(self, ctx: PhaseContext) -> Completed:
        return Completed({"output": ctx.case.get("output")})


class ExactMatchJudge:
    submits = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def execute(self, ctx: PhaseContext) -> Completed:
        matched = _texts_match(ctx.case.get("output"), ctx.case.get("expected"))
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
                "reason": "match" if matched else "mismatch",
                "status": "success",
            }
            for criterion_id in criteria
        ]
        return Completed({"rows": rows})


class AsyncSubmitJudge:
    submits = True

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def execute(self, ctx: PhaseContext) -> SubmitRequest:
        return SubmitRequest({"kind": "external_score", "item_id": ctx.item_id})


class BudgetWindow:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}


class LeasePair:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}


class RubricTrigger:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def should_trigger(self, row: dict[str, Any], rubric: dict[str, Any]) -> bool:
        if row.get("status") != "success" or row.get("score") is None:
            return False
        threshold = self.config.get("threshold", rubric.get("attribution_threshold"))
        if threshold is None:
            return False
        return float(row["score"]) < float(threshold)


class AttributionNone:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def attribute(self, ctx: PhaseContext) -> dict[str, Any]:
        return {"status": "success", "detail": {"skipped": True}}


class DispatchDisabled:
    blocks_completion = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.blocks_completion = bool(self.config.get("blocks_completion", False))


class WeightedMean:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def aggregate(self, rows: list[dict[str, Any]], rubric: dict[str, Any]) -> float | None:
        weights = {
            item["id"]: float(item.get("weight", 1))
            for item in rubric.get("criteria") or []
        }
        total = 0.0
        weight_sum = 0.0
        for row in rows:
            if row.get("status") != "success" or row.get("score") is None:
                continue
            weight = weights.get(row["criterion_id"], 1.0)
            total += float(row["score"]) * weight
            weight_sum += weight
        if weight_sum == 0:
            return None
        return total / weight_sum


class AlwaysTrue:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def ready(self, view: dict[str, Any]) -> bool:
        return True


class StagesReady:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def ready(self, view: dict[str, Any]) -> bool:
        return bool(view.get("input_sealed"))


class MetricsReport:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def render(self, view: dict[str, Any]) -> dict[str, Any]:
        return {"summary": view.get("metrics") or {}}


class LocalOnlyAuth:
    submits_external = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def verify(self, body: bytes, headers: dict[str, str]) -> bool:
        return False


class HmacAuth:
    submits_external = True

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def verify(self, body: bytes, headers: dict[str, str]) -> bool:
        ref = str(self.config.get("secret_ref") or "callback")
        secret = os.environ.get(f"EVALBENCH_SECRET_{ref.upper()}", "")
        if not secret:
            return False
        provided = headers.get("x-signature") or headers.get("X-Signature") or ""
        digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(digest, provided)


class StreamingSampler:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def decide(self, message: dict[str, Any]) -> dict[str, Any]:
        name = str(message.get("agent_name") or "")
        required = self.config.get("agent_name")
        if required and name != required:
            return {"accept": False, "sample": False, "status": "filtered"}
        percent = int(self.config.get("sample_percent", 100))
        bucket = int(hashlib.sha256(str(message.get("trace_id", "")).encode()).hexdigest()[:8], 16) % 100
        if bucket >= percent:
            return {"accept": True, "sample": False, "status": "sampled_out"}
        return {"accept": True, "sample": True, "status": "evaluating"}
