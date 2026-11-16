"""Replay ports. Scheduling stays in the kernel; this module only rebuilds a turn."""

from __future__ import annotations

from typing import Any

from benchmark.kernel.agentrun import replay_turn
from benchmark.kernel.types import Completed, PhaseContext, TerminalFailure


class ParentRunCases:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}


class ReplayCaseLimits:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}


class EvalReplayRequest:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def build(self, ctx: PhaseContext) -> dict[str, Any]:
        snapshot = ctx.snapshot
        return {
            "agent_id": snapshot.get("replay_agent_id"),
            "subscription_id": snapshot.get("replay_subscription_id"),
            "agent_name": snapshot.get("replay_agent_name"),
            "header_overrides": snapshot.get("replay_header_overrides") or {},
            "body_overrides": snapshot.get("replay_body_overrides") or {},
            "dispatch_domain_id": (snapshot.get("dispatch_domain") or {}).get("domain_id"),
        }


class EvalReplayExecute:
    """Regenerate a trajectory on the in-process graph. Credentials stay outside it."""

    submits = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def execute(self, ctx: PhaseContext) -> Completed | TerminalFailure:
        if "user_bearer_checkout" in (ctx.snapshot.get("credential_requirements") or []):
            if not ctx.credential_ready:
                return TerminalFailure("auth_expired")
        answer = replay_turn(ctx.case.get("input"))
        output = {
            "text": answer,
            "replayed": True,
            "agent_id": ctx.snapshot.get("replay_agent_id"),
        }
        overrides = ctx.snapshot.get("replay_body_overrides") or {}
        if isinstance(overrides, dict):
            output.update(
                {
                    key: value
                    for key, value in overrides.items()
                    if key not in {"text", "authorization", "token", "bearer"}
                }
            )
        return Completed({"output": output, "audit": {"request": ctx.request or {}, "runtime": "langgraph"}})


class ReuseParentOutput:
    submits = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def execute(self, ctx: PhaseContext) -> Completed | TerminalFailure:
        if ctx.case.get("output") is None:
            return TerminalFailure("judge_only_missing_parent_output")
        return Completed({"output": ctx.case.get("output"), "audit": {"reused": True}})
