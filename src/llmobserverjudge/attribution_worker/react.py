"""ReAct attributor with stub policy + LangGraph loop (design §8.1)."""

from __future__ import annotations

from typing import Any

from llmobserverjudge.attribution_worker.checkpoint import InMemoryCheckpointStore
from llmobserverjudge.attribution_worker.context import AttributionContext
from llmobserverjudge.attribution_worker.tools import BuiltinToolRuntime
from llmobserverjudge.contracts.models import AttributionJob, AttributionResult


class StubReActPolicy:
    """Deterministic tool sequence — no LLM (STEP-07)."""

    async def next_step(
        self, *, rounds: int, tools: BuiltinToolRuntime, observations: list[dict[str, Any]]
    ) -> tuple[str, str, dict[str, Any], float]:
        """Return (thought, action, args, llm_self_confidence)."""
        summary = tools.ctx.trace_summary
        status = str(summary.get("status") or "OK").upper()
        score = tools.ctx.score_snapshot.get("score")
        tags = summary.get("tags") or []

        if rounds == 1:
            return (
                "Inspect trace summary for status and tags",
                "get_trace_summary",
                {},
                0.45,
            )
        if rounds == 2:
            if status == "ERROR" or any(
                str(s.get("status", "")).upper() == "ERROR" for s in tools.ctx.spans
            ):
                return (
                    "Look for error spans as root evidence",
                    "get_error_spans",
                    {},
                    0.65,
                )
            return (
                "Inspect tool calls for failure signals",
                "get_tool_calls",
                {},
                0.60,
            )
        if rounds == 3:
            return (
                "Search similar historical bad cases",
                "search_similar_cases",
                {},
                0.80,
            )

        # Submit with inferred root cause
        root = _infer_root_cause(status=status, tags=tags, score=score, observations=observations)
        evidence = _collect_evidence(tools, observations)
        return (
            f"Conclude root_cause={root}",
            "submit_attribution",
            {
                "root_cause": root,
                "confidence": 0.9,
                "evidence": evidence,
                "summary": f"stub react for {tools.ctx.eval_agent_id}",
            },
            0.92,
        )


class ReActAttributor:
    EARLY_STOP_CONFIDENCE = 0.85
    EARLY_STOP_MIN_ROUNDS = 3
    MAX_ROUNDS = 20
    CHECKPOINT_EVERY = 3

    def __init__(
        self,
        *,
        policy: Any | None = None,
        checkpoint: InMemoryCheckpointStore | None = None,
        max_rounds: int | None = None,
    ) -> None:
        self.policy = policy or StubReActPolicy()
        self.checkpoint = checkpoint or InMemoryCheckpointStore()
        self.max_rounds = max_rounds or self.MAX_ROUNDS

    async def analyze(self, job: AttributionJob, ctx: AttributionContext) -> AttributionResult:
        from llmobserverjudge.attribution_worker.langgraph_react import run_react_langgraph

        return await run_react_langgraph(attributor=self, job=job, ctx=ctx)

    def _estimate_confidence(
        self,
        *,
        llm_self: float,
        conclusions: list[str],
        tools_called: list[str],
        observations: list[dict[str, Any]],
    ) -> float:
        llm_self = max(0.0, min(1.0, llm_self))
        consistency = _consistency(conclusions)
        rule_check = _rule_check(tools_called, observations, conclusions)
        return round(0.4 * llm_self + 0.3 * consistency + 0.3 * rule_check, 4)


def _consistency(conclusions: list[str]) -> float:
    if len(conclusions) < 2:
        return 0.5
    a, b = conclusions[-2], conclusions[-1]
    if a == b:
        return 1.0
    # rough normalized disagreement
    return 0.2


def _rule_check(
    tools_called: list[str],
    observations: list[dict[str, Any]],
    conclusions: list[str],
) -> float:
    score = 0.0
    if any(t in tools_called for t in ("get_span_list", "get_trace_summary", "get_error_spans")):
        score += 0.34
    if "get_tool_calls" in tools_called or any(
        "tool" in str(o.get("action", "")) for o in observations
    ):
        score += 0.33
    if conclusions and conclusions[-1] not in ("", "UNKNOWN"):
        score += 0.33
    return min(1.0, score)


def _infer_root_cause(
    *,
    status: str,
    tags: list[Any] | Any,
    score: Any,
    observations: list[dict[str, Any]],
) -> str:
    if isinstance(tags, str):
        tags = [tags]
    tags = list(tags or [])
    if status == "ERROR":
        return "RUNTIME_ERROR"
    if any(t in tags for t in ("toxicity", "jailbreak", "pii_leak", "safety")):
        return "SAFETY_VIOLATION"
    if score is not None:
        try:
            if float(score) < 0.5:
                return "LOW_QUALITY_OUTPUT"
        except (TypeError, ValueError):
            pass
    for o in reversed(observations):
        obs = o.get("observation") or {}
        similar = obs.get("similar") or []
        if similar and similar[0].get("root_cause_hint"):
            return str(similar[0]["root_cause_hint"])
    return "PROMPT_MISMATCH"


def _collect_evidence(
    tools: BuiltinToolRuntime, observations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = [
        {"type": "score_snapshot", "data": dict(tools.ctx.score_snapshot)},
        {"type": "trace_status", "data": tools.ctx.trace_summary.get("status")},
    ]
    for o in observations[-3:]:
        evidence.append(
            {
                "type": "react_step",
                "round": o.get("round"),
                "action": o.get("action"),
            }
        )
    return evidence
