"""Builtin ReAct tools for attribution (design §8.2)."""

from __future__ import annotations

from typing import Any

from llmobserverjudge.attribution_worker.context import AttributionContext

BUILTIN_TOOLS = frozenset(
    {
        "get_trace_summary",
        "get_span_list",
        "get_span_detail",
        "get_tool_calls",
        "get_context",
        "get_error_spans",
        "get_eval_history",
        "search_similar_cases",
        "get_prompt_version",
        "submit_attribution",
    }
)


class BuiltinToolRuntime:
    def __init__(self, ctx: AttributionContext) -> None:
        self.ctx = ctx
        self.tools_called: list[str] = []
        self.submitted: dict[str, Any] | None = None

    async def invoke(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        if name not in BUILTIN_TOOLS:
            raise ValueError(f"unknown builtin tool: {name}")
        self.tools_called.append(name)
        handler = getattr(self, f"_tool_{name}")
        return await handler(args)

    async def _tool_get_trace_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        return {"trace_summary": dict(self.ctx.trace_summary)}

    async def _tool_get_span_list(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        return {
            "spans": [
                {
                    "span_id": s.get("span_id"),
                    "name": s.get("name"),
                    "status": s.get("status"),
                    "kind": s.get("kind"),
                }
                for s in self.ctx.spans
            ]
        }

    async def _tool_get_span_detail(self, args: dict[str, Any]) -> dict[str, Any]:
        span_id = args.get("span_id")
        for s in self.ctx.spans:
            if s.get("span_id") == span_id:
                return {"span": dict(s)}
        return {"span": None, "error": "not_found"}

    async def _tool_get_tool_calls(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        calls = []
        for s in self.ctx.spans:
            tc = s.get("tool_calls")
            if tc:
                calls.append({"span_id": s.get("span_id"), "tool_calls": tc})
        summary_tc = self.ctx.trace_summary.get("tool_calls")
        if summary_tc:
            calls.append({"span_id": None, "tool_calls": summary_tc})
        return {"tool_calls": calls}

    async def _tool_get_context(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        return {
            "score_snapshot": dict(self.ctx.score_snapshot),
            "eval_agent_id": self.ctx.eval_agent_id,
        }

    async def _tool_get_error_spans(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        errors = [s for s in self.ctx.spans if str(s.get("status", "")).upper() == "ERROR"]
        if not errors and str(self.ctx.trace_summary.get("status", "")).upper() == "ERROR":
            errors = [{"from_summary": True, "status": "ERROR"}]
        return {"error_spans": errors}

    async def _tool_get_eval_history(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        return {"history": [], "note": "stub_empty"}

    async def _tool_search_similar_cases(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        label = self.ctx.score_snapshot.get("label") or "unknown"
        return {
            "similar": [
                {"case_id": "sim-stub-1", "label": label, "root_cause_hint": "PROMPT_MISMATCH"}
            ]
        }

    async def _tool_get_prompt_version(self, args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        return {
            "prompt_version_id": self.ctx.trace_summary.get("prompt_version_id"),
        }

    async def _tool_submit_attribution(self, args: dict[str, Any]) -> dict[str, Any]:
        self.submitted = {
            "root_cause": str(args.get("root_cause") or "UNKNOWN"),
            "confidence": float(args.get("confidence") or 0.0),
            "evidence": list(args.get("evidence") or []),
            "summary": str(args.get("summary") or ""),
        }
        return {"submitted": True, **self.submitted}
