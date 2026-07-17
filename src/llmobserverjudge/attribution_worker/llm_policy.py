"""LLM-backed ReAct policy with Stub fallback (design §8.1)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from llmobserverjudge.attribution_worker.react import StubReActPolicy
from llmobserverjudge.attribution_worker.tools import BUILTIN_TOOLS, BuiltinToolRuntime
from llmobserverjudge.shared.openai_chat import (
    OpenAIChatClient,
    chat_client_from_env,
    extract_json_object,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Bad Case attribution ReAct agent.
Each turn choose ONE builtin tool and return JSON only:
{"thought":"...","action":"<tool_name>","args":{},"confidence":0.0-1.0}

Builtin tools:
- get_trace_summary, get_span_list, get_span_detail, get_tool_calls
- get_context, get_error_spans, get_eval_history, search_similar_cases
- get_prompt_version, submit_attribution

For submit_attribution args must include:
{"root_cause":"...","confidence":0.0-1.0,"evidence":[],"summary":"..."}

Prefer gathering evidence before submit. Do not invent tool names.
"""


class LLMReActPolicy:
    """Call OpenAI-compatible chat; invalid/failed → StubReActPolicy."""

    def __init__(
        self,
        chat: OpenAIChatClient,
        *,
        fallback: StubReActPolicy | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.chat = chat
        self.fallback = fallback or StubReActPolicy()
        self.temperature = temperature
        self.last_source: str = "llm"  # llm | stub_fallback

    async def next_step(
        self,
        *,
        rounds: int,
        tools: BuiltinToolRuntime,
        observations: list[dict[str, Any]],
    ) -> tuple[str, str, dict[str, Any], float]:
        try:
            content = await self.chat.chat(
                self._messages(rounds, tools, observations),
                temperature=self.temperature,
            )
            parsed = extract_json_object(content)
            if not parsed:
                raise ValueError("unparseable react step")
            action = str(parsed.get("action") or "").strip()
            if action not in BUILTIN_TOOLS:
                raise ValueError(f"unknown action: {action}")
            args = parsed.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            thought = str(parsed.get("thought") or "")
            conf = float(parsed.get("confidence") or 0.5)
            conf = max(0.0, min(1.0, conf))
            # Ensure submit has root_cause
            if action == "submit_attribution" and not args.get("root_cause"):
                raise ValueError("submit_attribution missing root_cause")
            self.last_source = "llm"
            return thought, action, args, conf
        except Exception as e:
            logger.warning("llm_react_fallback round=%s err=%s", rounds, e)
            self.last_source = "stub_fallback"
            return await self.fallback.next_step(
                rounds=rounds, tools=tools, observations=observations
            )

    def _messages(
        self,
        rounds: int,
        tools: BuiltinToolRuntime,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        ctx = tools.ctx
        history = observations[-6:]
        user = {
            "round": rounds,
            "eval_agent_id": ctx.eval_agent_id,
            "score_snapshot": ctx.score_snapshot,
            "trace_summary": ctx.trace_summary,
            "tools_called": tools.tools_called,
            "recent_observations": history,
        }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Context:\n" + json.dumps(user, ensure_ascii=False, default=str),
            },
        ]


def react_policy_from_env(
    *,
    env: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
    enforce_ssrf: bool | None = None,
) -> StubReActPolicy | LLMReActPolicy:
    """Prefer LLMJUDGE_REACT_*; fall back to LLMJUDGE_JUDGE_*; else Stub."""
    e = env if env is not None else dict(os.environ)
    chat = chat_client_from_env(
        env=e, prefix="LLMJUDGE_REACT", client=client, enforce_ssrf=enforce_ssrf
    )
    if chat is None:
        chat = chat_client_from_env(
            env=e, prefix="LLMJUDGE_JUDGE", client=client, enforce_ssrf=enforce_ssrf
        )
    if chat is None:
        return StubReActPolicy()
    return LLMReActPolicy(chat)
