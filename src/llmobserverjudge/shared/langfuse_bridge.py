"""Optional Langfuse bridge for Prompt resolve + span mirror (P2).

When LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are unset, all helpers no-op and
the in-house PromptService / Dashboard remain the source of truth.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client: Any | None = None
_tried = False


def langfuse_enabled() -> bool:
    return bool(
        (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
        and (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    )


def get_langfuse_client() -> Any | None:
    global _client, _tried
    if not langfuse_enabled():
        return None
    if _tried:
        return _client
    _tried = True
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"].strip(),
            secret_key=os.environ["LANGFUSE_SECRET_KEY"].strip(),
            host=(os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").strip(),
        )
        return _client
    except Exception:
        logger.warning("langfuse_client_unavailable", exc_info=True)
        _client = None
        return None


async def try_get_prompt_content(prompt_id: str) -> str | None:
    """Fetch prompt text from Langfuse by name; None → fall back to local store."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        # langfuse prompt API is sync in most versions
        prompt = client.get_prompt(prompt_id)
        if prompt is None:
            return None
        compile = getattr(prompt, "compile", None)
        if callable(compile):
            return str(compile())
        return str(getattr(prompt, "prompt", None) or getattr(prompt, "content", None) or "")
    except Exception:
        logger.debug("langfuse_get_prompt_failed id=%s", prompt_id, exc_info=True)
        return None


def try_export_span(record: Any) -> None:
    """Best-effort mirror of a local SpanRecord into Langfuse (non-blocking)."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        trace = client.trace(
            id=getattr(record, "trace_id", None),
            name=getattr(record, "name", None),
            metadata={
                "span_id": getattr(record, "span_id", None),
                "parent_span_id": getattr(record, "parent_span_id", None),
                "kind": getattr(record, "kind", None),
                "status": getattr(record, "status", None),
            },
        )
        if hasattr(trace, "span"):
            trace.span(
                id=getattr(record, "span_id", None),
                name=getattr(record, "name", None),
                input=getattr(record, "input", None),
                output=getattr(record, "output", None),
            )
    except Exception:
        logger.debug("langfuse_export_span_failed", exc_info=True)
