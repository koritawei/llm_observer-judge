"""W3C TraceContext helpers (SDD-IG-10 §3.2)."""

from __future__ import annotations

import hashlib
import secrets
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class SpanContext:
    trace_id: str
    span_id: str
    sampled: bool


_current: ContextVar[SpanContext | None] = ContextVar("llmj_span_ctx", default=None)


def generate_trace_id() -> str:
    return secrets.token_hex(16)


def generate_span_id() -> str:
    return secrets.token_hex(8)


def should_sample(trace_id: str, rate: float = 1.0) -> bool:
    """TraceIDRatioBased with MD5 — deterministic across processes."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.md5(trace_id.encode()).hexdigest()
    return int(digest, 16) % 10000 < int(rate * 10000)


def get_current_context() -> SpanContext | None:
    return _current.get()


def set_current_context(ctx: SpanContext | None):
    return _current.set(ctx)


def reset_current_context(token) -> None:
    _current.reset(token)


def inject_context(headers: dict[str, str]) -> dict[str, str]:
    ctx = get_current_context()
    if ctx and ctx.sampled:
        headers["traceparent"] = f"00-{ctx.trace_id}-{ctx.span_id}-01"
    return headers


def extract_context(headers: dict[str, str]) -> SpanContext | None:
    tp = headers.get("traceparent") or headers.get("Traceparent") or ""
    parts = tp.split("-")
    if len(parts) != 4:
        return None
    return SpanContext(
        trace_id=parts[1],
        span_id=parts[2],
        sampled=(parts[3] == "01"),
    )
