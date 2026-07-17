"""@observe decorator — OpenTelemetry spans + Gateway buffer export."""

from __future__ import annotations

import functools
import inspect
import json
from typing import Any, Callable, TypeVar

from opentelemetry.trace import Status, StatusCode

from llmobserverjudge.sdk.config import get_config
from llmobserverjudge.sdk.context import (
    SpanContext,
    reset_current_context,
    set_current_context,
)
from llmobserverjudge.sdk.otel import get_tracer, record_exception

F = TypeVar("F", bound=Callable[..., Any])


def _safe_attr(value: Any) -> str:
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = repr(value)
    return text[:4000]


def _sync_local_ctx(span: Any):
    """Keep legacy SpanContext ContextVar in sync for inject/extract helpers."""
    ctx = span.get_span_context()
    return set_current_context(
        SpanContext(
            trace_id=format(ctx.trace_id, "032x"),
            span_id=format(ctx.span_id, "016x"),
            sampled=bool(ctx.trace_flags.sampled),
        )
    )


def observe(name: str | None = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not get_config().enabled:
                    return await func(*args, **kwargs)
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    token = _sync_local_ctx(span)
                    span.set_attribute("llmj.kind", "CHAIN")
                    span.set_attribute("llmj.input", _safe_attr((args, kwargs)))
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("llmj.output", _safe_attr(result))
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as exc:
                        record_exception(span, exc)
                        raise
                    finally:
                        reset_current_context(token)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not get_config().enabled:
                return func(*args, **kwargs)
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                token = _sync_local_ctx(span)
                span.set_attribute("llmj.kind", "CHAIN")
                span.set_attribute("llmj.input", _safe_attr((args, kwargs)))
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("llmj.output", _safe_attr(result))
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    record_exception(span, exc)
                    raise
                finally:
                    reset_current_context(token)

        return sync_wrapper  # type: ignore[return-value]

    return decorator
