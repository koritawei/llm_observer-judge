"""@observe decorator — sync/async (design §3.1.2)."""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, TypeVar

from llmobserverjudge.sdk.config import get_config
from llmobserverjudge.sdk.context import (
    SpanContext,
    generate_span_id,
    generate_trace_id,
    get_current_context,
    reset_current_context,
    set_current_context,
    should_sample,
)
from llmobserverjudge.sdk.runtime import push_span
from llmobserverjudge.sdk.span_record import SpanRecord

F = TypeVar("F", bound=Callable[..., Any])


def _capture_span(span_name: str) -> tuple[SpanRecord, Any]:
    cfg = get_config()
    parent = get_current_context()
    if parent:
        trace_id = parent.trace_id
        parent_span_id = parent.span_id
        sampled = parent.sampled
    else:
        trace_id = generate_trace_id()
        parent_span_id = None
        sampled = should_sample(trace_id, cfg.sample_rate)

    span_id = generate_span_id()
    record = SpanRecord(
        name=span_name,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        sampled=sampled,
        kind="CHAIN",
    )
    token = set_current_context(
        SpanContext(trace_id=trace_id, span_id=span_id, sampled=sampled)
    )
    return record, token


def observe(name: str | None = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not get_config().enabled:
                    return await func(*args, **kwargs)
                record, token = _capture_span(span_name)
                record.input = (args, kwargs)
                start = time.perf_counter_ns()
                try:
                    result = await func(*args, **kwargs)
                    record.output = result
                    record.status = "OK"
                    return result
                except Exception as exc:
                    record.status = "ERROR"
                    record.error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    record.duration_ms = int((time.perf_counter_ns() - start) / 1_000_000)
                    push_span(record)
                    reset_current_context(token)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not get_config().enabled:
                return func(*args, **kwargs)
            record, token = _capture_span(span_name)
            record.input = (args, kwargs)
            start = time.perf_counter_ns()
            try:
                result = func(*args, **kwargs)
                record.output = result
                record.status = "OK"
                return result
            except Exception as exc:
                record.status = "ERROR"
                record.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record.duration_ms = int((time.perf_counter_ns() - start) / 1_000_000)
                push_span(record)
                reset_current_context(token)

        return sync_wrapper  # type: ignore[return-value]

    return decorator
