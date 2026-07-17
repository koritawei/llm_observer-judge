"""OpenTelemetry thin layer for the SDK (replaces hand-rolled TraceContext-only path)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

_PROVIDER_SET = False


class GatewaySpanExporter(SpanExporter):
    """Export finished OTel spans into the existing SDK ring buffer → Gateway ingest."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        from llmobserverjudge.sdk.runtime import push_span
        from llmobserverjudge.sdk.span_record import SpanRecord

        for span in spans:
            try:
                ctx = span.get_span_context()
                parent = span.parent
                attrs = dict(span.attributes or {})
                status = "OK" if span.status.status_code != StatusCode.ERROR else "ERROR"
                duration_ms = 0
                if span.end_time is not None and span.start_time is not None:
                    duration_ms = int((span.end_time - span.start_time) / 1_000_000)
                record = SpanRecord(
                    name=span.name,
                    trace_id=format(ctx.trace_id, "032x"),
                    span_id=format(ctx.span_id, "016x"),
                    parent_span_id=(
                        format(parent.span_id, "016x")
                        if parent is not None and parent.span_id
                        else None
                    ),
                    sampled=bool(ctx.trace_flags.sampled),
                    kind=str(attrs.get("llmj.kind") or "CHAIN"),
                    status=status,
                    error=(str(span.status.description) if span.status.description else None),
                    duration_ms=duration_ms,
                    input=attrs.get("llmj.input"),
                    output=attrs.get("llmj.output"),
                )
                push_span(record)
                try:
                    from llmobserverjudge.shared.langfuse_bridge import try_export_span

                    try_export_span(record)
                except Exception:
                    pass
            except Exception:
                logger.debug("otel_export_span_failed", exc_info=True)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def ensure_tracer_provider() -> TracerProvider:
    global _PROVIDER_SET
    provider = trace.get_tracer_provider()
    if _PROVIDER_SET and isinstance(provider, TracerProvider):
        return provider
    resource = Resource.create({"service.name": "llmobserverjudge-sdk"})
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(SimpleSpanProcessor(GatewaySpanExporter()))
    trace.set_tracer_provider(tp)
    _PROVIDER_SET = True
    return tp


def get_tracer(name: str = "llmobserverjudge.sdk") -> trace.Tracer:
    ensure_tracer_provider()
    return trace.get_tracer(name)


def record_exception(span: Any, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
