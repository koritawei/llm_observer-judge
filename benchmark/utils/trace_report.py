"""agent-eval → Langfuse **report** helpers (write path).

One HTTP eval should produce **one** Langfuse root. Call
:func:`eval_report_root` at ``EvaluatorService.evaluate`` entry, then nest
children with :func:`start_report_child_span` (resolve, tools, …).

Credentials: ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` via
:func:`get_report_client` (same singleton as ``LangfuseMiddleware``).
Never construct a fresh ``Langfuse(...)`` here — that opens a second root.

Always pass ``public_key=LANGFUSE_PUBLIC_KEY`` into ``get_client``. After
dataset eval registers a second SDK client (evaluated project), bare
``get_client()`` returns a tracing-disabled stub and drops report spans.
"""

import json
import os
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from benchmark.libs.agenthost.observability import TraceRequestScope, bind_trace_parent_observation
from benchmark.libs.agenthost.observability.logger import get_logger

if TYPE_CHECKING:
    from langfuse import Langfuse

    from benchmark.settings.setting import LocalSpanLogSettings

logger = get_logger("agent_eval.trace_report")


def _local_span_log_settings() -> "LocalSpanLogSettings":
    # The config package imports score helpers; read them only after the report module finishes initializing, to avoid a cycle.
    from benchmark.settings.setting import LocalSpanLogSettings  # noqa: PLC0415

    return LocalSpanLogSettings()


def _local_span_log_enabled() -> bool:
    return _local_span_log_settings().enabled


def _local_span_log_max_chars() -> int:
    return _local_span_log_settings().max_chars


@dataclass
class ReportTraceCapture:
    """HTTP response metadata, separate from the shared evaluator verdict."""

    trace_id: str | None = None
    _root_seen: bool = False


_report_trace_capture: ContextVar[ReportTraceCapture | None] = ContextVar(
    "agent_eval_report_trace_capture", default=None
)


@contextmanager
def capture_report_trace_id() -> Iterator[ReportTraceCapture]:
    """Capture this request's first report root; restore scope even on failure."""
    capture = ReportTraceCapture()
    token = _report_trace_capture.set(capture)
    try:
        yield capture
    finally:
        _report_trace_capture.reset(token)


def _safe_json_preview(value: object) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    max_chars = _local_span_log_max_chars()
    if len(text) > max_chars:
        return text[:max_chars] + "…<truncated>"
    return text


def _log_local_span_event(*, event: str, name: str, payload: object | None = None) -> None:
    if not _local_span_log_enabled():
        return
    logger.info(
        "trace_report.local_span event=%s name=%s payload=%s",
        event,
        name,
        _safe_json_preview(payload or {}),
    )


class _LocalLoggingSpan:
    """Proxy span that mirrors update/end payloads to local logs when enabled."""

    def __init__(
        self, inner: object | None, *, name: str, input_value: object | None = None
    ) -> None:
        self._inner = inner
        self._name = name
        _log_local_span_event(event="start", name=name, payload={"input": input_value or {}})

    def __getattr__(self, name: str) -> object:
        if self._inner is None:
            raise AttributeError(name)
        return getattr(self._inner, name)

    def update(self, **kwargs: object) -> object | None:
        _log_local_span_event(event="update", name=self._name, payload=kwargs)
        if self._inner is None:
            return None
        return self._inner.update(**kwargs)

    def end(self) -> object | None:
        _log_local_span_event(event="end", name=self._name)
        if self._inner is None:
            return None
        return self._inner.end()


class ReportSpan(Protocol):
    """Minimal Langfuse span surface used by report helpers."""

    def update(self, **kwargs: object) -> object: ...

    def end(self) -> object: ...


def get_report_client() -> "Langfuse":
    """Return the agent-eval **report** Langfuse client (write path).

    Pins ``LANGFUSE_PUBLIC_KEY`` so multi-project processes (dataset eval +
    Middleware) do not hit the SDK's cross-project leakage guard.
    """

    from langfuse import get_client  # noqa: PLC0415

    public_key = os.environ.get("TRACE_PUBLIC_KEY") or None
    secret_key = os.environ.get("TRACE_SECRET_KEY")
    if public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    if secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    tracing_env = os.environ.get("TRACE_TRACING_ENVIRONMENT")
    if tracing_env:
        os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = tracing_env
    timeout = os.environ.get("TRACE_TIMEOUT")
    if timeout:
        os.environ["LANGFUSE_TIMEOUT"] = timeout
    return get_client(public_key=public_key)


def get_report_trace_id(root: object | None) -> str | None:
    """Return a valid ID only for a recording report root.

    Keep SDK-specific inspection here: a disabled SDK's NoOpTracer may inherit
    a valid parent ID, so an ambient get_current_trace_id() alone is insufficient.
    SDK lookup failures must not interrupt evaluation.
    """
    if root is None:
        return None
    try:
        candidate = getattr(root, "trace_id", None)
        span = getattr(root, "_otel_span", None)
        if (
            isinstance(candidate, str)
            and re.fullmatch(r"[0-9a-f]{32}", candidate)
            and candidate != "0" * 32
            and span is not None
            and span.is_recording() is True
        ):
            return candidate
    except Exception:
        logger.debug("trace_report.trace_id_unavailable", exc_info=True)
    return None


@contextmanager
def with_current_node_otel_parent() -> Iterator[None]:
    """Attach the active LangGraph node CHAIN as OTel current parent.

    Langfuse CallbackHandler node spans are often **not** the OTel current span.
    ``LangfuseMiddleware`` stores them in a request ContextVar; attach that span
    before ``start_observation`` so leaf tool/report children nest under
    ``preprocess`` / ``judge_*`` instead of becoming root siblings.
    """

    parent: Any | None = None
    try:
        from benchmark.libs.agenthost.observability.middleware.trace_middleware import (  # noqa: PLC0415
            TraceMiddleware,
        )

        parent = TraceMiddleware.get_current_node_otel_span()
    except Exception:
        logger.debug("trace_report.node_parent_lookup_failed", exc_info=True)
        parent = None

    if parent is None:
        yield
        return

    try:
        from opentelemetry import context as otel_context_api  # noqa: PLC0415
        from opentelemetry import trace as otel_trace_api  # noqa: PLC0415
    except Exception:
        logger.debug("trace_report.otel_import_failed", exc_info=True)
        yield
        return

    token = otel_context_api.attach(otel_trace_api.set_span_in_context(parent))
    try:
        yield
    finally:
        try:
            otel_context_api.detach(token)
        except Exception:
            logger.debug("trace_report.otel_detach_failed", exc_info=True)


@contextmanager
def _exported_parent_bindings(observation: object | None) -> Iterator[None]:
    """Fresh request scope and exported parent for one evaluate()."""

    if isinstance(observation, _LocalLoggingSpan):
        observation = observation._inner
    with ExitStack() as stack:
        try:
            stack.enter_context(TraceRequestScope().bind())
        except Exception:
            logger.warning("trace_report.request_scope_failed", exc_info=True)
        if observation is not None:
            try:
                stack.enter_context(bind_trace_parent_observation(observation))
            except Exception:
                logger.debug("trace_report.parent_bind_failed", exc_info=True)
        yield


@contextmanager
def eval_report_root(
    *,
    name: str,
    input_value: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Open the single agent-eval report root for one ``evaluate()`` call.

    Yields the observation object, or ``None`` when reporting is unavailable.
    Graph/LLM callbacks are bound to this exported root so they nest here
    instead of becoming a separate incomplete trace.
    """

    try:
        client = get_report_client()
        cm = client.start_as_current_observation(
            name=name,
            as_type="span",
            input=input_value or {},
            metadata=metadata or {},
        )
    except Exception:
        logger.debug("trace_report.root_open_failed", exc_info=True)
        yield _LocalLoggingSpan(
            None, name=name, input_value={"input": input_value or {}, "metadata": metadata or {}}
        )
        return

    try:
        observation = cm.__enter__()
        observation = _LocalLoggingSpan(
            observation,
            name=name,
            input_value={"input": input_value or {}, "metadata": metadata or {}},
        )
    except Exception:
        logger.debug("trace_report.root_enter_failed", exc_info=True)
        yield _LocalLoggingSpan(
            None, name=name, input_value={"input": input_value or {}, "metadata": metadata or {}}
        )
        return

    capture = _report_trace_capture.get()
    if capture is not None and not capture._root_seen:
        capture._root_seen = True
        capture.trace_id = get_report_trace_id(observation)

    try:
        with _exported_parent_bindings(observation):
            yield observation
    except BaseException as exc:
        try:
            cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            logger.debug("trace_report.root_exit_failed", exc_info=True)
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.debug("trace_report.root_exit_failed", exc_info=True)


def start_report_child_span(
    *,
    name: str,
    input_value: dict[str, Any] | None = None,
) -> ReportSpan | None:
    """Start a child span under the active report root, or ``None``.

    Skips when there is no ``get_current_trace_id()`` so we never create an
    orphan second root (e.g. unit tests calling ``before_eval`` alone).
    """

    try:
        client = get_report_client()
        get_trace_id = getattr(client, "get_current_trace_id", None)
        if not callable(get_trace_id) or not get_trace_id():
            return _LocalLoggingSpan(None, name=name, input_value=input_value or {})
        with with_current_node_otel_parent():
            span = client.start_observation(
                name=name,
                as_type="span",
                input=input_value or {},
            )
        return _LocalLoggingSpan(span, name=name, input_value=input_value or {})
    except Exception:
        logger.debug("trace_report.child_start_failed name=%s", name, exc_info=True)
        return _LocalLoggingSpan(None, name=name, input_value=input_value or {})


@contextmanager
def report_child_scope(
    *,
    name: str,
    input_value: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Open a **container** child observation and set it as OTel current.

    Use for nested work (triage/attribution LangGraph, model, tools) that must
    parent under this span. Unlike :func:`start_report_child_span` (leaf spans
    that you ``finish`` manually), this uses ``start_as_current_observation``.

    Independent of sync vs async scheduling: callers must already be inside
    :func:`eval_report_root` (or another current observation).
    """

    try:
        client = get_report_client()
        get_trace_id = getattr(client, "get_current_trace_id", None)
        if not callable(get_trace_id) or not get_trace_id():
            yield _LocalLoggingSpan(None, name=name, input_value=input_value or {})
            return
        cm = client.start_as_current_observation(
            name=name,
            as_type="span",
            input=input_value or {},
        )
        observation = cm.__enter__()
        observation = _LocalLoggingSpan(observation, name=name, input_value=input_value or {})
    except Exception:
        logger.debug("trace_report.child_scope_open_failed name=%s", name, exc_info=True)
        yield _LocalLoggingSpan(None, name=name, input_value=input_value or {})
        return

    try:
        with ExitStack() as stack:
            if observation._inner is not None:
                try:
                    stack.enter_context(bind_trace_parent_observation(observation._inner))
                except Exception:
                    logger.debug(
                        "trace_report.child_scope_parent_bind_failed name=%s",
                        name,
                        exc_info=True,
                    )
            yield observation
    except BaseException as exc:
        try:
            cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            logger.debug(
                "trace_report.child_scope_exit_failed name=%s",
                name,
                exc_info=True,
            )
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.debug(
                "trace_report.child_scope_exit_failed name=%s",
                name,
                exc_info=True,
            )


def finish_report_child_span(
    span: ReportSpan | None,
    *,
    output: dict[str, Any] | None = None,
    exc: BaseException | None = None,
) -> None:
    """Update and end a child span; no-op when ``span`` is ``None``."""

    if span is None:
        return
    try:
        if exc is not None:
            span.update(level="ERROR", status_message=str(exc))
        elif output is not None:
            span.update(output=output)
    except Exception:
        logger.debug("trace_report.child_update_failed", exc_info=True)
    finally:
        try:
            end = getattr(span, "end", None)
            if callable(end):
                end()
        except Exception:
            logger.debug("trace_report.child_end_failed", exc_info=True)


__all__ = [
    "ReportSpan",
    "ReportTraceCapture",
    "capture_report_trace_id",
    "eval_report_root",
    "finish_report_child_span",
    "get_report_client",
    "get_report_trace_id",
    "report_child_scope",
    "start_report_child_span",
    "with_current_node_otel_parent",
]
