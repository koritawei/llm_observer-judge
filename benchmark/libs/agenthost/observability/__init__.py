from contextlib import contextmanager
from contextvars import ContextVar

_bound_span: ContextVar[object | None] = ContextVar("trace_parent_span", default=None)


def _observation_span(observation: object | None) -> object | None:
    if observation is None:
        return None
    return getattr(observation, "_otel_span", None)


class TraceRequestScope:
    def __init__(self, *_args, **_kwargs):
        pass

    @contextmanager
    def bind(self):
        yield self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@contextmanager
def bind_trace_parent_observation(observation: object | None = None, *_args, **_kwargs):
    token = _bound_span.set(_observation_span(observation))
    try:
        yield _bound_span.get()
    finally:
        _bound_span.reset(token)


@contextmanager
def activate_trace_parent_context(*_args, **_kwargs):
    span = _bound_span.get()
    if span is None:
        yield None
        return
    from opentelemetry import context as otel_context_api
    from opentelemetry import trace as otel_trace_api

    token = otel_context_api.attach(otel_trace_api.set_span_in_context(span))
    try:
        yield span
    finally:
        otel_context_api.detach(token)


@contextmanager
def suppress_sensitive_io_observability():
    yield
