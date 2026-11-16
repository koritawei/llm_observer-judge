from contextlib import contextmanager
from contextvars import ContextVar, Token

_request_context: ContextVar[dict] = ContextVar("benchmark_request_context", default={})


def init_request_context(*values: dict, **kwargs) -> Token:
    merged: dict = {}
    for item in values:
        if isinstance(item, dict):
            merged.update(item)
    merged.update(kwargs)
    return _request_context.set(merged)


@contextmanager
def request_context(**values):
    token = init_request_context(**values)
    try:
        yield dict(_request_context.get())
    finally:
        clear_request_context(token)


def update_request_context(**values) -> None:
    current = dict(_request_context.get())
    current.update(values)
    _request_context.set(current)


def clear_request_context(token: Token | None = None) -> None:
    if token is None:
        _request_context.set({})
        return
    _request_context.reset(token)
