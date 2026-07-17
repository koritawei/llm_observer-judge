"""SDK @observe + context + flush to Gateway."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from llmobserverjudge.gateway.app import create_app
from llmobserverjudge.gateway.auth import ApiKeyAuthenticator
from llmobserverjudge.sdk import configure, flush, observe, shutdown
from llmobserverjudge.sdk.context import (
    extract_context,
    get_current_context,
    inject_context,
    should_sample,
)
from llmobserverjudge.sdk.runtime import get_buffer, reset_runtime_for_tests
from llmobserverjudge.shared.mq import InMemoryMQ


@pytest.fixture(autouse=True)
def _reset_sdk():
    reset_runtime_for_tests()
    yield
    reset_runtime_for_tests()


def test_should_sample_deterministic() -> None:
    tid = "0123456789abcdef0123456789abcdef"
    assert should_sample(tid, 1.0) is True
    assert should_sample(tid, 0.0) is False
    a = should_sample(tid, 0.5)
    b = should_sample(tid, 0.5)
    assert a == b


def test_observe_sync_pushes_span() -> None:
    @observe("unit_sync")
    def add(x: int, y: int) -> int:
        assert get_current_context() is not None
        return x + y

    assert add(1, 2) == 3
    assert get_current_context() is None
    assert len(get_buffer()) == 1
    span = get_buffer().drain(1)[0]
    assert span.name == "unit_sync"
    assert span.status == "OK"
    assert len(span.trace_id) == 32
    assert len(span.span_id) == 16


@pytest.mark.asyncio
async def test_observe_async_error_status() -> None:
    @observe()
    async def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await boom()
    span = get_buffer().drain(1)[0]
    assert span.status == "ERROR"
    assert "ValueError" in (span.error or "")


def test_parent_based_sampling() -> None:
    @observe("parent")
    def outer() -> str:
        @observe("child")
        def inner() -> str:
            return "ok"

        return inner()

    outer()
    spans = get_buffer().drain(10)
    assert len(spans) == 2
    parent = next(s for s in spans if s.name == "parent")
    child = next(s for s in spans if s.name == "child")
    assert child.trace_id == parent.trace_id
    assert child.parent_span_id == parent.span_id


def test_inject_extract_traceparent() -> None:
    @observe("svc")
    def work() -> dict[str, str]:
        return inject_context({})

    headers = work()
    assert "traceparent" in headers
    ctx = extract_context(headers)
    assert ctx is not None
    assert ctx.sampled is True


@pytest.mark.asyncio
async def test_flush_to_gateway() -> None:
    from llmobserverjudge.contracts.topics import Topic
    from llmobserverjudge.sdk.runtime import ensure_sender, get_buffer, push_span
    from llmobserverjudge.sdk.span_record import SpanRecord

    mq = InMemoryMQ()
    app = create_app(mq=mq, authenticator=ApiKeyAuthenticator({"k": "t1"}))
    configure(api_key="k", endpoint="http://test", flush_max_batch=50)

    sender = await ensure_sender()
    await sender.http.aclose()
    sender.http = AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    )
    sender._owns_http = True

    push_span(
        SpanRecord(
            name="flush_me",
            trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            started_at=datetime.now(UTC),
        )
    )
    n = await flush()
    assert n == 1
    assert get_buffer().metrics.sent_spans_total == 1
    assert await mq.depth(Topic.SPANS) == 1
    await shutdown()
