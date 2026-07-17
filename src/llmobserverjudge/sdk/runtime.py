"""SDK runtime singleton — buffer + sender lifecycle."""

from __future__ import annotations

from llmobserverjudge.sdk.buffer import BoundedRingBuffer
from llmobserverjudge.sdk.config import SDKConfig, configure, get_config
from llmobserverjudge.sdk.sender import AsyncSender
from llmobserverjudge.sdk.span_record import SpanRecord

_buffer: BoundedRingBuffer | None = None
_sender: AsyncSender | None = None


def get_buffer() -> BoundedRingBuffer:
    global _buffer
    if _buffer is None:
        _buffer = BoundedRingBuffer(capacity=get_config().buffer_size)
    return _buffer


def get_sender() -> AsyncSender | None:
    return _sender


async def ensure_sender() -> AsyncSender:
    global _sender, _buffer
    cfg = get_config()
    if _buffer is None or _buffer.metrics.buffer_size != cfg.buffer_size:
        _buffer = BoundedRingBuffer(capacity=cfg.buffer_size)
    if _sender is None:
        _sender = AsyncSender(cfg, _buffer)
        _sender.start_background()
    return _sender


def push_span(span: SpanRecord) -> None:
    if not get_config().enabled:
        return
    get_buffer().push(span)


async def flush() -> int:
    sender = await ensure_sender()
    return await sender.flush()


async def shutdown() -> None:
    global _sender
    if _sender is not None:
        await _sender.shutdown()
        _sender = None


def reset_runtime_for_tests() -> None:
    """Test helper — clear singleton state."""
    global _buffer, _sender
    _buffer = None
    _sender = None
    configure(
        api_key="",
        endpoint="http://127.0.0.1:8080",
        buffer_size=4096,
        flush_interval_s=10.0,
        flush_max_batch=50,
        sample_rate=1.0,
        enabled=True,
    )
