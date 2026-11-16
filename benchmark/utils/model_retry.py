"""Shared backoff retry for transient model errors (rate limit / 5xx / timeout)."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Protocol, Self, TypeVar

from benchmark.libs.agenthost.models.errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelTimeoutError,
    ProviderError,
)
from benchmark.libs.agenthost.observability.logger import get_logger
from anthropic import APIConnectionError as AnthropicAPIConnectionError

logger = get_logger("agent_eval.model_retry")

ResultT = TypeVar("ResultT")

_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 5.0
_BACKOFF_MAX_SECONDS = 40.0
_JITTER_RATIO = 0.2
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_SERVER_ERROR_MAX = 599
_RATE_LIMIT_MARKERS = (
    "qps",
    "qpm",
    "tpm",
    "rate limit",
    "too many requests",
    "insufficient_quota",
    "exceeded your current quota",
    "token-limit",
    "limit_burst_rate",
    "rate increased too quickly",
    "每秒钟请求次数",
    "每分钟请求次数",
    "每分钟消耗token",
    "请求频率",
)


class ChatModelDelegate(Protocol):
    """Minimal LangChain ChatModel surface required by the retry wrapper."""

    async def ainvoke(self, *args: object, **kwargs: object) -> object: ...

    def bind_tools(self, *args: object, **kwargs: object) -> Self: ...


def is_retryable_model_error(exc: BaseException) -> bool:
    """Recognize model errors that can be retried with backoff: rate limit, 5xx, timeout, ProviderError."""

    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (
        status == _HTTP_TOO_MANY_REQUESTS
        or _HTTP_SERVER_ERROR_MIN <= status <= _HTTP_SERVER_ERROR_MAX
    ):
        return True
    if isinstance(exc, AuthenticationError):
        return False
    if isinstance(exc, InvalidRequestError):
        message = str(exc).lower()
        return any(marker in message for marker in _RATE_LIMIT_MARKERS)
    if isinstance(exc, (ProviderError, ModelTimeoutError, AnthropicAPIConnectionError)):
        return True
    return False


def is_rate_limit_error(exc: BaseException) -> bool:
    """Compatibility alias; the meaning is now :func:`is_retryable_model_error`."""

    return is_retryable_model_error(exc)


async def call_with_rate_limit_retry(
    operation: Callable[[], Awaitable[ResultT]],
    *,
    operation_name: str,
) -> ResultT:
    """Exponential backoff for retryable model errors; other errors are re-raised as-is."""

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await operation()
        except Exception as exc:
            if not is_retryable_model_error(exc) or attempt >= _MAX_ATTEMPTS:
                raise
            base_delay = min(
                _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                _BACKOFF_MAX_SECONDS,
            )
            delay = base_delay + random.uniform(0.0, base_delay * _JITTER_RATIO)
            logger.warning(
                "model.retry operation=%s attempt=%d max_attempts=%d "
                "delay_s=%.2f error_type=%s status_code=%s",
                operation_name,
                attempt,
                _MAX_ATTEMPTS,
                delay,
                type(exc).__name__,
                getattr(exc, "status_code", None),
            )
            await asyncio.sleep(delay)

    raise AssertionError("unreachable")


class RetryingChatModel:
    """Inject transient-error backoff for agent-eval LangChain models in one place."""

    def __init__(self, delegate: ChatModelDelegate) -> None:
        self._delegate = delegate

    async def ainvoke(self, *args: object, **kwargs: object) -> object:
        """Call the underlying ``ainvoke`` and retry with backoff on rate limit / 5xx / timeout."""

        async def invoke() -> object:
            return await self._delegate.ainvoke(*args, **kwargs)

        return await call_with_rate_limit_retry(
            invoke,
            operation_name="llm.ainvoke",
        )

    def bind_tools(self, *args: object, **kwargs: object) -> Self:
        """After binding tools, still return a model that can back off."""

        return RetryingChatModel(self._delegate.bind_tools(*args, **kwargs))

    def __getattr__(self, name: str) -> object:
        """Proxy the rest of the LangChain model surface transparently to the underlying object."""

        return getattr(self._delegate, name)


__all__ = [
    "RetryingChatModel",
    "call_with_rate_limit_retry",
    "is_rate_limit_error",
    "is_retryable_model_error",
]
