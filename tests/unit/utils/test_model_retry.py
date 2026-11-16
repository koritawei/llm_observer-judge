"""Default transient-error backoff for agent-eval models."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx  # noqa: TID251 -- construct native Anthropic SDK exceptions in tests
import pytest
from benchmark.libs.agenthost.models.errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelTimeoutError,
    ProviderError,
)
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APITimeoutError as AnthropicAPITimeoutError,
)
from anthropic import (
    AuthenticationError as AnthropicAuthenticationError,
)
from anthropic import (
    InternalServerError as AnthropicInternalServerError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from benchmark.utils.model_retry import RetryingChatModel, is_retryable_model_error


@pytest.mark.asyncio
async def test_retrying_chat_model_retries_gateway_qpm_error(monkeypatch) -> None:
    attempts = 0
    delays: list[float] = []

    class _RateLimitedThenSuccess:
        async def ainvoke(self, *_args: object, **_kwargs: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise InvalidRequestError("Error code: 400 - 每分钟请求次数超过当前AK申请QPM值:60")
            return "ok"

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    model = RetryingChatModel(_RateLimitedThenSuccess())

    assert await model.ainvoke(["prompt"]) == "ok"
    assert attempts == 2
    assert len(delays) == 1
    assert 5.0 <= delays[0] <= 6.0


@pytest.mark.asyncio
async def test_retrying_chat_model_retries_gateway_tpm_error(monkeypatch) -> None:
    attempts = 0
    delays: list[float] = []

    class _TpmLimitedThenSuccess:
        async def ainvoke(self, *_args: object, **_kwargs: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise InvalidRequestError(
                    "Error code: 400 - 每分钟消耗token数超过当前AK申请TPM值:1000000"
                )
            return "ok"

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    model = RetryingChatModel(_TpmLimitedThenSuccess())

    assert await model.ainvoke(["prompt"]) == "ok"
    assert attempts == 2
    assert len(delays) == 1
    assert 5.0 <= delays[0] <= 6.0


@pytest.mark.asyncio
async def test_retrying_chat_model_retries_insufficient_quota(monkeypatch) -> None:
    attempts = 0
    delays: list[float] = []

    class _QuotaLimitedThenSuccess:
        async def ainvoke(self, *_args: object, **_kwargs: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise InvalidRequestError(
                    'Error code: 400 - {"error":{"message":"You exceeded your '
                    "current quota, please check your plan and billing details. "
                    "For details, see: https://help.aliyun.com/zh/model-studio/"
                    'error-code#token-limit","type":"insufficient_quota",'
                    '"code":"insufficient_quota"}}'
                )
            return "ok"

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    model = RetryingChatModel(_QuotaLimitedThenSuccess())

    assert await model.ainvoke(["prompt"]) == "ok"
    assert attempts == 2
    assert len(delays) == 1
    assert 5.0 <= delays[0] <= 6.0


@pytest.mark.asyncio
async def test_retrying_chat_model_retries_provider_502(monkeypatch) -> None:
    attempts = 0
    delays: list[float] = []

    class _BadGatewayThenSuccess:
        async def ainvoke(self, *_args: object, **_kwargs: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderError("Bad Gateway", provider="openai", status_code=502)
            return "ok"

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    model = RetryingChatModel(_BadGatewayThenSuccess())

    assert await model.ainvoke(["prompt"]) == "ok"
    assert attempts == 2
    assert len(delays) == 1
    assert 5.0 <= delays[0] <= 6.0


@pytest.mark.asyncio
async def test_retrying_chat_model_retries_timeout(monkeypatch) -> None:
    attempts = 0

    class _TimeoutThenSuccess:
        async def ainvoke(self, *_args: object, **_kwargs: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ModelTimeoutError("request timed out", provider="openai")
            return "ok"

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    model = RetryingChatModel(_TimeoutThenSuccess())

    assert await model.ainvoke(["prompt"]) == "ok"
    assert attempts == 2


@pytest.mark.asyncio
async def test_retrying_chat_model_does_not_retry_regular_invalid_request(
    monkeypatch,
) -> None:
    delegate = AsyncMock()
    delegate.ainvoke.side_effect = InvalidRequestError("messages is required")
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    model = RetryingChatModel(delegate)

    with pytest.raises(InvalidRequestError, match="messages is required"):
        await model.ainvoke(["prompt"])

    delegate.ainvoke.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrying_chat_model_does_not_retry_auth_error(monkeypatch) -> None:
    delegate = AsyncMock()
    delegate.ainvoke.side_effect = AuthenticationError(
        "invalid api key",
        provider="openai",
        status_code=401,
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    model = RetryingChatModel(delegate)

    with pytest.raises(AuthenticationError):
        await model.ainvoke(["prompt"])

    delegate.ainvoke.assert_awaited_once()
    sleep.assert_not_awaited()


def test_is_retryable_model_error_covers_5xx_and_rate_limit() -> None:
    assert is_retryable_model_error(ProviderError("boom", status_code=502))
    assert is_retryable_model_error(ProviderError("transport failed"))
    assert is_retryable_model_error(ModelTimeoutError("timeout"))
    assert is_retryable_model_error(
        InvalidRequestError("每分钟消耗token数超过当前AK申请TPM值:1000000")
    )
    assert is_retryable_model_error(
        InvalidRequestError(
            'Error code: 400 - {"error":{"code":"insufficient_quota","type":"insufficient_quota"}}'
        )
    )
    assert not is_retryable_model_error(InvalidRequestError("messages is required"))
    assert not is_retryable_model_error(AuthenticationError("unauthorized", status_code=401))


def test_is_retryable_model_error_covers_anthropic_transport_errors() -> None:
    request = httpx.Request("POST", "https://example.test/v1/messages")
    rate_limit_response = httpx.Response(429, request=request)
    server_error_response = httpx.Response(503, request=request)
    auth_response = httpx.Response(401, request=request)

    assert is_retryable_model_error(AnthropicAPIConnectionError(request=request))
    assert is_retryable_model_error(AnthropicAPITimeoutError(request=request))
    assert is_retryable_model_error(
        AnthropicRateLimitError("rate limited", response=rate_limit_response, body=None)
    )
    assert is_retryable_model_error(
        AnthropicInternalServerError(
            "service unavailable",
            response=server_error_response,
            body=None,
        )
    )
    assert not is_retryable_model_error(
        AnthropicAuthenticationError(
            "invalid api key",
            response=auth_response,
            body=None,
        )
    )


def test_bind_tools_keeps_retry_wrapper() -> None:
    delegate = MagicMock()
    bound_delegate = MagicMock()
    delegate.bind_tools.return_value = bound_delegate
    model = RetryingChatModel(delegate)

    bound = model.bind_tools([{"name": "trace"}])

    assert isinstance(bound, RetryingChatModel)
    delegate.bind_tools.assert_called_once_with([{"name": "trace"}])
