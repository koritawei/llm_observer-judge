"""Unit tests for submit_failure — §8.2 retryable / force_fail classification + pending_callback retry."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.queue.submit_failure import (
    RunningJobWrite,
    add_running_with_retry,
    is_retryable_callback_error,
    is_retryable_submit_failure,
    should_release_on_submit_failure,
)
from benchmark.utils.http_transport import ConnectError, HTTPStatusError


def test_retryable_timeout_and_transport() -> None:
    assert is_retryable_submit_failure(TimeoutError("x")) is True
    assert is_retryable_submit_failure(ConnectError("boom")) is True
    assert is_retryable_submit_failure(ConnectionError("net")) is True


def test_retryable_http_5xx_and_429() -> None:
    resp_500 = MagicMock()
    resp_500.status_code = 500
    assert (
        is_retryable_submit_failure(
            HTTPStatusError("err", request=MagicMock(), response=resp_500),
        )
        is True
    )

    resp_429 = MagicMock()
    resp_429.status_code = 429
    assert (
        is_retryable_submit_failure(
            HTTPStatusError("err", request=MagicMock(), response=resp_429),
        )
        is True
    )


def test_non_retryable_4xx_and_config() -> None:
    resp_400 = MagicMock()
    resp_400.status_code = 400
    assert (
        is_retryable_submit_failure(
            HTTPStatusError("bad", request=MagicMock(), response=resp_400),
        )
        is False
    )
    assert (
        is_retryable_submit_failure(
            RuntimeError("Aone judge client not configured"),
        )
        is False
    )
    assert is_retryable_submit_failure(ValueError("bad payload")) is False
    assert (
        is_retryable_submit_failure(
            AttributeError("'str' object has no attribute 'get'"),
        )
        is False
    )


def test_callback_error_enrich_miss_is_non_retryable() -> None:
    assert (
        is_retryable_callback_error(
            "ValueError: 适配后全部 response 为空，请检查 enrich 命中",
        )
        is False
    )


def test_callback_error_empty_criterion_scores_is_non_retryable() -> None:
    """A fake-success empty dimension score → not retryable (ieval_external-20260909:58)."""
    assert is_retryable_callback_error("empty_criterion_scores") is False


def test_callback_error_timeout_is_retryable() -> None:
    assert is_retryable_callback_error("TimeoutError: rubric scoring timed out") is True
    assert is_retryable_callback_error("HTTP 503 Service Unavailable") is True
    assert is_retryable_callback_error("rate limit exceeded 429") is True


def test_callback_error_empty_not_retryable() -> None:
    assert is_retryable_callback_error(None) is False
    assert is_retryable_callback_error("") is False
    assert is_retryable_callback_error("   ") is False


def test_should_release_only_when_not_accepted() -> None:
    exc = RuntimeError("x")
    assert should_release_on_submit_failure(exc, notification_accepted=False) is True
    assert should_release_on_submit_failure(exc, notification_accepted=True) is False


@pytest.mark.asyncio
async def test_add_running_with_retry_succeeds_after_transient() -> None:
    q = MagicMock()
    q.add = AsyncMock(side_effect=[RuntimeError("transient"), None])

    await add_running_with_retry(
        q,
        RunningJobWrite(
            job_id="jid",
            name="pending_callback",
            data={"a": 1},
            attempts=3,
        ),
    )
    assert q.add.await_count == 2


@pytest.mark.asyncio
async def test_add_running_with_retry_exhausts() -> None:
    q = MagicMock()
    q.add = AsyncMock(side_effect=RuntimeError("always"))

    with pytest.raises(RuntimeError, match="always"):
        await add_running_with_retry(
            q,
            RunningJobWrite(
                job_id="jid",
                name="pending_callback",
                data={},
                attempts=2,
            ),
        )
    assert q.add.await_count == 2
