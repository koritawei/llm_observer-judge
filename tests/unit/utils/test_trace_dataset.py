"""Unit tests for Langfuse dataset fetch timeout / retry helper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.utils.http_transport import ReadTimeout
from benchmark.utils.trace_dataset import (
    TRACE_FETCH_ATTEMPTS,
    TRACE_FETCH_PAGE_SIZE,
    TRACE_HTTP_TIMEOUT_S,
    TraceDatasetError,
    create_project_trace_client,
    fetch_dataset_items,
)
from langfuse.api.core.api_error import ApiError


@pytest.mark.asyncio
async def test_fetch_dataset_items_success() -> None:
    items = [SimpleNamespace(id="i1"), SimpleNamespace(id="i2")]
    lf = MagicMock()
    lf.get_dataset = MagicMock(return_value=SimpleNamespace(items=items))

    result = await fetch_dataset_items(lf, "ds")

    assert result == items
    lf.get_dataset.assert_called_once_with(
        "ds",
        fetch_items_page_size=TRACE_FETCH_PAGE_SIZE,
    )


@pytest.mark.asyncio
async def test_fetch_dataset_items_retries_read_timeout() -> None:
    items = [SimpleNamespace(id="i1")]
    lf = MagicMock()
    lf.get_dataset = MagicMock(
        side_effect=[ReadTimeout("timed out"), SimpleNamespace(items=items)],
    )

    with patch("benchmark.utils.trace_dataset.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await fetch_dataset_items(lf, "ds")

    assert result == items
    assert lf.get_dataset.call_count == 2
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_dataset_items_exhausts_retries() -> None:
    lf = MagicMock()
    lf.get_dataset = MagicMock(side_effect=ReadTimeout("timed out"))

    with patch("benchmark.utils.trace_dataset.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(TraceDatasetError) as exc_info:
            await fetch_dataset_items(lf, "ds")

    assert lf.get_dataset.call_count == TRACE_FETCH_ATTEMPTS
    # The original httpx exception is attached as __cause__, so upper-layer logs can tell which timeout it was
    assert isinstance(exc_info.value.__cause__, ReadTimeout)


@pytest.mark.asyncio
async def test_fetch_dataset_items_does_not_retry_api_error() -> None:
    lf = MagicMock()
    lf.get_dataset = MagicMock(side_effect=ApiError(status_code=404, body="missing"))

    with pytest.raises(ApiError) as exc_info:
        await fetch_dataset_items(lf, "ds")

    assert exc_info.value.status_code == 404
    lf.get_dataset.assert_called_once()


def test_create_project_trace_client_sets_timeout() -> None:
    with patch("benchmark.utils.trace_dataset.Langfuse") as trace_cls:
        create_project_trace_client(
            public_key="pk",
            secret_key="sk",
            base_url="http://lf",
        )

    trace_cls.assert_called_once_with(
        public_key="pk",
        secret_key="sk",
        base_url="http://lf",
        timeout=TRACE_HTTP_TIMEOUT_S,
    )


def test_create_project_trace_client_honors_custom_timeout() -> None:
    with patch("benchmark.utils.trace_dataset.Langfuse") as trace_cls:
        create_project_trace_client(
            public_key="pk",
            secret_key="sk",
            base_url="http://lf",
            timeout=120,
        )

    trace_cls.assert_called_once_with(
        public_key="pk",
        secret_key="sk",
        base_url="http://lf",
        timeout=120,
    )


@pytest.mark.asyncio
async def test_fetch_dataset_items_honors_custom_attempts_and_page_size() -> None:
    items = [SimpleNamespace(id="i1")]
    lf = MagicMock()
    lf.get_dataset = MagicMock(return_value=SimpleNamespace(items=items))

    await fetch_dataset_items(lf, "ds", attempts=5, page_size=100)

    lf.get_dataset.assert_called_once_with(
        "ds",
        fetch_items_page_size=100,
    )
