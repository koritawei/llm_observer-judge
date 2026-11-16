"""report-providers API unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.routes.report_providers import (
    ReportProviderUpsertRequest,
    delete_report_provider,
    get_report_provider,
    list_report_providers,
    upsert_report_provider,
)
from benchmark.domain.report_provider_config import ReportProviderEntry, ReportProvidersConfig
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_upsert_requires_endpoint_for_external() -> None:
    with pytest.raises(HTTPException) as ei:
        await upsert_report_provider(
            "search_eval",
            ReportProviderUpsertRequest(provider="external_http", endpoint_url=""),
        )
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_upsert_rejects_non_http_url() -> None:
    with pytest.raises(HTTPException) as ei:
        await upsert_report_provider(
            "search_eval",
            ReportProviderUpsertRequest(
                provider="external_http",
                endpoint_url="ftp://x",
            ),
        )
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_upsert_rejects_private_endpoint_url() -> None:
    with pytest.raises(HTTPException) as ei:
        await upsert_report_provider(
            "search_eval",
            ReportProviderUpsertRequest(
                provider="external_http",
                endpoint_url="http://127.0.0.1:8080/report",
            ),
        )
    assert ei.value.status_code == 400
    assert "loopback" in str(ei.value.detail).lower() or "private" in str(ei.value.detail).lower()


@pytest.mark.asyncio
async def test_upsert_create() -> None:
    created = MagicMock()
    created.benchmark_name = "search_eval"
    created.provider = "external_http"
    created.endpoint_url = "https://biz.example/r"
    created.bearer_token = "t"
    created.timeout_ms = 60_000
    created.fallback_to_platform = True
    created.enabled = True
    created.description = "d"

    with (
        patch(
            "benchmark.routes.report_providers.BenchmarkReportProvider.filter",
            return_value=MagicMock(first=AsyncMock(return_value=None)),
        ),
        patch(
            "benchmark.routes.report_providers.BenchmarkReportProvider.create",
            AsyncMock(return_value=created),
        ) as create,
    ):
        out = await upsert_report_provider(
            "search_eval",
            ReportProviderUpsertRequest(
                provider="external_http",
                endpoint_url="https://biz.example/r",
                bearer_token="t",
                description="d",
            ),
        )
    create.assert_awaited_once()
    assert out["data"]["benchmark_name"] == "search_eval"
    assert out["data"]["has_bearer_token"] is True
    assert "bearer_token" not in out["data"]


@pytest.mark.asyncio
async def test_list_merges_remote_config() -> None:
    row = MagicMock()
    row.benchmark_name = "a"
    row.provider = "external_http"
    row.endpoint_url = "https://a"
    row.bearer_token = None
    row.timeout_ms = 1_000
    row.fallback_to_platform = True
    row.enabled = True
    row.description = None

    remote_config = ReportProvidersConfig(
        providers=[
            ReportProviderEntry(
                benchmark_name="b",
                provider="external_http",
                endpoint_url="https://b",
                enabled=True,
            ),
        ],
    )

    qs = MagicMock()
    qs.order_by = MagicMock(side_effect=lambda *_a, **_k: AsyncMock(return_value=[row])())

    with (
        patch(
            "benchmark.routes.report_providers.BenchmarkReportProvider.all",
            return_value=qs,
        ),
        patch(
            "benchmark.routes.report_providers.fetch_remote_config_report_providers",
            return_value=remote_config,
        ),
    ):
        out = await list_report_providers()
    names = {i["benchmark_name"] for i in out["data"]}
    assert names == {"a", "b"}


@pytest.mark.asyncio
async def test_delete_not_found() -> None:
    with patch(
        "benchmark.routes.report_providers.BenchmarkReportProvider.filter",
        return_value=MagicMock(delete=AsyncMock(return_value=0)),
    ):
        with pytest.raises(HTTPException) as ei:
            await delete_report_provider("missing")
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_get_report_provider() -> None:
    resolved = ReportProviderEntry(
        benchmark_name="x",
        provider="platform",
        enabled=False,
    )
    with (
        patch(
            "benchmark.routes.report_providers.BenchmarkReportProvider.filter",
            return_value=MagicMock(first=AsyncMock(return_value=None)),
        ),
        patch(
            "benchmark.routes.report_providers.resolve_report_provider",
            AsyncMock(return_value=resolved),
        ),
    ):
        out = await get_report_provider("x")
    assert out["data"]["registered"] is None
    assert out["data"]["external_active"] is False
