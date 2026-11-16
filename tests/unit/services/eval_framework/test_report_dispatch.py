"""External report provider: domain parsing + HTTP client + dispatch."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.domain.report_provider_config import (
    ReportProviderEntry,
    ReportProvidersConfig,
    entry_public_dict,
    resolve_report_provider,
)
from benchmark.reporting.framework.external_report_client import (
    ExternalReportClient,
    ExternalReportError,
)
from benchmark.reporting.framework.report_dispatch import dispatch_report_generation


@pytest.mark.asyncio
async def test_resolve_report_provider_defaults_to_platform() -> None:
    with patch(
        "benchmark.domain.report_provider_config.fetch_remote_config_report_providers",
        return_value=ReportProvidersConfig(providers=[]),
    ):
        with patch(
            "benchmark.records.models.benchmark_report_provider.BenchmarkReportProvider.filter",
            return_value=MagicMock(first=AsyncMock(return_value=None)),
        ):
            entry = await resolve_report_provider("unknown_bm")
    assert entry.provider == "platform"
    assert not entry.is_external_active()


@pytest.mark.asyncio
async def test_resolve_report_provider_from_db_row() -> None:
    row = MagicMock()
    row.benchmark_name = "search_eval"
    row.provider = "external_http"
    row.endpoint_url = "https://biz.example/report"
    row.bearer_token = "tok"
    row.timeout_ms = 30_000
    row.fallback_to_platform = True
    row.enabled = True
    row.description = "biz"
    with patch(
        "benchmark.records.models.benchmark_report_provider.BenchmarkReportProvider.filter",
        return_value=MagicMock(first=AsyncMock(return_value=row)),
    ):
        entry = await resolve_report_provider("search_eval")
    assert entry.is_external_active()
    assert entry.endpoint_url == "https://biz.example/report"
    assert entry.bearer_token == "tok"


@pytest.mark.asyncio
async def test_resolve_disabled_db_forces_platform() -> None:
    row = MagicMock()
    row.benchmark_name = "search_eval"
    row.provider = "external_http"
    row.endpoint_url = "https://biz.example/report"
    row.bearer_token = None
    row.timeout_ms = 30_000
    row.fallback_to_platform = True
    row.enabled = False
    row.description = None
    with patch(
        "benchmark.records.models.benchmark_report_provider.BenchmarkReportProvider.filter",
        return_value=MagicMock(first=AsyncMock(return_value=row)),
    ):
        entry = await resolve_report_provider("search_eval")
    assert entry.provider == "platform"
    assert not entry.is_external_active()


def test_entry_public_dict_hides_token() -> None:
    entry = ReportProviderEntry(
        benchmark_name="b",
        provider="external_http",
        endpoint_url="https://x",
        bearer_token="secret",
    )
    pub = entry_public_dict(entry)
    assert "bearer_token" not in pub
    assert pub["has_bearer_token"] is True


@pytest.mark.asyncio
async def test_external_report_client_success(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ReportProviderEntry(
        benchmark_name="b",
        provider="external_http",
        endpoint_url="https://biz.example/r",
        bearer_token="t",
        timeout_ms=5_000,
    )

    class _Resp:
        status_code = 200
        text = '{"report_url":"https://cdn/r.html"}'

        def json(self) -> dict[str, Any]:
            return {"report_url": "https://cdn/r.html", "html": "<html>ok</html>"}

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> _Resp:
            assert url == "https://biz.example/r"
            assert headers["Authorization"] == "Bearer t"
            assert json["benchmark_name"] == "b"
            assert "report_spec" in json
            return _Resp()

    monkeypatch.setattr(
        "benchmark.reporting.framework.external_report_client.AsyncClient",
        _Client,
    )

    result = await ExternalReportClient().generate(
        provider=provider,
        spec={"run_id": "r1", "benchmark_name": "b", "final_score": 1.5},
    )
    assert result.report_url == "https://cdn/r.html"
    assert result.html == "<html>ok</html>"


@pytest.mark.asyncio
async def test_external_report_client_rejects_non_str_report_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A report_url that is a dict/list must not be str()'d into a garbage URL."""
    provider = ReportProviderEntry(
        benchmark_name="b",
        provider="external_http",
        endpoint_url="https://biz.example/r",
    )

    class _Resp:
        status_code = 200
        text = '{"report_url":{"url":"https://cdn/r.html"}}'

        def json(self) -> dict[str, Any]:
            return {"report_url": {"url": "https://cdn/r.html"}}

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def post(self, *a: object, **k: object) -> _Resp:
            return _Resp()

    monkeypatch.setattr(
        "benchmark.reporting.framework.external_report_client.AsyncClient",
        _Client,
    )

    with pytest.raises(ExternalReportError, match="report_url or non-empty html"):
        await ExternalReportClient().generate(
            provider=provider,
            spec={"run_id": "r1", "benchmark_name": "b"},
        )


@pytest.mark.asyncio
async def test_external_report_client_rejects_empty_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ReportProviderEntry(
        benchmark_name="b",
        provider="external_http",
        endpoint_url="https://biz.example/r",
    )

    class _Resp:
        status_code = 200
        text = "{}"

        def json(self) -> dict[str, Any]:
            return {}

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def post(self, *a: object, **k: object) -> _Resp:
            return _Resp()

    monkeypatch.setattr(
        "benchmark.reporting.framework.external_report_client.AsyncClient",
        _Client,
    )

    with pytest.raises(ExternalReportError, match="report_url or non-empty html"):
        await ExternalReportClient().generate(
            provider=provider,
            spec={"run_id": "r1", "benchmark_name": "b"},
        )


@pytest.mark.asyncio
async def test_dispatch_uses_external_then_skips_webdev() -> None:
    provider = ReportProviderEntry(
        benchmark_name="b",
        provider="external_http",
        endpoint_url="https://biz.example/r",
        enabled=True,
    )
    webdev = AsyncMock()
    external = AsyncMock()
    external.generate = AsyncMock(
        return_value=MagicMock(report_url="https://ext/r", html="<html/>"),
    )
    with patch(
        "benchmark.reporting.framework.report_dispatch.resolve_report_provider",
        AsyncMock(return_value=provider),
    ):
        html, url = await dispatch_report_generation(
            spec={"benchmark_name": "b", "run_id": "r1"},
            webdev=webdev,
            external_client=external,
        )
    assert url == "https://ext/r"
    assert html == "<html/>"
    webdev.generate_report.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_fallback_to_platform_on_external_error() -> None:
    provider = ReportProviderEntry(
        benchmark_name="b",
        provider="external_http",
        endpoint_url="https://biz.example/r",
        fallback_to_platform=True,
        enabled=True,
    )
    webdev = AsyncMock()
    webdev.generate_report = AsyncMock(return_value=("<html>p</html>", "https://plat/r"))
    external = AsyncMock()
    external.generate = AsyncMock(side_effect=ExternalReportError("boom"))
    with patch(
        "benchmark.reporting.framework.report_dispatch.resolve_report_provider",
        AsyncMock(return_value=provider),
    ):
        html, url = await dispatch_report_generation(
            spec={"benchmark_name": "b", "run_id": "r1"},
            webdev=webdev,
            external_client=external,
        )
    assert url == "https://plat/r"
    assert html == "<html>p</html>"


@pytest.mark.asyncio
async def test_dispatch_no_fallback_returns_empty() -> None:
    provider = ReportProviderEntry(
        benchmark_name="b",
        provider="external_http",
        endpoint_url="https://biz.example/r",
        fallback_to_platform=False,
        enabled=True,
    )
    webdev = AsyncMock()
    external = AsyncMock()
    external.generate = AsyncMock(side_effect=ExternalReportError("boom"))
    with patch(
        "benchmark.reporting.framework.report_dispatch.resolve_report_provider",
        AsyncMock(return_value=provider),
    ):
        html, url = await dispatch_report_generation(
            spec={"benchmark_name": "b", "run_id": "r1"},
            webdev=webdev,
            external_client=external,
        )
    assert html == ""
    assert url is None
    webdev.generate_report.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_fallback_on_unexpected_external_error() -> None:
    """When the external call raises something other than ExternalReportError, it can still degrade to the platform report."""
    provider = ReportProviderEntry(
        benchmark_name="b",
        provider="external_http",
        endpoint_url="https://biz.example/r",
        fallback_to_platform=True,
        enabled=True,
    )
    webdev = AsyncMock()
    webdev.generate_report = AsyncMock(return_value=("<html>p</html>", "https://plat/r"))
    external = AsyncMock()
    external.generate = AsyncMock(side_effect=RuntimeError("unexpected"))
    with patch(
        "benchmark.reporting.framework.report_dispatch.resolve_report_provider",
        AsyncMock(return_value=provider),
    ):
        html, url = await dispatch_report_generation(
            spec={"benchmark_name": "b", "run_id": "r1"},
            webdev=webdev,
            external_client=external,
        )
    assert url == "https://plat/r"
    assert html == "<html>p</html>"
