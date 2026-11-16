"""HTTP client for external reports.

Contract (implemented by the business service)::

    POST {endpoint_url}
    Authorization: Bearer <token>   # optional
    Content-Type: application/json

    Request::
        {
          "run_id": "...",
          "benchmark_name": "...",
          "report_spec": { ... }   # full ReportSpec (includes case_results)
        }

    Response 200::
        {
          "report_url": "https://...",   # preferred: already-hosted report URL
          "html": "<!DOCTYPE html>..."  # optional: the platform may upload it to OSS
        }

At least one of ``report_url`` or non-empty ``html`` must be returned, otherwise the call fails.
"""

from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.report_provider_config import ReportProviderEntry
from benchmark.utils.http_transport import AsyncClient, HTTPError

logger = get_logger(__name__)

#: HTTP status codes greater than or equal to this value are treated as failed responses.
_HTTP_ERROR_STATUS_MIN = 400


class ExternalReportResult:
    """Result of an external report call."""

    __slots__ = ("report_url", "html", "raw")

    def __init__(
        self,
        *,
        report_url: str | None,
        html: str | None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.report_url = report_url
        self.html = html
        self.raw = raw or {}


class ExternalReportError(Exception):
    """External report call failed."""


class ExternalReportClient:
    """Call the business external report service."""

    async def generate(
        self,
        *,
        provider: ReportProviderEntry,
        spec: dict[str, Any],
    ) -> ExternalReportResult:
        if not provider.endpoint_url:
            raise ExternalReportError("endpoint_url is empty")

        run_id = str(spec.get("run_id") or "")
        benchmark_name = str(spec.get("benchmark_name") or provider.benchmark_name)
        payload = {
            "run_id": run_id,
            "benchmark_name": benchmark_name,
            "report_spec": spec,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if provider.bearer_token:
            headers["Authorization"] = f"Bearer {provider.bearer_token}"

        timeout_s = max(provider.timeout_ms, 1000) / 1000.0
        logger.info(
            "external_report.request benchmark=%s run_id=%s url=%s timeout_s=%.1f",
            benchmark_name,
            run_id,
            provider.endpoint_url,
            timeout_s,
        )
        try:
            async with AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(
                    provider.endpoint_url,
                    json=payload,
                    headers=headers,
                )
        except HTTPError as exc:
            raise ExternalReportError(f"http error: {exc}") from exc

        if resp.status_code >= _HTTP_ERROR_STATUS_MIN:
            body_preview = (resp.text or "")[:300]
            raise ExternalReportError(
                f"status={resp.status_code} body={body_preview!r}",
            )

        try:
            data = resp.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise ExternalReportError(f"invalid json response: {exc}") from exc

        if not isinstance(data, dict):
            raise ExternalReportError("response must be a JSON object")

        report_url = data.get("report_url") or data.get("url")
        html = data.get("html") or data.get("html_content")
        # Accept only a real string URL; non-str values such as dict/list must not be str()'d into a URL
        report_url_s = (
            report_url.strip() if isinstance(report_url, str) and report_url.strip() else None
        )
        html_s = str(html) if isinstance(html, str) and html.strip() else None
        if not report_url_s and not html_s:
            raise ExternalReportError(
                "response must include report_url or non-empty html",
            )

        logger.info(
            "external_report.ok benchmark=%s run_id=%s has_url=%s html_len=%d",
            benchmark_name,
            run_id,
            bool(report_url_s),
            len(html_s or ""),
        )
        return ExternalReportResult(
            report_url=report_url_s,
            html=html_s,
            raw=data,
        )


__all__ = [
    "ExternalReportClient",
    "ExternalReportError",
    "ExternalReportResult",
]
