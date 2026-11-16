"""Report generation dispatch: external Provider first, with fallback to webdev on failure.

Shared by ``BenchmarkRunner._generate_report`` and
``ExternalBenchmarkRunner._generate_external_report``.
"""

from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.report_provider_config import resolve_report_provider
from benchmark.reporting.framework.external_report_client import (
    ExternalReportClient,
    ExternalReportError,
)

logger = get_logger(__name__)


class _WebdevLike(Protocol):
    async def generate_report(
        self,
        spec: dict[str, Any],
    ) -> tuple[str, str | None]: ...


async def dispatch_report_generation(
    *,
    spec: dict[str, Any],
    webdev: _WebdevLike | None,
    external_client: ExternalReportClient | None = None,
) -> tuple[str, str | None]:
    """Generate a report according to the Provider configuration.

    Returns:
        ``(html, report_url)`` — aligned with ``WebdevReportClient.generate_report``.
        When an external provider returns only a URL, ``html`` may be an empty string
        (a business-hosted page).
    """
    benchmark_name = str(spec.get("benchmark_name") or "")
    provider = await resolve_report_provider(benchmark_name)

    if provider.is_external_active():
        client = external_client or ExternalReportClient()
        try:
            result = await client.generate(provider=provider, spec=spec)
            html = result.html or ""
            report_url = result.report_url
            logger.info(
                "report_dispatch.external_ok benchmark=%s run_id=%s report_url=%s html_len=%d",
                benchmark_name,
                spec.get("run_id"),
                report_url,
                len(html),
            )
            return html, report_url
        except ExternalReportError as exc:
            logger.error(
                "report_dispatch.external_failed benchmark=%s run_id=%s fallback=%s err=%s",
                benchmark_name,
                spec.get("run_id"),
                provider.fallback_to_platform,
                exc,
            )
            if not provider.fallback_to_platform:
                return "", None
        except Exception:
            logger.error(
                "report_dispatch.external_unexpected benchmark=%s run_id=%s fallback=%s",
                benchmark_name,
                spec.get("run_id"),
                provider.fallback_to_platform,
                exc_info=True,
            )
            if not provider.fallback_to_platform:
                return "", None

    if webdev is None:
        logger.error(
            "report_dispatch.no_webdev benchmark=%s run_id=%s",
            benchmark_name,
            spec.get("run_id"),
        )
        return "", None

    return await webdev.generate_report(spec)


__all__ = ["dispatch_report_generation"]
