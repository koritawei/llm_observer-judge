"""BenchmarkReportProvider lifecycle observer.

Emits domain-level observation logs when an external report-service registration
row is saved or deleted. Registered automatically at application startup via
``DatabaseConfig.observers``.

Notes:
  - Observers only emit observation logs and do not carry online business logic
    (constitution §4.6); ``bearer_token`` credentials must **not** be logged.
"""

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.models.observer import ModelObserver

from benchmark.records.models.benchmark_report_provider import BenchmarkReportProvider

logger = get_logger("agent_eval.observers.benchmark_report_provider")


class BenchmarkReportProviderObserver(ModelObserver[BenchmarkReportProvider]):
    """Lifecycle observer for BenchmarkReportProvider.

    Responsibilities:
      - **on_save**: record a registration change (benchmark / provider / enabled), at debug level.
      - **on_delete**: record registration deletion, at info level (audit).
    """

    async def on_save(self, instance: BenchmarkReportProvider) -> None:
        """Post-save: record external report-service registration / update."""
        logger.debug(
            "benchmark_report_provider saved: benchmark=%s provider=%s enabled=%s "
            "fallback_to_platform=%s",
            instance.benchmark_name,
            instance.provider,
            instance.enabled,
            instance.fallback_to_platform,
        )

    async def on_delete(self, instance: BenchmarkReportProvider) -> None:
        """Registration deleted (= fall back to platform report publishing); record an audit event."""
        logger.info(
            "benchmark_report_provider deleted: benchmark=%s provider=%s",
            instance.benchmark_name,
            instance.provider,
        )


__all__ = ["BenchmarkReportProviderObserver"]
