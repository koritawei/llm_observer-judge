"""Langfuse client factory for cluster-report performance collection.

Reuses the langfuse_client HTTP transport-pool factory; the business side only receives a ready client handle.
"""

from collections.abc import Callable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.trace.http_backend import (
    TraceHttpBackend,
    TraceHttpClient,
    build_shared_http_client,
)
from benchmark.trace.schema import TraceEndpoint

logger = get_logger(__name__)

#: HTTP timeout for performance collection (looser than online eval: batch observation fetches are offline).
DEFAULT_PERF_HTTP_TIMEOUT_S = 30.0


PerfClientSelector = Callable[[str], TraceHttpBackend | None]


def build_perf_client_selector(
    instances: dict[str, TraceEndpoint],
    *,
    project_id: str = "",
    timeout_s: float = DEFAULT_PERF_HTTP_TIMEOUT_S,
) -> tuple[PerfClientSelector | None, TraceHttpClient | None]:
    """Build a performance-client selector that shares an HTTP connection pool.

    A project passed by the caller wins over the default ``project_id``; if neither hits, fall back to the first list item and warn.
    """

    if not instances:
        logger.info("cluster_report.perf_client_absent no langfuse instances configured")
        return None, None

    http_client = build_shared_http_client(timeout_s)

    def select(requested_project_id: str = "") -> TraceHttpBackend:
        selected_project_id = requested_project_id.strip() or project_id.strip()
        instance = instances.get(selected_project_id) if selected_project_id else None
        if instance is None:
            fallback_id, instance = next(iter(instances.items()))
            if selected_project_id:
                logger.warning(
                    "cluster_report.perf_project_not_found requested=%s fallback=%s "
                    "(凭证可能不匹配，性能块可能整体失败)",
                    selected_project_id,
                    fallback_id,
                )
            elif len(instances) > 1:
                logger.warning(
                    "cluster_report.perf_project_unset instances=%d fallback=%s "
                    "(多实例请设置 AGENT_EVAL_CLUSTER_REPORT_PROJECT_ID)",
                    len(instances),
                    fallback_id,
                )
        return TraceHttpBackend(instance, http_client)

    return select, http_client


def build_perf_client(
    instances: dict[str, TraceEndpoint],
    *,
    project_id: str = "",
    timeout_s: float = DEFAULT_PERF_HTTP_TIMEOUT_S,
) -> tuple[TraceHttpBackend | None, TraceHttpClient | None]:
    """Compatibility entry that builds a performance-collection client for a single default project."""

    selector, http_client = build_perf_client_selector(
        instances,
        project_id=project_id,
        timeout_s=timeout_s,
    )
    return (selector("") if selector is not None else None), http_client


__all__ = [
    "DEFAULT_PERF_HTTP_TIMEOUT_S",
    "PerfClientSelector",
    "build_perf_client",
    "build_perf_client_selector",
]
