"""Langfuse dataset fetch — longer HTTP timeout + retries.

The SDK default timeout is 5s (``LANGFUSE_TIMEOUT``). ``get_dataset`` paginates
``dataset_items.list``; a single slow page on pre/prod Langfuse raises
``httpx.ReadTimeout`` and aborts the whole benchmark / dataset-eval run.
Transient HTTP errors are retried, then surfaced to callers as
:class:`LangfuseDatasetError` so that upper layers never touch ``httpx``.
"""

import asyncio
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError

from benchmark.settings.trace_endpoints import TraceEndpoint, load_trace_instances
from benchmark.settings.setting import settings
from benchmark.utils.http_transport import HTTPError

logger = get_logger("agent_eval.utils.langfuse_dataset")

TRACE_HTTP_TIMEOUT_S = 60
TRACE_FETCH_ATTEMPTS = 3
TRACE_FETCH_PAGE_SIZE = 50


class TraceDatasetError(RuntimeError):
    """Langfuse dataset fetch failed — raised by this module after transient HTTP retries are exhausted.

    ``__cause__`` is the original ``httpx`` exception (timeout / transport error, and similar); upper layers use it as the
    signal to mark the run unreachable (distinct from an SDK ``ApiError`` business error, which
    the caller catches directly). Exception knowledge is closed over here; the business layer does not import httpx.
    """


def create_project_trace_client(
    *,
    public_key: str,
    secret_key: str,
    base_url: str,
    timeout: float | None = None,
) -> Langfuse:
    """Build a per-project SDK client with a dataset-safe HTTP timeout.

    ``timeout`` falls back to ``LANGFUSE_HTTP_TIMEOUT_S`` (60s) when unset.
    ``LangfuseResourceManager`` is a singleton keyed by ``public_key``; the
    timeout is applied only on first construction for that key.
    """
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        timeout=timeout or TRACE_HTTP_TIMEOUT_S,
    )


class TraceEndpointNotFoundError(ValueError):
    """``project_id`` has no configured Langfuse instance (``settings.langfuse_instances``)."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(f"Langfuse instance not found for projectId: {project_id}")


def create_trace_client_for_project(
    project_id: str,
    *,
    timeout: float | None = None,
) -> tuple[Langfuse, TraceEndpoint]:
    """Resolve instance credentials by ``project_id`` and build an SDK client (shared by benchmark / dataset).

    ``timeout`` defaults to ``settings.langfuse_http_timeout_s``; the instance is also returned so
    the caller can tell apart fetches of the same dataset on different instances in logs.

    Raises:
        LangfuseInstanceNotFoundError: ``project_id`` has no matching instance.
    """
    instances = load_trace_instances(settings.trace_instances)
    instance = instances.get(project_id)
    if instance is None:
        raise TraceEndpointNotFoundError(project_id)
    client = create_project_trace_client(
        public_key=instance.public_key,
        secret_key=instance.secret_key.get_secret_value(),
        base_url=instance.base_url,
        timeout=timeout or settings.trace_http_timeout_s,
    )
    return client, instance


async def fetch_dataset_items(
    lf: Langfuse,
    dataset_name: str,
    *,
    attempts: int | None = None,
    page_size: int | None = None,
) -> list[Any]:
    """Fetch all dataset items, retrying transient HTTP / timeout errors.

    ``attempts`` / ``page_size`` fall back to ``LANGFUSE_FETCH_ATTEMPTS`` (3)
    and ``LANGFUSE_FETCH_PAGE_SIZE`` (50) when unset. Exhausted retries raise
    :class:`LangfuseDatasetError`; SDK ``ApiError`` is re-raised as-is.
    """
    n_attempts = attempts or TRACE_FETCH_ATTEMPTS
    n_page_size = page_size or TRACE_FETCH_PAGE_SIZE
    last_http_error: HTTPError | None = None
    for attempt in range(1, n_attempts + 1):
        try:
            dataset = await asyncio.to_thread(
                lf.get_dataset,
                dataset_name,
                fetch_items_page_size=n_page_size,
            )
            return list(dataset.items)
        except ApiError:
            raise
        except HTTPError as exc:
            last_http_error = exc
            logger.warning(
                "langfuse.dataset_fetch_retry dataset=%s attempt=%d/%d error=%s",
                dataset_name,
                attempt,
                n_attempts,
                type(exc).__name__,
            )
            if attempt >= n_attempts:
                break
            await asyncio.sleep(min(2 ** (attempt - 1), 8))

    # Falling out of the loop without return means every retry failed; last_http_error must not be None
    # (the except branch ran at least once). Use an explicit check instead of assert so -O cannot strip it.
    if last_http_error is None:  # pragma: no cover - logically unreachable
        raise RuntimeError("fetch_dataset_items exited loop without error")
    raise TraceDatasetError(
        f"Langfuse unreachable: {type(last_http_error).__name__}"
    ) from last_http_error
