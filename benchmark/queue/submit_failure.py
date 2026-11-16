"""submit_failure — classify async submit / callback failures (design doc §8.2 / §8.3).

- ``is_retryable_submit_failure``: network / timeout / 429 / 5xx → retryable (running→wait)
- ``is_retryable_callback_error``: classify ieval callback ``error`` text
  (defined in :mod:`benchmark.utils.callback_retry`, shared with the benchmark callback
  schema; this package is infrastructure-only and must not depend on business packages)
- ``should_release_on_submit_failure``: release the slot only if Aone did not accept the job;
  if it was accepted, hold the slot until the callback
- ``add_running_with_retry``: backoff when a pending_callback write fails transiently
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Queue

from benchmark.utils.callback_retry import is_retryable_callback_error
from benchmark.utils.http_transport import classify_httpx_retryable

logger = get_logger("agent_eval.eval_queue.submit_failure")

#: Default retry count for pending_callback writes (design §8.2)
_DEFAULT_RUNNING_ADD_ATTEMPTS = 5


def is_retryable_submit_failure(exc: BaseException) -> bool:
    """Whether an Aone submit / local registration failure is retryable.

    Retryable: timeout, connection error, 429, 5xx, and other transient TransportError.
    Not retryable: 4xx (auth / parameters), missing config (client not configured).
    """
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)):
        return True
    # httpx errors (raised by Aone submit / job status calls) are classified inside aone_client
    httpx_retryable = classify_httpx_retryable(exc)
    if httpx_retryable is not None:
        return httpx_retryable
    msg = str(exc).lower()
    if "not configured" in msg or "not configured" in type(exc).__name__.lower():
        return False
    # AttributeError: a programming error (for example ``'str'.get``); not retryable, same as TypeError
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return False
    # Unknown exceptions: retry conservatively (still bounded by max_attempts / the time window)
    return True


def should_release_on_submit_failure(
    exc: BaseException,
    *,
    notification_accepted: bool,
) -> bool:
    """If Aone already accepted the job, hold the slot until the callback / TTL; otherwise release it, then requeue or
    fail terminally.
    """
    return not notification_accepted


@dataclass(frozen=True)
class RunningJobWrite:
    """Write spec for ``add_running_with_retry`` (FUNC-PARAMS-001 parameter folding)."""

    job_id: str
    name: str
    data: dict[str, Any]
    attempts: int = _DEFAULT_RUNNING_ADD_ATTEMPTS


async def add_running_with_retry(
    running_queue: Queue,
    write: RunningJobWrite,
) -> None:
    """Retry a transient failure while writing pending_callback / inflight.

    When retries are exhausted, raise the last exception. The caller decides whether to
    requeue, force_fail, or alert and keep the slot.
    """
    job_id = write.job_id
    name = write.name
    data = write.data
    attempts = write.attempts
    last_exc: BaseException | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            await running_queue.add(
                name,
                data,
                {
                    "jobId": job_id,
                    "removeOnComplete": False,
                    "removeOnFail": False,
                },
            )
            if attempt > 1:
                logger.info(
                    "running_add.retry_ok job_id=%s name=%s attempt=%d",
                    job_id,
                    name,
                    attempt,
                )
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "running_add.retry job_id=%s name=%s attempt=%d/%d err=%s",
                job_id,
                name,
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                await asyncio.sleep(0.05 * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc


__all__ = [
    "RunningJobWrite",
    "add_running_with_retry",
    "is_retryable_callback_error",
    "is_retryable_submit_failure",
    "should_release_on_submit_failure",
]
