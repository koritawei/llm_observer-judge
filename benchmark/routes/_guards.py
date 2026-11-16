"""Shared API-layer guards.

Guards for the benchmark / dataset processing switches (RemoteConfig
``agent-eval.service.switches`` fields ``benchmark_disabled`` / ``dataset_disabled`` —
two independent switches, default ``true`` means do not process; the old combined field
``benchmark_dataset_disabled`` is no longer read and is not compatible).

Also a BullMQ queue-stack degradation guard: after ``_ls_bullmq_init_failed_fallback``
degrades silently (Redis not configured, and so on), accepting work would produce a
"202 false success" — the run is created, no worker consumes it, and it sits until the
stale healer reaps it. A degraded instance rejects trigger endpoints with 503 so the
caller learns immediately.
"""

from fastapi import HTTPException, Request

from benchmark.domain.service_switches import is_benchmark_disabled, is_dataset_disabled


def raise_if_benchmark_disabled() -> None:
    """Reject with 503 when the benchmark switch is on (RemoteConfig ``benchmark_disabled=true``).

    Only for benchmark create/trigger endpoints (run / run-pipeline / run-inline /
    runs / run-external, and so on). In-flight append/finalize/callbacks and GET
    queries are not blocked — the switch means "reject tasks that have not started;
    tasks already running finish". State comes from RemoteConfig and is checked live
    on each action (short TTL cache); no process restart is required.
    """
    if is_benchmark_disabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "benchmark processing is disabled "
                "(RemoteConfig agent-eval.service.switches: benchmark_disabled=true)"
            ),
        )


def raise_if_dataset_disabled() -> None:
    """Reject with 503 when the dataset switch is on (RemoteConfig ``dataset_disabled=true``).

    Only for dataset create/trigger endpoints (eval / eval-inline, and so on). In-flight
    append/finalize/callbacks and GET queries are not blocked — the switch means "reject
    tasks that have not started; tasks already running finish". State comes from
    RemoteConfig and is checked live on each action (short TTL cache); no process
    restart is required.
    """
    if is_dataset_disabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "dataset processing is disabled "
                "(RemoteConfig agent-eval.service.switches: dataset_disabled=true)"
            ),
        )


def raise_if_eval_queue_degraded(request: Request) -> None:
    """Reject trigger endpoints with 503 when the BullMQ queue stack is degraded.

    A run accepted while degraded has no worker (processedCount stays 0 and the stale
    healer eventually marks it failed). Trigger endpoints reject up front so the caller
    gets a clear error instead of a "202 false success". Only the explicit fallback flag
    is consulted, so a lightweight app that never assembled a queue stack is not rejected
    by mistake. GET queries and in-flight callbacks are not blocked (queued work is
    finished by the healer).
    """
    degraded_reason = getattr(request.app.state, "eval_queue_degraded_reason", None)
    if degraded_reason:
        raise HTTPException(
            status_code=503,
            detail=(
                "BullMQ eval queue degraded; benchmark runs would never be "
                f"consumed by workers (reason: {degraded_reason}). Check Redis config "
                "(EVAL_QUEUE_REDIS_URL / EVAL_QUEUE_REDIS_KVSTORE_INSTANCE_ID / "
                "EVAL_QUEUE_REDIS_HOST) and restart."
            ),
        )


__all__ = [
    "raise_if_benchmark_disabled",
    "raise_if_dataset_disabled",
    "raise_if_eval_queue_degraded",
]
