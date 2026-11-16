"""Central definitions for agent-eval business constants.

Improves readability for people and models, and keeps magic numbers from scattering through the code.

Usage:
    from benchmark.constant import DEFAULT_PAGE_SIZE
"""

#: Default page size for paginated queries
DEFAULT_PAGE_SIZE = 20

#: Default max_concurrency used when benchmark streaming finalizes
#: (fallback when config_json does not pass it explicitly)
DEFAULT_MAX_CONCURRENCY = 1

#: Max workers executing concurrently per node (shared by benchmark batch and realtime streaming)
#: Set to 1 → a node runs only one run's worker at a time, avoiding multi-worker concurrency
MAX_CONCURRENT_WORKERS = 1

#: Run scanner poll interval (seconds) — scans non-terminal runs that still have pending tasks
RUN_SCANNER_INTERVAL_S = 10.0

#: Default runtime environment id
DEFAULT_ENVIRONMENT = "development"

#: RemoteConfig cache TTL (seconds); set to 0 to disable the cache
CACHE_TTL_SEC = 300

#: Langfuse truncation marker (written by ClickHouseWriter when a field exceeds 1MB)
TRUNCATION_MARKER = "[TRUNCATED: Field exceeded size limit]"

#: Initial backoff interval for consume retries (milliseconds)
CONSUMER_BACKOFF_INITIAL_MS = 1000

#: Maximum backoff interval for consume retries (milliseconds)
CONSUMER_BACKOFF_MAXIMUM_MS = 30000

#: Rate-limiter time window (seconds) — rate limit per minute
RATE_LIMITER_TIME_PERIOD_S = 60.0

#: Trace ID dedup cache TTL (seconds) — the same trace_id is processed only once within this window
#: Langfuse publishes multiple Kafka messages with the same trace_id over a trace's lifetime
#: 30 min covers the full lifecycle of most traces (create→update→complete),
#: so a long trace's Kafka messages are not sampled twice or sent to the LLM judge twice
TRACE_DEDUP_TTL_S = 1_800

#: Task-level heartbeat timeout (seconds) — a running task older than this may be reset to pending.
#: Must be greater than EVALUATE_TIMEOUT_S (benchmark 300s),
#: otherwise a task can be reset for heartbeat timeout while an evaluator call is in flight and run twice.
#: 600s (10 min) leaves a 60s buffer over the longest 540s evaluator call.
TASK_STALE_TIMEOUT_S = 600

#: Run-level heartbeat timeout (seconds) — a non-terminal run older than this may be recovered as FAILED.
#: Used by recover_stale_runs at startup: after a crash and restart, a run whose heartbeat exceeds this threshold
#: is marked FAILED. Must be greater than EVALUATE_TIMEOUT_S, otherwise a crash during an evaluator call
#: followed by restart marks the run dead too early and the scanner cannot resume the remaining tasks.
#: 600s gives the scanner a 10 min window to re-dispatch workers on the new process.
RUN_STALE_TIMEOUT_S = 600


__all__ = [
    "CACHE_TTL_SEC",
    "CONSUMER_BACKOFF_INITIAL_MS",
    "CONSUMER_BACKOFF_MAXIMUM_MS",
    "DEFAULT_ENVIRONMENT",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_PAGE_SIZE",
    "MAX_CONCURRENT_WORKERS",
    "RUN_SCANNER_INTERVAL_S",
    "RATE_LIMITER_TIME_PERIOD_S",
    "RUN_STALE_TIMEOUT_S",
    "TASK_STALE_TIMEOUT_S",
    "TRACE_DEDUP_TTL_S",
    "TRUNCATION_MARKER",
]
