"""eval_runtime — shared worker-runtime pieces for benchmark batch and realtime streaming.

Run-level lifecycle (heartbeat / fail / stale recovery) and the benchmark multi-node heartbeat loop.
Generic asyncio heartbeats live in :mod:`benchmark.libs.async_runtime`.
"""

from benchmark.libs.async_runtime import (
    RUN_HEARTBEAT_INTERVAL_S,
    HeartbeatSpec,
    cancel_and_wait,
    heartbeat_loop,
)
from benchmark.lifecycle.run_lifecycle import (
    RUN_STALE_TIMEOUT_S,
    STALE_RECOVERY_REASON,
    RunLifecycleBinding,
    fail_eval_run,
    fail_inflight_runs,
    fail_run,
    heartbeat_run,
    recover_stale_runs,
)

__all__ = [
    "RUN_HEARTBEAT_INTERVAL_S",
    "RUN_STALE_TIMEOUT_S",
    "STALE_RECOVERY_REASON",
    "HeartbeatSpec",
    "RunLifecycleBinding",
    "cancel_and_wait",
    "fail_eval_run",
    "fail_inflight_runs",
    "fail_run",
    "heartbeat_loop",
    "heartbeat_run",
    "recover_stale_runs",
]
