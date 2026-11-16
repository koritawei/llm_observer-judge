"""eval_workers — assembly layer for eval-task workers and runtime orchestration.

Dependency direction (package-level one-way; reverse imports forbidden)::

    eval_workers → eval_queue (queue infrastructure) / benchmark /
    eval_framework / eval_runtime

This package is the top assembly point of the dependency graph: worker / healer /
external_runner depend on both queue infrastructure (slots, retry, requeue) and
benchmark business services, so they were moved out of ``eval_queue`` /
``benchmark.ieval`` into their own package, removing the former package-level
``benchmark ↔ eval_queue`` cycle.

Module layout:
- ``worker_skeleton`` excluded (lives in :mod:`benchmark.queue`, pure infrastructure)
- ``benchmark_worker`` — benchmark batch-eval processor
- ``external_worker`` — async-callback eval processor + attribution processor
- ``running_healer`` — crash/restart recovery (heartbeat + orphan running + TTL)
- ``finalize_trigger`` — atomic CAS completion detection + report trigger
- ``external_runner`` — ``ExternalBenchmarkRunner`` external-trigger orchestrator
- ``daily_scheduler`` — daily OSS data schedule trigger (composes external_runner)
"""

from benchmark.workers.attribution_worker import (
    create_attribution_wait_worker,
)
from benchmark.workers.benchmark_worker import create_benchmark_worker
from benchmark.workers.external_runner import ExternalBenchmarkRunner
from benchmark.workers.external_worker import create_external_wait_worker
from benchmark.workers.finalize_trigger import try_finalize_external
from benchmark.workers.running_healer import RunningHealer

__all__ = [
    "ExternalBenchmarkRunner",
    "RunningHealer",
    "create_attribution_wait_worker",
    "create_benchmark_worker",
    "create_external_wait_worker",
    "try_finalize_external",
]
