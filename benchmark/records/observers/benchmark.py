"""Benchmark lifecycle observer.

Emits domain-level observation logs when a BenchmarkRun is saved or deleted.
Registered automatically at application startup via ``DatabaseConfig.observers``.

Notes:
  - ``BenchmarkRunItem`` is written with ``bulk_create`` and cleaned up by run
    cascade delete. It does not pass through the ``save()`` lifecycle, so it has
    no observer (attaching one would not fire).
  - Observers only emit observation logs and do not carry online business logic
    (constitution §4.6).
"""

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.models.observer import ModelObserver

from benchmark.records.models.benchmark_run import BenchmarkRun

logger = get_logger("agent_eval.observers.benchmark")


class BenchmarkRunObserver(ModelObserver[BenchmarkRun]):
    """Lifecycle observer for BenchmarkRun.

    Triggered directly by the post-save call inside the ``BenchmarkRun.save()`` override.

    Responsibilities:
      - **on_save**: record the run save event (status + evaluation progress), at debug level.
      - **on_delete**: record the run delete event, for audit.
    """

    async def on_save(self, instance: BenchmarkRun) -> None:
        """Post-save: record run status and criterion progress."""
        logger.debug(
            "benchmark_run saved: run_id=%s status=%s benchmark=%s service=%s completed=%s/%s",
            instance.run_id,
            instance.status,
            instance.benchmark_name,
            instance.service_name,
            instance.completed_evaluators,
            instance.total_evaluators,
        )

    async def on_delete(self, instance: BenchmarkRun) -> None:
        """Fires after a BenchmarkRun is deleted and records the delete event."""
        logger.info(
            "benchmark_run deleted: run_id=%s benchmark=%s",
            instance.run_id,
            instance.benchmark_name,
        )


__all__ = ["BenchmarkRunObserver"]
