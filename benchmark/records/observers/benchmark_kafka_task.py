"""BenchmarkKafkaTask lifecycle observer.

Emits domain-level observation logs when a streaming task-mirror row is saved or deleted.
Registered automatically at application startup via ``DatabaseConfig.observers``.

Notes:
  - Mirror rows are written with ``update_or_create``: the UPDATE branch does not pass
    through the ``save()`` lifecycle (only the INSERT branch goes through ``create()``
    and fires ``on_save``); delete goes through ``row.delete()`` and fires ``on_delete``.
  - Observers only emit observation logs and do not carry online business logic
    (constitution §4.6).
"""

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.models.observer import ModelObserver

from benchmark.records.models.benchmark_kafka_task import BenchmarkKafkaTask

logger = get_logger("agent_eval.observers.benchmark_kafka_task")


class BenchmarkKafkaTaskObserver(ModelObserver[BenchmarkKafkaTask]):
    """Lifecycle observer for BenchmarkKafkaTask.

    Responsibilities:
      - **on_save**: record mirror-row creation (task name / declared state / consumer group),
        at debug level.
      - **on_delete**: record mirror-row deletion (= streaming task deletion), at info level (audit).
    """

    async def on_save(self, instance: BenchmarkKafkaTask) -> None:
        """Post-save: record mirror-row creation (INSERT branch)."""
        logger.debug(
            "benchmark_kafka_task saved: task_id=%s state=%s source=%s group_id=%s topics=%s",
            instance.task_id,
            instance.state,
            instance.source,
            instance.group_id,
            instance.topics,
        )

    async def on_delete(self, instance: BenchmarkKafkaTask) -> None:
        """Mirror row deleted (= streaming task removed from the declaration); record an audit event."""
        logger.info(
            "benchmark_kafka_task deleted: task_id=%s state=%s source=%s",
            instance.task_id,
            instance.state,
            instance.source,
        )


__all__ = ["BenchmarkKafkaTaskObserver"]
