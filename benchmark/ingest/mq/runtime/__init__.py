"""MQ runtime layer — config + lifecycle orchestration + bounded concurrency pool + consume-only QueueManager.

The four modules each have one job:

- :mod:`~benchmark.ingest.mq.runtime.config` — :class:`EvalMQConfig` + load/project helpers.
- :mod:`~benchmark.ingest.mq.runtime.service` — :class:`EvalMQConsumerService`: consume loop +
  worker pool + lifecycle orchestration.
- :mod:`~benchmark.ingest.mq.runtime.worker` — :class:`EvalWorkerPool`: fixed concurrency + bounded queue.
- :mod:`~benchmark.ingest.mq.runtime.queue_manager` — :class:`ConsumerOnlyMetaqQueueManager`.

External callers import from this package facade; the implementation is split across the four modules above.
"""

from benchmark.ingest.mq.runtime.config import (
    ConsumerTopicConfig,
    EvalMQConfig,
    load_mq_config,
    resolve_namesrv_domain,
    should_start_consumer,
    to_backend_configuration,
)
from benchmark.ingest.mq.runtime.queue_manager import ConsumerOnlyMetaqQueueManager
from benchmark.ingest.mq.runtime.service import EvalMQConsumerService
from benchmark.ingest.mq.runtime.worker import EvalWorkerPool

__all__ = [
    "ConsumerOnlyMetaqQueueManager",
    "ConsumerTopicConfig",
    "EvalMQConfig",
    "EvalMQConsumerService",
    "EvalWorkerPool",
    "load_mq_config",
    "resolve_namesrv_domain",
    "should_start_consumer",
    "to_backend_configuration",
]
