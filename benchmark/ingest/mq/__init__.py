"""agent-eval MQ consumer module.

A RocketMQ consumer built on the monorepo shared library ``libs/message_queue`` (MetaqBackend + QueueManager +
MQConsumer): it consumes eval tasks on the ``luoshu_agent_eval`` topic and,
after decode, triggers the eval pipeline in :class:`benchmark.services.evaluation.EvaluatorService`.

Four design goals:

- **Availability**: enable switch + IP canary + graceful shutdown + alerts on consume-loop exceptions.
- **Reliability**: each task retries with backoff until success, and the offset advances only after success; missing evaluator name / permanent errors are skipped.
- **Extensibility**: tolerant payload normalization + handler / topic registration seams.
- **Bounded resources**: a fixed concurrent worker pool + a bounded queue + blocking enqueue backpressure (**no unbounded memory growth**).

Public entry: :class:`EvalMQConsumerService` (assembled by the ``benchmark.service.main`` lifespan).
"""

from benchmark.ingest.mq.enrichment import Enricher, enrich, register_enricher
from benchmark.ingest.mq.executor import EvalMessageExecutor
from benchmark.ingest.mq.protocol import (
    BaseTaskMessage,
    ProtocolConverter,
    TolerantEvalCodec,
    convert_for_source,
    register_converter,
)
from benchmark.ingest.mq.runtime import (
    EvalMQConfig,
    EvalMQConsumerService,
    EvalWorkerPool,
    load_mq_config,
    resolve_namesrv_domain,
    should_start_consumer,
    to_backend_configuration,
)

# Import each source package so its converter / enricher self-registers (must follow the framework imports;
# isort: skip pins the position so alphabetical reordering cannot break init order).
from benchmark.ingest.mq import consumers  # noqa: F401  # isort: skip
from benchmark.ingest.mq.consumers.luoshu import (  # isort: skip
    LuoshuConverter,
    LuoshuTraceEnricher,
)

__all__ = [
    "BaseTaskMessage",
    "Enricher",
    "EvalMQConfig",
    "EvalMQConsumerService",
    "EvalMessageExecutor",
    "EvalWorkerPool",
    "LuoshuConverter",
    "LuoshuTraceEnricher",
    "ProtocolConverter",
    "TolerantEvalCodec",
    "convert_for_source",
    "enrich",
    "load_mq_config",
    "register_converter",
    "register_enricher",
    "resolve_namesrv_domain",
    "should_start_consumer",
    "to_backend_configuration",
]
