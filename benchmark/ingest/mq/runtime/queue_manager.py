"""Consumer-only QueueManager — avoid a duplicate producer start on the shared-library MetaQ consume path.

:class:`~message_queue.QueueManager` in ``libs/message_queue`` creates **two** ``MetaqBackend``
instances for METAQ inside ``register_consumer``: one extra default/producer backend (built via
``_get_or_create_backend`` and stored in ``_backends``, used only for the InMemory check) plus one
backend actually used by the consumer. They share the same ``producer_group``; ``bootstrap()`` starts the former's
producer and ``consumer.start()`` starts the latter's, but rocketmq-client-cpp does not allow the same producer group
to be registered twice **in one process**, so it raises
``The producer group[...] has been created before``.

agent-eval only consumes and never produces, so this class overrides :meth:`_create_consumer_backend`: for METAQ it no longer
creates and stores the extra backend via ``_get_or_create_backend``, and builds only the one the consumer uses. Then
``_backends`` stays empty → ``bootstrap()`` starts no producer → only ``consumer.start()`` starts
**one** producer, with no duplicate and no conflict. Other backends (InMemory / Swift / Kafka) delegate to the parent,
and behavior is unchanged.

⚠️ This class depends on shared-library internals (``_create_consumer_backend`` / ``_backend_configs`` /
``message_queue._backends.metaq.MetaqBackend``). If ``libs/message_queue`` refactors those internals,
this class must be updated in step. **The real fix belongs in the shared library: lazy-start the MetaQ producer** (connect on the first ``send``),
and this class can be removed then.
"""

from benchmark.libs.mq import BackendType, QueueManager
from benchmark.libs.mq._backends.base import BaseBackend
from benchmark.libs.mq._backends.metaq import MetaqBackend


class ConsumerOnlyMetaqQueueManager(QueueManager):
    """Consume-only QueueManager: the METAQ consume path does not create an extra producer backend."""

    def _create_consumer_backend(self, backend_type: BackendType) -> BaseBackend:
        if backend_type == BackendType.METAQ:
            # Skip the parent's _get_or_create_backend (it would store an extra producer
            # backend in the same group; once bootstrap starts it, it collides with this consumer backend's producer).
            return MetaqBackend(self._backend_configs[backend_type])
        return super()._create_consumer_backend(backend_type)


__all__ = ["ConsumerOnlyMetaqQueueManager"]
