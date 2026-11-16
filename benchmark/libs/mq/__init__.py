"""Kafka and in-memory queues with the message_queue names the service already uses."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Generic, TypeVar


class BackendType(str, Enum):
    KAFKA = "kafka"
    METAQ = "metaq"
    MEMORY = "memory"


class Env(str, Enum):
    LOCAL = "local"
    DAILY = "daily"
    PRE = "pre"
    PROD = "prod"


class ConsumeError(Exception):
    pass


class CommitError(Exception):
    pass


class RetryExhaustedError(ConsumeError):
    pass


@dataclass
class RawMessage:
    topic: str
    payload: Any = None
    key: bytes | None = None
    partition: int = 0
    offset: int = 0
    timestamp_ms: int = 0
    headers: list | None = None
    value: Any = None

    def __post_init__(self) -> None:
        if self.payload in (None, b"") and self.value is not None:
            self.payload = self.value


PayloadT = TypeVar("PayloadT")


@dataclass
class ConsumedMessage(Generic[PayloadT]):
    topic: str
    payload: Any = None
    key: Any = None
    headers: list | None = None
    partition: int = 0
    offset: int = 0
    timestamp_ms: int = 0

    def json(self) -> Any:
        body = self.payload
        if isinstance(body, (bytes, bytearray)):
            return json.loads(body.decode("utf-8"))
        if isinstance(body, str):
            return json.loads(body)
        return body


class TopicDefinition(Generic[PayloadT]):
    def __init__(self, name: str = "", topic: str = "", payload_type: Any = None, **_extra: Any):
        self.name = name or topic
        self.topic = topic or name
        self.payload_type = payload_type


class Configuration:
    def __init__(
        self,
        bootstrap_servers: str = "127.0.0.1:9092",
        group_id: str = "benchmark",
        topics: list[str] | None = None,
        options: dict[str, Any] | None = None,
        env: Env = Env.LOCAL,
        backend: BackendType = BackendType.MEMORY,
        **extra: Any,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = list(topics or [])
        self.options = options or {}
        self.env = env
        self.backend = backend
        self.service_name = str(extra.get("service_name") or "")
        self.security_protocol = str(extra.get("security_protocol") or "")
        self.extra = extra
        for key, value in extra.items():
            setattr(self, key, value)


class JsonCodec:
    def encode(self, value: Any) -> bytes:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    def decode(self, payload: bytes, target_type: Any = None) -> Any:
        raw = json.loads(payload.decode("utf-8"))
        if target_type is None:
            return raw
        if hasattr(target_type, "model_validate"):
            return target_type.model_validate(raw)
        return raw


class BackoffPolicy(str, Enum):
    EXPONENTIAL = "exponential"
    CONSTANT = "constant"


class DefaultBackoffPerformer:
    def __init__(
        self,
        base_s: float | None = None,
        max_s: float | None = None,
        *,
        policy: BackoffPolicy | None = None,
        initial_backoff_ms: int = 200,
        maximum_backoff_ms: int = 5000,
        maximum_retries: int = 0,
    ):
        self.policy = policy or BackoffPolicy.EXPONENTIAL
        self.initial_ms = int(base_s * 1000) if base_s is not None else initial_backoff_ms
        self.maximum_ms = int(max_s * 1000) if max_s is not None else maximum_backoff_ms
        self.maximum_retries = maximum_retries
        self.attempt = 0

    async def wait(self) -> None:
        delay_ms = min(self.maximum_ms, self.initial_ms * (2**self.attempt))
        self.attempt += 1
        await asyncio.sleep(delay_ms / 1000)

    def reset(self) -> None:
        self.attempt = 0

    async def perform(self, attempt: Callable[[], Awaitable[bool]]) -> None:
        tries = 0
        while True:
            if await attempt():
                self.reset()
                return
            tries += 1
            if self.maximum_retries and tries >= self.maximum_retries:
                raise RetryExhaustedError("retries exhausted")
            await self.wait()


BackoffPerformer = DefaultBackoffPerformer
Processor = Callable[[ConsumedMessage], Awaitable[None]]


class AioKafkaBackend:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self._consumer = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        from aiokafka import AIOKafkaConsumer

        self._consumer = AIOKafkaConsumer(
            *self.configuration.topics,
            bootstrap_servers=self.configuration.bootstrap_servers,
            group_id=self.configuration.group_id,
            enable_auto_commit=False,
        )
        await self._consumer.start()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
        if self._consumer is not None:
            await self._consumer.stop()

    async def poll(self, timeout_s: float = 1) -> ConsumedMessage | None:
        if self._consumer is None:
            raise ConsumeError("consumer is not started")
        record = await self._consumer.getone()
        del timeout_s
        return ConsumedMessage(
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            timestamp_ms=record.timestamp,
            key=record.key,
            payload=record.value,
            headers=list(record.headers or []),
        )

    async def commit(self) -> None:
        if self._consumer is not None:
            await self._consumer.commit()


class MQConsumer:
    def __init__(
        self,
        backend: AioKafkaBackend | None = None,
        group_id: str = "",
        poll_timeout_s: float = 1,
        auto_commit: bool = False,
        backoff: DefaultBackoffPerformer | None = None,
        codec: Any = None,
        queue: asyncio.Queue | None = None,
    ):
        self.backend = backend
        self.group_id = group_id
        self.poll_timeout_s = poll_timeout_s
        self.auto_commit = auto_commit
        self.backoff = backoff or DefaultBackoffPerformer()
        self.codec = codec
        self._queue = queue
        self._processors: dict[str, Processor] = {}
        self._topics: dict[str, TopicDefinition] = {}
        self._running = False
        self._offset = 0

    def subscribe(self, topic: TopicDefinition, processor: Processor) -> None:
        self._processors[topic.topic] = processor
        self._topics[topic.topic] = topic
        if self.backend is not None and topic.topic not in self.backend.configuration.topics:
            self.backend.configuration.topics.append(topic.topic)

    async def start(self) -> None:
        self._running = True
        if self.backend is not None and getattr(self.backend, "_consumer", None) is None:
            if self.backend.configuration.topics:
                await self.backend.start()

    def stop(self) -> None:
        self._running = False

    async def commit_message(self, _message: ConsumedMessage) -> None:
        if self.backend is not None:
            await self.backend.commit()

    async def commit(self) -> None:
        await self.commit_message(ConsumedMessage(topic=""))

    async def poll(self, timeout_s: float | None = None) -> ConsumedMessage | None:
        timeout = self.poll_timeout_s if timeout_s is None else timeout_s
        if self._queue is None:
            if self.backend is None:
                return None
            return await self.backend.poll(timeout)
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return None
        topic_name = next(iter(self._topics), "")
        topic = self._topics.get(topic_name)
        payload = item
        if self.codec is not None and isinstance(item, (bytes, bytearray, str)):
            data = item.encode("utf-8") if isinstance(item, str) else bytes(item)
            payload = self.codec.decode(data, topic.payload_type if topic else None)
        elif self.codec is not None and not isinstance(item, (bytes, bytearray, str)):
            data = self.codec.encode(item)
            payload = self.codec.decode(data, topic.payload_type if topic else None)
        self._offset += 1
        return ConsumedMessage(topic=topic_name, payload=payload, partition=0, offset=self._offset)

    async def run(self) -> None:
        self._running = True
        try:
            while self._running:
                message = await self.poll(self.poll_timeout_s)
                if message is None:
                    continue
                processor = self._processors.get(message.topic)
                if processor is None:
                    continue
                try:
                    await processor(message)
                    if self.auto_commit:
                        await self.commit_message(message)
                    self.backoff.reset()
                except Exception:
                    await self.backoff.wait()
                    raise
        finally:
            self._running = False

    def on_error(self, error: Exception) -> None:
        del error


class MetaqBackend(AioKafkaBackend):
    """The old broker name. Local and tests use the in-memory queue."""


class MetaqConsumerBackend(MetaqBackend):
    def __init__(self, configuration: Configuration):
        super().__init__(configuration)


class BaseBackend(AioKafkaBackend):
    pass


class QueueManager:
    def __init__(self, configuration: Configuration | None = None):
        self.configuration = configuration or Configuration()
        self._queues: dict[str, asyncio.Queue] = {}
        self._consumers: list[MQConsumer] = []
        self._backend_configs = {
            BackendType.METAQ: self.configuration,
            BackendType.KAFKA: self.configuration,
            BackendType.MEMORY: self.configuration,
        }

    def _queue_for(self, topic: str) -> asyncio.Queue:
        return self._queues.setdefault(topic, asyncio.Queue())

    def register_producer(self, topic: TopicDefinition) -> None:
        self._queue_for(topic.topic)

    def register_consumer(
        self,
        topic: TopicDefinition,
        processor: Processor,
        group_id: str = "",
        poll_timeout_s: float = 1,
        auto_commit: bool = False,
        codec: Any = None,
    ) -> MQConsumer:
        consumer = MQConsumer(
            group_id=group_id,
            poll_timeout_s=poll_timeout_s,
            auto_commit=auto_commit,
            codec=codec,
            queue=self._queue_for(topic.topic),
        )
        consumer.subscribe(topic, processor)
        self._consumers.append(consumer)
        return consumer

    async def publish(self, topic: TopicDefinition, payload: Any) -> None:
        await self._queue_for(topic.topic).put(payload)

    async def bootstrap(self) -> None:
        return None

    async def shutdown(self) -> None:
        for consumer in self._consumers:
            consumer.stop()

    def consumer(self, group_id: str) -> MQConsumer:
        return MQConsumer(AioKafkaBackend(self.configuration), group_id)

    def _create_consumer_backend(self, backend_type: BackendType) -> BaseBackend:
        return MetaqBackend(self._backend_configs[backend_type])
