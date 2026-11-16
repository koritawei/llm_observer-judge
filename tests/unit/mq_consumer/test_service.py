"""End-to-end unit tests for EvalMQConsumerService (InMemory backend; no rocketmq/broker).

Inject the service with ``QueueManager(Configuration())`` (env=LOCAL → InMemoryBackend).
The producer and consumer share one MessageStore. This verifies the full chain publish → consume loop → bounded-concurrency worker pool
→ handler → EvaluatorService.evaluate, focused on the "bounded concurrency, and a task retries until
success before advancing" semantics:

- Ordinary messages are all processed (global order is not guaranteed under concurrency).
- Inside a single worker: a 5xx failure retries until success before the next message is processed (order preserved).
- Under concurrency: each task retries until success on its own, without affecting the others.
- A 4xx permanent failure → skip (advance), and do not retry forever occupying a worker.
- A missing evaluator name → a bad route → skip (not evaluated, but already consumed, and advance).
"""

import asyncio

import pytest
from benchmark.evaluators.base import EvaluatorRequest, EvaluatorResult
from benchmark.ingest.mq.protocol import register_converter
from benchmark.ingest.mq.protocol.dispatch import _CONVERTERS
from benchmark.ingest.mq.runtime import ConsumerTopicConfig, EvalMQConfig, EvalMQConsumerService
from benchmark.ingest.mq.runtime.service import _RETRY_ALERT_ATTEMPTS, _alert_attempts
from benchmark.services.evaluation import EvalRunOptions
from benchmark.utils.exceptions import (
    EVALUATION_FAILED_CODE,
    MODEL_NOT_FOUND_CODE,
    EvaluatorError,
    evaluator_status,
)
from benchmark.libs.mq import Configuration, QueueManager, TopicDefinition

from tests.unit.mq_consumer.stubs import STUB_SOURCE, StubConverter, StubTaskMessage

_TOPIC_NAME = "luoshu_agent_eval_pre"


@pytest.fixture(autouse=True)
def _register_stub_converter():
    """Temporarily register StubConverter for this module by its source (messages map back through a codec round trip).

    Routing is topic → source → converter: the source key in the consumers config is :data:`STUB_SOURCE`,
    and the codec locates this converter from it. It is removed from the global :data:`_CONVERTERS` afterward, so it does not leak into other tests.
    """

    register_converter(StubConverter())
    try:
        yield
    finally:
        _CONVERTERS.pop(STUB_SOURCE, None)


class _ScriptedEvaluatorService:
    """Script behavior by trace_id and record every evaluate call (including retries).

    - ``fail_n_times(trace_id, n)``: that trace raises 5xx (retryable) for the first n calls, then succeeds.
    - ``permanent_fail(trace_id)``: that trace always raises 4xx (permanent; it should be skipped and not retried).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []  # Record trace_id in call order
        self._remaining_fails: dict[str, int] = {}
        self._permanent: set[str] = set()

    def fail_n_times(self, trace_id: str, n: int) -> None:
        self._remaining_fails[trace_id] = n

    def permanent_fail(self, trace_id: str) -> None:
        self._permanent.add(trace_id)

    async def evaluate(
        self,
        evaluator_name: str,
        request: EvaluatorRequest,
        *,
        options: EvalRunOptions | None = None,
    ) -> EvaluatorResult:
        trace_id = request.trace_id or ""
        self.calls.append(trace_id)
        if trace_id in self._permanent:
            raise EvaluatorError(
                code=evaluator_status(MODEL_NOT_FOUND_CODE),
                error_type=MODEL_NOT_FOUND_CODE,
                message="permanent",
            )
        left = self._remaining_fails.get(trace_id, 0)
        if left > 0:
            self._remaining_fails[trace_id] = left - 1
            raise EvaluatorError(
                code=evaluator_status(EVALUATION_FAILED_CODE),
                error_type=EVALUATION_FAILED_CODE,
                message="transient",
            )
        return EvaluatorResult(score=0.9, reason="ok")


def _config(*, worker_concurrency: int = 2) -> EvalMQConfig:
    # The backoff is set very small, so "retry until success" converges quickly in the test.
    return EvalMQConfig(
        consumers={STUB_SOURCE: ConsumerTopicConfig(topic=_TOPIC_NAME, consumer_group="CID-test")},
        poll_timeout_s=0.02,
        worker_concurrency=worker_concurrency,
        worker_queue_size=10,
        retry_initial_backoff_ms=5,
        retry_max_backoff_ms=20,
    )


def _build(
    fake: _ScriptedEvaluatorService, *, worker_concurrency: int = 2
) -> tuple[QueueManager, TopicDefinition[StubTaskMessage], EvalMQConsumerService]:
    topic: TopicDefinition[StubTaskMessage] = TopicDefinition(
        name=_TOPIC_NAME, payload_type=StubTaskMessage
    )
    manager = QueueManager(Configuration())  # LOCAL → InMemoryBackend
    manager.register_producer(topic)
    service = EvalMQConsumerService(
        evaluator_service=fake,
        config=_config(worker_concurrency=worker_concurrency),
        env="local",
        queue_manager=manager,
    )
    return manager, topic, service


async def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.02) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


async def test_publish_triggers_evaluation() -> None:
    fake = _ScriptedEvaluatorService()
    manager, topic, service = _build(fake)
    await service.start()
    try:
        await manager.publish(
            topic,
            StubTaskMessage(
                evaluator_name="judge_example",
                input={"query": "x"},
                output={"answer": "y"},
                trace_id="t-1",
                project_id="p",
                agent_name="a",
            ),
        )
        assert await _wait_until(lambda: fake.calls == ["t-1"])
    finally:
        await service.stop()
        await manager.shutdown()


async def test_multiple_messages_all_processed() -> None:
    fake = _ScriptedEvaluatorService()
    manager, topic, service = _build(fake, worker_concurrency=2)
    await service.start()
    try:
        for i in range(5):
            await manager.publish(
                topic,
                StubTaskMessage(evaluator_name="judge_example", trace_id=f"t-{i}"),
            )
        assert await _wait_until(lambda: len(fake.calls) == 5)
        # Under concurrency, global order is not guaranteed, but every message must be processed.
        assert set(fake.calls) == {f"t-{i}" for i in range(5)}
    finally:
        await service.stop()
        await manager.shutdown()


async def test_single_worker_retries_until_success_before_advancing() -> None:
    # A single worker: the current task retries on failure until it succeeds, and only then advances to the next one (order preserved).
    fake = _ScriptedEvaluatorService()
    fake.fail_n_times("A", 2)  # A returns 5xx the first two times and succeeds the third
    manager, topic, service = _build(fake, worker_concurrency=1)
    await service.start()
    try:
        await manager.publish(topic, StubTaskMessage(evaluator_name="judge_example", trace_id="A"))
        await manager.publish(topic, StubTaskMessage(evaluator_name="judge_example", trace_id="B"))
        assert await _wait_until(lambda: "B" in fake.calls)
        assert fake.calls.count("A") == 3  # 2 failures + 1 success
        first_b = fake.calls.index("B")
        assert set(fake.calls[:first_b]) == {"A"}  # Every attempt of A happens before B
        assert fake.calls[first_b:] == ["B"]
    finally:
        await service.stop()
        await manager.shutdown()


async def test_concurrent_tasks_each_retry_until_success() -> None:
    # Concurrent: each task retries until it succeeds, and they do not block each other.
    fake = _ScriptedEvaluatorService()
    fake.fail_n_times("A", 2)  # A needs 3 attempts
    manager, topic, service = _build(fake, worker_concurrency=2)
    await service.start()
    try:
        await manager.publish(topic, StubTaskMessage(evaluator_name="judge_example", trace_id="A"))
        await manager.publish(topic, StubTaskMessage(evaluator_name="judge_example", trace_id="B"))
        assert await _wait_until(lambda: fake.calls.count("A") == 3 and "B" in fake.calls)
        assert fake.calls.count("B") == 1  # B succeeds once
    finally:
        await service.stop()
        await manager.shutdown()


async def test_permanent_failure_skipped_not_retried() -> None:
    fake = _ScriptedEvaluatorService()
    fake.permanent_fail("A")  # A is always 4xx → skip, and do not retry
    manager, topic, service = _build(fake, worker_concurrency=1)
    await service.start()
    try:
        await manager.publish(topic, StubTaskMessage(evaluator_name="judge_example", trace_id="A"))
        await manager.publish(topic, StubTaskMessage(evaluator_name="judge_example", trace_id="B"))
        assert await _wait_until(lambda: "B" in fake.calls)
        assert fake.calls.count("A") == 1  # A permanent error is attempted only once and does not occupy a worker
        assert fake.calls == ["A", "B"]
    finally:
        await service.stop()
        await manager.shutdown()


async def test_message_without_evaluator_name_is_skipped() -> None:
    fake = _ScriptedEvaluatorService()
    manager, topic, service = _build(fake, worker_concurrency=1)
    await service.start()
    try:
        # A missing evaluator name → a bad route → skip (not evaluated, but already consumed); the next one is evaluated normally. A single worker preserves order.
        await manager.publish(topic, StubTaskMessage(evaluator_name="", trace_id="skip"))
        await manager.publish(topic, StubTaskMessage(evaluator_name="judge_example", trace_id="ok"))
        assert await _wait_until(lambda: fake.calls == ["ok"])
    finally:
        await service.stop()
        await manager.shutdown()


@pytest.mark.parametrize(
    ("retry_max_attempts", "expected"),
    [
        (20, 10),  # Current config in all three environments: warn on the 10th attempt, far earlier than giving up on the 20th
        (40, 20),
        (6, 3),
        (2, 1),
        (1, 1),  # It must not be 0, or the first failure is already reported as ERROR
    ],
)
def test_alert_attempts_derived_from_finite_retry_cap(
    retry_max_attempts: int, expected: int
) -> None:
    assert _alert_attempts(retry_max_attempts) == expected
    # The warning must be strictly earlier than giving up (with a limit of 1 they can only happen together).
    assert expected <= retry_max_attempts


def test_alert_attempts_falls_back_to_constant_for_infinite_retry() -> None:
    # retry_max_attempts=0 = retry forever, so there is no "give up" to compare against → use a fixed value.
    assert _alert_attempts(0) == _RETRY_ALERT_ATTEMPTS
