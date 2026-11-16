"""Unit tests for the StreamingBenchmarkRunner lifecycle (§3.2 drain / §6.2 state machine / §6.3 topic diff)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.streaming.config import (
    KafkaConsumerConfig,
    StreamingKafkaConfig,
)
from benchmark.streaming.runner import StreamingBenchmarkRunner
from benchmark.streaming.tasks import StreamingTaskSpec

pytestmark = pytest.mark.asyncio


class _FakeConsumer:
    """The run loop continues until stop() is called (successful drain path)."""

    def __init__(self, *, hang: bool = False) -> None:
        self._running = False
        self.started = False
        self.closed = False
        self._stop = asyncio.Event()
        self._hang = hang

    async def start(self) -> None:
        self.started = True
        self._running = True

    async def run(self) -> None:
        if self._hang:
            await asyncio.Event().wait()  # Never returns (drain timeout path)
            return
        await self._stop.wait()
        self._running = False

    def stop(self) -> None:
        self._stop.set()

    async def close(self) -> None:
        self.closed = True


def _kafka_config(*, concurrency: int = 2) -> StreamingKafkaConfig:
    return StreamingKafkaConfig(
        consumer=KafkaConsumerConfig(
            topics=["langfuse.records"],
            group_id="agent-eval-judge",
            bootstrap_servers="localhost:9092",
        ),
        topics=["langfuse.records"],
        concurrency=concurrency,
        evaluate_timeout_s=0,
        max_msgs_per_min=0.0,
    )


def _runner(consumers: list[_FakeConsumer]) -> StreamingBenchmarkRunner:
    return StreamingBenchmarkRunner(
        evaluator_service=AsyncMock(),
        config_store=MagicMock(),
        kafka_config=_kafka_config(),
        kafka_enabled=True,
    )


def _spec(name: str = "t1", **updates: object) -> StreamingTaskSpec:
    payload: dict[str, object] = {
        "name": name,
        "topics": ["topic-a", "topic-b"],
        "group_id": "group-x",
    }
    payload.update(updates)
    return StreamingTaskSpec(**payload)  # type: ignore[arg-type]


async def test_start_task_creates_per_topic_instances() -> None:
    made: list[_FakeConsumer] = []

    def _factory(**kwargs: object) -> _FakeConsumer:
        consumer = _FakeConsumer()
        made.append(consumer)
        return consumer

    runner = _runner(made)
    with patch(
        "benchmark.streaming.runner.create_streaming_consumer",
        side_effect=_factory,
    ):
        task = await runner.start_task(_spec())
    # topics(2) × concurrency(2) = 4 instances; each instance subscribes to only one topic
    assert len(made) == 4
    assert task.instance_count == 4
    assert task.subscribed_topics == ["topic-a", "topic-b"]
    assert task.is_running
    assert runner.registry.get("t1") is task


async def test_start_task_is_idempotent_when_running() -> None:
    runner = _runner([])
    with patch(
        "benchmark.streaming.runner.create_streaming_consumer",
        side_effect=lambda **kw: _FakeConsumer(),
    ) as factory:
        await runner.start_task(_spec())
        again = await runner.start_task(_spec())
    assert factory.call_count == 4  # The second round does not rebuild
    assert again.is_running


async def test_stop_task_drains_and_removes() -> None:
    runner = _runner([])
    with patch(
        "benchmark.streaming.runner.create_streaming_consumer",
        side_effect=lambda **kw: _FakeConsumer(),
    ):
        await runner.start_task(_spec())
        task = runner.registry.get("t1")
        assert task is not None
        await runner.stop_task("t1", remove=True)
    assert runner.registry.get("t1") is None
    assert task.instance_count == 0
    # After drain, stop_events / run_tasks are cleared
    assert task.stop_events == {}
    assert task.run_tasks == {}


async def test_pause_then_resume_cycle() -> None:
    runner = _runner([])
    with patch(
        "benchmark.streaming.runner.create_streaming_consumer",
        side_effect=lambda **kw: _FakeConsumer(),
    ):
        await runner.start_task(_spec())
        assert await runner.pause_task("t1") is True
        task = runner.registry.get("t1")
        assert task is not None
        assert task.runtime_state == "paused"
        assert not task.is_running
        assert await runner.resume_task("t1") is True
        task = runner.registry.get("t1")
        assert task is not None
        assert task.runtime_state == "running"
        assert task.is_running


async def test_pause_unknown_task_returns_false() -> None:
    runner = _runner([])
    assert await runner.pause_task("ghost") is False
    assert await runner.resume_task("ghost") is False


async def test_add_and_remove_topics_incremental() -> None:
    runner = _runner([])
    with patch(
        "benchmark.streaming.runner.create_streaming_consumer",
        side_effect=lambda **kw: _FakeConsumer(),
    ):
        task = await runner.start_task(_spec(topics=["topic-a"]))
        assert task.subscribed_topics == ["topic-a"]
        await runner.add_topics("t1", ["topic-b", "topic-a"])  # Duplicate elements are idempotent
        assert task.subscribed_topics == ["topic-a", "topic-b"]
        await runner.remove_topics("t1", ["topic-a"])
        assert task.subscribed_topics == ["topic-b"]
        assert task.is_running  # An unchanged instance is not rebuilt


async def test_drain_timeout_counts_stop_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    runner = _runner([])

    def _factory(**kwargs: object) -> _FakeConsumer:
        return _FakeConsumer(hang=True)

    with patch(
        "benchmark.streaming.runner.create_streaming_consumer",
        side_effect=_factory,
    ):
        task = await runner.start_task(_spec(topics=["topic-a"]))
        with caplog.at_level(logging.WARNING, logger="agent_eval.benchmark.streaming.funnel"):
            await runner.stop_task("t1", timeout=0.05, remove=True)
    # topics(1) × concurrency(2) = 2 hung instances are cancelled → stop_dropped=2;
    # flush_funnel at the end of drain prints and resets, so assert the log line rather than the counter after it is cleared.
    assert task.evaluator._funnel.stop_dropped == 0
    funnel_logs = [r.message for r in caplog.records if "Judge funnel:" in r.message]
    assert len(funnel_logs) == 1
    assert "stop_dropped=2" in funnel_logs[0]


async def test_start_owns_sidecar_and_does_not_start_kafka() -> None:
    runner = StreamingBenchmarkRunner(
        evaluator_service=AsyncMock(),
        config_store=MagicMock(),
        kafka_config=_kafka_config(),
        settings=MagicMock(),
    )
    sentinel = object()
    with patch(
        "benchmark.streaming.runner.start_sidecar_ingest",
        new=AsyncMock(return_value=sentinel),
    ) as start_sidecar:
        await runner.start()
    start_sidecar.assert_awaited_once()
    assert start_sidecar.await_args.kwargs["benchmark_judge"] is not None
    assert runner.sidecar_service is sentinel
    assert runner._reconciler_task is None
    assert runner.registry.names() == []


async def test_start_task_refuses_kafka_when_disabled() -> None:
    runner = StreamingBenchmarkRunner(
        evaluator_service=AsyncMock(),
        config_store=MagicMock(),
        kafka_config=_kafka_config(),
    )
    with pytest.raises(RuntimeError, match="kafka consumption is disabled"):
        await runner.start_task(_spec())


async def test_start_mounts_kafka_reconciler_when_enabled() -> None:
    runner = StreamingBenchmarkRunner(
        evaluator_service=AsyncMock(),
        config_store=MagicMock(),
        kafka_config=_kafka_config(),
        settings=MagicMock(),
        kafka_enabled=True,
    )
    reconciler = MagicMock()
    reconciler.reconcile_once = AsyncMock()
    reconciler.run = AsyncMock()
    with (
        patch(
            "benchmark.streaming.runner.start_sidecar_ingest",
            new=AsyncMock(return_value=object()),
        ),
        patch(
            "benchmark.streaming.runner.StreamingTopicReconciler",
            return_value=reconciler,
        ),
    ):
        await runner.start()
    reconciler.reconcile_once.assert_awaited_once()
    assert runner._reconciler is reconciler
    assert runner._reconciler_task is not None
    await runner._reconciler_task
