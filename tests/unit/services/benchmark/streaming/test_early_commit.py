"""Unit tests for Kafka early-commit wrapper in create_streaming_consumer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.streaming.consumer import KafkaConsumerConfig, create_streaming_consumer
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.libs.mq.consumer import ConsumedMessage

pytestmark = pytest.mark.asyncio


def _make_msg(*, offset: int = 42) -> ConsumedMessage[RecordStreamMessage]:
    record = TraceRecord(
        id="trace-1",
        project_id="proj-1",
        environment="prod",
        metadata={},
    )
    payload = RecordStreamMessage(
        record_type="trace",
        record=record.model_dump(),
    )
    return ConsumedMessage(
        topic="langfuse.records",
        payload=payload,
        key="proj-1:trace-1",
        offset=offset,
        partition=0,
    )


def _config() -> KafkaConsumerConfig:
    return KafkaConsumerConfig(
        topics=["langfuse.records"],
        group_id="test-group",
        bootstrap_servers="localhost:9092",
        auto_commit=True,
    )


async def test_holdback_runs_after_commit_before_processor() -> None:
    """commit → holdback → processor, even when the record is still fresh."""
    call_order: list[str] = []

    async def processor(msg: ConsumedMessage[RecordStreamMessage]) -> None:
        call_order.append("processor")

    async def fake_holdback(*args: object, **kwargs: object) -> float:
        call_order.append("holdback")
        return 110.0

    with (
        patch("benchmark.streaming.consumer.AioKafkaBackend"),
        patch("benchmark.streaming.consumer.MQConsumer") as mock_mq_cls,
        patch(
            "benchmark.streaming.consumer.kafka_message_holdback_s", return_value=110.0
        ),
        patch(
            "benchmark.streaming.consumer.wait_kafka_message_holdback",
            side_effect=fake_holdback,
        ),
    ):
        consumer_inst = MagicMock()
        consumer_inst.commit_message = AsyncMock(
            side_effect=lambda msg: call_order.append("commit"),
        )
        mock_mq_cls.return_value = consumer_inst

        create_streaming_consumer(config=_config(), processor=processor)
        _topic_def, wrapped = consumer_inst.subscribe.call_args[0]

        await wrapped(_make_msg())

    assert call_order == ["commit", "holdback", "processor"]


async def test_early_commit_before_processor_and_swallows_errors() -> None:
    """commit_message runs before processor; processor errors do not propagate."""
    call_order: list[str] = []

    async def failing_processor(msg: ConsumedMessage[RecordStreamMessage]) -> None:
        call_order.append("processor")
        raise RuntimeError("judge boom")

    with (
        patch("benchmark.streaming.consumer.AioKafkaBackend"),
        patch("benchmark.streaming.consumer.MQConsumer") as mock_mq_cls,
    ):
        consumer_inst = MagicMock()
        consumer_inst.commit_message = AsyncMock(
            side_effect=lambda msg: call_order.append("commit"),
        )
        mock_mq_cls.return_value = consumer_inst

        create_streaming_consumer(config=_config(), processor=failing_processor)

        assert consumer_inst.subscribe.call_count == 1
        _topic_def, wrapped = consumer_inst.subscribe.call_args[0]

        await wrapped(_make_msg())

    assert call_order == ["commit", "processor"]


async def test_early_commit_propagates_commit_failure() -> None:
    """If commit_message fails, error propagates and processor is not called."""
    processor = AsyncMock()

    with (
        patch("benchmark.streaming.consumer.AioKafkaBackend"),
        patch("benchmark.streaming.consumer.MQConsumer") as mock_mq_cls,
    ):
        consumer_inst = MagicMock()
        consumer_inst.commit_message = AsyncMock(
            side_effect=RuntimeError("broker down"),
        )
        mock_mq_cls.return_value = consumer_inst

        create_streaming_consumer(config=_config(), processor=processor)
        _topic_def, wrapped = consumer_inst.subscribe.call_args[0]

        with pytest.raises(RuntimeError, match="broker down"):
            await wrapped(_make_msg())

    processor.assert_not_awaited()
