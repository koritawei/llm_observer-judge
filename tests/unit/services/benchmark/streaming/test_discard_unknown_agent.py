"""Unit tests: Kafka BenchmarkStreamingEvaluator discards unknown / missing agent_name."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorDeps,
)
from benchmark.streaming.metadata import is_discard_agent_name
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.libs.mq.consumer import ConsumedMessage


def _make_msg(record: TraceRecord) -> ConsumedMessage[RecordStreamMessage]:
    payload = RecordStreamMessage(
        record_type="trace",
        record=record.model_dump(),
    )
    return ConsumedMessage(
        topic="langfuse.records",
        payload=payload,
        key=f"{record.project_id}:{record.id}",
        offset=100,
        partition=0,
    )


def _make_evaluator(
    evaluator_service: AsyncMock, config_store: MagicMock
) -> BenchmarkStreamingEvaluator:
    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(metadata={}, skip_trace_resolve=True)

    return BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=evaluator_service,
            request_converter=_converter,
        ),
    )


class TestIsDiscardAgentName:
    def test_none_and_blank(self) -> None:
        assert is_discard_agent_name(None) is True
        assert is_discard_agent_name("") is True
        assert is_discard_agent_name("   ") is True

    def test_unknown_sentinel_case_insensitive(self) -> None:
        assert is_discard_agent_name("unknown") is True
        assert is_discard_agent_name("Unknown") is True
        assert is_discard_agent_name(" UNKNOWN ") is True

    def test_valid_names(self) -> None:
        assert is_discard_agent_name("amap_agent") is False
        assert is_discard_agent_name("vehicle-agent") is False


class TestDiscardUnknownAgentName:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metadata",
        [
            {"agent_name": "unknown"},
            {"agent_name": "Unknown"},
            {"agent_name": "  unknown  "},
            {},
            {"agent_name": ""},
            {"agent_name": "   "},
        ],
    )
    async def test_skips_before_evaluate(self, metadata: dict) -> None:
        """A trace missing a name is not evaluated; the discard point has moved later to S3' (one decision after the entry is fetched)."""
        config_store = MagicMock()
        config_store.get_configs.return_value = [
            EvalConfigEntry(evaluatorName="example", sampling=1.0),
        ]
        evaluator_service = AsyncMock()
        evaluator = _make_evaluator(evaluator_service, config_store)

        record = TraceRecord(
            id="t-unknown",
            project_id="proj-1",
            environment="prod",
            metadata=metadata,
        )
        with patch(
            "benchmark.streaming.consumer.get_agent_names_by_project_id",
            return_value=[],
        ):
            await evaluator(_make_msg(record))

        evaluator_service.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_registry_resolve_unknown_still_discarded(self) -> None:
        """RemoteConfig registry returning sentinel 'unknown' must also discard."""
        config_store = MagicMock()
        evaluator_service = AsyncMock()
        evaluator = _make_evaluator(evaluator_service, config_store)

        record = TraceRecord(
            id="t-reg",
            project_id="proj-1",
            environment="prod",
            metadata={},
        )
        with patch(
            "benchmark.streaming.consumer.get_agent_names_by_project_id",
            return_value=["unknown"],
        ):
            await evaluator(_make_msg(record))

        evaluator_service.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_agent_name_continues(self) -> None:
        config_store = MagicMock()
        config_store.get_configs.return_value = [
            EvalConfigEntry(evaluatorName="example", sampling=1.0),
        ]
        evaluator_service = AsyncMock()
        evaluator = _make_evaluator(evaluator_service, config_store)

        record = TraceRecord(
            id="t-ok",
            project_id="proj-1",
            environment="prod",
            metadata={"agent_name": "amap_agent"},
            input={"q": "test"},
            output={"a": "resp"},
        )
        await evaluator(_make_msg(record))

        assert evaluator_service.evaluate.call_count == 1

    @pytest.mark.asyncio
    async def test_legacy_agentName_key_still_accepted(self) -> None:
        """Upstream Langfuse traces may still carry camelCase agentName."""
        config_store = MagicMock()
        config_store.get_configs.return_value = [
            EvalConfigEntry(evaluatorName="example", sampling=1.0),
        ]
        evaluator_service = AsyncMock()
        evaluator = _make_evaluator(evaluator_service, config_store)

        record = TraceRecord(
            id="t-legacy",
            project_id="proj-1",
            environment="prod",
            metadata={"agentName": "amap_agent"},
            input={"q": "test"},
            output={"a": "resp"},
        )
        await evaluator(_make_msg(record))

        assert evaluator_service.evaluate.call_count == 1
