"""Unit tests for BenchmarkStreamingEvaluator B5 error handling strategy.

Verifies the domain-driven error strategy in the B5 evaluate loop
(after Kafka early-commit the whole message is not retried, and fan-out continues per entry):

1. ``TimeoutError`` → skip config (continue)
2. ``EvaluatorError`` with ``code < 500`` → permanent → skip config
3. ``EvaluatorError`` with ``code >= 500`` → transient → skip config
4. Other exceptions (429, network) → transient → skip config

The key scenario from production: a trace record whose ``metadata``
lacks ``agent_name`` causes ``EvaluatorError(metadata_missing)``.  This is a
permanent error — the message is immutable, retrying won't add
``agent_name``.  The fix ensures it is skipped, not retried 3×.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorConfig,
    BenchmarkStreamingEvaluatorDeps,
)
from benchmark.streaming.schemas import (
    RecordStreamMessage,
    TraceRecord,
)
from benchmark.utils.exceptions import (
    EVALUATION_FAILED_CODE,
    LLM_FAILED_CODE,
    METADATA_MISSING_CODE,
    EvaluatorError,
    evaluator_status,
)
from benchmark.libs.mq.consumer import ConsumedMessage

pytestmark = pytest.mark.asyncio


# ── Helpers ─────────────────────────────────────────────────────────────


_TRACE_NAME = "shared.stream.event"


def _make_config(
    evaluator_name: str = "example",
    sampling: float = 1.0,
    environment: list[str] | None = None,
) -> EvalConfigEntry:
    # When several entries share a project, one with no name is dropped by retain_*; these tests always set a name.
    return EvalConfigEntry(
        evaluatorName=evaluator_name,
        sampling=sampling,
        environment=environment,
        name=[_TRACE_NAME],
    )


def _make_trace_record(
    trace_id: str = "trace-001",
    project_id: str = "proj-001",
    environment: str = "prod",
    metadata: dict | None = None,
) -> TraceRecord:
    return TraceRecord(
        id=trace_id,
        project_id=project_id,
        name=_TRACE_NAME,
        environment=environment,
        metadata=metadata if metadata is not None else {"agent_name": "agent-a"},
        input={"q": "test"},
        output={"a": "resp"},
    )


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
    *,
    evaluator_service: AsyncMock,
    config_store: MagicMock,
    evaluate_timeout_s: int = 0,
) -> BenchmarkStreamingEvaluator:
    """Build a BenchmarkStreamingEvaluator with mocked dependencies."""

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(metadata={}, skip_trace_resolve=True)

    return BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=evaluator_service,
            request_converter=_converter,
        ),
        config=BenchmarkStreamingEvaluatorConfig(evaluate_timeout_s=evaluate_timeout_s),
    )


# ── Permanent errors: should skip config (no raise) ─────────────────────


class TestPermanentEvaluatorErrors:
    """400-level EvaluatorErrors are permanent — skip config, don't raise."""

    @pytest.mark.parametrize(
        ("error_type", "expected_code"),
        [
            (METADATA_MISSING_CODE, 400),
            ("model_not_found", 400),
            ("evaluator_prompt_not_found", 400),
            ("trace_not_found", 400),
        ],
    )
    async def test_400_level_error_skips_config(self, error_type: str, expected_code: int) -> None:
        """EvaluatorError with code < 500 → skip config, no exception raised."""

        config = _make_config(evaluator_name="example")
        config_store = MagicMock()
        config_store.get_configs.return_value = [config]

        evaluator_service = AsyncMock()
        evaluator_service.evaluate.side_effect = EvaluatorError(
            code=evaluator_status(error_type),
            error_type=error_type,
            message=f"test {error_type}",
        )

        evaluator = _make_evaluator(
            evaluator_service=evaluator_service,
            config_store=config_store,
        )

        record = _make_trace_record()
        # evaluate raises metadata_missing (permanent) — skip, no raise
        msg = _make_msg(record)

        # Should NOT raise — permanent error is skipped
        await evaluator(msg)

        # evaluate was called exactly once (not retried)
        assert evaluator_service.evaluate.call_count == 1

    async def test_metadata_missing_does_not_retry(self) -> None:
        """The production scenario: metadata_missing must not trigger retry.

        Before the fix, this would raise → MQConsumer retries 3× → all fail
        identically → retry exhausted.  After the fix, it skips immediately.
        """

        config = _make_config()
        config_store = MagicMock()
        config_store.get_configs.return_value = [config]

        evaluator_service = AsyncMock()
        evaluator_service.evaluate.side_effect = EvaluatorError(
            code=400,
            error_type=METADATA_MISSING_CODE,
            message="required metadata missing: agent_name",
        )

        evaluator = _make_evaluator(
            evaluator_service=evaluator_service,
            config_store=config_store,
        )

        record = _make_trace_record()
        msg = _make_msg(record)

        # Must not raise
        await evaluator(msg)

        # Called exactly once — no retry
        assert evaluator_service.evaluate.call_count == 1

    async def test_perman_error_continues_to_next_config(self) -> None:
        """If config A fails permanently, config B should still be evaluated."""

        config_a = _make_config(evaluator_name="evaluator_a")
        config_b = _make_config(evaluator_name="evaluator_b")
        config_store = MagicMock()
        config_store.get_configs.return_value = [config_a, config_b]

        evaluator_service = AsyncMock()
        # First call fails permanently, second succeeds
        evaluator_service.evaluate.side_effect = [
            EvaluatorError(
                code=400,
                error_type=METADATA_MISSING_CODE,
                message="missing agent_name",
            ),
            None,  # success
        ]

        evaluator = _make_evaluator(
            evaluator_service=evaluator_service,
            config_store=config_store,
        )

        record = _make_trace_record()
        msg = _make_msg(record)

        # Must not raise — config A skipped, config B evaluated
        await evaluator(msg)

        assert evaluator_service.evaluate.call_count == 2


# ── Transient errors: skip config, continue fan-out (early-commit) ───────


class TestTransientEvaluatorErrors:
    """A 500+ or network error is recorded as transient and does not raise (there is no Kafka retry after early-commit)."""

    @pytest.mark.parametrize(
        ("error_type", "expected_code"),
        [
            (LLM_FAILED_CODE, 502),
            (EVALUATION_FAILED_CODE, 500),
        ],
    )
    async def test_500_level_error_skips_config(self, error_type: str, expected_code: int) -> None:
        """EvaluatorError with code >= 500 → skip config, no raise."""

        config = _make_config()
        config_store = MagicMock()
        config_store.get_configs.return_value = [config]

        evaluator_service = AsyncMock()
        evaluator_service.evaluate.side_effect = EvaluatorError(
            code=expected_code,
            error_type=error_type,
            message=f"test {error_type}",
        )

        evaluator = _make_evaluator(
            evaluator_service=evaluator_service,
            config_store=config_store,
        )

        record = _make_trace_record()
        msg = _make_msg(record)

        await evaluator(msg)

        assert evaluator_service.evaluate.call_count == 1

    async def test_generic_exception_skips_config(self) -> None:
        """Non-EvaluatorError exceptions (429, network) → skip, no raise."""

        config = _make_config()
        config_store = MagicMock()
        config_store.get_configs.return_value = [config]

        evaluator_service = AsyncMock()
        evaluator_service.evaluate.side_effect = ConnectionError("network down")

        evaluator = _make_evaluator(
            evaluator_service=evaluator_service,
            config_store=config_store,
        )

        record = _make_trace_record()
        msg = _make_msg(record)

        await evaluator(msg)
        assert evaluator_service.evaluate.call_count == 1

    async def test_transient_continues_to_next_config(self) -> None:
        """After Config A fails transiently, Config B should still be evaluated."""

        config_a = _make_config(evaluator_name="evaluator_a")
        config_b = _make_config(evaluator_name="evaluator_b")
        config_store = MagicMock()
        config_store.get_configs.return_value = [config_a, config_b]

        evaluator_service = AsyncMock()
        evaluator_service.evaluate.side_effect = [
            EvaluatorError(
                code=502,
                error_type=LLM_FAILED_CODE,
                message="LLM service unavailable",
            ),
            None,
        ]

        evaluator = _make_evaluator(
            evaluator_service=evaluator_service,
            config_store=config_store,
        )

        record = _make_trace_record()
        msg = _make_msg(record)

        await evaluator(msg)

        assert evaluator_service.evaluate.call_count == 2

    async def test_timeout_error_skips_config(self) -> None:
        """TimeoutError → permanent → skip config."""

        config = _make_config()
        config_store = MagicMock()
        config_store.get_configs.return_value = [config]

        evaluator_service = AsyncMock()
        evaluator_service.evaluate.side_effect = TimeoutError("LLM timed out")

        evaluator = _make_evaluator(
            evaluator_service=evaluator_service,
            config_store=config_store,
            evaluate_timeout_s=5,
        )

        record = _make_trace_record()
        msg = _make_msg(record)

        # Must not raise — timeout is permanent
        await evaluator(msg)
        assert evaluator_service.evaluate.call_count == 1


# ── Mixed scenario: permanent then transient ────────────────────────────


class TestMixedErrorScenario:
    """Both permanent and transient are skipped, and fan-out is not interrupted."""

    async def test_permanent_then_transient_both_skipped(self) -> None:
        """Config A permanent and Config B transient → both are skipped, and nothing is raised."""

        config_a = _make_config(evaluator_name="evaluator_a")
        config_b = _make_config(evaluator_name="evaluator_b")
        config_store = MagicMock()
        config_store.get_configs.return_value = [config_a, config_b]

        evaluator_service = AsyncMock()
        evaluator_service.evaluate.side_effect = [
            EvaluatorError(
                code=400,
                error_type=METADATA_MISSING_CODE,
                message="missing agent_name",
            ),
            EvaluatorError(
                code=502,
                error_type=LLM_FAILED_CODE,
                message="LLM service unavailable",
            ),
        ]

        evaluator = _make_evaluator(
            evaluator_service=evaluator_service,
            config_store=config_store,
        )

        record = _make_trace_record()
        msg = _make_msg(record)

        await evaluator(msg)

        assert evaluator_service.evaluate.call_count == 2
