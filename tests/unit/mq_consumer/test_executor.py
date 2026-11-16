"""EvalMessageExecutor unit tests — routing, conversion, and error policy (using a fake EvaluatorService).

Routing: the message carries evaluator_name (aliases are normalized by the schema); missing → skip (a bad route; the handler does not guess).
Evaluability: after enrichment, a non-None ``unevaluatable_reason()`` → skip before evaluation (incomplete data is a permanent defect).
Error policy: a 4xx permanent error → skip (return normally); a 5xx retryable error → raise.
Reporting: one report root per message, and the terminal state (scored / skipped / needs retry) must land on root.output.
"""

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from benchmark.evaluators.base import EvaluatorRequest, EvaluatorResult
from benchmark.ingest.mq.enrichment import EnrichmentUnavailableError
from benchmark.ingest.mq.executor import EvalMessageExecutor
from benchmark.services.evaluation import EvalRunOptions
from benchmark.utils.exceptions import (
    EVALUATION_FAILED_CODE,
    MODEL_NOT_FOUND_CODE,
    EvaluatorError,
    evaluator_status,
)

from tests.unit.mq_consumer.stubs import StubTaskMessage


class _FakeEvaluatorService:
    """Record evaluate calls; the return value or the error raised can be configured."""

    def __init__(
        self, *, result: EvaluatorResult | None = None, error: Exception | None = None
    ) -> None:
        self.calls: list[tuple[str, EvaluatorRequest]] = []
        self.options: list[EvalRunOptions | None] = []
        self._result = result or EvaluatorResult(score=1.0, reason="ok")
        self._error = error

    async def evaluate(
        self,
        evaluator_name: str,
        request: EvaluatorRequest,
        *,
        options: EvalRunOptions | None = None,
    ) -> EvaluatorResult:
        self.calls.append((evaluator_name, request))
        self.options.append(options)
        if self._error is not None:
            raise self._error
        return self._result


@pytest.fixture
def report_root(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the MQ chain's report root with an assertable mock (do not use the real Langfuse SDK)."""

    root = MagicMock()
    captured: dict[str, Any] = {}

    @contextmanager
    def fake_root(
        *,
        name: str,
        input_value: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        captured.update(name=name, input_value=input_value, metadata=metadata)
        yield root

    monkeypatch.setattr("benchmark.ingest.mq.executor.eval_report_root", fake_root)
    root.captured = captured
    return root


async def test_report_root_carries_message_identifiers_and_consume_outcome(
    report_root: MagicMock,
) -> None:
    # A message-level root carries only the message's own identity plus the consume terminal state; score/evaluation status belongs to evaluate
    # and its existing report nodes, and must not be repeated here.
    fake = _FakeEvaluatorService(result=EvaluatorResult(score=0.5, reason="meh"))
    msg = StubTaskMessage(
        evaluator_name="custom",
        trace_id="t-1",
        project_id="p-1",
        metadata={"source": "luoshu"},
    )
    await EvalMessageExecutor(fake).handle(msg)

    captured = report_root.captured
    assert captured["name"] == "agent-eval.luoshu"
    # Identity fields are for fast filtering.
    input_value = captured["input_value"]
    assert input_value["source"] == "luoshu"
    assert input_value["trace_id"] == "t-1"
    assert input_value["project_id"] == "p-1"
    assert input_value["generate_time_b"] is None
    # The full raw payload MQ received must be reported (even if it is later skipped or fails, and it does not depend on evaluate succeeding).
    assert input_value["message"]["trace_id"] == "t-1"
    assert input_value["message"]["metadata"] == {"source": "luoshu"}
    # metadata carries pipeline_source, so the Langfuse side can filter the metaq chain by channel.
    assert captured["metadata"] == {"pipeline_source": "metaq"}
    report_root.update.assert_called_once_with(output={"status": "evaluated"})


async def test_report_root_records_skip_reason(report_root: MagicMock) -> None:
    # A message skipped by the gate must still show on the trace "why it was not evaluated", and be marked ERROR (evaluate did not run).
    fake = _FakeEvaluatorService()
    await EvalMessageExecutor(fake).handle(
        _UnevaluatableMessage(evaluator_name="custom", trace_id="t-1")
    )

    assert fake.calls == []
    payload = report_root.update.call_args.kwargs
    assert payload["level"] == "ERROR"
    assert payload["output"] == {
        "status": "skipped",
        "reason": "unevaluatable",
        "detail": "langfuse trace miss",
    }


async def test_report_root_marks_error_on_retry_path(report_root: MagicMock) -> None:
    # A failure that needs a retry must be marked ERROR on the root, or a retry storm cannot be seen from the Langfuse side.
    err = EvaluatorError(
        code=evaluator_status(EVALUATION_FAILED_CODE),
        error_type=EVALUATION_FAILED_CODE,
        message="boom",
    )
    fake = _FakeEvaluatorService(error=err)
    with pytest.raises(EvaluatorError):
        await EvalMessageExecutor(fake).handle(
            StubTaskMessage(evaluator_name="custom", trace_id="t-1")
        )

    payload = report_root.update.call_args.kwargs
    assert payload["level"] == "ERROR"
    assert payload["output"]["status"] == "retry"


async def test_handle_keeps_evaluator_service_own_trace_report() -> None:
    # The MQ layer must not turn off evaluate's existing reporting: evaluation/attribution nodes (including the score) are produced by it,
    # and MQ only owns the outer message-level root. Lock that enable_langfuse_report=False is no longer passed.
    fake = _FakeEvaluatorService()
    handler = EvalMessageExecutor(fake)
    await handler.handle(StubTaskMessage(evaluator_name="custom", trace_id="t-1"))
    assert len(fake.options) == 1
    opts = fake.options[0]
    assert opts is not None
    assert opts.enable_trace_report is True


async def test_handle_invokes_evaluator_with_converted_request() -> None:
    fake = _FakeEvaluatorService()
    handler = EvalMessageExecutor(fake)
    await handler.handle(
        StubTaskMessage(
            evaluator_name="custom",
            input={"query": "x"},
            output={"answer": "y"},
            trace_id="t-1",
            project_id="p",
            agent_name="a",
        )
    )
    assert len(fake.calls) == 1
    name, req = fake.calls[0]
    # The message carries an evaluator name → follow it first (do not fall back to the default).
    assert name == "custom"
    assert req.trace_id == "t-1"
    assert req.project_id == "p"
    assert req.agent_name == "a"


async def test_handle_skips_when_no_evaluator_name() -> None:
    # The message has no evaluator_name → a bad route → skip (do not evaluate; the luoshu route is filled in by the converter).
    fake = _FakeEvaluatorService()
    handler = EvalMessageExecutor(fake)
    await handler.handle(StubTaskMessage(evaluator_name="   ", input={"q": 1}, trace_id="t"))
    assert fake.calls == []


class _UnevaluatableMessage(StubTaskMessage):
    """Simulate a source message that is "still missing required data after enrichment" (for example luoshu missed in Langfuse)."""

    def unevaluatable_reason(self) -> str | None:
        return "langfuse trace miss"


async def test_handle_skips_unevaluatable_message_before_evaluating() -> None:
    # Incomplete data is a permanent defect: it never enters evaluation (it does not spend retry budget), and it does not raise (already consumed, offset committed).
    fake = _FakeEvaluatorService()
    handler = EvalMessageExecutor(fake)
    await handler.handle(_UnevaluatableMessage(evaluator_name="custom", trace_id="t-1"))
    assert fake.calls == []


async def test_handle_evaluates_when_message_declares_itself_evaluatable() -> None:
    # The base class allows by default (unevaluatable_reason returns None) → existing source behavior stays.
    msg = StubTaskMessage(evaluator_name="custom", trace_id="t-1")
    assert msg.unevaluatable_reason() is None
    fake = _FakeEvaluatorService()
    await EvalMessageExecutor(fake).handle(msg)
    assert len(fake.calls) == 1


class _TransientEnrichMessage(StubTaskMessage):
    """A message whose enrichment failed transiently (an external service blipped)."""

    def enrich_retry_reason(self) -> str | None:
        return "langfuse query failed transiently"


async def test_handle_raises_on_transient_enrich_failure() -> None:
    """A transient enrichment failure → raise so the worker retries, instead of skipping it as a permanent defect.

    The old implementation collapsed both "query timed out" and "the trace really does not exist" into "the enrichment field is None", so
    a Langfuse blip silently dropped the batch. This case locks that it is no longer dropped.
    """
    fake = _FakeEvaluatorService()
    handler = EvalMessageExecutor(fake)
    msg = _TransientEnrichMessage(evaluator_name="custom", trace_id="t-1")

    with pytest.raises(EnrichmentUnavailableError, match="transiently"):
        await handler.handle(msg)

    # The retry gate is before evaluation: it must not spend LLM budget
    assert fake.calls == []


async def test_transient_gate_precedes_unevaluatable_gate() -> None:
    """When both gates hit, retry wins — otherwise a transient failure would still be skipped as a permanent defect."""

    class _Both(_TransientEnrichMessage):
        def unevaluatable_reason(self) -> str | None:
            return "also unevaluatable"

    fake = _FakeEvaluatorService()
    with pytest.raises(EnrichmentUnavailableError):
        await EvalMessageExecutor(fake).handle(_Both(evaluator_name="custom", trace_id="t-1"))
    assert fake.calls == []


async def test_handle_reraises_retriable_eval_error() -> None:
    # 5xx (EVALUATION_FAILED_CODE=500) → retryable → raise, and let the consume loop retry.
    err = EvaluatorError(
        code=evaluator_status(EVALUATION_FAILED_CODE),
        error_type=EVALUATION_FAILED_CODE,
        message="boom",
    )
    fake = _FakeEvaluatorService(error=err)
    handler = EvalMessageExecutor(fake)
    with pytest.raises(EvaluatorError):
        await handler.handle(StubTaskMessage(evaluator_name="custom", trace_id="t"))
    assert len(fake.calls) == 1


async def test_handle_skips_permanent_eval_error() -> None:
    # 4xx (MODEL_NOT_FOUND_CODE=400) → a permanent error → skip (do not raise), so an infinite retry does not stall the queue.
    err = EvaluatorError(
        code=evaluator_status(MODEL_NOT_FOUND_CODE),
        error_type=MODEL_NOT_FOUND_CODE,
        message="unknown evaluator",
    )
    fake = _FakeEvaluatorService(error=err)
    handler = EvalMessageExecutor(fake)
    await handler.handle(StubTaskMessage(evaluator_name="nope", trace_id="t"))
    assert len(fake.calls) == 1
