"""EvaluatorService.after_eval — when triage_service=None keep the current behavior; when it is not None, await it synchronously."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.evaluators.base import EvaluatorRequest, EvaluatorResult
from benchmark.services.evaluation import EvaluatorService, _triage_span_output
from benchmark.attribution.service.service import TriageOutcome
from benchmark.utils.exceptions import EVALUATION_FAILED_CODE, EvaluatorError


def _make_service(*, triage_service=None) -> EvaluatorService:
    svc = EvaluatorService(
        runtime=MagicMock(),
        trace_record_service=MagicMock(),
        triage_service=triage_service,
        source_score_client_provider=MagicMock(),
    )
    svc._trace_record_service.upsert = AsyncMock()
    return svc


def test_triage_span_distinguishes_attribution_and_dispatch_status() -> None:
    request = EvaluatorRequest(trace_id="tr-nonsafety", agent_name="vehicle", project_id="proj")
    outcome = TriageOutcome(
        status="skipped",
        reason="non_safety_aggregate_only",
        attribution=SimpleNamespace(
            conclusion_type="agent_badcase",
            category="模型输出|幻觉",
            root_cause_code="poi_tag_grafting",
            confidence=0.85,
            needs_human_review=False,
        ),
        notification_task=SimpleNamespace(
            status="skipped",
            message="non_safety_aggregate_via_cluster_report",
            workitem_id=None,
            payload=None,
        ),
    )

    output = _triage_span_output("vehicle_agent", request, outcome)

    assert "status" not in output
    assert output["attribution_status"] == "completed"
    assert output["dispatch_status"] == "skipped"
    assert output["dispatch_policy"] == "aggregate_via_cluster_report"


@pytest.mark.asyncio
async def test_no_triage_service_only_upserts() -> None:
    """When triage_service=None, after_eval only persists, and does not trigger attribution."""
    svc = _make_service()
    request = EvaluatorRequest(trace_id="t1", agent_name="a", project_id="p")
    result = EvaluatorResult(score=1.0)

    await svc.after_eval("example", request, result)

    svc._trace_record_service.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_triage_service_awaits_run_before_return() -> None:
    """When triage_service is not None, after_eval returns only after awaiting triage.run."""
    triage = MagicMock()
    triage.run = AsyncMock(return_value=TriageOutcome(status="created"))

    svc = _make_service(triage_service=triage)
    request = EvaluatorRequest(trace_id="t1", agent_name="a", project_id="p")
    result = EvaluatorResult(score=0.5)

    await svc.after_eval("example", request, result)

    triage.run.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    ["attribution_failed", "dispatch_failed"],
)
async def test_triage_failure_outcome_raises_evaluator_error(status: str) -> None:
    """A soft-failure status from run() must be raised, and must not be HTTP 200 again."""
    triage = MagicMock()
    triage.run = AsyncMock(
        return_value=TriageOutcome(status=status, error="agent blew up"),
    )
    svc = _make_service(triage_service=triage)

    with pytest.raises(EvaluatorError) as exc_info:
        await svc.after_eval(
            "example",
            EvaluatorRequest(trace_id="t-fail", agent_name="a", project_id="p"),
            EvaluatorResult(score=0.1),
        )

    assert exc_info.value.error_type == EVALUATION_FAILED_CODE
    assert status in exc_info.value.message
    assert "agent blew up" in exc_info.value.message


@pytest.mark.asyncio
async def test_triage_skipped_outcome_does_not_fail_request() -> None:
    """A skipped result such as good/watch still counts as success."""
    triage = MagicMock()
    triage.run = AsyncMock(
        return_value=TriageOutcome(status="skipped", reason="good"),
    )
    svc = _make_service(triage_service=triage)

    await svc.after_eval(
        "example",
        EvaluatorRequest(trace_id="t-skip", agent_name="a", project_id="p"),
        EvaluatorResult(score=9.0),
    )
    triage.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_triage_runs_under_report_root_child_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the sync path, triage still opens the triage_attribution container span inside eval_report_root."""
    from benchmark.services import evaluation

    report_context: ContextVar[str | None] = ContextVar("report_context", default=None)
    child_span = MagicMock()

    @contextmanager
    def fake_root(**_kwargs: object):
        token = report_context.set("active-root")
        try:
            yield MagicMock()
        finally:
            report_context.reset(token)

    @contextmanager
    def fake_child_scope(**kwargs: object):
        assert report_context.get() == "active-root", "child_scope must open inside root"
        assert kwargs["name"] == "triage_attribution"
        token = report_context.set("active-triage")
        try:
            yield child_span
        finally:
            report_context.reset(token)

    async def run_under_scope(*_args: object, **_kwargs: object) -> TriageOutcome:
        assert report_context.get() == "active-triage"
        return TriageOutcome(status="created")

    triage = MagicMock()
    triage.run = AsyncMock(side_effect=run_under_scope)
    svc = _make_service(triage_service=triage)
    monkeypatch.setattr(evaluation, "eval_report_root", fake_root)
    monkeypatch.setattr(evaluation, "report_child_scope", fake_child_scope)
    monkeypatch.setattr(svc, "_select", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        svc,
        "_eval",
        AsyncMock(return_value=EvaluatorResult(score=0.5, reason="needs triage")),
    )

    await svc.evaluate(
        "example",
        EvaluatorRequest(
            trace_id="t-context",
            agent_name="a",
            project_id="p",
            input={},
            output={},
            metadata={"trace_id": "t-context"},
            skip_trace_resolve=True,
        ),
    )
    child_span.update.assert_called_once_with(
        output={
            "evaluator_name": "example",
            "trace_id": "t-context",
            "attribution_status": "not_started",
        },
    )


@pytest.mark.asyncio
async def test_evaluate_waits_for_triage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.services import evaluation

    triage_started = asyncio.Event()
    release_triage = asyncio.Event()

    async def slow_run(*_args: object, **_kwargs: object) -> TriageOutcome:
        triage_started.set()
        await release_triage.wait()
        return TriageOutcome(status="created")

    triage = MagicMock()
    triage.run = AsyncMock(side_effect=slow_run)
    svc = _make_service(triage_service=triage)
    monkeypatch.setattr(evaluation, "eval_report_root", MagicMock())
    monkeypatch.setattr(svc, "_select", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        svc,
        "_eval",
        AsyncMock(return_value=EvaluatorResult(score=0.5, reason="needs triage")),
    )

    evaluate_task = asyncio.create_task(
        svc.evaluate(
            "example",
            EvaluatorRequest(
                trace_id="t-wait",
                agent_name="a",
                project_id="p",
                input={},
                output={},
                metadata={"trace_id": "t-wait"},
                skip_trace_resolve=True,
            ),
        )
    )
    await asyncio.wait_for(triage_started.wait(), timeout=1.0)

    assert not evaluate_task.done()
    assert not release_triage.is_set()

    release_triage.set()
    await asyncio.wait_for(evaluate_task, timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "triage_error",
    [RuntimeError("triage failed"), asyncio.CancelledError()],
    ids=["exception", "cancelled"],
)
async def test_triage_error_finishes_span_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
    triage_error: BaseException,
) -> None:
    from benchmark.services import evaluation

    triage = MagicMock()
    triage.run = AsyncMock(side_effect=triage_error)
    svc = _make_service(triage_service=triage)
    child_span = MagicMock()
    seen_errors: list[BaseException] = []

    @contextmanager
    def fake_child_scope(**_kwargs: object):
        try:
            yield child_span
        except BaseException as exc:
            seen_errors.append(exc)
            raise

    monkeypatch.setattr(evaluation, "report_child_scope", fake_child_scope)

    request = EvaluatorRequest(trace_id="t-error", agent_name="a", project_id="p")
    result = EvaluatorResult(score=0.1)

    if isinstance(triage_error, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await svc.after_eval("example", request, result)
        assert exc_info.value is triage_error
    else:
        with pytest.raises(EvaluatorError) as exc_info:
            await svc.after_eval("example", request, result)
        assert exc_info.value.error_type == EVALUATION_FAILED_CODE
        assert exc_info.value.__cause__ is triage_error

    assert seen_errors == [triage_error]


@pytest.mark.asyncio
async def test_evaluate_holds_semaphore_through_triage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """judge and triage share _llm_semaphore: the slot is not released before attribution finishes."""
    from benchmark.services import evaluation

    triage_started = asyncio.Event()
    release_triage = asyncio.Event()
    semaphore_held_during_triage = asyncio.Event()

    async def slow_run(*_args: object, **_kwargs: object) -> TriageOutcome:
        triage_started.set()
        # semaphore value 0 means the slot is still held by this evaluate().
        if svc._llm_semaphore is not None and svc._llm_semaphore._value == 0:
            semaphore_held_during_triage.set()
        await release_triage.wait()
        return TriageOutcome(status="created")

    triage = MagicMock()
    triage.run = AsyncMock(side_effect=slow_run)
    svc = EvaluatorService(
        runtime=MagicMock(),
        trace_record_service=MagicMock(),
        triage_service=triage,
        llm_max_concurrency=1,
        source_score_client_provider=MagicMock(),
    )
    svc._trace_record_service.upsert = AsyncMock()
    monkeypatch.setattr(evaluation, "eval_report_root", MagicMock())
    monkeypatch.setattr(svc, "_select", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        svc,
        "_eval",
        AsyncMock(return_value=EvaluatorResult(score=0.5, reason="needs triage")),
    )

    evaluate_task = asyncio.create_task(
        svc.evaluate(
            "example",
            EvaluatorRequest(
                trace_id="t-sem",
                agent_name="a",
                project_id="p",
                input={},
                output={},
                metadata={"trace_id": "t-sem"},
                skip_trace_resolve=True,
            ),
        )
    )
    await asyncio.wait_for(triage_started.wait(), timeout=1.0)
    await asyncio.wait_for(semaphore_held_during_triage.wait(), timeout=1.0)
    assert svc._llm_semaphore is not None
    assert svc._llm_semaphore._value == 0

    release_triage.set()
    await asyncio.wait_for(evaluate_task, timeout=1.0)
    assert svc._llm_semaphore._value == 1


@pytest.mark.asyncio
async def test_same_evaluator_runs_in_parallel() -> None:
    """Calling after_eval directly can still run in parallel; the concurrency limit is the _llm_semaphore on evaluate()."""
    in_flight = 0
    max_in_flight = 0
    both_started = asyncio.Event()
    release_all = asyncio.Event()

    async def slow_run(*_args: object, **_kwargs: object) -> TriageOutcome:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        if in_flight >= 2:
            both_started.set()
        await release_all.wait()
        in_flight -= 1
        return TriageOutcome(status="created")

    triage = MagicMock()
    triage.run = AsyncMock(side_effect=slow_run)
    svc = _make_service(triage_service=triage)

    task1 = asyncio.create_task(
        svc.after_eval(
            "example",
            EvaluatorRequest(trace_id="t1", agent_name="a", project_id="p"),
            EvaluatorResult(score=0.1),
        )
    )
    task2 = asyncio.create_task(
        svc.after_eval(
            "example",
            EvaluatorRequest(trace_id="t2", agent_name="a", project_id="p"),
            EvaluatorResult(score=0.1),
        )
    )

    await asyncio.wait_for(both_started.wait(), timeout=2.0)
    assert max_in_flight == 2
    assert triage.run.await_count == 2

    release_all.set()
    await asyncio.wait_for(asyncio.gather(task1, task2), timeout=2.0)


@pytest.mark.asyncio
async def test_after_eval_passes_enriched_metrics_to_storage_and_triage():
    triage = MagicMock()
    triage.run = AsyncMock(return_value=TriageOutcome(status="created"))
    svc = _make_service(triage_service=triage)
    request = EvaluatorRequest(trace_id="metrics", agent_name="vehicle", project_id="p")
    enriched = EvaluatorResult(score=0.5, extra={"metric_observations": [{"value": 12}]})
    svc._enrich_vehicle_metrics = AsyncMock(return_value=enriched)
    await svc.after_eval("vehicle_agent", request, EvaluatorResult(score=0.5))
    svc._enrich_vehicle_metrics.assert_awaited_once()
    assert (
        svc._trace_record_service.upsert.await_args.kwargs["judge_json"]["extra"] == enriched.extra
    )
    assert triage.run.await_args.args[0].evaluator_result is enriched
