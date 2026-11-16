"""Unit tests for LsCaseReviewAfterRunHook — it does not fire when attribution=None."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from benchmark.labeling.case_review_hook import LsCaseReviewAfterRunHook
from benchmark.attribution.service.service import AfterRunInput, TriageOutcome


def _make_hook() -> tuple[LsCaseReviewAfterRunHook, AsyncMock, AsyncMock]:
    dispatcher = AsyncMock()
    dispatcher.dispatch_case_review.return_value = True
    inner = AsyncMock()
    inner.return_value = TriageOutcome(status="created")
    hook = LsCaseReviewAfterRunHook(dispatcher=dispatcher, inner=inner)
    return hook, dispatcher, inner


def _agent_response():
    return SimpleNamespace(traceid="t1", status="SUCCESS", content_parts=[])


def _attribution():
    return SimpleNamespace(
        root_cause="tool_error",
        category="Toolbox",
        confidence=0.85,
        reasoning="tool failed",
    )


class TestLsCaseReviewAfterRunHook:
    async def test_dispatches_when_attribution_present(self) -> None:
        hook, dispatcher, inner = _make_hook()
        outcome = await hook(
            AfterRunInput(
                evaluator_name="eval-1",
                evaluator_request=SimpleNamespace(trace_id="t1"),
                attribution_response=_agent_response(),
                attribution=_attribution(),
            ),
        )
        assert outcome.status == "created"
        inner.assert_called_once()
        dispatcher.dispatch_case_review.assert_called_once()

    async def test_skips_when_attribution_none(self) -> None:
        hook, dispatcher, inner = _make_hook()
        outcome = await hook(
            AfterRunInput(
                evaluator_name="eval-1",
                evaluator_request=SimpleNamespace(trace_id="t1"),
                attribution_response=_agent_response(),
                attribution=None,
            ),
        )
        assert outcome.status == "created"
        inner.assert_called_once()
        dispatcher.dispatch_case_review.assert_not_called()

    async def test_dispatch_failure_does_not_affect_outcome(self) -> None:
        hook, dispatcher, inner = _make_hook()
        dispatcher.dispatch_case_review.side_effect = RuntimeError("boom")
        outcome = await hook(
            AfterRunInput(
                evaluator_name="eval-1",
                evaluator_request=SimpleNamespace(trace_id="t1"),
                attribution_response=_agent_response(),
                attribution=_attribution(),
            ),
        )
        assert outcome.status == "created"

    async def test_inner_hook_called_first(self) -> None:
        hook, dispatcher, inner = _make_hook()
        call_order = []
        inner.side_effect = lambda *args, **kw: (
            call_order.append("inner") or TriageOutcome(status="created")
        )
        dispatcher.dispatch_case_review.side_effect = lambda **kw: (
            call_order.append("dispatch") or True
        )
        await hook(
            AfterRunInput(
                evaluator_name="eval-1",
                evaluator_request=SimpleNamespace(trace_id="t1"),
                attribution_response=_agent_response(),
                attribution=_attribution(),
            ),
        )
        assert call_order == ["inner", "dispatch"]

    async def test_dispatches_for_dataset_eval_metadata(self) -> None:
        """Dataset-eval still pushes to Label Studio (filter via trace_meta_json)."""
        hook, dispatcher, inner = _make_hook()
        outcome = await hook(
            AfterRunInput(
                evaluator_name="eval-1",
                evaluator_request=SimpleNamespace(
                    trace_id="dataset:item-1",
                    metadata={"dataset_item_id": "item-1", "dataset_name": "nav/hard"},
                ),
                attribution_response=_agent_response(),
                attribution=_attribution(),
            ),
        )
        assert outcome.status == "created"
        inner.assert_called_once()
        dispatcher.dispatch_case_review.assert_called_once()
