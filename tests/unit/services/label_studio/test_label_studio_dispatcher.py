"""Unit tests for LabelStudioDispatcher — ContextVar reads + direct push + swallowed exceptions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from benchmark.labeling.context import (
    clear_triage_judge_context,
    set_triage_judge_context,
)
from benchmark.labeling.dispatcher import LabelStudioDispatcher


def _make_dispatcher() -> LabelStudioDispatcher:
    client = MagicMock()
    client.import_tasks.return_value = {"task_ids": [1]}
    resolver = AsyncMock()
    resolver.resolve.return_value = 42
    trace_svc = AsyncMock()
    return LabelStudioDispatcher(
        client=client,
        project_resolver=resolver,
        trace_record_service=trace_svc,
    )


def _eval_request(
    trace_id="t1",
    project_id="p1",
    agent_name="agent-a",
    metadata=None,
):
    return SimpleNamespace(
        trace_id=trace_id,
        project_id=project_id,
        agent_name=agent_name,
        input={"query": "hello"},
        output={"answer": "world"},
        metadata=metadata or {"trace_record_id": "uuid-test-1"},
    )


def _attribution():
    return SimpleNamespace(
        root_cause="tool_error",
        category="Toolbox",
        confidence=0.85,
        reasoning="tool failed",
    )


class TestDispatchCaseReview:
    async def test_success_with_context(self) -> None:
        dispatcher = _make_dispatcher()
        result = SimpleNamespace(score=0.3, reason="bad answer")
        set_triage_judge_context(evaluator_result=result, triage_level="severe_bad")
        try:
            ok = await dispatcher.dispatch_case_review(
                evaluator_name="eval-1",
                evaluator_request=_eval_request(),
                attribution=_attribution(),
            )
            assert ok is True
            dispatcher._client.import_tasks.assert_called_once()
            payload = dispatcher._client.import_tasks.call_args[0][1][0]
            assert payload["data"]["judge_score"] == "0.3"
            assert payload["data"]["judge_reason"] == "bad answer"
            assert payload["data"]["triage_level"] == "severe_bad"
            assert payload["data"]["root_cause"] == "tool_error"
            assert payload["data"]["stage"] == "badcase_review"
            assert payload["data"]["trace_record_id"] == "uuid-test-1"
        finally:
            clear_triage_judge_context()

    async def test_ctx_none_degrades_gracefully(self) -> None:
        dispatcher = _make_dispatcher()
        clear_triage_judge_context()
        ok = await dispatcher.dispatch_case_review(
            evaluator_name="eval-1",
            evaluator_request=_eval_request(),
            attribution=_attribution(),
        )
        assert ok is True
        payload = dispatcher._client.import_tasks.call_args[0][1][0]
        assert payload["data"]["judge_score"] == ""
        assert payload["data"]["triage_level"] == ""

    async def test_push_failure_swallowed(self) -> None:
        dispatcher = _make_dispatcher()
        dispatcher._client.import_tasks.side_effect = RuntimeError("LS down")
        clear_triage_judge_context()
        ok = await dispatcher.dispatch_case_review(
            evaluator_name="eval-1",
            evaluator_request=_eval_request(),
            attribution=_attribution(),
        )
        assert ok is False

    async def test_json_preview_truncation(self) -> None:
        dispatcher = _make_dispatcher()
        clear_triage_judge_context()
        req = _eval_request()
        req.input = "x" * 5000
        await dispatcher.dispatch_case_review(
            evaluator_name="eval-1",
            evaluator_request=req,
            attribution=_attribution(),
        )
        payload = dispatcher._client.import_tasks.call_args[0][1][0]
        assert len(payload["data"]["agent_input"]) <= 2000

    async def test_dataset_metadata_passed_to_resolver(self) -> None:
        dispatcher = _make_dispatcher()
        clear_triage_judge_context()
        req = _eval_request(
            metadata={
                "trace_record_id": "uuid-test-1",
                "dataset_name": "ds-vehicle",
                "dataset_run_name": "run-001",
                "dataset_item_id": "item-42",
            },
        )
        await dispatcher.dispatch_case_review(
            evaluator_name="eval-1",
            evaluator_request=req,
            attribution=_attribution(),
        )
        identity = dispatcher._project_resolver.resolve.call_args.args[0]
        assert identity.dataset_name == "ds-vehicle"
        assert identity.run_name == "run-001"
        payload = dispatcher._client.import_tasks.call_args[0][1][0]
        assert payload["data"]["dataset_name"] == "ds-vehicle"
        assert payload["data"]["dataset_run_name"] == "run-001"
        assert payload["data"]["dataset_item_id"] == "item-42"

    async def test_no_dataset_metadata_omits_dataset_keys_in_payload(self) -> None:
        dispatcher = _make_dispatcher()
        clear_triage_judge_context()
        await dispatcher.dispatch_case_review(
            evaluator_name="eval-1",
            evaluator_request=_eval_request(),
            attribution=_attribution(),
        )
        payload = dispatcher._client.import_tasks.call_args[0][1][0]
        assert "dataset_name" not in payload["data"]
        assert "dataset_run_name" not in payload["data"]
        assert "dataset_item_id" not in payload["data"]

    async def test_no_dataset_metadata_uses_default_naming(self) -> None:
        dispatcher = _make_dispatcher()
        clear_triage_judge_context()
        await dispatcher.dispatch_case_review(
            evaluator_name="eval-1",
            evaluator_request=_eval_request(),
            attribution=_attribution(),
        )
        identity = dispatcher._project_resolver.resolve.call_args.args[0]
        assert identity.dataset_name is None
        assert identity.run_name is None
