"""Unit tests for external_runner attribution assembly — a missing response_text column falls back to result (production-incident regression).

The production ieval data-pipeline CSV puts the model output in the ``result`` column (there is no ``response_text``
column); the old assembly read only ``response_text`` → response was empty → the attribution worker's
required-field check blocked it (``attribution.submit_missing_fields``, a non-retryable force_fail)
→ attribution was never submitted, and ``attribution_job_id`` stayed empty.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.schema import JudgeCallback
from benchmark.workers.external_runner import (
    ExternalBenchmarkRunner,
    ExternalRunnerDeps,
    JudgeCallbackCase,
    _AsyncAttributionRoute,
)

_ATTR_REF = "business.runs.ieval.external_execution:IevalExternalAttribution"


def _runner() -> ExternalBenchmarkRunner:
    runner = ExternalBenchmarkRunner(
        benchmark_service=MagicMock(),
        storage_client=None,
        deps=ExternalRunnerDeps(queue_manager=MagicMock()),
    )
    runner._benchmark_service.increment_attribution_expected_once = AsyncMock(
        return_value=(1, {}, True),
    )
    return runner


def _case(item_output: object) -> JudgeCallbackCase:
    return JudgeCallbackCase(
        payload=JudgeCallback(
            run_id="bench-20260907",
            item_id="bench-20260907:0",
            criterion_scores={},
        ),
        run_id="bench-20260907",
        item_id="bench-20260907:0",
        source_trace_id="judge-job-1",
        item_input={"query": "q-1", "adiu": "a-1"},
        item_output=item_output,
        ri_row=None,
    )


async def _capture_response(
    runner: ExternalBenchmarkRunner,
    case: JudgeCallbackCase,
) -> str:
    """Call _enqueue_async_attribution_case and capture the response field of the enqueued spec."""
    captured: dict[str, object] = {}

    async def _capture_spec(spec: object) -> None:
        captured["spec"] = spec

    runner._enqueue_attribution_case = _capture_spec  # type: ignore[method-assign]
    await runner._enqueue_async_attribution_case(case, _AsyncAttributionRoute())
    return captured["spec"].response


@pytest.mark.asyncio
async def test_response_falls_back_to_result_column() -> None:
    """response_text is missing → fall back to the result column (the production ieval CSV shape)."""
    response = await _capture_response(
        _runner(),
        _case({"response_text": "", "result": "模型输出文本"}),
    )
    assert response == "模型输出文本"


@pytest.mark.asyncio
async def test_response_prefers_response_text_when_present() -> None:
    """When response_text is non-empty it wins (no fallback)."""
    response = await _capture_response(
        _runner(),
        _case({"response_text": "正文", "result": "结果"}),
    )
    assert response == "正文"


@pytest.mark.asyncio
async def test_response_empty_when_both_missing() -> None:
    """Both columns are missing → an empty string (the worker-side required-field check blocks it, and force_fail is the terminal fallback)."""
    response = await _capture_response(_runner(), _case({"tool_calls_json": "[]"}))
    assert response == ""


@pytest.mark.asyncio
async def test_enqueue_uses_attribution_expected_once() -> None:
    """After enqueue, +expected is idempotent per item (so a repeat does not raise the ceiling)."""
    runner = _runner()
    runner._enqueue_attribution_case = AsyncMock()  # type: ignore[method-assign]
    await runner._enqueue_async_attribution_case(
        _case({"result": "out"}),
        _AsyncAttributionRoute(class_ref=_ATTR_REF),
    )
    runner._enqueue_attribution_case.assert_awaited_once()
    assert runner._enqueue_attribution_case.await_args.args[0].class_ref == _ATTR_REF
    runner._benchmark_service.increment_attribution_expected_once.assert_awaited_once_with(
        "bench-20260907",
        "bench-20260907:0",
    )


@pytest.mark.asyncio
async def test_empty_criterion_scores_routes_to_failure_not_attribution() -> None:
    """No error + empty criterion_scores → a non-retryable failure, with no fusion and no attribution."""
    runner = _runner()
    runner._handle_judge_callback_failure = AsyncMock()  # type: ignore[method-assign]
    runner._load_run_with_status = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            SimpleNamespace(status="running", config_json={}, benchmark_name="ieval_external"),
            "running",
            {},
        ),
    )
    runner._resolve_callback_item_id = AsyncMock(  # type: ignore[method-assign]
        return_value=("bench-20260907:58", True),
    )
    runner._release_and_promote_next_judge = AsyncMock()  # type: ignore[method-assign]

    with (
        patch(
            "benchmark.workers.external_runner.BenchmarkRunItem.filter",
            return_value=MagicMock(first=AsyncMock(return_value=None)),
        ),
        patch(
            "benchmark.workers.external_runner.load_rubric_from_config",
            return_value=[],
        ),
        patch(
            "benchmark.workers.external_runner.load_aggregation_from_config",
            return_value=MagicMock(),
        ),
    ):
        await runner._handle_judge_callback(
            JudgeCallback(
                run_id="bench-20260907",
                item_id="bench-20260907:58",
                job_id="judge-job-58",
                criterion_scores={},
            ),
        )

    runner._handle_judge_callback_failure.assert_awaited_once()
    fail_case = runner._handle_judge_callback_failure.await_args.args[0]
    assert fail_case.payload.error == "empty_criterion_scores"
    assert fail_case.payload.is_retryable is False
    runner._benchmark_service.increment_attribution_expected_once.assert_not_awaited()
