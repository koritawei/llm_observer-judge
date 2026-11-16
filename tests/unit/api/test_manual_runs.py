"""manual-runs only allocates a run_id; the request body and the accept flow stay as they are."""

from unittest.mock import MagicMock

import pytest
from benchmark.routes.benchmark import (
    ManualRunDeps,
    _pin_dated_oss_object_key,
    run_manual_benchmark,
)
from benchmark.runs.schema import ExternalBenchmarkRequest, ExternalRunAcceptedResponse
from fastapi import HTTPException


def _degraded_request(reason: str = "RuntimeError: Redis connection not configured") -> MagicMock:
    """Simulate a Request whose BullMQ stack is degraded (fallback set to the degradation reason)."""
    request = MagicMock()
    request.app.state.eval_queue_manager = None
    request.app.state.eval_queue_degraded_reason = reason
    return request


def _healthy_request() -> MagicMock:
    """Simulate a Request whose queue stack is healthy (no degradation marker)."""
    request = MagicMock()
    request.app.state.eval_queue_manager = object()
    request.app.state.eval_queue_degraded_reason = None
    return request


def _deps() -> ManualRunDeps:
    """Stand-in for the three-service aggregate dependency (direct route calls; service methods are replaced on the stub)."""
    return ManualRunDeps(
        benchmark_service=MagicMock(),
        runtime=MagicMock(),
        external_runner=MagicMock(),
    )


@pytest.mark.asyncio
async def test_manual_route_rejects_caller_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("benchmark.routes.benchmark.raise_if_benchmark_disabled", lambda: None)
    payload = ExternalBenchmarkRequest(
        benchmark_name="ieval_external",
        date="20261010",
        run_id_suffix="q1",
    )
    with pytest.raises(HTTPException) as exc:
        await run_manual_benchmark(payload, _healthy_request(), _deps(), None)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_manual_route_assigns_next_sequence_without_rewriting_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("benchmark.routes.benchmark.raise_if_benchmark_disabled", lambda: None)

    async def _list(benchmark_name: str, date: str, *, namespace: str = "m") -> list[str]:
        assert benchmark_name == "ieval_external"
        assert date == "20261010"
        assert namespace == "m"
        return ["ieval_external-20261010-m1"]

    seen: dict[str, object] = {}

    async def _accept(
        payload: ExternalBenchmarkRequest,
        accept: object,
    ) -> ExternalRunAcceptedResponse:
        assigned = accept.assigned_run_id
        seen["suffix"] = payload.run_id_suffix
        seen["date"] = payload.date
        seen["assigned"] = assigned
        seen["data_date"] = accept.data_date
        seen["reuse"] = accept.reuse_existing
        return ExternalRunAcceptedResponse(
            run_id=str(assigned),
            benchmark_name=payload.benchmark_name,
            status="initializing",
        )

    monkeypatch.setattr("benchmark.routes.benchmark.list_manual_run_ids", _list)
    monkeypatch.setattr("benchmark.routes.benchmark._accept_external_run", _accept)

    payload = ExternalBenchmarkRequest(benchmark_name="ieval_external", date="20261010")
    response = await run_manual_benchmark(payload, _healthy_request(), _deps(), None)

    assert isinstance(response, ExternalRunAcceptedResponse)
    assert response.run_id == "ieval_external-20261010-m2"
    assert seen == {
        "suffix": None,
        "date": "20261010",
        "assigned": "ieval_external-20261010-m2",
        "data_date": "20261010",
        "reuse": False,
    }
    assert payload.run_id_suffix is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "expected_namespace"),
    [("immediate", "m"), ("daily", "s"), ("weekly", "s"), ("cron", "s")],
)
async def test_manual_route_uses_task_trigger_namespace(
    monkeypatch: pytest.MonkeyPatch,
    trigger: str,
    expected_namespace: str,
) -> None:
    monkeypatch.setattr("benchmark.routes.benchmark.raise_if_benchmark_disabled", lambda: None)

    async def _find_task(task_id: str) -> MagicMock:
        assert task_id == "task-scheduled"
        return MagicMock(trigger=trigger)

    async def _list(benchmark_name: str, date: str, *, namespace: str = "m") -> list[str]:
        assert namespace == expected_namespace
        return [
            f"{benchmark_name}-{date}-m9",
            f"{benchmark_name}-{date}-s1",
        ]

    async def _accept(
        payload: ExternalBenchmarkRequest,
        accept: object,
    ) -> ExternalRunAcceptedResponse:
        return ExternalRunAcceptedResponse(
            run_id=str(accept.assigned_run_id),
            benchmark_name=payload.benchmark_name,
            status="initializing",
        )

    monkeypatch.setattr("benchmark.domain.benchmark_console.find_task_document", _find_task)
    monkeypatch.setattr("benchmark.routes.benchmark.list_manual_run_ids", _list)
    monkeypatch.setattr("benchmark.routes.benchmark._accept_external_run", _accept)
    payload = ExternalBenchmarkRequest(
        benchmark_name="ieval_external",
        date="20261010",
        eval_task_id="task-scheduled",
    )
    response = await run_manual_benchmark(payload, _healthy_request(), _deps(), None)
    assert isinstance(response, ExternalRunAcceptedResponse)
    expected_seq = 10 if expected_namespace == "m" else 2
    assert response.run_id == f"ieval_external-20261010-{expected_namespace}{expected_seq}"


@pytest.mark.asyncio
async def test_manual_route_explicit_s_namespace_does_not_require_task_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("benchmark.routes.benchmark.raise_if_benchmark_disabled", lambda: None)

    async def _list(benchmark_name: str, date: str, *, namespace: str = "m") -> list[str]:
        assert namespace == "s"
        return []

    async def _accept(
        payload: ExternalBenchmarkRequest,
        accept: object,
    ) -> ExternalRunAcceptedResponse:
        return ExternalRunAcceptedResponse(
            run_id=str(accept.assigned_run_id),
            benchmark_name=payload.benchmark_name,
            status="initializing",
        )

    monkeypatch.setattr("benchmark.routes.benchmark.list_manual_run_ids", _list)
    monkeypatch.setattr("benchmark.routes.benchmark._accept_external_run", _accept)
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval_v2",
        date="20261031",
        manual_run_namespace="s",
    )
    response = await run_manual_benchmark(payload, _healthy_request(), _deps(), None)
    assert isinstance(response, ExternalRunAcceptedResponse)
    assert response.run_id == "search_eval_v2-20261031-s1"


def test_pin_dated_prefix_without_duplicate_date() -> None:
    """Immediate / scheduled: when the prefix already contains the data date, pin the object key so the loader does not apply the date twice."""
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval_v2",
        date="20261031",
        oss_prefix="search-eval/benchmark_result/20261031",
        manual_run_namespace="s",
    )
    _pin_dated_oss_object_key(payload)
    assert payload.oss_object_key == (
        "search-eval/benchmark_result/20261031/benchmark_final_results.csv"
    )

    payload.oss_object_key = "already/snapshot.csv"
    _pin_dated_oss_object_key(payload)
    assert payload.oss_object_key == "already/snapshot.csv"

    # Immediate trigger: after the page pre-expands ${yyyymmdd} and forwards a dated prefix, it must still be pinned
    immediate = ExternalBenchmarkRequest(
        benchmark_name="search_eval_v2",
        date="20261103",
        oss_prefix="search-eval/benchmark_result/20261103",
    )
    _pin_dated_oss_object_key(immediate)
    assert immediate.oss_object_key == (
        "search-eval/benchmark_result/20261103/benchmark_final_results.csv"
    )

    # Pre-expanded offset macro: a tail segment ≠ data_date must also be pinned, to avoid .../20261102/20261103/...
    offset = ExternalBenchmarkRequest(
        benchmark_name="search_eval_v2",
        date="20261103",
        oss_prefix="search-eval/benchmark_result/20261102",
    )
    _pin_dated_oss_object_key(offset)
    assert offset.oss_object_key == (
        "search-eval/benchmark_result/20261102/benchmark_final_results.csv"
    )

    non_dated = ExternalBenchmarkRequest(
        benchmark_name="search_eval_v2",
        date="20261031",
        oss_prefix="search-eval/benchmark_result",
        manual_run_namespace="s",
    )
    _pin_dated_oss_object_key(non_dated)
    assert non_dated.oss_object_key is None


@pytest.mark.asyncio
async def test_manual_route_rejects_when_eval_queue_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the queue stack is degraded (worker not started), reject with 503 and do not produce a fake 202 success.

    A run accepted while degraded has no worker to consume it (processedCount stays 0 and the stale
    recovery job eventually reaps it). On rejection the error carries the degradation reason and Redis config guidance.
    """
    monkeypatch.setattr("benchmark.routes.benchmark.raise_if_benchmark_disabled", lambda: None)
    accepted = MagicMock()
    monkeypatch.setattr("benchmark.routes.benchmark._accept_external_run", accepted)

    payload = ExternalBenchmarkRequest(benchmark_name="ieval_external", date="20261010")
    with pytest.raises(HTTPException) as exc:
        await run_manual_benchmark(payload, _degraded_request(), _deps(), None)
    assert exc.value.status_code == 503
    detail = str(exc.value.detail)
    assert "degraded" in detail
    assert "Redis connection not configured" in detail
    accepted.assert_not_called()
