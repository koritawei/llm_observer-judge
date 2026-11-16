"""Unit tests for unified case-worker routing (design doc §8.2/§8.3).

Locks acceptance point 6: the same ``benchmark-cases`` worker routes to the sync / async judge implementation
by the ``judge.execution`` in the run's ``config_json`` snapshot;
the default (a legacy run with no snapshot key) = the production sync behavior is unchanged.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.queue.queue_factory import BENCHMARK_CASES, BENCHMARK_RUNNING
from benchmark.workers import benchmark_worker
from benchmark.workers.benchmark_worker import (
    BenchmarkWorkerDeps,
    _finalize_eval_type,
    process_benchmark_case,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Isolate the module-level per-run cache (so tests do not pollute each other)."""
    benchmark_worker._run_exec_cfg_cache.clear()
    benchmark_worker._run_rubric_cache.clear()
    benchmark_worker._run_local_impl_cache.clear()
    yield
    benchmark_worker._run_exec_cfg_cache.clear()
    benchmark_worker._run_rubric_cache.clear()
    benchmark_worker._run_local_impl_cache.clear()


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.sync_retry_max_duration_ms = 1000
    settings.sync_cluster_soft_cap = 0
    settings.sync_slots_per_run_per_node = 1
    settings.sync_slot_busy_delay_ms = 100
    settings.sync_slot_ttl_s = 60
    return settings


def _deps(run_config: dict[str, Any], *, with_factory: bool = False) -> BenchmarkWorkerDeps:
    run = MagicMock()
    run.config_json = run_config
    run_service = MagicMock()
    run_service.get_run = AsyncMock(return_value=run)
    factory = MagicMock() if with_factory else None
    if factory is not None:
        factory.get_judge_client.return_value = MagicMock(name="judge_client")
    return BenchmarkWorkerDeps(
        run_service=run_service,
        evaluator_service=MagicMock(),
        triage_service=MagicMock(),
        settings=_settings(),
        external_client_factory=factory,
    )


def _job(run_id: str = "run-1") -> MagicMock:
    job = MagicMock()
    job.data = {"run_id": run_id, "item_id": f"{run_id}:0"}
    return job


async def _capture_handlers(
    monkeypatch: pytest.MonkeyPatch,
    deps: BenchmarkWorkerDeps,
) -> dict[str, Any]:
    """Run process_benchmark_case and capture the handlers passed to the skeleton."""
    captured: dict[str, Any] = {}

    async def _fake_skeleton(job: Any, token: str, runtime: Any, **_: Any) -> None:
        captured["handlers"] = runtime.handlers

    monkeypatch.setattr(benchmark_worker, "process_sync_eval_case", _fake_skeleton)
    await process_benchmark_case(
        _job(),
        "token",
        queue_manager=MagicMock(),
        deps=deps,
    )
    return captured


@pytest.mark.asyncio
async def test_route_sync_default_without_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy run (no judge snapshot key) → sync-semantics handlers (production behavior unchanged)."""
    captured = await _capture_handlers(monkeypatch, _deps({}))
    handlers = captured["handlers"]
    assert handlers.eval_type == "benchmark"
    assert handlers.wait_queue_name == BENCHMARK_CASES
    assert handlers.running_queue_name == BENCHMARK_RUNNING
    # sync semantics: release the slot on completion / remove running on success (the opposite of async, which keeps the slot)
    assert handlers.hold_slot_on_success is False
    assert handlers.remove_running_on_success is True
    assert handlers.cluster_slot_kind == "sync"


@pytest.mark.asyncio
async def test_route_sync_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = await _capture_handlers(monkeypatch, _deps({"judge": {"execution": "sync"}}))
    assert captured["handlers"].cluster_slot_kind == "sync"


@pytest.mark.asyncio
async def test_route_async_uses_unified_queue_with_hold_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """judge.execution=async → async handlers (keep the slot and wait for the callback) go through the unified queue."""
    deps = _deps(
        {"judge": {"execution": "async", "target": None, "dims": "D0"}},
        with_factory=True,
    )
    captured = await _capture_handlers(monkeypatch, deps)
    handlers = captured["handlers"]
    assert handlers.wait_queue_name == BENCHMARK_CASES
    assert handlers.running_queue_name == BENCHMARK_RUNNING
    # async semantics: keep the slot and wait for the callback / promote to pending_callback rather than removing it
    assert handlers.hold_slot_on_success is True
    assert handlers.remove_running_on_success is False
    assert handlers.cluster_slot_kind == "judge"
    # The per-run target is resolved through the factory (None → the process default URL)
    factory: Any = deps.external_client_factory
    assert factory is not None
    factory.get_judge_client.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_route_async_per_run_target_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-run judge.target URL is passed through to the factory (§6.3)."""
    url = "https://aone.example.com/api/pipelines/777/trigger"
    deps = _deps({"judge": {"execution": "async", "target": url}}, with_factory=True)
    await _capture_handlers(monkeypatch, deps)
    factory2: Any = deps.external_client_factory
    assert factory2 is not None
    factory2.get_judge_client.assert_called_once_with(url)


@pytest.mark.asyncio
async def test_route_async_without_factory_raises() -> None:
    """An async judge but the factory is not configured → an explicit error (no silent degrade)."""
    deps = _deps({"judge": {"execution": "async"}}, with_factory=False)
    with pytest.raises(RuntimeError, match="AoneClientFactory not configured"):
        await process_benchmark_case(
            _job(),
            "token",
            queue_manager=MagicMock(),
            deps=deps,
        )


def test_finalize_eval_type_routing() -> None:
    """Finalize count-model routing: either judge or attribution being async → external."""
    assert _finalize_eval_type({}, {}) == "benchmark"
    assert _finalize_eval_type({"execution": "sync"}, {"execution": "none"}) == "benchmark"
    assert _finalize_eval_type({"execution": "async"}, {}) == "external"
    # The new combination: sync judge + async attribution → the external counter model
    assert _finalize_eval_type({"execution": "sync"}, {"execution": "async"}) == "external"
