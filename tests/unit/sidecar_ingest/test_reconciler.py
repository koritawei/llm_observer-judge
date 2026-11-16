"""Hot-reconcile tests for the sidecar stable identity."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry, EvalConfigFile, EvalConfigStore
from benchmark.ingest.sidecar.config import SidecarTraceConfig
from benchmark.ingest.sidecar.service import SidecarIngestDeps, SidecarIngestService


def _entry(**kwargs: object) -> EvalConfigEntry:
    return EvalConfigEntry.model_validate(
        {
            "evaluatorName": "example",
            "sampling": 1.0,
            "ingest": "sidecar",
            **kwargs,
        }
    )


def _backend() -> MagicMock:
    backend = MagicMock()
    backend.start = AsyncMock()
    backend.close = AsyncMock()
    backend.subscribe_with_expressions = AsyncMock()
    backend.poll = AsyncMock(return_value=None)
    return backend


def _service(store: EvalConfigStore, factory) -> SidecarIngestService:
    reader = MagicMock()
    reader.open = AsyncMock()
    reader.close = AsyncMock()
    return SidecarIngestService(
        SidecarIngestDeps(
            benchmark_judge=MagicMock(),
            config=SidecarTraceConfig(enabled=True),
            env="daily",
            lindorm_dsn="mysql://sidecar",
            config_store=store,
            reader=reader,
            backend_factory=factory,
        )
    )


@pytest.mark.asyncio
async def test_reconcile_adds_keeps_and_removes_consumer_without_restart() -> None:
    current: dict[str, list[EvalConfigEntry]] = {}
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.side_effect = lambda: EvalConfigFile(configs=current)
    backends: list[MagicMock] = []

    def factory() -> MagicMock:
        backend = _backend()
        backends.append(backend)
        return backend

    service = _service(store, factory)
    await service.start()
    assert not backends

    current["p-a"] = [_entry()]
    await service.reconcile_once()
    assert len(backends) == 1
    await service.reconcile_once()
    assert len(backends) == 1

    current.clear()
    await service.reconcile_once()
    backends[0].close.assert_awaited()
    assert not service._consumers
    await service.stop()


@pytest.mark.asyncio
async def test_eval_snapshot_refresh_triggers_immediate_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a page activates or retires an evaluator and refreshes the snapshot, the consumer should come up or down immediately, without waiting 30s."""
    from benchmark.domain import runtime_config_store as store_mod

    monkeypatch.setattr(store_mod, "_eval_snapshot_listeners", [])
    current: dict[str, list[EvalConfigEntry]] = {}
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.side_effect = lambda: EvalConfigFile(configs=current)
    backends: list[MagicMock] = []

    def factory() -> MagicMock:
        backend = _backend()
        backends.append(backend)
        return backend

    service = _service(store, factory)
    await service.start()
    assert service._snapshot_listener_registered
    assert not backends

    current["p-a"] = [_entry()]
    # Simulate the tail notification of refresh_eval_config_snapshot (fire-and-forget; it does not block the caller)
    await store_mod._notify_eval_snapshot_listeners()
    pending = list(service._pending_snapshot_reconciles)
    assert pending, "snapshot 通知应调度后台 reconcile"
    await asyncio.gather(*pending)
    assert len(backends) == 1
    assert service.active_consumer_count == 1

    current.clear()
    await store_mod._notify_eval_snapshot_listeners()
    pending = list(service._pending_snapshot_reconciles)
    await asyncio.gather(*pending)
    assert service.active_consumer_count == 0
    backends[0].close.assert_awaited()

    await service.stop()
    assert not service._snapshot_listener_registered
    # After stop, it no longer responds
    current["p-a"] = [_entry()]
    await store_mod._notify_eval_snapshot_listeners()
    assert not service._pending_snapshot_reconciles
    assert service.active_consumer_count == 0


@pytest.mark.asyncio
async def test_same_evaluator_streams_share_one_acl_cid() -> None:
    """Several streams of the same project+evaluator share one ACL allowlist CID, and the subscription is not restarted."""
    first = _entry(agentRegistrationId="reg-v1", streamInstanceId="ds-a")
    second = _entry(agentRegistrationId="reg-v2", streamInstanceId="ds-b")
    current = {"p-a": [first, second]}
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.side_effect = lambda: EvalConfigFile(configs=current)
    backends: list[MagicMock] = []

    def factory() -> MagicMock:
        backend = _backend()
        backends.append(backend)
        return backend

    service = _service(store, factory)
    await service.start()
    assert len(backends) == 1
    assert service.active_consumer_count == 1
    group = backends[0].subscribe_with_expressions.await_args.args[1]
    assert group == "CID_sidecar_trace_notify_p-a_example"
    assert "reg-v1" not in group
    assert "ds-a" not in group

    current["p-a"] = [second, first]
    await service.reconcile_once()
    assert len(backends) == 1

    current["p-a"] = [second]
    await service.reconcile_once()
    assert len(service._consumers) == 1
    remaining = next(iter(service._consumers))
    assert remaining.project_id == "p-a"
    assert remaining.evaluator_name == "example"
    await service.stop()


@pytest.mark.asyncio
async def test_single_start_failure_does_not_block_other_entries() -> None:
    current = {"p-a": [_entry(evaluatorName="good")]}
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.side_effect = lambda: EvalConfigFile(configs=current)
    backends: list[MagicMock] = []

    failing = _backend()
    failing.start = AsyncMock(side_effect=RuntimeError("start failed"))
    calls = {"count": 0}

    def factory() -> MagicMock:
        calls["count"] += 1
        if calls["count"] == 2:
            return failing
        backend = _backend()
        backends.append(backend)
        return backend

    service = _service(store, factory)
    await service.start()
    current["p-a"].append(_entry(evaluatorName="bad"))

    await service.reconcile_once()

    assert len(service._consumers) == 1
    assert next(iter(service._consumers)).evaluator_name == "good"
    failing.close.assert_awaited()
    await service.stop()
