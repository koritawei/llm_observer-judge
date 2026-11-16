"""Sidecar ingest service: one MetaQ consumer per eval config."""

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry, EvalConfigFile, EvalConfigStore
from benchmark.ingest.sidecar.config import SidecarTraceConfig
from benchmark.ingest.sidecar.processor import SidecarProcessor
from benchmark.ingest.sidecar.service import SidecarIngestService
from benchmark.libs.mq import ConsumeError, RawMessage
from benchmark.libs.mq._backends.metaq.backend import MetaqConsumerBackend


def _entry(**kwargs: object) -> EvalConfigEntry:
    payload = {
        "evaluatorName": "amap_agent",
        "sampling": 1.0,
        "ingest": "sidecar",
        **kwargs,
    }
    return EvalConfigEntry.model_validate(payload)


def _backend() -> MagicMock:
    backend = MagicMock()
    backend.start = AsyncMock()
    backend.close = AsyncMock()
    backend.subscribe_with_expressions = AsyncMock()
    backend.poll = AsyncMock(return_value=None)
    return backend


def _service(
    *,
    store: EvalConfigStore,
    reader: MagicMock,
    backend_factory: Callable[[], MetaqConsumerBackend],
    lindorm_dsn: str = "mysql://sidecar",
) -> SidecarIngestService:
    from benchmark.ingest.sidecar.service import SidecarIngestDeps

    return SidecarIngestService(
        SidecarIngestDeps(
            benchmark_judge=MagicMock(),
            config=SidecarTraceConfig(enabled=True),
            env="daily",
            lindorm_dsn=lindorm_dsn,
            config_store=store,
            reader=reader,
            backend_factory=backend_factory,
        )
    )


@pytest.mark.asyncio
async def test_start_skips_when_no_sidecar_entries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile()
    reader = MagicMock()
    reader.open = AsyncMock()
    factory = MagicMock()
    service = _service(store=store, reader=reader, backend_factory=factory)
    with caplog.at_level("WARNING"):
        await service.start()
    reader.open.assert_not_awaited()
    factory.assert_not_called()
    assert service.active_consumer_count == 0
    assert any("sidecar.enabled_but_no_ingest_entries" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_start_skips_kafka_only_remote_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(configs={"p-a": [_entry(ingest="kafka")]})
    reader = MagicMock()
    reader.open = AsyncMock()
    factory = MagicMock()
    service = _service(store=store, reader=reader, backend_factory=factory)
    with caplog.at_level("WARNING"):
        await service.start()
    reader.open.assert_not_awaited()
    factory.assert_not_called()
    assert service.active_consumer_count == 0
    assert any("sidecar.enabled_but_no_ingest_entries" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_start_refuses_empty_lindorm_dsn() -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(configs={"p-a": [_entry()]})
    reader = MagicMock()
    reader.open = AsyncMock()
    factory = MagicMock()
    service = _service(store=store, reader=reader, backend_factory=factory, lindorm_dsn="")
    with pytest.raises(RuntimeError, match="SIDECAR_TRACE_LINDORM_DSN"):
        await service.start()
    reader.open.assert_not_awaited()
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_start_consumer_closes_backend_on_cancelled_error() -> None:
    """When subscribe is interrupted by CancelledError, still close the backend, so a connection is not left orphaned."""
    from benchmark.ingest.sidecar.consumer_group import SidecarConsumerIdentity

    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(configs={"p-a": [_entry()]})
    reader = MagicMock()
    reader.open = AsyncMock()
    backends: list[MagicMock] = []

    def factory() -> MagicMock:
        backend = _backend()
        backend.subscribe_with_expressions = AsyncMock(side_effect=asyncio.CancelledError())
        backends.append(backend)
        return backend

    service = _service(store=store, reader=reader, backend_factory=factory)
    identity = SidecarConsumerIdentity.from_entry("p-a", _entry())
    with pytest.raises(asyncio.CancelledError):
        await service._start_consumer(identity)
    assert service.active_consumer_count == 0
    assert len(backends) == 1
    backends[0].close.assert_awaited()


@pytest.mark.asyncio
async def test_start_one_consumer_per_eval_entry() -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(
        configs={
            "p-a": [_entry()],
            "p-b": [_entry(evaluatorName="example", environment=["prod"])],
        }
    )
    reader = MagicMock()
    reader.open = AsyncMock()
    reader.close = AsyncMock()
    backends: list[MagicMock] = []

    def factory() -> MagicMock:
        backend = _backend()
        backends.append(backend)
        return backend

    service = _service(store=store, reader=reader, backend_factory=factory)
    await service.start()
    assert service.active_consumer_count == 2
    assert len(backends) == 2
    topics = {tuple(backend.subscribe_with_expressions.await_args.args[0]) for backend in backends}
    groups = {backend.subscribe_with_expressions.await_args.args[1] for backend in backends}
    assert topics == {
        ("sidecar_trace_notify_p-a",),
        ("sidecar_trace_notify_p-b",),
    }
    assert groups == {
        "CID_sidecar_trace_notify_p-a_amap_agent",
        "CID_sidecar_trace_notify_p-b_example",
    }
    await service.stop()
    for backend in backends:
        backend.close.assert_awaited()


@pytest.mark.asyncio
async def test_start_subscribes_only_sidecar_entries() -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(
        configs={
            "p-a": [
                _entry(ingest="kafka"),
                _entry(evaluatorName="example"),
            ],
            "p-b": [_entry(ingest="kafka", evaluatorName="example")],
        }
    )
    reader = MagicMock()
    reader.open = AsyncMock()
    reader.close = AsyncMock()
    backends: list[MagicMock] = []

    def factory() -> MagicMock:
        backend = _backend()
        backends.append(backend)
        return backend

    service = _service(store=store, reader=reader, backend_factory=factory)
    await service.start()
    assert len(backends) == 1
    assert backends[0].subscribe_with_expressions.await_args.args[0] == {
        "sidecar_trace_notify_p-a": None,
    }
    assert backends[0].subscribe_with_expressions.await_args.args[1] == (
        "CID_sidecar_trace_notify_p-a_example"
    )
    await service.stop()


def test_ack_timeout_exceeds_handler_timeout() -> None:
    from benchmark.ingest.sidecar.service import _to_configuration

    cfg = SidecarTraceConfig(handler_timeout_s=300)
    extra = _to_configuration(cfg, "daily").extra
    assert float(extra["ack_timeout"]) > cfg.handler_timeout_s


@pytest.mark.asyncio
async def test_long_evaluate_is_acked_before_it_finishes() -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(configs={"p-a": [_entry()]})
    store.get_configs.return_value = [_entry()]
    reader = MagicMock()
    reader.open = AsyncMock()
    reader.close = AsyncMock()
    raw = RawMessage(
        topic="sidecar_trace_notify_p-a",
        partition=0,
        offset=1,
        key=None,
        value=b"{}",
        headers=[],
    )
    backend = _backend()
    polls = {"n": 0}

    async def _poll(_timeout: float) -> RawMessage | None:
        polls["n"] += 1
        if polls["n"] == 1:
            return raw
        await asyncio.sleep(0.05)
        return None

    order: list[str] = []

    async def _settle(*_args: object, **_kwargs: object) -> None:
        order.append("ack")

    backend.poll = AsyncMock(side_effect=_poll)
    backend.settle_message = AsyncMock(side_effect=_settle)
    service = _service(store=store, reader=reader, backend_factory=lambda: backend)
    await service.start()

    async def _handle(*_args: object, **_kwargs: object) -> None:
        assert order == ["ack"]
        await asyncio.sleep(0.05)
        order.append("eval")

    assert service._processor is not None
    service._processor.handle = _handle  # type: ignore[method-assign]
    for _ in range(50):
        if "eval" in order:
            break
        await asyncio.sleep(0.01)
    assert order == ["ack", "eval"]
    await service.stop()


@pytest.mark.asyncio
async def test_poll_consume_error_keeps_loop_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("benchmark.ingest.sidecar.service._POLL_ERROR_BACKOFF_S", 0)
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(configs={"p-a": [_entry()]})
    reader = MagicMock()
    reader.open = AsyncMock()
    reader.close = AsyncMock()
    backend = _backend()
    backend.poll = AsyncMock(side_effect=ConsumeError("poll failed"))
    service = _service(store=store, reader=reader, backend_factory=lambda: backend)
    await service.start()
    for _ in range(50):
        if backend.poll.await_count >= 2:
            break
        await asyncio.sleep(0.01)
    assert backend.poll.await_count >= 2
    await service.stop()


@pytest.mark.asyncio
async def test_stop_closes_remaining_backends_if_one_fails() -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(
        configs={"p-a": [_entry(), _entry(evaluatorName="example")]}
    )
    reader = MagicMock()
    reader.open = AsyncMock()
    reader.close = AsyncMock()
    backends: list[MagicMock] = []

    def factory() -> MagicMock:
        backend = _backend()
        backends.append(backend)
        return backend

    service = _service(store=store, reader=reader, backend_factory=factory)
    await service.start()
    assert len(backends) == 2
    backends[0].close = AsyncMock(side_effect=RuntimeError("close fail"))
    await service.stop()
    backends[1].close.assert_awaited()


@pytest.mark.asyncio
async def test_message_uses_refreshed_remote_config_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(configs={"p-a": [_entry(sampling=0.0005)]})
    refreshed = _entry(sampling=1.0, environment=["prod"])
    store.get_configs.return_value = [refreshed]
    reader = MagicMock()
    reader.open = AsyncMock()
    reader.close = AsyncMock()
    backend = _backend()
    message = RawMessage(
        topic="sidecar_trace_notify_p-a",
        partition=0,
        offset=1,
        key=None,
        value=b"{}",
        headers=[],
    )
    polls = {"sent": False}
    release = asyncio.Event()

    async def _poll(_timeout: float) -> RawMessage | None:
        if not polls["sent"]:
            polls["sent"] = True
            return message
        await release.wait()
        return None

    backend.poll = AsyncMock(side_effect=_poll)
    backend.settle_message = AsyncMock()
    seen: list[object] = []

    async def _handle(
        _self: SidecarProcessor,
        _raw: object,
        *,
        config: object,
        project_id: str,
    ) -> None:
        del project_id
        seen.append(config)

    monkeypatch.setattr(SidecarProcessor, "handle", _handle)
    service = _service(store=store, reader=reader, backend_factory=lambda: backend)
    await service.start()
    for _ in range(50):
        if seen:
            break
        await asyncio.sleep(0.01)
    release.set()
    await service.stop()
    assert seen == [refreshed]
    backend.settle_message.assert_awaited()


@pytest.mark.asyncio
async def test_message_acks_when_entry_no_longer_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MagicMock(spec=EvalConfigStore)
    store.get_config_file.return_value = EvalConfigFile(configs={"p-a": [_entry()]})
    store.get_configs.return_value = [_entry(ingest="kafka")]
    reader = MagicMock()
    reader.open = AsyncMock()
    reader.close = AsyncMock()
    backend = _backend()
    message = RawMessage(
        topic="sidecar_trace_notify_p-a",
        partition=0,
        offset=2,
        key=None,
        value=b"{}",
        headers=[],
    )
    polls = {"sent": False}
    release = asyncio.Event()

    async def _poll(_timeout: float) -> RawMessage | None:
        if not polls["sent"]:
            polls["sent"] = True
            return message
        await release.wait()
        return None

    backend.poll = AsyncMock(side_effect=_poll)
    backend.settle_message = AsyncMock()
    handled = {"n": 0}

    async def _handle(*_args: object, **_kwargs: object) -> None:
        handled["n"] += 1

    monkeypatch.setattr(SidecarProcessor, "handle", _handle)
    service = _service(store=store, reader=reader, backend_factory=lambda: backend)
    await service.start()
    for _ in range(50):
        if backend.settle_message.await_count:
            break
        await asyncio.sleep(0.01)
    release.set()
    await service.stop()
    assert handled["n"] == 0
    assert backend.settle_message.await_args.kwargs["accepted"] is True
