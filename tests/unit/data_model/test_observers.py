"""Registration and trigger tests for Benchmark lifecycle observers.

Covers:
  - ``register_database()`` auto-discovery binds observers to the matching models;
  - a model's ``save()`` override really triggers ``on_save``;
  - paths that bypass save, such as ``filter().update()``, do not trigger observers;
  - the run_items table has no observer (it is written via bulk_create / update_or_create, and attaching one would not fire anyway).
"""

from unittest.mock import AsyncMock

from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import (
    BenchmarkRun,
    BenchmarkRunItem,
    TraceRecord,
)
from benchmark.records.observers import (
    BenchmarkRunObserver,
    TraceRecordObserver,
)


class TestObserverRegistration:
    """conftest's sqlite init goes through register_database → observers are registered automatically."""

    async def test_benchmark_observers_registered(self) -> None:
        assert isinstance(BenchmarkRun.get_observer(), BenchmarkRunObserver)

    async def test_trace_record_observer_still_registered(self) -> None:
        assert isinstance(TraceRecord.get_observer(), TraceRecordObserver)

    async def test_item_models_have_no_observer(self) -> None:
        """The run_items table is written via bulk_create and does not mix in ObserverMixin (avoid dead code)."""
        assert not hasattr(BenchmarkRunItem, "get_observer")


class TestBenchmarkObserverTrigger:
    async def test_run_create_and_save_trigger_on_save(self, monkeypatch) -> None:
        observer = BenchmarkRun.get_observer()
        spy = AsyncMock(wraps=observer.on_save)
        monkeypatch.setattr(observer, "on_save", spy)

        run = await BenchmarkRun.create(
            run_id="bm_test_obs_1",
            benchmark_name="chat-bench",
            service_name="amap_chat",
            dataset_name="ds-1",
            status=BenchmarkStatus.INITIALIZING,
            total_samples=10,
            total_evaluators=2,
        )
        spy.assert_awaited_once()
        call = spy.await_args
        assert call is not None
        assert call.args[0].run_id == run.run_id

        run.status = BenchmarkStatus.RUNNING
        await run.save()
        assert spy.await_count == 2

    async def test_queryset_update_bypasses_observer(self, monkeypatch) -> None:
        """filter().update() bypasses the save() lifecycle and does not trigger observers."""
        observer = BenchmarkRun.get_observer()
        spy = AsyncMock(wraps=observer.on_save)
        monkeypatch.setattr(observer, "on_save", spy)

        run = await BenchmarkRun.create(
            run_id="bm_test_obs_bypass",
            benchmark_name="chat-bench",
            service_name="amap_chat",
            dataset_name="ds-1",
            status=BenchmarkStatus.INITIALIZING,
            total_samples=1,
            total_evaluators=1,
        )
        spy.reset_mock()

        await BenchmarkRun.filter(id=run.id).update(status=BenchmarkStatus.RUNNING)
        spy.assert_not_awaited()
