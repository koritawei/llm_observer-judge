"""Unit tests for streaming task mirror persistence (declared-state mirror of benchmark_kafka_tasks)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.records.database import create_database_config
from benchmark.records.models import BenchmarkKafkaTask
from benchmark.streaming import persistence as p
from benchmark.streaming.tasks import StreamingTaskSpec

pytestmark = pytest.mark.asyncio


def _spec(name: str, **updates: object) -> StreamingTaskSpec:
    payload: dict[str, object] = {
        "name": name,
        "topics": ["langfuse.records"],
        "group_id": "group-x",
    }
    payload.update(updates)
    return StreamingTaskSpec(**payload)  # type: ignore[arg-type]


class _Row:
    """Fake ORM row (only task_id + delete)."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.delete = AsyncMock()


class _AsyncIter:
    def __init__(self, items: list[_Row]) -> None:
        self._items = items

    def __aiter__(self):
        async def _gen():
            for item in self._items:
                yield item

        return _gen()


def _model_mock(existing: list[_Row]) -> MagicMock:
    model = MagicMock()
    model.all.return_value = _AsyncIter(existing)
    model.update_or_create = AsyncMock()
    return model


class TestModelMeta:
    def test_table_name(self) -> None:
        assert BenchmarkKafkaTask._meta.db_table == "benchmark_kafka_tasks"

    def test_task_id_unique(self) -> None:
        field = BenchmarkKafkaTask._meta.fields_map["task_id"]
        assert field.unique is True

    def test_model_registered_in_database_config(self) -> None:
        """The model must be registered in the models list of create_database_config.

        Regression lock (2026-09-28 production incident): when benchmark_kafka_task was left unregistered,
        Tortoise.init did not bind a connection for it, and reconciler mirror sync reported
        ``default_connection ... cannot be None`` (fail-soft WARNING,
        and the mirror table was unusable). This case locks the registration list directly so a newly added model cannot slip through again.
        """
        config = create_database_config(db_url="sqlite:///./data.db")
        assert "benchmark.records.models.benchmark_kafka_task" in config.models


class TestSpecFingerprint:
    def test_fingerprint_stable_for_same_spec(self) -> None:
        assert p.spec_fingerprint(_spec("t1")) == p.spec_fingerprint(_spec("t1"))

    def test_fingerprint_changes_with_spec(self) -> None:
        assert p.spec_fingerprint(_spec("t1")) != p.spec_fingerprint(
            _spec("t1", sampling=0.5),
        )
        assert p.spec_fingerprint(_spec("t1")) != p.spec_fingerprint(
            _spec("t1", state="paused"),
        )

    def test_fingerprint_length(self) -> None:
        assert len(p.spec_fingerprint(_spec("t1"))) == 16


async def test_sync_upserts_specs_and_deletes_stale() -> None:
    stale = _Row("stale-task")
    model = _model_mock([stale, _Row("t1")])
    specs = {
        "t1": _spec("t1", rubric_class="amap_agent_kafka_rubric"),
        "t2": _spec("t2", state="paused", sampling=0.5),
    }
    with patch.object(p, "BenchmarkKafkaTask", model):
        ok = await p.sync_streaming_tasks(specs)
    assert ok is True
    stale.delete.assert_awaited_once()
    assert model.update_or_create.await_count == 2
    by_name = {
        call.kwargs["task_id"]: call.kwargs["defaults"]
        for call in model.update_or_create.await_args_list
    }
    t1 = by_name["t1"]
    assert t1["source"] == "remote_config"
    assert t1["state"] == "running"
    assert t1["topics"] == ["langfuse.records"]
    assert t1["group_id"] == "group-x"
    assert t1["rubric_class"] == "amap_agent_kafka_rubric"
    assert t1["spec_fingerprint"] == p.spec_fingerprint(specs["t1"])
    assert t1["last_reconciled_at"] is not None
    t2 = by_name["t2"]
    assert t2["state"] == "paused"
    assert t2["sampling"] == 0.5


async def test_sync_marks_env_source() -> None:
    model = _model_mock([])
    with patch.object(p, "BenchmarkKafkaTask", model):
        ok = await p.sync_streaming_tasks({"t1": _spec("t1")}, source="env")
    assert ok is True
    defaults = model.update_or_create.await_args.kwargs["defaults"]
    assert defaults["source"] == "env"


async def test_sync_empty_specs_deletes_all() -> None:
    rows = [_Row("t1"), _Row("t2")]
    model = _model_mock(rows)
    with patch.object(p, "BenchmarkKafkaTask", model):
        ok = await p.sync_streaming_tasks({})
    assert ok is True
    for row in rows:
        row.delete.assert_awaited_once()
    model.update_or_create.assert_not_awaited()


async def test_sync_failure_is_fail_soft() -> None:
    model = MagicMock()
    model.all.side_effect = RuntimeError("relation benchmark_kafka_tasks does not exist")
    with patch.object(p, "BenchmarkKafkaTask", model):
        ok = await p.sync_streaming_tasks({"t1": _spec("t1")})
    assert ok is False  # A missing table / DB failure does not block (degrade to WARNING)
