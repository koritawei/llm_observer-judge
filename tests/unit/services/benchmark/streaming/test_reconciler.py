"""Unit tests for StreamingTopicReconciler (§6 reconcile / §11 empty-set protection / §8 env fallback)."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.streaming_config import StreamingConfig, StreamingTaskConfig
from benchmark.streaming import reconciler as rec_mod
from benchmark.streaming.config import (
    KafkaConsumerConfig,
    StreamingKafkaConfig,
)
from benchmark.streaming.reconciler import StreamingTopicReconciler
from benchmark.streaming.registry import RunningTask, StreamingTaskRegistry
from benchmark.streaming.tasks import (
    ENV_FALLBACK_TASK_NAME,
    StreamingTaskSpec,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def mock_sync() -> Generator[AsyncMock, None, None]:
    """Mirror-persist mock (every case prevents a real DB call; a named reference can be asserted synchronously)."""
    sync = AsyncMock(return_value=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rec_mod, "sync_streaming_tasks", sync)
        yield sync


# ── fakes ────────────────────────────────────────────────────────────────


class _FakeRunner:
    """Recording stand-in for the runner primitives (where reconciler diff assertions land)."""

    def __init__(self) -> None:
        self.registry = StreamingTaskRegistry()
        self.started: list[StreamingTaskSpec] = []
        self.stopped: list[tuple[str, bool]] = []
        self.topics_added: list[tuple[str, list[str]]] = []
        self.topics_removed: list[tuple[str, list[str]]] = []

    async def start_task(self, spec: StreamingTaskSpec) -> RunningTask:
        self.started.append(spec)
        task = RunningTask(spec=spec, evaluator=MagicMock(), runtime_state="running")
        task.consumers = {topic: [MagicMock(_running=True)] for topic in spec.topics}
        self.registry.upsert(task)
        return task

    async def stop_task(
        self,
        name: str,
        *,
        timeout: float = 60.0,
        remove: bool = True,
        mark: StreamingTaskSpec | None = None,
    ) -> RunningTask | None:
        self.stopped.append((name, remove))
        task = self.registry.get(name)
        if task is None:
            return None
        task.consumers = {}
        task.run_tasks = {}
        if remove:
            self.registry.remove(name)
        elif mark is not None:
            task.spec = mark
        return task

    async def add_topics(self, name: str, topics: list[str]) -> None:
        self.topics_added.append((name, topics))

    async def remove_topics(self, name: str, topics: list[str], **kwargs: object) -> None:
        self.topics_removed.append((name, topics))

    def build_evaluator(self, spec: StreamingTaskSpec) -> MagicMock:
        return MagicMock()


def _kafka_config(*, shadow: bool = False) -> StreamingKafkaConfig:
    return StreamingKafkaConfig(
        consumer=KafkaConsumerConfig(
            topics=["langfuse.records"],
            group_id="agent-eval-judge",
            bootstrap_servers="localhost:9092",
        ),
        topics=["langfuse.records"],
        concurrency=1,
        evaluate_timeout_s=0,
        max_msgs_per_min=0.0,
        shadow=shadow,
    )


def _reconciler(
    runner: _FakeRunner,
    config: StreamingConfig | None,
    *,
    shadow: bool = False,
) -> StreamingTopicReconciler:
    return StreamingTopicReconciler(
        runner=runner,
        kafka_config=_kafka_config(shadow=shadow),
        fetch_config=lambda: config,
    )


def _task(name: str, **updates: object) -> StreamingTaskConfig:
    payload: dict[str, object] = {
        "name": name,
        "topics": ["langfuse.records"],
        "group_id": "group-x",
        "state": "running",
    }
    payload.update(updates)
    return StreamingTaskConfig.model_validate(payload)


def _config(
    *tasks: StreamingTaskConfig, enabled: bool = True, allow_empty: bool = False
) -> StreamingConfig:
    return StreamingConfig(
        enabled=enabled,
        allow_empty_tasks=allow_empty,
        tasks=list(tasks),
    )


# ── diff's six operations (§6.2 / §6.3)──────────────────────────────────────────


async def test_new_running_task_starts_instances() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1")))
    await rec.reconcile_once()
    assert [s.name for s in runner.started] == ["t1"]
    assert runner.registry.get("t1") is not None


async def test_new_paused_task_registered_without_start() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1", state="paused")))
    await rec.reconcile_once()
    assert runner.started == []
    task = runner.registry.get("t1")
    assert task is not None
    assert task.runtime_state == "paused"


async def test_deleted_task_stopped_and_removed() -> None:
    runner = _FakeRunner()
    await runner.start_task(
        StreamingTaskSpec(name="t1", topics=["langfuse.records"], group_id="group-x"),
    )
    runner.started.clear()
    rec = _reconciler(runner, _config(_task("t2")))
    await rec.reconcile_once()
    assert ("t1", True) in runner.stopped
    assert runner.registry.get("t1") is None
    assert [s.name for s in runner.started] == ["t2"]


async def test_running_to_paused_drains_and_marks() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1")))
    await rec.reconcile_once()
    rec_pause = _reconciler(runner, _config(_task("t1", state="paused")))
    await rec_pause.reconcile_once()
    assert ("t1", False) in runner.stopped
    task = runner.registry.get("t1")
    assert task is not None
    assert task.runtime_state == "paused"


async def test_paused_to_running_resumes() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1", state="paused")))
    await rec.reconcile_once()
    runner.started.clear()
    rec_resume = _reconciler(runner, _config(_task("t1")))
    await rec_resume.reconcile_once()
    assert [s.name for s in runner.started] == ["t1"]


async def test_topics_diff_is_element_level() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1", topics=["a", "b"])))
    await rec.reconcile_once()
    rec2 = _reconciler(runner, _config(_task("t1", topics=["b", "c"])))
    await rec2.reconcile_once()
    assert runner.topics_added == [("t1", ["c"])]
    assert runner.topics_removed == [("t1", ["a"])]
    # An unchanged topic does not trigger a rebuild
    assert len(runner.started) == 1


async def test_exec_param_change_rebuilds_task() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1", sampling=1.0)))
    await rec.reconcile_once()
    rec2 = _reconciler(runner, _config(_task("t1", sampling=0.5)))
    await rec2.reconcile_once()
    assert ("t1", True) in runner.stopped
    assert [s.name for s in runner.started] == ["t1", "t1"]


# ── empty-set protection (§11)───────────────────────────────────────────────────────


async def test_empty_convergence_protected_without_allow_flag() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1")))
    await rec.reconcile_once()
    runner.started.clear()
    runner.stopped.clear()
    # enabled=false and allow_empty_tasks not set explicitly → keep the current state
    rec_off = _reconciler(runner, _config(enabled=False))
    await rec_off.reconcile_once()
    assert runner.stopped == []
    assert runner.registry.get("t1") is not None
    # tasks=[] is the same
    rec_empty = _reconciler(runner, _config())
    await rec_empty.reconcile_once()
    assert runner.stopped == []
    assert runner.registry.get("t1") is not None


async def test_empty_convergence_confirmed_with_allow_flag() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1")))
    await rec.reconcile_once()
    rec_off = _reconciler(runner, _config(enabled=False, allow_empty=True))
    await rec_off.reconcile_once()
    assert ("t1", True) in runner.stopped
    assert runner.registry.names() == []


async def test_parse_failure_keeps_current_state() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1")))
    await rec.reconcile_once()
    runner.started.clear()
    runner.stopped.clear()
    # fetch returns None (keeping the last successful config on parse failure is handled by the domain layer;
    # None reaching the reconciler means no declaration is available → existing tasks are kept)
    rec_none = _reconciler(runner, None)
    await rec_none.reconcile_once()
    assert runner.stopped == []
    assert runner.registry.get("t1") is not None


# ── per-entry validation (§5.2)─────────────────────────────────────────────────────


async def test_null_group_at_most_one() -> None:
    runner = _FakeRunner()
    rec = _reconciler(
        runner,
        _config(
            _task("t1", group_id=None),
            _task("t2", group_id=None, topics=["other"]),
        ),
    )
    await rec.reconcile_once()
    assert [s.name for s in runner.started] == ["t1"]


async def test_same_group_same_topic_rejected() -> None:
    runner = _FakeRunner()
    rec = _reconciler(
        runner,
        _config(
            _task("t1", group_id="g1"),
            _task("t2", group_id="g1", topics=["langfuse.records"]),
        ),
    )
    await rec.reconcile_once()
    assert [s.name for s in runner.started] == ["t1"]


async def test_sampling_zero_rejected() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1", sampling=0.0)))
    await rec.reconcile_once()
    assert runner.started == []


async def test_batch_alias_rejected_for_streaming() -> None:
    runner = _FakeRunner()
    rec = _reconciler(
        runner,
        _config(_task("t1", judge_class="ieval_external_judge")),
    )
    await rec.reconcile_once()
    assert runner.started == []


async def test_streaming_alias_subset_accepted() -> None:
    runner = _FakeRunner()
    rec = _reconciler(
        runner,
        _config(
            _task(
                "t1",
                judge_class="evaluator_service_judge",
                attribution_class="no_attribution",
                rubric_class="amap_agent_kafka_rubric",
            ),
        ),
    )
    await rec.reconcile_once()
    assert [s.name for s in runner.started] == ["t1"]
    spec = runner.started[0]
    assert spec.judge_class == "evaluator_service_judge"
    assert spec.attribution_class == "no_attribution"
    assert spec.attribution_execution == "none"
    assert spec.judge_execution == "sync"


async def test_amap_stage_aliases_and_sync_async_combo() -> None:
    """A three-stage alias plus a sync judge / async attribution combination can pass reconciliation."""
    runner = _FakeRunner()
    rec = _reconciler(
        runner,
        _config(
            _task(
                "amap_agent_daily",
                judge_class="evaluator_service_judge",
                judge_execution="sync",
                triage_class="amap_agent_triage",
                attribution_class="amap_agent_attribution",
                attribution_execution="async",
                rubric_class="amap_agent_kafka_rubric",
            ),
        ),
    )
    await rec.reconcile_once()
    assert [s.name for s in runner.started] == ["amap_agent_daily"]
    spec = runner.started[0]
    assert spec.triage_class == "amap_agent_triage"
    assert spec.attribution_class == "amap_agent_attribution"
    assert spec.judge_execution == "sync"
    assert spec.attribution_execution == "async"
    assert spec.no_attribution is False


async def test_async_judge_on_evaluator_service_rejected() -> None:
    runner = _FakeRunner()
    rec = _reconciler(
        runner,
        _config(
            _task(
                "t1",
                judge_class="evaluator_service_judge",
                judge_execution="async",
            ),
        ),
    )
    await rec.reconcile_once()
    assert runner.started == []


# ── env fallback (§8) and shadow group (P2)───────────────────────────────────


async def test_env_fallback_starts_when_no_remote_config_config() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, None)
    await rec.reconcile_once()
    assert [s.name for s in runner.started] == [ENV_FALLBACK_TASK_NAME]
    spec = runner.started[0]
    assert spec.topics == ["langfuse.records"]
    assert spec.group_id == "agent-eval-judge"


async def test_env_fallback_retired_when_remote_config_activates() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, None)
    await rec.reconcile_once()
    runner.started.clear()
    rec2 = _reconciler(runner, _config(_task("t1")))
    await rec2.reconcile_once()
    assert (ENV_FALLBACK_TASK_NAME, True) in runner.stopped
    assert [s.name for s in runner.started] == ["t1"]


async def test_shadow_forces_independent_group() -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1", group_id=None)), shadow=True)
    await rec.reconcile_once()
    assert runner.started[0].group_id == "agent-eval-judge-shadow"


async def test_interval_floor_applied() -> None:
    runner = _FakeRunner()
    cfg = _config(_task("t1"))
    cfg = cfg.model_copy(update={"reconcile_interval_s": 3})
    rec = _reconciler(runner, cfg)
    await rec.reconcile_once()
    assert rec.interval_s == 10


# ── declared-state mirror persistence (§5.3 revision / §6.7)───────────────────────────────────────


async def test_remote_config_reconcile_syncs_mirror(mock_sync: AsyncMock) -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(_task("t1"), _task("t2", group_id="g2")))
    await rec.reconcile_once()
    mock_sync.assert_awaited_once()
    (specs,) = mock_sync.await_args.args
    assert set(specs) == {"t1", "t2"}
    assert mock_sync.await_args.kwargs["source"] == "remote_config"


async def test_env_fallback_syncs_mirror_with_env_source(mock_sync: AsyncMock) -> None:
    runner = _FakeRunner()
    rec = _reconciler(runner, None)
    await rec.reconcile_once()
    mock_sync.assert_awaited_once()
    (specs,) = mock_sync.await_args.args
    assert set(specs) == {ENV_FALLBACK_TASK_NAME}
    assert mock_sync.await_args.kwargs["source"] == "env"


async def test_empty_protection_does_not_sync(mock_sync: AsyncMock) -> None:
    """When empty-set protection fires (keep the current state), do not persist — the mirror matches the declared state."""
    runner = _FakeRunner()
    rec = _reconciler(runner, _config(enabled=False))
    await rec.reconcile_once()
    mock_sync.assert_not_awaited()
