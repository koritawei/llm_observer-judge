"""WebuiTaskScheduler: a task document → a cron job → fire the run when the time arrives."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from benchmark.workers import webui_task_scheduler as wts_mod
from benchmark.workers.scheduler_hub import SchedulerHub
from benchmark.workers.webui_task_scheduler import (
    WebuiTaskScheduler,
    build_task_trigger,
)


def _task(**overrides: object) -> SimpleNamespace:
    body: dict[str, object] = {
        "id": "task-1",
        "status": "active",
        "trigger": "daily",
        "cron_expression": None,
        "agent_id": "ba-1",
        "dataset_id": "ds-1",
    }
    body.update(overrides)
    return SimpleNamespace(**body)


def _scheduler(hub: SchedulerHub) -> WebuiTaskScheduler:
    from benchmark.workers.webui_task_scheduler import WebuiTaskSchedulerDeps

    return WebuiTaskScheduler(
        WebuiTaskSchedulerDeps(
            hub=hub,
            benchmark_service=AsyncMock(),
            runtime=AsyncMock(),
            external_runner=AsyncMock(),
        )
    )


@pytest.mark.asyncio
async def test_batch_execution_aliases_reads_declared_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch judge/attribution alias declared by the component automatically enters the request (search_eval's local constraint)."""
    from benchmark.runs.common.agent_capabilities import CapabilityBindingView

    class _Services:
        benchmark_capabilities = (
            CapabilityBindingView(
                pipeline="batch",
                kind="judge",
                execution="sync",
                provider="local",
                alias="search_eval_judge",
            ),
            CapabilityBindingView(
                pipeline="batch",
                kind="attribution",
                execution="none",
                provider="none",
                alias="no_attribution",
            ),
        )

    async def _fake_ensure(name: str, runtime: object) -> _Services:
        return _Services()

    monkeypatch.setattr(
        "benchmark.runs.common.dynamic_component.ensure_dynamic_benchmark_component",
        _fake_ensure,
    )
    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    aliases = await scheduler._batch_execution_aliases("search_eval")
    assert aliases == {
        "judge_class": "search_eval_judge",
        "attribution_class": "no_attribution",
    }

    # No capability chain declared (a page dynamic agent) → an empty dict, and the resolver's original behavior is kept
    class _Bare:
        pass

    async def _bare_ensure(name: str, runtime: object) -> _Bare:
        return _Bare()

    monkeypatch.setattr(
        "benchmark.runs.common.dynamic_component.ensure_dynamic_benchmark_component",
        _bare_ensure,
    )
    assert await scheduler._batch_execution_aliases("dyn_agent") == {}
    await hub.shutdown()


def test_build_task_trigger_maps_daily_weekly_and_cron() -> None:
    daily = build_task_trigger("daily", None)
    weekly = build_task_trigger("weekly", None)
    custom = build_task_trigger("cron", "0 2 * * 1-5")
    assert daily is not None and weekly is not None and custom is not None
    assert "hour='2'" in str(daily)
    assert "day_of_week='mon'" in str(weekly)
    # An illegal / missing cron returns None (the caller skips registration)
    assert build_task_trigger("cron", None) is None
    assert build_task_trigger("cron", "bad cron") is None
    assert build_task_trigger("immediate", None) is None


@pytest.mark.asyncio
async def test_sync_jobs_registers_updates_and_removes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    # The end of sync calls catch_up; this case tests only cron add/update/remove, and isolates the backfill
    monkeypatch.setattr(scheduler, "catch_up_missed_tasks", AsyncMock())
    tasks = [_task(), _task(id="task-2", trigger="cron", cron_expression="*/5 * * * *")]
    monkeypatch.setattr(wts_mod, "list_schedulable_tasks", AsyncMock(return_value=tasks))

    await scheduler.sync_jobs()
    assert hub.has_job("webui-eval-task:task-1")
    assert hub.has_job("webui-eval-task:task-2")
    assert len(scheduler._signatures) == 2

    # The same signature is not registered twice (so next_run_time does not jitter)
    monkeypatch.setattr(wts_mod, "list_schedulable_tasks", AsyncMock(return_value=tasks))
    await scheduler.sync_jobs()
    assert len(scheduler._signatures) == 2

    # The task is deleted → the job is removed
    monkeypatch.setattr(
        wts_mod,
        "list_schedulable_tasks",
        AsyncMock(return_value=[_task()]),
    )
    await scheduler.sync_jobs()
    assert hub.has_job("webui-eval-task:task-1")
    assert not hub.has_job("webui-eval-task:task-2")
    await scheduler.stop()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_sync_jobs_disables_invalid_cron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An illegal cron must remove an already registered job, so it does not keep firing on the historical expression."""
    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    monkeypatch.setattr(scheduler, "catch_up_missed_tasks", AsyncMock())
    good = _task(id="task-1", trigger="cron", cron_expression="*/5 * * * *")
    monkeypatch.setattr(wts_mod, "list_schedulable_tasks", AsyncMock(return_value=[good]))
    await scheduler.sync_jobs()
    assert hub.has_job("webui-eval-task:task-1")
    assert "webui-eval-task:task-1" in scheduler._signatures

    broken = _task(id="task-1", trigger="cron", cron_expression="not a cron")
    monkeypatch.setattr(wts_mod, "list_schedulable_tasks", AsyncMock(return_value=[broken]))
    await scheduler.sync_jobs()
    assert not hub.has_job("webui-eval-task:task-1")
    assert "webui-eval-task:task-1" not in scheduler._signatures
    await scheduler.stop()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_sync_jobs_invokes_catch_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan A: the end of every leader sync must catch_up, so an add_once that a non-leader skipped is not left unretried forever."""
    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    catch_up = AsyncMock()
    monkeypatch.setattr(scheduler, "catch_up_missed_tasks", catch_up)
    monkeypatch.setattr(wts_mod, "list_schedulable_tasks", AsyncMock(return_value=[]))

    await scheduler.sync_jobs()
    catch_up.assert_awaited_once()
    await scheduler.stop()
    await hub.shutdown()


def test_daily_trigger_next_slot_before_and_after_fire() -> None:
    """daily 02:00: after the slot → 02:00 the next day; after midnight and before the slot → 02:00 today (being on time the next day relies on cron)."""
    from datetime import datetime

    tz = wts_mod._schedule_tz()
    trigger = build_task_trigger("daily", None)
    assert trigger is not None
    after_slot = datetime(2026, 11, 4, 16, 0, tzinfo=tz)
    before_slot = datetime(2026, 11, 5, 1, 0, tzinfo=tz)
    assert trigger.get_next_fire_time(None, after_slot) == datetime(2026, 11, 5, 2, 0, tzinfo=tz)
    assert trigger.get_next_fire_time(None, before_slot) == datetime(2026, 11, 5, 2, 0, tzinfo=tz)


@pytest.mark.asyncio
async def test_restart_after_today_done_keeps_next_cron_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated sync after today's run has finished (simulating a process restart):

    - catch_up does not fire again (there is a same-day run)
    - cron next_run = the CronTrigger's next slot (the next day goes through cron on time, and does not rely on LocalEval/catch_up)
    """
    from datetime import datetime

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    fire = AsyncMock()
    monkeypatch.setattr(scheduler, "fire_task", fire)
    monkeypatch.setattr(
        wts_mod,
        "list_schedulable_tasks",
        AsyncMock(return_value=[_task(id="task-daily", trigger="daily")]),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="oss", name="daily-ds")),
    )
    tz = wts_mod._schedule_tz()
    # From catch_up's point of view: today 16:00, the slot has passed and a run already exists
    frozen_now = datetime(2026, 11, 4, 16, 0, tzinfo=tz)
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(now=lambda _tz=None: frozen_now),
    )
    scheduler._benchmark_service.list_runs = AsyncMock(
        return_value=[
            SimpleNamespace(
                run_id="search_eval-20261104",
                config_json={"data_date": "20261104"},
            ),
        ],
    )

    await scheduler.sync_jobs()
    # Clear the signature to simulate syncing again after a process restart
    scheduler._signatures.clear()
    await scheduler.sync_jobs()

    fire.assert_not_called()
    next_run = hub.next_run_time("webui-eval-task:task-daily")
    assert next_run is not None
    # APScheduler computes next_run from the wall clock; align it with the real CronTrigger's next slot
    trigger = build_task_trigger("daily", None)
    assert trigger is not None
    expected = trigger.get_next_fire_time(None, datetime.now(tz))
    assert next_run == expected
    await scheduler.stop()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_before_slot_catchup_idle_cron_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past midnight but before today's slot: catch_up is a no-op; cron is mounted and has a next_run (the on-time trigger surface)."""
    from datetime import datetime

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    fire = AsyncMock()
    monkeypatch.setattr(scheduler, "fire_task", fire)
    monkeypatch.setattr(
        wts_mod,
        "list_schedulable_tasks",
        AsyncMock(return_value=[_task(id="task-daily", trigger="daily")]),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="oss", name="daily-ds")),
    )
    tz = wts_mod._schedule_tz()
    frozen_now = datetime(2026, 11, 5, 1, 0, tzinfo=tz)
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(now=lambda _tz=None: frozen_now),
    )
    scheduler._benchmark_service.list_runs = AsyncMock(return_value=[])

    await scheduler.sync_jobs()

    fire.assert_not_called()
    next_run = hub.next_run_time("webui-eval-task:task-daily")
    assert next_run is not None
    trigger = build_task_trigger("daily", None)
    assert trigger is not None
    assert next_run == trigger.get_next_fire_time(None, datetime.now(tz))
    await scheduler.stop()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_fire_task_skips_stopped_and_realtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    accept = AsyncMock()
    monkeypatch.setattr(scheduler, "_accept_run", accept)

    monkeypatch.setattr(wts_mod, "find_task_document", AsyncMock(return_value=None))
    await scheduler.fire_task("task-1")
    accept.assert_not_called()

    stopped = _task(status="stopped")
    monkeypatch.setattr(wts_mod, "find_task_document", AsyncMock(return_value=stopped))
    await scheduler.fire_task("task-1")
    accept.assert_not_called()

    realtime = _task()
    monkeypatch.setattr(wts_mod, "find_task_document", AsyncMock(return_value=realtime))
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="realtime", name="rt-ds")),
    )
    await scheduler.fire_task("task-1")
    accept.assert_not_called()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_fire_task_triggers_run_and_records_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    from benchmark.runs.manual_run_id import scheduled_run_id

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    monkeypatch.setattr(wts_mod, "find_task_document", AsyncMock(return_value=_task()))
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="oss", name="daily-ds")),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_agent_document_by_id",
        AsyncMock(return_value=SimpleNamespace(name="search_eval")),
    )
    monkeypatch.setattr(
        scheduler,
        "_batch_execution_aliases",
        AsyncMock(
            return_value={
                "judge_class": "search_eval_judge",
                "attribution_class": "no_attribution",
            },
        ),
    )
    fixed_now = datetime(2026, 10, 30, 2, 0, tzinfo=wts_mod._schedule_tz())
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(now=lambda _tz=None: fixed_now),
    )
    expected_run_id = scheduled_run_id("search_eval", "20261030")
    accept = AsyncMock(return_value=True)
    monkeypatch.setattr(scheduler, "_accept_run", accept)
    record = AsyncMock()
    monkeypatch.setattr(scheduler, "_record_run", record)

    await scheduler.fire_task("task-1")

    accept.assert_awaited_once()
    run_id = accept.call_args.args[0]
    assert run_id == expected_run_id
    assert run_id == "search_eval-20261030"
    assert "-t-" not in run_id
    assert "-m" not in run_id
    payload = accept.call_args.kwargs["payload"]
    assert payload.eval_task_id == "task-1"
    assert payload.dataset_id == "ds-1"
    assert payload.agent_registration_id == "ba-1"
    # A batch evaluation no longer limits the sample count: it carries no limit, and the server runs the full set
    assert payload.limit is None
    # The execution class name is carried with the request (the search_eval rubric requires a local sync judge)
    assert payload.judge_class == "search_eval_judge"
    assert payload.attribution_class == "no_attribution"
    record.assert_awaited_once_with("task-1", expected_run_id)
    await hub.shutdown()


@pytest.mark.asyncio
async def test_fire_task_run_id_uses_dataset_partition_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dataset oss_partition_date drives the date segment of run_id."""
    from datetime import datetime

    from benchmark.runs.manual_run_id import scheduled_run_id

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    monkeypatch.setattr(wts_mod, "find_task_document", AsyncMock(return_value=_task()))
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(
            return_value=SimpleNamespace(
                source="oss",
                name="daily-ds",
                oss_partition_date="20261110",
                oss_prefix="ieval-service/benchmark/benchmark_result",
            ),
        ),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_agent_document_by_id",
        AsyncMock(return_value=SimpleNamespace(name="ieval_external")),
    )
    monkeypatch.setattr(scheduler, "_batch_execution_aliases", AsyncMock(return_value={}))
    fixed_now = datetime(2026, 11, 11, 2, 0, tzinfo=wts_mod._schedule_tz())
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(now=lambda _tz=None: fixed_now),
    )
    expected = scheduled_run_id("ieval_external", "20261110")
    accept = AsyncMock(return_value=True)
    monkeypatch.setattr(scheduler, "_accept_run", accept)
    monkeypatch.setattr(scheduler, "_record_run", AsyncMock())

    await scheduler.fire_task("task-1")

    assert accept.call_args.args[0] == expected
    assert accept.call_args.kwargs["date"] == "20261110"
    await hub.shutdown()


def test_resolve_fire_storage_expands_yyyymmdd_minus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``${yyyymmdd-1}`` is expanded once against the schedule day, and the partition date drives data_date."""
    from datetime import datetime

    monkeypatch.setattr(
        wts_mod,
        "get_default_download_prefix",
        lambda: "oss://bucket/default-prefix",
    )
    now = datetime(2026, 11, 12, 2, 0, tzinfo=wts_mod._schedule_tz())
    dataset = SimpleNamespace(
        source="oss",
        name="ieval-ds",
        oss_partition_date=None,
        oss_prefix=("ieval-service/benchmark/benchmark_result/${yyyymmdd-1}"),
    )

    storage = wts_mod._resolve_fire_storage(dataset, now=now)

    assert storage is not None
    assert storage.data_date == "20261111"
    assert storage.oss_prefix == ("ieval-service/benchmark/benchmark_result/20261111")
    assert storage.oss_object_key == (
        "ieval-service/benchmark/benchmark_result/20261111/benchmark_final_results.csv"
    )


@pytest.mark.asyncio
async def test_fire_task_run_id_uses_expanded_oss_date_macro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When oss_prefix contains ``${yyyymmdd-1}``, run_id / data_date use the expanded partition date."""
    from datetime import datetime

    from benchmark.runs.manual_run_id import scheduled_run_id

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    monkeypatch.setattr(
        wts_mod,
        "get_default_download_prefix",
        lambda: "oss://bucket/default-prefix",
    )
    monkeypatch.setattr(wts_mod, "find_task_document", AsyncMock(return_value=_task()))
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(
            return_value=SimpleNamespace(
                source="oss",
                name="ieval-ds",
                oss_partition_date=None,
                oss_prefix=("ieval-service/benchmark/benchmark_result/${yyyymmdd-1}"),
            ),
        ),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_agent_document_by_id",
        AsyncMock(return_value=SimpleNamespace(name="ieval_external")),
    )
    monkeypatch.setattr(scheduler, "_batch_execution_aliases", AsyncMock(return_value={}))
    fixed_now = datetime(2026, 11, 12, 2, 0, tzinfo=wts_mod._schedule_tz())
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(now=lambda _tz=None: fixed_now),
    )
    expected = scheduled_run_id("ieval_external", "20261111")
    accept = AsyncMock(return_value=True)
    monkeypatch.setattr(scheduler, "_accept_run", accept)
    monkeypatch.setattr(scheduler, "_record_run", AsyncMock())

    await scheduler.fire_task("task-1")

    assert accept.call_args.args[0] == expected
    assert accept.call_args.kwargs["date"] == "20261111"
    payload = accept.call_args.kwargs["payload"]
    assert payload.oss_prefix == ("ieval-service/benchmark/benchmark_result/20261111")
    assert payload.oss_object_key == (
        "ieval-service/benchmark/benchmark_result/20261111/benchmark_final_results.csv"
    )
    await hub.shutdown()


@pytest.mark.asyncio
async def test_catch_up_skips_when_macro_partition_run_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a run already exists for the partition date ``${yyyymmdd-1}`` expects, catchup does not misfire against the schedule day."""
    from datetime import datetime

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    fire = AsyncMock()
    monkeypatch.setattr(scheduler, "fire_task", fire)
    monkeypatch.setattr(
        wts_mod,
        "get_default_download_prefix",
        lambda: "oss://bucket/default-prefix",
    )
    monkeypatch.setattr(
        wts_mod,
        "list_schedulable_tasks",
        AsyncMock(return_value=[_task(id="task-macro")]),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(
            return_value=SimpleNamespace(
                source="oss",
                name="ieval-ds",
                oss_partition_date=None,
                oss_prefix=("ieval-service/benchmark/benchmark_result/${yyyymmdd-1}"),
            ),
        ),
    )
    scheduler._benchmark_service.list_runs = AsyncMock(
        return_value=[
            SimpleNamespace(
                run_id="ieval_external-20261111",
                config_json={"data_date": "20261111"},
            ),
        ],
    )

    class _PastTrigger:
        def get_next_fire_time(self, _prev: object, _now: object) -> datetime:
            return datetime(2026, 11, 12, 2, 0, tzinfo=wts_mod._schedule_tz())

    monkeypatch.setattr(wts_mod, "build_task_trigger", lambda *_a, **_k: _PastTrigger())
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(
            now=lambda _tz=None: datetime(2026, 11, 12, 16, 0, tzinfo=wts_mod._schedule_tz()),
        ),
    )

    await scheduler.catch_up_missed_tasks()
    fire.assert_not_called()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_fire_task_skips_record_when_accept_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed 409 / JSONResponse → do not write the task pointer back."""
    from datetime import datetime

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    monkeypatch.setattr(wts_mod, "find_task_document", AsyncMock(return_value=_task()))
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="oss", name="daily-ds")),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_agent_document_by_id",
        AsyncMock(return_value=SimpleNamespace(name="search_eval")),
    )
    monkeypatch.setattr(scheduler, "_batch_execution_aliases", AsyncMock(return_value={}))
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(
            now=lambda _tz=None: datetime(2026, 10, 30, 2, 0, tzinfo=wts_mod._schedule_tz()),
        ),
    )
    monkeypatch.setattr(scheduler, "_accept_run", AsyncMock(return_value=False))
    record = AsyncMock()
    monkeypatch.setattr(scheduler, "_record_run", record)

    await scheduler.fire_task("task-1")
    record.assert_not_called()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_same_agent_tasks_share_remote_config_style_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two tasks of the same agent share a RemoteConfig-shaped run_id on the same day (reuse_existing)."""
    from datetime import datetime

    from benchmark.runs.manual_run_id import scheduled_run_id

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await hub.start()
    fixed_now = datetime(2026, 10, 30, 2, 0, tzinfo=wts_mod._schedule_tz())
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(now=lambda _tz=None: fixed_now),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="oss", name="ds")),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_agent_document_by_id",
        AsyncMock(return_value=SimpleNamespace(name="ieval_external")),
    )
    monkeypatch.setattr(scheduler, "_batch_execution_aliases", AsyncMock(return_value={}))
    seen: list[str] = []

    async def _accept(run_id: str, **_kwargs: object) -> bool:
        seen.append(run_id)
        return True

    monkeypatch.setattr(scheduler, "_accept_run", _accept)
    monkeypatch.setattr(scheduler, "_record_run", AsyncMock())

    monkeypatch.setattr(
        wts_mod,
        "find_task_document",
        AsyncMock(return_value=_task(id="task-a", dataset_id="ds-a")),
    )
    await scheduler.fire_task("task-a")
    monkeypatch.setattr(
        wts_mod,
        "find_task_document",
        AsyncMock(return_value=_task(id="task-b", dataset_id="ds-b")),
    )
    await scheduler.fire_task("task-b")

    expected = scheduled_run_id("ieval_external", "20261030")
    assert seen == [expected, expected]
    assert expected == "ieval_external-20261030"
    await hub.shutdown()


@pytest.mark.asyncio
async def test_catch_up_fires_due_batch_without_today_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup backfill: fire_task when today's slot has passed, it is not realtime, and there is no run yet today."""
    from datetime import datetime

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    fire = AsyncMock()
    monkeypatch.setattr(scheduler, "fire_task", fire)
    monkeypatch.setattr(
        wts_mod,
        "list_schedulable_tasks",
        AsyncMock(return_value=[_task(id="task-due")]),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="oss", name="daily-ds")),
    )
    scheduler._benchmark_service.list_runs = AsyncMock(return_value=[])

    class _PastTrigger:
        def get_next_fire_time(self, _prev: object, _now: object) -> datetime:
            return datetime(2026, 10, 30, 2, 0, tzinfo=wts_mod._schedule_tz())

    monkeypatch.setattr(wts_mod, "build_task_trigger", lambda *_a, **_k: _PastTrigger())
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(
            now=lambda _tz=None: datetime(2026, 10, 30, 16, 0, tzinfo=wts_mod._schedule_tz()),
        ),
    )

    await scheduler.catch_up_missed_tasks()
    fire.assert_awaited_once_with("task-due")
    await hub.shutdown()


@pytest.mark.asyncio
async def test_catch_up_skips_before_today_slot_even_if_yesterday_due(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart before today's 11:00: even if yesterday should have fired at 11:00, do not start today's run early."""
    from datetime import datetime

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    fire = AsyncMock()
    monkeypatch.setattr(scheduler, "fire_task", fire)
    monkeypatch.setattr(
        wts_mod,
        "list_schedulable_tasks",
        AsyncMock(
            return_value=[
                _task(id="task-cron-11", trigger="cron", cron_expression="0 11 * * *"),
            ]
        ),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="oss", name="daily-ds")),
    )
    scheduler._benchmark_service.list_runs = AsyncMock(return_value=[])
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(
            now=lambda _tz=None: datetime(2026, 10, 31, 8, 0, tzinfo=wts_mod._schedule_tz()),
        ),
    )

    await scheduler.catch_up_missed_tasks()
    fire.assert_not_called()
    await hub.shutdown()


def test_today_fire_already_due_false_before_slot() -> None:
    """A real CronTrigger: today 08:00, cron=11:00 → today's slot has not arrived."""
    from datetime import datetime

    trigger = build_task_trigger("cron", "0 11 * * *")
    assert trigger is not None
    tz = wts_mod._schedule_tz()
    now = datetime(2026, 10, 31, 8, 0, tzinfo=tz)
    assert wts_mod._today_fire_already_due(trigger, now, tz) is False


def test_today_fire_already_due_true_after_slot() -> None:
    """A real CronTrigger: today 12:00, cron=11:00 → today's slot has passed."""
    from datetime import datetime

    trigger = build_task_trigger("cron", "0 11 * * *")
    assert trigger is not None
    tz = wts_mod._schedule_tz()
    now = datetime(2026, 10, 31, 12, 0, tzinfo=tz)
    assert wts_mod._today_fire_already_due(trigger, now, tz) is True


@pytest.mark.asyncio
async def test_catch_up_skips_when_today_run_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a run already exists today (including failed), the startup backfill does not retry."""
    from datetime import datetime

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    fire = AsyncMock()
    monkeypatch.setattr(scheduler, "fire_task", fire)
    monkeypatch.setattr(
        wts_mod,
        "list_schedulable_tasks",
        AsyncMock(return_value=[_task(id="task-done")]),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="oss", name="daily-ds")),
    )
    scheduler._benchmark_service.list_runs = AsyncMock(
        return_value=[
            SimpleNamespace(
                run_id="search_eval-20261030",
                config_json={"data_date": "20261030"},
            ),
        ],
    )

    class _PastTrigger:
        def get_next_fire_time(self, _prev: object, _now: object) -> datetime:
            return datetime(2026, 10, 30, 2, 0, tzinfo=wts_mod._schedule_tz())

    monkeypatch.setattr(wts_mod, "build_task_trigger", lambda *_a, **_k: _PastTrigger())
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(
            now=lambda _tz=None: datetime(2026, 10, 30, 16, 0, tzinfo=wts_mod._schedule_tz()),
        ),
    )

    await scheduler.catch_up_missed_tasks()
    fire.assert_not_called()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_start_registers_startup_catchup_job() -> None:
    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    await scheduler.start()
    assert hub.has_job("webui-eval-task-startup-catchup")
    await scheduler.stop()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_catch_up_skips_realtime_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A realtime dataset does not participate in the startup backfill."""
    from datetime import datetime

    hub = SchedulerHub()
    scheduler = _scheduler(hub)
    fire = AsyncMock()
    monkeypatch.setattr(scheduler, "fire_task", fire)
    monkeypatch.setattr(
        wts_mod,
        "list_schedulable_tasks",
        AsyncMock(return_value=[_task(id="task-rt")]),
    )
    monkeypatch.setattr(
        wts_mod,
        "find_dataset_document",
        AsyncMock(return_value=SimpleNamespace(source="realtime", name="rt-ds")),
    )
    scheduler._benchmark_service.list_runs = AsyncMock(return_value=[])

    class _PastTrigger:
        def get_next_fire_time(self, _prev: object, _now: object) -> datetime:
            return datetime(2026, 10, 30, 2, 0, tzinfo=wts_mod._schedule_tz())

    monkeypatch.setattr(wts_mod, "build_task_trigger", lambda *_a, **_k: _PastTrigger())
    monkeypatch.setattr(
        wts_mod,
        "datetime",
        SimpleNamespace(
            now=lambda _tz=None: datetime(2026, 10, 30, 16, 0, tzinfo=wts_mod._schedule_tz()),
        ),
    )

    await scheduler.catch_up_missed_tasks()
    fire.assert_not_called()
    await hub.shutdown()
