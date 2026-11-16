"""console stop cascades to terminate a batch run (API layer).

A batch task executes as a benchmark run plus BullMQ jobs — if stop only updates the task
document, the run keeps going. These cases lock the stop_eval_task cascade drain (delete jobs +
release slots + stop_run→stopped) and the degraded path (skip when a dependency is missing;
a realtime-stream task with no run does not trigger it).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.routes import benchmark_console as console
from benchmark.domain.benchmark_console import ConsoleError
from benchmark.runs import stop_inflight as stop_inflight_mod
from benchmark.runs.run_stop import StopReportReadiness
from benchmark.queue.queue_factory import BENCHMARK_CASES, EXTERNAL_RUNNING


def _ready_snap(
    *,
    ready: bool = True,
    running_items: int = 0,
    attribution_pending: int = 0,
) -> StopReportReadiness:
    return StopReportReadiness(
        ready=ready,
        running_items=running_items,
        attribution_pending=attribution_pending,
    )


def _fake_job(
    run_id: str,
    *,
    slot_key: str | None = "node-1:judge",
    kind: str = "judge",
    state: str = "waiting",
    name: str = "inflight",
):
    job = MagicMock()
    job.id = f"job-{run_id}-{state}-{name}"
    job.name = name
    job._bullmq_state = state
    data: dict[str, Any] = {"run_id": run_id}
    if slot_key:
        data["slot_key"] = slot_key
    if kind:
        data["cluster_kind"] = kind
    job.data = data
    job.remove = AsyncMock()
    return job


async def _jobs_for_types(queue: Any, types: list[str]) -> list[Any]:
    """Filter by BullMQ state, for the stop-run active/parked inventory unit tests."""
    return [
        job
        for job in getattr(queue, "_jobs", [])
        if getattr(job, "_bullmq_state", "waiting") in types
    ]


def _fake_request(qm: Any = None, service: Any = None) -> MagicMock:
    request = MagicMock()
    state = MagicMock()
    state.benchmark_service = service
    state.eval_queue_manager = qm
    request.app.state = state
    return request


def _fake_qm(jobs_by_queue: dict[str, list[Any]]) -> MagicMock:
    qm = MagicMock()

    def _get_queue(name: str) -> MagicMock:
        queue = MagicMock()
        queue._jobs = jobs_by_queue.get(name, [])
        return queue

    qm.get_queue.side_effect = _get_queue
    return qm


@pytest.mark.asyncio
async def test_stop_task_terminates_manual_run() -> None:
    """Batch stop: clear wait; keep active/pending_callback → waiting_inflight."""
    judge_job = _fake_job("bm-1", state="active")
    # A sync placeholder inflight (parked) must be kept and counted as in-flight; only unknown names may be cleared
    sync_placeholder = _fake_job(
        "bm-1",
        state="waiting",
        name="inflight",
        slot_key=None,
        kind="",
    )
    parked_orphan = _fake_job(
        "bm-1",
        state="waiting",
        name="orphan-unknown",
        slot_key=None,
        kind="",
    )
    wait_job = _fake_job("bm-1", slot_key=None, kind="", state="waiting")
    other_job = _fake_job("other-run", state="waiting")  # Jobs belonging to other runs are left alone
    qm = _fake_qm(
        {
            EXTERNAL_RUNNING: [judge_job, sync_placeholder, parked_orphan],
            BENCHMARK_CASES: [wait_job, other_job],
        },
    )
    service = MagicMock()
    service.fail_run = AsyncMock()
    service.get_run = AsyncMock(
        return_value=MagicMock(
            status="running",
            report_meta_json={},
            evaluated_items=3,
            failed_items=0,
        ),
    )
    service.list_runs = AsyncMock(return_value=[])
    request = _fake_request(qm=qm, service=service)

    with (
        patch.object(console, "list_queue_jobs", side_effect=_jobs_for_types),
        patch.object(stop_inflight_mod, "list_queue_jobs", side_effect=_jobs_for_types),
        patch.object(console, "release_slot", new=AsyncMock()) as rel_slot,
        patch.object(console, "release_cluster_slot", new=AsyncMock()) as rel_cluster,
        patch.object(console, "apply_stop_task") as stop_task,
        patch.object(console, "mark_stop_requested", new=AsyncMock()) as mark,
        patch.object(console, "publish_stop_signal", new=AsyncMock()),
        patch.object(console, "align_expectations_for_stop", new=AsyncMock()) as align,
        patch.object(
            console,
            "inspect_stop_report_readiness",
            new=AsyncMock(return_value=_ready_snap()),
        ),
        patch.object(console, "finalize_as_stopped", new=AsyncMock()) as fin,
        patch.object(console, "_force_partial_finalize", new=AsyncMock()) as force,
    ):
        mark.return_value = {
            "ok": True,
            "noop": False,
            "status": "running",
            "completed_cases": 3,
        }
        align.return_value = {
            "stop_target_count": 5,
            "inflight_total": 2,  # active + sync inflight placeholder
            "inflight_judge": 2,
            "inflight_attribution": 0,
            "processed": 3,
        }

        async def _stop(task_id: str) -> dict[str, Any]:
            return {"id": "task-1", "status": "stopped", "manualRunId": "bm-1"}

        stop_task.side_effect = _stop
        result = await console.stop_eval_task("task-1", request)

    # The task document may be stopped; this run stays running while it waits on in-flight work
    assert result["status"] == "stopped"
    judge_job.remove.assert_not_called()  # active is kept
    sync_placeholder.remove.assert_not_called()  # the sync inflight placeholder is kept
    parked_orphan.remove.assert_awaited_once()  # unknown parked jobs may be cleared
    wait_job.remove.assert_awaited_once()
    other_job.remove.assert_not_called()
    rel_slot.assert_not_awaited()  # wait has no slot
    rel_cluster.assert_not_awaited()
    service.fail_run.assert_not_awaited()
    mark.assert_awaited()
    align.assert_awaited()
    # Snapshot processed before inventorying inflight; the sync placeholder counts
    assert align.await_args.kwargs.get("processed_snapshot") == 3
    assert align.await_args.kwargs.get("inflight_judge") == 2
    fin.assert_not_awaited()
    force.assert_not_awaited()
    stop_run = result.get("stopRun") or {}
    assert stop_run.get("reportStatus") == "waiting_inflight"
    assert stop_run.get("status") == "running"


@pytest.mark.asyncio
async def test_terminate_finalizes_when_no_inflight() -> None:
    """Nothing running in flight and items already terminal: clear wait, then immediately emit a staged report + stopped."""
    wait_job = _fake_job("bm-ready", slot_key=None, kind="", state="waiting")
    qm = _fake_qm({BENCHMARK_CASES: [wait_job]})
    service = MagicMock()
    service.get_run = AsyncMock(
        return_value=MagicMock(
            status="stopped",
            report_meta_json={"report_status": "ready"},
            evaluated_items=5,
            failed_items=0,
        ),
    )
    request = _fake_request(qm=qm, service=service)
    with (
        patch.object(console, "list_queue_jobs", side_effect=_jobs_for_types),
        patch.object(stop_inflight_mod, "list_queue_jobs", side_effect=_jobs_for_types),
        patch.object(console, "mark_stop_requested", new=AsyncMock()) as mark,
        patch.object(console, "publish_stop_signal", new=AsyncMock()),
        patch.object(console, "align_expectations_for_stop", new=AsyncMock()) as align,
        patch.object(
            console,
            "inspect_stop_report_readiness",
            new=AsyncMock(return_value=_ready_snap()),
        ),
        patch.object(console, "finalize_as_stopped", new=AsyncMock()) as fin,
        patch.object(console, "_force_partial_finalize", new=AsyncMock()) as force,
    ):
        mark.return_value = {
            "ok": True,
            "noop": False,
            "status": "running",
            "completed_cases": 5,
        }
        align.return_value = {"stop_target_count": 5, "inflight_total": 0, "processed": 5}
        info = await console._terminate_manual_run(request, "bm-ready")

    wait_job.remove.assert_awaited_once()
    force.assert_awaited_once()
    fin.assert_awaited_once()
    assert info["status"] == "stopped"
    assert info["reportStatus"] == "ready"


@pytest.mark.asyncio
async def test_terminate_waits_when_running_items_remain() -> None:
    """The queue has nothing in flight but the DB still has running items: must not emit a report or finalize."""
    qm = _fake_qm({})
    service = MagicMock()
    service.get_run = AsyncMock(
        return_value=MagicMock(
            status="running",
            report_meta_json={},
            evaluated_items=2,
            failed_items=0,
        ),
    )
    request = _fake_request(qm=qm, service=service)
    with (
        patch.object(console, "list_queue_jobs", side_effect=_jobs_for_types),
        patch.object(stop_inflight_mod, "list_queue_jobs", side_effect=_jobs_for_types),
        patch.object(console, "mark_stop_requested", new=AsyncMock()) as mark,
        patch.object(console, "publish_stop_signal", new=AsyncMock()),
        patch.object(console, "align_expectations_for_stop", new=AsyncMock()) as align,
        patch.object(
            console,
            "inspect_stop_report_readiness",
            new=AsyncMock(
                return_value=_ready_snap(ready=False, running_items=4),
            ),
        ),
        patch.object(console, "finalize_as_stopped", new=AsyncMock()) as fin,
        patch.object(console, "_force_partial_finalize", new=AsyncMock()) as force,
    ):
        mark.return_value = {
            "ok": True,
            "noop": False,
            "status": "running",
            "completed_cases": 2,
        }
        align.return_value = {"stop_target_count": 2, "inflight_total": 0, "processed": 2}
        info = await console._terminate_manual_run(request, "bm-items")

    fin.assert_not_awaited()
    force.assert_not_awaited()
    assert info["status"] == "running"
    assert info["reportStatus"] == "waiting_inflight"
    assert info["runningItems"] == 4


@pytest.mark.asyncio
async def test_terminate_waits_when_attribution_pending() -> None:
    """Items are terminal but attribution is incomplete: stop-run must not emit a report."""
    qm = _fake_qm({})
    service = MagicMock()
    service.get_run = AsyncMock(
        return_value=MagicMock(
            status="running",
            report_meta_json={},
            evaluated_items=3,
            failed_items=0,
        ),
    )
    request = _fake_request(qm=qm, service=service)
    with (
        patch.object(console, "list_queue_jobs", side_effect=_jobs_for_types),
        patch.object(stop_inflight_mod, "list_queue_jobs", side_effect=_jobs_for_types),
        patch.object(console, "mark_stop_requested", new=AsyncMock()) as mark,
        patch.object(console, "publish_stop_signal", new=AsyncMock()),
        patch.object(console, "align_expectations_for_stop", new=AsyncMock()) as align,
        patch.object(
            console,
            "inspect_stop_report_readiness",
            new=AsyncMock(
                return_value=_ready_snap(
                    ready=False,
                    running_items=0,
                    attribution_pending=2,
                ),
            ),
        ),
        patch.object(console, "finalize_as_stopped", new=AsyncMock()) as fin,
        patch.object(console, "_force_partial_finalize", new=AsyncMock()) as force,
    ):
        mark.return_value = {
            "ok": True,
            "noop": False,
            "status": "running",
            "completed_cases": 3,
        }
        align.return_value = {"stop_target_count": 3, "inflight_total": 0, "processed": 3}
        info = await console._terminate_manual_run(request, "bm-attr")

    fin.assert_not_awaited()
    force.assert_not_awaited()
    assert info["reportStatus"] == "waiting_inflight"
    assert info["attributionPending"] == 2
    assert info["runningItems"] == 0


@pytest.mark.asyncio
async def test_terminate_defers_when_inflight_count_uncertain() -> None:
    """Inventory list_queue_jobs fails: must not align(0)/finalize; degrade to waiting_inflight."""
    wait_job = _fake_job("bm-uncertain", slot_key=None, kind="", state="waiting")
    qm = _fake_qm({BENCHMARK_CASES: [wait_job]})
    service = MagicMock()
    service.get_run = AsyncMock(
        return_value=MagicMock(
            status="running",
            report_meta_json={},
            evaluated_items=2,
            failed_items=0,
        ),
    )
    request = _fake_request(qm=qm, service=service)

    async def _list_ok_or_fail(queue: Any, types: list[str]) -> list[Any]:
        # The drain path (console.list_queue_jobs) can still succeed; the inventory path must fail
        return await _jobs_for_types(queue, types)

    with (
        patch.object(console, "list_queue_jobs", side_effect=_list_ok_or_fail),
        patch.object(
            stop_inflight_mod,
            "list_queue_jobs",
            new=AsyncMock(side_effect=RuntimeError("redis blip")),
        ),
        patch.object(console, "mark_stop_requested", new=AsyncMock()) as mark,
        patch.object(console, "publish_stop_signal", new=AsyncMock()),
        patch.object(console, "align_expectations_for_stop", new=AsyncMock()) as align,
        patch.object(console, "finalize_as_stopped", new=AsyncMock()) as fin,
        patch.object(console, "_force_partial_finalize", new=AsyncMock()) as force,
    ):
        mark.return_value = {
            "ok": True,
            "noop": False,
            "status": "running",
            "completed_cases": 2,
        }
        info = await console._terminate_manual_run(request, "bm-uncertain")

    wait_job.remove.assert_awaited_once()
    align.assert_not_awaited()
    fin.assert_not_awaited()
    force.assert_not_awaited()
    assert info["status"] == "running"
    assert info["reportStatus"] == "waiting_inflight"
    assert info["inflightJobs"] is None
    assert info["completedCases"] == 2


@pytest.mark.asyncio
async def test_stop_task_without_run_skips_terminate() -> None:
    """Realtime-stream task (no manualRunId): do not take the run-termination chain."""
    request = _fake_request(qm=MagicMock(), service=MagicMock())
    with (
        patch.object(console, "apply_stop_task") as stop_task,
        patch.object(console, "_terminate_manual_run", new=AsyncMock()) as terminate,
        patch.object(
            console,
            "_trigger_realtime_stopped_report",
            new=AsyncMock(return_value=None),
        ),
    ):

        async def _stop(task_id: str) -> dict[str, Any]:
            return {"status": "stopped"}

        stop_task.side_effect = _stop
        result = await console.stop_eval_task("task-rt", request)

    assert result["status"] == "stopped"
    terminate.assert_not_awaited()


@pytest.mark.asyncio
async def test_realtime_stopped_report_requires_exact_registration_match() -> None:
    """Match via find_report_item; an empty registry or a miss must not publish."""
    target = MagicMock()
    scheduler = MagicMock()
    scheduler.find_report_item = AsyncMock(side_effect=[None, target])
    scheduler.publish_stopped_window = AsyncMock(return_value="run-stopped-1")
    request = MagicMock()
    request.app.state.trace_report_scheduler = scheduler

    # An empty registry → None, and lookup/publish must not be called
    assert (
        await console._trigger_realtime_stopped_report(
            request,
            {"id": "t1", "agentRegistrationId": ""},
        )
        is None
    )
    scheduler.find_report_item.assert_not_awaited()
    scheduler.publish_stopped_window.assert_not_awaited()

    # lookup miss → None
    assert (
        await console._trigger_realtime_stopped_report(
            request,
            {"id": "t1b", "agentRegistrationId": "reg-miss"},
        )
        is None
    )
    scheduler.publish_stopped_window.assert_not_awaited()

    # exact match → publish target
    run_id = await console._trigger_realtime_stopped_report(
        request,
        {
            "id": "t2",
            "agentRegistrationId": "reg-a",
            "streamInstanceId": "si-a",
        },
    )
    assert run_id == "run-stopped-1"
    scheduler.find_report_item.assert_awaited_with(
        registration_id="reg-a",
        stream_instance_id="si-a",
    )
    scheduler.publish_stopped_window.assert_awaited_once_with(target)


@pytest.mark.asyncio
async def test_stop_run_scheduled_without_manual_run_is_noop() -> None:
    """A scheduled task with no manualRunId and no owned active run does not call apply_stop_task; the document stays as it is."""
    request = _fake_request(service=MagicMock())
    request.app.state.benchmark_service.list_runs = AsyncMock(return_value=[])
    request.app.state.benchmark_service.get_run = AsyncMock(return_value=None)
    with (
        patch.object(console, "get_task", new=AsyncMock()) as get_task,
        patch.object(console, "apply_stop_task", new=AsyncMock()) as stop_task,
        patch.object(console, "_terminate_manual_run", new=AsyncMock()) as terminate,
    ):
        get_task.return_value = {
            "id": "task-daily",
            "status": "active",
            "evalMode": "dataset-replay",
            "trigger": "daily",
        }
        result = await console.stop_eval_task_run("task-daily", request)

    assert result["reportStatus"] == "noop_no_active_run"
    assert result["status"] == "active"
    stop_task.assert_not_awaited()
    terminate.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_run_resolves_active_run_by_task_id() -> None:
    """When the document has no manualRunId, find the active run by eval_task_id and terminate it."""
    active = MagicMock()
    active.status = "running"
    active.run_id = "bm-scheduled-today"
    active.id = "pk-scheduled-today"
    service = MagicMock()
    service.get_run = AsyncMock(return_value=None)
    service.list_runs = AsyncMock(return_value=[active])
    request = _fake_request(service=service)
    with (
        patch.object(console, "get_task", new=AsyncMock()) as get_task,
        patch.object(console, "apply_stop_task", new=AsyncMock()) as stop_task,
        patch.object(
            console,
            "_terminate_manual_run",
            new=AsyncMock(return_value={"status": "stopped", "reportStatus": "ready"}),
        ) as terminate,
    ):
        get_task.return_value = {
            "id": "task-daily",
            "status": "active",
            "evalMode": "dataset-replay",
            "trigger": "daily",
            "manualRunPk": "pk-stale-old",
        }
        result = await console.stop_eval_task_run("task-daily", request)

    stop_task.assert_not_awaited()
    terminate.assert_awaited_once_with(request, "bm-scheduled-today")
    assert result["runId"] == "bm-scheduled-today"
    # Fallback resolution must return the live run's pk and must not keep a stale manualRunPk from the document.
    assert result["runPk"] == "pk-scheduled-today"
    assert result["status"] == "stopped"


@pytest.mark.asyncio
async def test_stop_run_keeps_doc_pointer_when_get_run_raises() -> None:
    """When get_run raises, keep manualRunId and do not turn the whole stop-run into a 500."""
    service = MagicMock()
    service.get_run = AsyncMock(side_effect=RuntimeError("db down"))
    service.list_runs = AsyncMock(return_value=[])
    request = _fake_request(service=service)
    with (
        patch.object(console, "get_task", new=AsyncMock()) as get_task,
        patch.object(
            console,
            "_terminate_manual_run",
            new=AsyncMock(return_value={"status": "stopped", "reportStatus": "ready"}),
        ) as terminate,
    ):
        get_task.return_value = {
            "id": "task-1",
            "status": "active",
            "evalMode": "dataset-replay",
            "manualRunId": "bm-doc",
            "manualRunPk": "pk-doc",
        }
        result = await console.stop_eval_task_run("task-1", request)

    terminate.assert_awaited_once_with(request, "bm-doc")
    assert result["runId"] == "bm-doc"
    assert result["runPk"] == "pk-doc"
    assert result["status"] == "stopped"


@pytest.mark.asyncio
async def test_stop_run_clears_stale_run_pk_on_fallback() -> None:
    """When manualRunId is already terminal and fallback hits a new run, runPk and runId both refer to that new run."""
    stale = MagicMock()
    stale.status = "completed"
    stale.id = "pk-stale"
    live = MagicMock()
    live.status = "running"
    live.run_id = "bm-live"
    live.id = "pk-live"
    service = MagicMock()
    service.get_run = AsyncMock(return_value=stale)
    service.list_runs = AsyncMock(return_value=[live])
    request = _fake_request(service=service)
    with (
        patch.object(console, "get_task", new=AsyncMock()) as get_task,
        patch.object(
            console,
            "_terminate_manual_run",
            new=AsyncMock(return_value={"status": "stopped", "reportStatus": "ready"}),
        ) as terminate,
    ):
        get_task.return_value = {
            "id": "task-1",
            "status": "active",
            "evalMode": "dataset-replay",
            "manualRunId": "bm-stale",
            "manualRunPk": "pk-stale",
        }
        result = await console.stop_eval_task_run("task-1", request)

    terminate.assert_awaited_once_with(request, "bm-live")
    assert result["runId"] == "bm-live"
    assert result["runPk"] == "pk-live"


@pytest.mark.asyncio
async def test_stop_task_falls_back_to_task_owned_active_run() -> None:
    """When manualRunId is already terminal, stopping the task still cascades to runs that are still running under the same task name."""
    active = MagicMock()
    active.status = "running"
    active.run_id = "bm-live"
    service = MagicMock()
    service.get_run = AsyncMock(return_value=MagicMock(status="completed"))
    service.list_runs = AsyncMock(return_value=[active])
    request = _fake_request(service=service)
    with (
        patch.object(console, "apply_stop_task", new=AsyncMock()) as stop_task,
        patch.object(
            console,
            "_terminate_manual_run",
            new=AsyncMock(return_value={"status": "stopped"}),
        ) as terminate,
        patch.object(console, "_trigger_realtime_stopped_report", new=AsyncMock()),
    ):
        stop_task.return_value = {
            "id": "task-1",
            "status": "stopped",
            "manualRunId": "bm-old",
        }
        result = await console.stop_eval_task("task-1", request)

    terminate.assert_awaited_once_with(request, "bm-live")
    assert result["stopRun"]["status"] == "stopped"


@pytest.mark.asyncio
async def test_stop_run_realtime_triggers_stopped_window() -> None:
    """Realtime-stream stop-run: close the task window and trigger the *-stopped-* report lookup."""
    request = _fake_request()
    with (
        patch.object(console, "get_task", new=AsyncMock()) as get_task,
        patch.object(console, "apply_stop_task", new=AsyncMock()) as stop_task,
        patch.object(
            console,
            "_trigger_realtime_stopped_report",
            new=AsyncMock(return_value="rt-stopped-1"),
        ) as trigger,
    ):
        get_task.return_value = {
            "id": "task-rt",
            "status": "active",
            "evalMode": "realtime-span",
            "agentRegistrationId": "reg-a",
        }
        stop_task.return_value = {"status": "stopped", "id": "task-rt"}
        result = await console.stop_eval_task_run("task-rt", request)

    stop_task.assert_awaited_once_with("task-rt")
    trigger.assert_awaited()
    assert result["stopRun"]["runId"] == "rt-stopped-1"
    assert result["reportStatus"] == "generating"


@pytest.mark.asyncio
async def test_terminate_manual_run_status_when_run_missing() -> None:
    """When get_run is empty after finalize, status must be stopped, not the string None."""
    service = MagicMock()
    service.get_run = AsyncMock(return_value=None)
    request = _fake_request(qm=None, service=service)
    with (
        patch.object(console, "mark_stop_requested", new=AsyncMock()) as mark,
        patch.object(console, "publish_stop_signal", new=AsyncMock()),
        patch.object(
            console,
            "align_expectations_for_stop",
            new=AsyncMock(return_value={"stop_target_count": 0, "inflight_total": 0}),
        ),
        patch.object(
            console,
            "inspect_stop_report_readiness",
            new=AsyncMock(return_value=_ready_snap()),
        ),
        patch.object(console, "finalize_as_stopped", new=AsyncMock(return_value=False)),
        patch.object(console, "_force_partial_finalize", new=AsyncMock()) as force,
    ):
        mark.return_value = {
            "ok": True,
            "noop": False,
            "status": "running",
            "completed_cases": 0,
        }
        info = await console._terminate_manual_run(request, "bm-missing")

    assert info["status"] == "stopped"
    assert info["status"] != "None"
    # No completed case: forbid a forced finalize (avoid a fake 0-score report from pre-written running rows)
    force.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminate_skips_partial_report_when_no_completed_cases() -> None:
    """stop-run with completed_cases=0: persist stopped only, and do not emit a staged report."""
    service = MagicMock()
    service.get_run = AsyncMock(
        return_value=MagicMock(
            status="stopped",
            report_meta_json={"report_status": "skipped_no_cases"},
        ),
    )
    request = _fake_request(qm=None, service=service)
    with (
        patch.object(console, "mark_stop_requested", new=AsyncMock()) as mark,
        patch.object(console, "publish_stop_signal", new=AsyncMock()),
        patch.object(
            console,
            "align_expectations_for_stop",
            new=AsyncMock(return_value={"stop_target_count": 0, "inflight_total": 0}),
        ),
        patch.object(
            console,
            "inspect_stop_report_readiness",
            new=AsyncMock(return_value=_ready_snap()),
        ),
        patch.object(console, "finalize_as_stopped", new=AsyncMock(return_value=True)),
        patch.object(console, "_force_partial_finalize", new=AsyncMock()) as force,
    ):
        mark.return_value = {
            "ok": True,
            "noop": False,
            "status": "running",
            "completed_cases": 0,
        }
        info = await console._terminate_manual_run(request, "bm-early-stop")

    force.assert_not_awaited()
    assert info["reportStatus"] == "skipped_no_cases"
    assert info["completedCases"] == 0


@pytest.mark.asyncio
async def test_stop_task_degrades_when_dependencies_unavailable() -> None:
    """Queue degraded (qm=None): skip drain, but still write the stop latch and the stopped terminal state."""
    service = MagicMock()
    # get_run: resolve ownership + the processed snapshot + read status after finalize
    service.get_run = AsyncMock(
        side_effect=[
            MagicMock(status="running", report_meta_json={}, evaluated_items=0, failed_items=0),
            MagicMock(status="running", report_meta_json={}, evaluated_items=0, failed_items=0),
            MagicMock(status="stopped", report_meta_json={}, evaluated_items=0, failed_items=0),
        ],
    )
    service.list_runs = AsyncMock(return_value=[])
    request = _fake_request(qm=None, service=service)
    with (
        patch.object(console, "apply_stop_task") as stop_task,
        patch.object(console, "list_queue_jobs") as list_jobs,
        patch.object(console, "mark_stop_requested", new=AsyncMock()) as mark,
        patch.object(console, "publish_stop_signal", new=AsyncMock()),
        patch.object(
            console,
            "align_expectations_for_stop",
            new=AsyncMock(
                return_value={"stop_target_count": 0, "inflight_total": 0, "processed": 0},
            ),
        ),
        patch.object(
            console,
            "inspect_stop_report_readiness",
            new=AsyncMock(return_value=_ready_snap()),
        ),
        patch.object(console, "finalize_as_stopped", new=AsyncMock()),
        patch.object(console, "_force_partial_finalize", new=AsyncMock()) as force,
    ):
        mark.return_value = {
            "ok": True,
            "noop": False,
            "status": "running",
            "completed_cases": 0,
        }

        async def _stop(task_id: str) -> dict[str, Any]:
            return {"id": "task-1", "status": "stopped", "manualRunId": "bm-1"}

        stop_task.side_effect = _stop
        result = await console.stop_eval_task("task-1", request)

    assert result["status"] == "stopped"
    list_jobs.assert_not_awaited()  # If qm is missing, skip immediately and do not scan the queue
    force.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_task_console_error_maps_to_http_409() -> None:
    """Stopping a task that is not running → ConsoleError → HTTPException 409."""
    request = _fake_request()
    with (
        patch.object(console, "apply_stop_task") as stop_task,
        pytest.raises(Exception) as caught,
    ):
        stop_task.side_effect = ConsoleError(409, "只有运行中的任务可以停止")
        await console.stop_eval_task("task-done", request)

    assert caught.value.status_code == 409


@pytest.mark.asyncio
async def test_remove_task_does_not_touch_runs() -> None:
    """Deleting a task only soft-deletes the task document: it does not terminate the run, clear jobs, or mark failed.

    Evaluation results under that task name are kept and left untouched — even if the task carries a manualRunId,
    delete must not take the _terminate_manual_run chain (strictly separate from stop).
    """
    service = MagicMock()
    service.fail_run = AsyncMock()
    remove = AsyncMock(return_value={"deleted": True})
    with (
        patch.object(console, "apply_remove_task", new=remove),
        patch.object(console, "_terminate_manual_run", new=AsyncMock()) as terminate,
    ):
        result = await console.remove_task("task-1")

    assert result["deleted"] is True
    remove.assert_awaited_once_with("task-1")
    terminate.assert_not_awaited()
    service.fail_run.assert_not_called()
