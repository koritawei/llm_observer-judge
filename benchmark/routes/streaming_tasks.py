"""Benchmark streaming task management API (§6.5 auxiliary control plane).

The declarative config (RemoteConfig ``agent-eval.benchmark.streaming``) is the only
source of truth. This API only does **in-memory acceleration** and queries; it does not
persist. It does **not** offer create / delete / stop endpoints — add / delete / stop
must go through the RemoteConfig declaration, so there is not a second source of truth.
Writes (pause / resume) emit an audit log, and the response tells the caller to sync
RemoteConfig (otherwise a restart or cache refresh snaps back, §6.4).

This API is the future write path for the agent-park "eval config" tab (§10 P6: the page
reuses it directly and needs no new backend endpoint).
"""

from typing import Annotated, Any

from benchmark.libs.agenthost.observability.logger import get_logger
from fastapi import APIRouter, Depends, HTTPException, Request

from benchmark.domain.streaming_config import StreamingConfig, fetch_streaming_config
from benchmark.streaming.registry import RunningTask
from benchmark.streaming.runner import StreamingBenchmarkRunner

logger = get_logger("agent_eval.api.streaming_tasks")

router = APIRouter(prefix="/v1/benchmarks/streaming", tags=["benchmark-streaming"])


def get_streaming_runner(request: Request) -> StreamingBenchmarkRunner:
    """Resolve the streaming runner assembled by lifespan (not enabled → 503)."""
    runner = getattr(request.app.state, "streaming_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="Benchmark streaming is not enabled (BENCHMARK_STREAMING_ENABLED=false)",
        )
    return runner


StreamingRunnerDep = Annotated[StreamingBenchmarkRunner, Depends(get_streaming_runner)]

#: RemoteConfig sync hint after an in-memory write (§6.4 drift-visibility contract).
_SYNC_HINT = (
    "内存级变更仅对当前节点生效（多节点部署时其他节点消费实例不受影响）；"
    "请同步 RemoteConfig agent-eval.benchmark.streaming 的 tasks[].state，"
    "否则 reconciler 对账后按声明态回弹"
)


def _declared_view() -> dict[str, Any]:
    """RemoteConfig declared-state view (marked unavailable; not disguised as an empty set)."""
    cfg: StreamingConfig | None = fetch_streaming_config()
    if cfg is None:
        return {"available": False, "tasks": {}}
    return {
        "available": True,
        "enabled": cfg.enabled,
        "allow_empty_tasks": cfg.allow_empty_tasks,
        "reconcile_interval_s": cfg.reconcile_interval_s,
        "tasks": {
            task.name: {
                "state": task.state,
                "topics": task.topics,
                "group_id": task.group_id,
                "judge_class": task.judge_class,
                "attribution_class": task.attribution_class,
                "rubric_class": task.rubric_class,
                "sampling": task.sampling,
                "daily_report_enabled": task.daily_report.enabled,
            }
            for task in cfg.tasks
        },
    }


def _runtime_view(task: RunningTask) -> dict[str, Any]:
    """In-memory runtime view of one task (lag needs a broker admin query; field reserved)."""
    return {
        "name": task.spec.name,
        "runtime_state": task.runtime_state,
        "declared_topics": task.spec.topics,
        "subscribed_topics": task.subscribed_topics,
        "group_id": task.spec.group_id,
        "instances": task.instance_count,
        "is_running": task.is_running,
        "shadow": task.spec.shadow,
        "consumer_lag": None,
    }


@router.get("/tasks")
async def list_streaming_tasks(runner: StreamingRunnerDep) -> dict[str, Any]:
    """Task status: declared RemoteConfig state plus in-memory runtime (§6.5)."""
    reconciler = getattr(runner, "_reconciler", None)
    last_reconcile = getattr(reconciler, "last_reconcile_at", None)
    return {
        "declared": _declared_view(),
        "tasks": [_runtime_view(task) for task in runner.registry.list()],
        "last_reconcile_at": last_reconcile.isoformat() if last_reconcile else None,
    }


async def _apply_mem_action(
    name: str,
    *,
    action: str,
    runner: StreamingBenchmarkRunner,
) -> dict[str, Any]:
    """Shared in-memory pause / resume flow (idempotent, with an audit log).

    State-machine constraint (§6.2): a ``stopped`` task cannot be paused (recovery needs
    an explicit resume). Returning ok with no effect would mislead operators.
    """
    task = runner.registry.get(name)
    if task is None:
        raise HTTPException(status_code=404, detail=f"streaming task not found: {name}")
    if action == "pause":
        if task.runtime_state == "stopped":
            raise HTTPException(
                status_code=409,
                detail=f"streaming task {name} is stopped; pause not allowed (resume first)",
            )
        changed = await runner.pause_task(name) if task.runtime_state == "running" else True
    else:
        changed = await runner.resume_task(name) if task.runtime_state != "running" else True
    if not changed:
        raise HTTPException(status_code=409, detail=f"failed to {action} streaming task: {name}")
    after = runner.registry.get(name)
    return {
        "name": name,
        "action": action,
        "ok": True,
        "runtime_state": after.runtime_state if after is not None else None,
        "hint": _SYNC_HINT,
    }


@router.post("/tasks/{name}/pause")
async def pause_streaming_task(
    name: str,
    request: Request,
    runner: StreamingRunnerDep,
) -> dict[str, Any]:
    """In-memory pause (state=paused takes effect first; lag piles up and catches up on resume)."""
    logger.info(
        "streaming task audit: action=pause task=%s operator=api request_id=%s",
        name,
        request.headers.get("x-request-id", "-"),
    )
    return await _apply_mem_action(name, action="pause", runner=runner)


@router.post("/tasks/{name}/resume")
async def resume_streaming_task(
    name: str,
    request: Request,
    runner: StreamingRunnerDep,
) -> dict[str, Any]:
    """In-memory resume (catch up from the committed offset; sampling at that time decides eval)."""
    logger.info(
        "streaming task audit: action=resume task=%s operator=api request_id=%s",
        name,
        request.headers.get("x-request-id", "-"),
    )
    return await _apply_mem_action(name, action="resume", runner=runner)


__all__ = ["router"]
