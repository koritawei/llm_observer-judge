"""停止本次运行：门闩、幂等终态与阶段性报告编排（设计 §6.8）。

批量：保留原 ``run_id``，``config_json.stop_requested_at`` 为真相源；
finalize 收尾时 ``complete_run`` 见门闩则写 ``stopped`` + partial meta。
实时流窗口报告命名含 ``stopped`` 后缀，由调用方另行触发。

本模块为「stop latch core」原语：门闩写入、期望对齐（含在途
inflight）、partial meta、``stopped`` 终态；由控制台 ``stop-run``
与 finalize CAS 共同消费。
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models.benchmark_run import BenchmarkRun
from benchmark.records.models.benchmark_run_item import BenchmarkRunItem

logger = logging.getLogger("agent_eval.benchmark.run_stop")

STOP_REASON = "console_task_stopped"
STOP_KIND_MANUAL = "manual"
STOP_SIGNAL_CHANNEL = "agent-eval:run-stop"
#: healer 在 mark→drain→align 窗口内跳过「无在途则压 target」的宽限期。
#: 过短会与 console stop 竞态提前 CAS；过长仅延迟 stuck 兜底（可接受）。
STOP_RECONCILE_GRACE_S = 30
#: item 行仍在执行中的状态；stop-run 出报告前必须全部离开该集合
#: （变为 success / failed 等非 running）。
RUNNING_ITEM_STATUS = "running"

_TERMINAL = frozenset(
    {
        BenchmarkStatus.COMPLETED.value,
        BenchmarkStatus.FAILED.value,
        BenchmarkStatus.STOPPED.value,
    }
)


@dataclass(frozen=True)
class FinalizeStoppedWrite:
    """``finalize_as_stopped`` 入参（FUNC-PARAMS-001）。"""

    run_id: str
    final_score: float | None = None
    metrics_json: dict[str, Any] | None = None
    report_meta_json: dict[str, Any] | None = None
    duration_ms: int | None = None


def is_stop_requested(config: object) -> bool:
    """``config_json`` 是否已打停止门闩。"""
    if not isinstance(config, dict):
        return False
    return bool(config.get("stop_requested_at"))


def stop_rows_per_item(run: object, config: object) -> int:
    """stop_target / processed 的「每 item 行数」量纲（与 finalize CAS 对齐）。

    - ``triage_mode=fused``：每 item 1 行 → 1
    - ``per_eval`` / ``none``：每 item N 行（N=``total_evaluators``）→ N
    """
    triage = "fused"
    if isinstance(config, dict):
        triage = str(config.get("triage_mode") or "fused")
    if triage == "fused":
        return 1
    n = int(getattr(run, "total_evaluators", 0) or 0)
    if n <= 0 and isinstance(config, dict):
        n = int(config.get("total_evaluators") or 0)
    return max(n, 1)


def parse_stop_requested_at(config: object) -> datetime | None:
    """解析 ``stop_requested_at``；非法 / 缺失返回 ``None``。"""
    if not isinstance(config, dict):
        return None
    raw = config.get("stop_requested_at")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


def stop_reconcile_in_grace(
    config: object,
    *,
    now: datetime | None = None,
    grace_s: float = STOP_RECONCILE_GRACE_S,
) -> bool:
    """``stop_requested_at`` 距今是否仍在 healer reconcile 宽限期内。

    宽限期内 healer **不得** ``reconcile_stop_target_to_processed``：
    console 的 mark → drain wait → align 尚未完成时，真正在途可能暂为 0
    （wait job 尚未清、running 无可计占位），无条件压 target 会提前 CAS
    进 aggregating。时间戳缺失/非法视为不在宽限期（允许兜底，避免永久卡住）。
    """
    ts = parse_stop_requested_at(config)
    if ts is None:
        return False
    age_s = ((now or datetime.now(UTC)) - ts).total_seconds()
    return age_s < grace_s


async def count_running_benchmark_items(run_id: str) -> int:
    """统计仍 ``status=running`` 的 ``benchmark_run_items`` 行数。

    stop-run 阶段性报告门槛：队列 inflight 为空后，还须等全部 running
    item 变为 success/failed（或其它非 running）再出报告。
    """
    return int(
        await BenchmarkRunItem.filter(
            run_id=run_id,
            status=RUNNING_ITEM_STATUS,
        ).count(),
    )


@dataclass(frozen=True)
class StopReportReadiness:
    """stop-run 出报告就绪快照（FUNC-PARAMS-001）。"""

    ready: bool
    running_items: int = 0
    attribution_pending: int = 0


def _config_int(config: dict[str, Any], key: str) -> int:
    """读取 config 非负整型计数；非法 / 缺失 → 0。"""
    raw = config.get(key)
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _id_set(config: dict[str, Any], key: str) -> set[str]:
    """读取 config 中的 item id 列表为字符串集合。"""
    raw = config.get(key) or []
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw if item is not None and str(item)}


def stop_attribution_pending(config: object) -> int:
    """stop-run 仍待齐的归因条数；无归因配置 → 0（不阻塞出报告）。

    仅看已登记的 ``attribution_expected_*``：不拦截未启用归因的正常/
    stop 路径；也不改归因提交流程——只决定能否出报告。
    """
    if not isinstance(config, dict):
        return 0
    has_counter = "attribution_expected_count" in config
    expected_ids = _id_set(config, "attribution_expected_item_ids")
    if not has_counter and not expected_ids:
        return 0
    received = _config_int(config, "attribution_received_count")
    expected = _config_int(config, "attribution_expected_count")
    received_ids = _id_set(config, "attribution_received_item_ids")
    by_count = max(0, expected - received)
    by_ids = len(expected_ids - received_ids)
    return max(by_count, by_ids)


async def inspect_stop_report_readiness(run_id: str) -> StopReportReadiness:
    """stop-run 出报告门槛：running items 终态 + 已登记归因齐套。"""
    running = await count_running_benchmark_items(run_id)
    run = await BenchmarkRun.filter(run_id=run_id, deleted_at__isnull=True).first()
    config = run.config_json if run is not None and isinstance(run.config_json, dict) else {}
    attr_pending = stop_attribution_pending(config)
    return StopReportReadiness(
        ready=running == 0 and attr_pending == 0,
        running_items=running,
        attribution_pending=attr_pending,
    )


async def stop_items_ready_for_report(run_id: str) -> bool:
    """stop-run 是否可出报告：无 running item，且已登记归因全部到齐。"""
    return (await inspect_stop_report_readiness(run_id)).ready


async def mark_stop_requested(*, run_id: str, reason: str = STOP_REASON) -> dict[str, Any]:
    """幂等写入 ``stop_requested_at``；已终态则 noop。

    Returns:
        ``{ok, noop, status, stop_requested_at}``
    """
    run = await BenchmarkRun.filter(run_id=run_id, deleted_at__isnull=True).first()
    if run is None:
        return {"ok": False, "noop": True, "status": None, "reason": "run_not_found"}
    status = str(getattr(run.status, "value", run.status))
    if status in _TERMINAL:
        return {
            "ok": True,
            "noop": True,
            "status": status,
            "stop_requested_at": (
                run.config_json.get("stop_requested_at")
                if isinstance(run.config_json, dict)
                else None
            ),
        }
    now = datetime.now(UTC).isoformat()
    config = dict(run.config_json) if isinstance(run.config_json, dict) else {}
    if config.get("stop_requested_at"):
        return {
            "ok": True,
            "noop": True,
            "status": status,
            "stop_requested_at": config.get("stop_requested_at"),
        }
    config["stop_requested_at"] = now
    config["stop_reason"] = reason
    evaluated = int(run.evaluated_items or 0)
    failed = int(run.failed_items or 0)
    processed = evaluated + failed
    # 门闩可见即必须带 stop_target：finalize CAS 要求 target 存在。
    # 量纲与 CAS 一致（行数）：fused=item；per_eval/none=item*evaluators。
    # 临时值在 align 前不可达（除非已全部跑完）；align 按真正在途重写。
    rows_per = stop_rows_per_item(run, config)
    total_samples = int(getattr(run, "total_samples", 0) or 0)
    full_threshold = total_samples * rows_per
    provisional_target = max(full_threshold, processed + rows_per + 1)
    config["stop_target_count"] = provisional_target
    await BenchmarkRun.filter(run_id=run_id).update(config_json=config)
    logger.info(
        "benchmark.stop_requested run_id=%s status=%s processed=%d "
        "provisional_stop_target=%d rows_per_item=%d",
        run_id,
        status,
        processed,
        provisional_target,
        rows_per,
    )
    return {
        "ok": True,
        "noop": False,
        "status": status,
        "stop_requested_at": now,
        "completed_cases": processed,
        "stop_target_count": provisional_target,
    }


async def publish_stop_signal(
    redis: Any,  # noqa: ANN401 — 可选 pub/sub 客户端，缺省静默
    *,
    run_id: str,
    task_id: str | None = None,
) -> None:
    """尽力广播停止门闩；Redis 不可用时静默（DB 仍为真相源）。"""
    if redis is None:
        return
    try:
        payload = f"run_id={run_id};task_id={task_id or ''}"
        await redis.publish(STOP_SIGNAL_CHANNEL, payload)
    except Exception:
        logger.warning(
            "benchmark.stop_publish_failed run_id=%s",
            run_id,
            exc_info=True,
        )


def partial_report_meta(
    *,
    completed_cases: int,
    stop_requested_at: str | None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """【仅 stop】合并阶段性报告 meta（不覆盖 Pages/OSS URL）。

    写入 ``partial`` / ``completed_cases`` / ``stop_kind``；正常 completed/failed
    路径禁止调用本函数伪装 partial。
    """
    meta = dict(existing or {})
    meta["partial"] = True
    meta["stop_kind"] = STOP_KIND_MANUAL
    meta["completed_cases"] = completed_cases
    if stop_requested_at:
        meta["stopped_at"] = stop_requested_at
    if completed_cases <= 0:
        meta["report_status"] = "skipped_no_cases"
    else:
        meta["report_status"] = meta.get("report_status") or "ready"
    return meta


async def _stopped_finalize_gate(run_id: str, status: str) -> bool | None:
    """``finalize_as_stopped`` 前置门闩。

    Returns:
        ``True``/``False``：可直接作为函数返回值；``None``：继续写 stopped。
    """
    if status == BenchmarkStatus.STOPPED.value:
        return True
    if status in (BenchmarkStatus.COMPLETED.value, BenchmarkStatus.FAILED.value):
        return False
    if await stop_items_ready_for_report(run_id):
        return None
    running = await count_running_benchmark_items(run_id)
    logger.info(
        "benchmark.run_stopped_deferred run_id=%s running_items=%d — "
        "wait for item terminal before stopped/report",
        run_id,
        running,
    )
    return False


async def finalize_as_stopped(write: FinalizeStoppedWrite) -> bool:
    """将 run 标为 ``stopped`` 终态（幂等）。

    用于 0 case 跳过报告，或 finalize 路径写完报告后的终态落盘。
    CAS（§6.8.7）：仅非终态可转入 ``stopped``，避免覆盖
    ``completed``/``failed`` 及其报告 URL。
    仍有 ``benchmark_run_items.status=running`` 时拒绝落盘（stop-run
    必须等全部 item success/failed 后再出报告 / 标 stopped）。
    """
    run_id = write.run_id
    run = await BenchmarkRun.filter(run_id=run_id, deleted_at__isnull=True).first()
    if run is None:
        return False
    status = str(getattr(run.status, "value", run.status))
    gated = await _stopped_finalize_gate(run_id, status)
    if gated is not None:
        return gated
    config = run.config_json if isinstance(run.config_json, dict) else {}
    evaluated = int(run.evaluated_items or 0)
    failed = int(run.failed_items or 0)
    completed_cases = evaluated + failed
    stop_at = config.get("stop_requested_at") if isinstance(config, dict) else None
    existing_meta = run.report_meta_json if isinstance(run.report_meta_json, dict) else {}
    merged_meta = partial_report_meta(
        completed_cases=completed_cases,
        stop_requested_at=str(stop_at) if stop_at else None,
        existing={**existing_meta, **(write.report_meta_json or {})},
    )
    update: dict[str, Any] = {
        "status": BenchmarkStatus.STOPPED.value,
        "error_message": STOP_REASON,
        "finished_at": datetime.now(UTC),
        "report_meta_json": merged_meta,
    }
    if write.final_score is not None:
        update["final_score"] = write.final_score
    if write.metrics_json is not None:
        update["metrics_json"] = write.metrics_json
    if write.duration_ms is not None:
        update["duration_ms"] = write.duration_ms
    # 条件更新：终态行数=0，不覆盖并发 complete_run / fail_run
    updated = await (
        BenchmarkRun.filter(run_id=run_id, deleted_at__isnull=True)
        .exclude(status__in=list(_TERMINAL))
        .update(**update)
    )
    if updated:
        logger.info(
            "benchmark.run_stopped run_id=%s completed_cases=%d",
            run_id,
            completed_cases,
        )
        return True
    raced = await BenchmarkRun.filter(run_id=run_id, deleted_at__isnull=True).first()
    raced_status = str(getattr(raced.status, "value", raced.status)) if raced is not None else ""
    return raced_status == BenchmarkStatus.STOPPED.value


async def align_expectations_for_stop(
    run_id: str,
    *,
    inflight_judge: int = 0,
    inflight_attribution: int = 0,
    processed_snapshot: int | None = None,
) -> dict[str, int]:
    """【仅 stop】按「已收到 + 在途 running」对齐期望，再出阶段性报告。

    - external：``*_expected = *_received + inflight_*``，等在途回调到齐再 CAS。
    - sync：写 ``stop_target_count = processed + inflight_jobs * rows_per_item``。
      ``processed`` / CAS 左侧是**行数**（per_eval 每 item N 行）；
      ``inflight_*`` 是 **job/item 数**，必须乘 ``stop_rows_per_item``，
      否则 per_eval 会在约 ``ceil(K/N)`` 个在途 job 完成时提前 CAS。
    - ``processed_snapshot``：调用方在清点 inflight **之前**拍的已处理数，
      避免「清点→读行」窗口把同一 item 双计导致 stop_target 不可达。
    正常 completed/failed 禁止调用。
    """
    run = await BenchmarkRun.filter(run_id=run_id, deleted_at__isnull=True).first()
    if run is None or not isinstance(run.config_json, dict):
        return {
            "stop_target_count": 0,
            "inflight_total": 0,
            "processed": 0,
            "rows_per_item": 1,
        }
    config = dict(run.config_json)
    judge_in = max(0, int(inflight_judge))
    attr_in = max(0, int(inflight_attribution))
    inflight_total = judge_in + attr_in
    rows_per = stop_rows_per_item(run, config)
    if processed_snapshot is not None:
        processed = max(0, int(processed_snapshot))
    else:
        processed = int(run.evaluated_items or 0) + int(run.failed_items or 0)
    # 在途 job → 行：judge/sync case 按 rows_per；attribution job 通常 1 计数
    # 推进（不写 N 条 criterion），故 attr 不乘 evaluators。
    stop_target = processed + judge_in * rows_per + attr_in
    config["stop_target_count"] = stop_target
    if "judge_expected_count" in config:
        config["judge_expected_count"] = int(config.get("judge_received_count") or 0) + judge_in
    if "attribution_expected_count" in config:
        # 不得压低已登记 expected：Aone 等在途归因不在 BullMQ，
        # inflight_attr=0 时若写成 received 会漏等并提前出报告。
        # stop 只允许抬高到 received+queue_inflight；归因提交流程不变。
        attr_received = _config_int(config, "attribution_received_count")
        attr_expected = _config_int(config, "attribution_expected_count")
        config["attribution_expected_count"] = max(attr_expected, attr_received + attr_in)
    await BenchmarkRun.filter(run_id=run_id).update(config_json=config)
    logger.info(
        "benchmark.stop_expectations_aligned run_id=%s stop_target=%d "
        "inflight_judge=%d inflight_attr=%d processed=%d rows_per_item=%d",
        run_id,
        stop_target,
        judge_in,
        attr_in,
        processed,
        rows_per,
    )
    return {
        "stop_target_count": stop_target,
        "inflight_total": inflight_total,
        "inflight_judge": judge_in,
        "inflight_attribution": attr_in,
        "processed": processed,
        "rows_per_item": rows_per,
    }


async def reconcile_stop_target_to_processed(run_id: str) -> dict[str, int]:
    """【仅 stop 兜底】队列已空但 processed < stop_target 时，压到当前 processed。

    覆盖：inflight 与 processed 双计、parked 误计、回调丢弃等导致的不可达门闩。
    压齐后由调用方再触发 finalize CAS。
    """
    return await align_expectations_for_stop(run_id, inflight_judge=0, inflight_attribution=0)


async def align_external_expectations_for_stop(run_id: str) -> None:
    """兼容旧名：无在途时把期望压到已收到（``inflight=0``）。"""
    await align_expectations_for_stop(run_id)


def streaming_stopped_run_id(stem: str, *, now: datetime | None = None) -> str:
    """实时流停止窗口报告 run_id：``{stem}-{yyyymmdd}-stopped-{HHmmss}``。"""
    ts = now or datetime.now(UTC)
    ymd = ts.strftime("%Y%m%d")
    hms = ts.strftime("%H%M%S")
    base = (stem or "trace_report").strip() or "trace_report"
    return f"{base}-{ymd}-stopped-{hms}"


__all__ = [
    "RUNNING_ITEM_STATUS",
    "STOP_KIND_MANUAL",
    "STOP_REASON",
    "STOP_RECONCILE_GRACE_S",
    "STOP_SIGNAL_CHANNEL",
    "FinalizeStoppedWrite",
    "StopReportReadiness",
    "align_expectations_for_stop",
    "align_external_expectations_for_stop",
    "count_running_benchmark_items",
    "finalize_as_stopped",
    "inspect_stop_report_readiness",
    "is_stop_requested",
    "mark_stop_requested",
    "parse_stop_requested_at",
    "partial_report_meta",
    "publish_stop_signal",
    "reconcile_stop_target_to_processed",
    "stop_attribution_pending",
    "stop_items_ready_for_report",
    "stop_reconcile_in_grace",
    "stop_rows_per_item",
    "streaming_stopped_run_id",
]
