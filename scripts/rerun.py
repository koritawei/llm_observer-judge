#!/usr/bin/env python3
"""运维：完全重跑 run — 删除所有 items + 清队列 + 重新触发。

操作流程：
1. 从 PG 读取 run 信息 + config_json（用于重建请求）
2. Drain 6 个队列（EXTERNAL_RUNNING / BENCHMARK_RUNNING /
   ATTRIBUTION_RUNNING / EXTERNAL_WAIT / BENCHMARK_CASES /
   ATTRIBUTION_WAIT）删除该 run 的所有 job + 释放槽位
   （§8.3 队列归并：统一 run 的 judge job 驻留 benchmark-cases/running）
3. DEL 泄漏的 inflight Redis 键（per-node + cluster）
4. 删除 benchmark_run_items 所有行
5. 删除 benchmark_runs 行
6. POST /v1/benchmarks/run-external 重新触发
   （config_json 快照中的 rubric_class / judge_class / attribution_class /
   parse_interceptor_ref / limit 原样透传，保证重跑 profile 与原 run 一致）

默认 dry-run；加 ``--execute`` 才真正执行。

用法（优先包装脚本，线上会拼 agentgraph PYTHONPATH）::

  bash scripts/rerun.sh --run-id ieval_external-20260819
  bash scripts/rerun.sh --run-id ieval_external-20260819 --execute
"""

import argparse
import asyncio
import json
import sys
from typing import Any

import _bootstrap_quiet  # sys.path/.env/禁 MQ+healer+日调度+sidecar，须先于 app.* 执行
import redis.asyncio as aioredis
from benchmark.settings import Settings
from benchmark.records.models import BenchmarkRun, BenchmarkRunItem
from benchmark.domain.benchmark_console import find_task_document, save_task
from benchmark.domain.report_storage_config import (
    get_default_download_prefix,
)
from benchmark.libs.jobqueue.job_listing import list_queue_jobs
from benchmark.service.main import create_app
from benchmark.runs.common.agent_capabilities import (
    batch_trigger_execution_aliases,
)
from benchmark.runs.common.oss_date_macro import (
    expand_date_macros,
    has_date_macro,
    object_key_for_dated_prefix,
)
from benchmark.runs.common.rubric_classes import rubric_alias_for_ref
DailyDataLoader = None
from benchmark.runs.run_service import BenchmarkRunSpec
from benchmark.runs.schema import ExternalBenchmarkRequest
from benchmark.queue.inflight_slot import (
    release_cluster_slot,
    release_slot,
)
from benchmark.queue.queue_factory import (
    ATTRIBUTION_RUNNING,
    ATTRIBUTION_WAIT,
    BENCHMARK_CASES,
    BENCHMARK_RUNNING,
    EXTERNAL_RUNNING,
    EXTERNAL_WAIT,
    EvalQueueRegistry,
)
from fastapi import FastAPI

del _bootstrap_quiet  # 仅为副作用导入：已执行，释放模块名

_ALL_JOB_TYPES = ["waiting", "delayed", "active", "paused", "prioritized"]

#: run_id 中日期段的字符长度（``YYYYMMDD``）。
_DATE_PART_LEN = 8
#: ``{name}-{date}-{suffix}`` 按 ``-{date}-`` 切分后的段数。
_RUN_ID_DATE_SPLIT_PARTS = 2

_QUEUE_KIND_MAP = (
    (EXTERNAL_RUNNING, "judge"),
    (BENCHMARK_RUNNING, "judge"),
    (ATTRIBUTION_RUNNING, "attr"),
    (EXTERNAL_WAIT, ""),
    (BENCHMARK_CASES, ""),
    (ATTRIBUTION_WAIT, ""),
)


async def _drain_run(
    qm: EvalQueueRegistry,
    run_id: str,
    *,
    execute: bool,
) -> dict[str, int]:
    """扫描 4 个队列，删除该 run 的所有 job + 释放槽位。"""
    stats = {"jobs_removed": 0, "slots_released": 0, "cluster_released": 0}
    for queue_name, default_kind in _QUEUE_KIND_MAP:
        q = qm.get_queue(queue_name)
        try:
            jobs = await list_queue_jobs(q, _ALL_JOB_TYPES)
        except Exception as exc:
            print(f"  WARN: list {queue_name} failed: {exc}", file=sys.stderr)
            continue
        for job in jobs:
            if job is None:
                continue
            data = dict(job.data or {})
            if str(data.get("run_id") or "") != run_id:
                continue
            slot_key = data.get("slot_key")
            kind = str(data.get("cluster_kind") or default_kind)
            print(
                f"  [{queue_name}] job_id={job.id} name={job.name} "
                f"item={data.get('item_id')} slot_key={slot_key} kind={kind}",
            )
            if not execute:
                continue
            if slot_key:
                await release_slot(str(slot_key))
                stats["slots_released"] += 1
            if kind:
                await release_cluster_slot(kind)
                stats["cluster_released"] += 1
            try:
                await job.remove()
                stats["jobs_removed"] += 1
            except Exception as exc:
                print(f"    WARN: remove failed: {exc}", file=sys.stderr)
    return stats


async def _clear_inflight_keys(
    qm: EvalQueueRegistry,
    run_id: str,
    *,
    execute: bool,
) -> int:
    """直接 DEL 泄漏的 inflight Redis 键（per-node + cluster）。"""
    prefix = qm.prefix
    r = aioredis.from_url(qm.redis_url, decode_responses=True)
    keys: list[str] = []
    try:
        for kind in ("judge", "attr"):
            keys.append(f"{prefix}:inflight:cluster:{kind}")
        for pattern in (
            f"{prefix}:inflight:judge:*:{run_id}",
            f"{prefix}:inflight:attr:*:{run_id}",
        ):
            async for key in r.scan_iter(match=pattern, count=50):
                if key not in keys:
                    keys.append(key)
        print("Inflight keys:")
        for k in keys:
            val = await r.get(k)
            print(f"  {k} = {val!r}")
        if not execute:
            print("[dry-run] skip DEL inflight keys")
            return 0
        if keys:
            n = await r.delete(*keys)
            print(f"Deleted {n} inflight key(s)")
            return n
        print("No inflight keys to delete")
        return 0
    finally:
        await r.aclose()


async def _delete_pg_items(run_id: str, *, execute: bool) -> int:
    """删除 benchmark_run_items 所有行。"""
    count = await BenchmarkRunItem.filter(run_id=run_id).count()
    print(f"  benchmark_run_items: {count} row(s)")
    if not execute:
        print("  [dry-run] skip DELETE items")
        return 0
    deleted = await BenchmarkRunItem.filter(run_id=run_id).delete()
    print(f"  Deleted {deleted} item row(s)")
    return deleted


async def _delete_pg_run(run_id: str, *, execute: bool) -> bool:
    """删除 benchmark_runs 行。"""
    if not execute:
        print("  [dry-run] skip DELETE run row")
        return False
    deleted = await BenchmarkRun.filter(run_id=run_id).delete()
    print(f"  Deleted {deleted} run row(s)")
    return deleted > 0


def _apply_profile_snapshot(payload: dict[str, Any], cfg: dict[str, Any]) -> None:
    """config_json 快照中的 profile 键原样透传进重建请求（§9）。

    ``rubric`` JSON 覆盖仅 ``attribution_only`` 允许（与
    ``ExternalBenchmarkRequest`` 校验一致）；``judge_attribution`` 只用
    ``rubric_class``。``rubric_ref`` 为 null 的存量快照仍可能带 rubric 数组，
    原样回传会 422。
    """
    if payload.get("mode") == "attribution_only" and cfg.get("rubric"):
        payload["rubric"] = cfg["rubric"]
    if cfg.get("parse_interceptor_ref"):
        payload["parse_interceptor_ref"] = cfg["parse_interceptor_ref"]
    if cfg.get("aggregation"):
        payload["aggregation"] = cfg["aggregation"]
    if cfg.get("oss_limit"):
        payload["limit"] = cfg["oss_limit"]


def _pin_dated_oss_object_key(payload: dict[str, Any]) -> None:
    """prefix 末段已是 ``yyyymmdd`` 时补定 object key，避免 loader 再拼日期。

    不要求末段等于 ``date``：偏移宏预展开后的分区日同样需要钉死。
    """
    if payload.get("oss_object_key") or not payload.get("oss_prefix"):
        return
    pinned = object_key_for_dated_prefix(
        str(payload["oss_prefix"]),
        file_name=DailyDataLoader.FILE_NAME,
    )
    if pinned:
        payload["oss_object_key"] = pinned


def _date_from_run_id(run_id: str) -> str | None:
    """从 ``{name}-{yyyymmdd}[-suffix]`` 解析 8 位数据日。"""
    for part in run_id.split("-"):
        if len(part) == _DATE_PART_LEN and part.isdigit():
            return part
    return None


def _snapshot_rubric_class(cfg: dict[str, Any]) -> str | None:
    """读快照 rubric 别名；无 rubric_ref 时不回落 component_rubric 占位。"""
    rubric_class = cfg.get("rubric_class")
    if rubric_class:
        return str(rubric_class)
    if cfg.get("rubric_ref"):
        return rubric_alias_for_ref(cfg.get("rubric_ref"))
    return None


def _apply_oss_snapshot(payload: dict[str, Any], cfg: dict[str, Any]) -> None:
    """回放 OSS 定位字段；缺 object_key 且 prefix 已含数据日时补定。"""
    if cfg.get("oss_prefix"):
        payload["oss_prefix"] = cfg["oss_prefix"]
    if cfg.get("oss_object_key"):
        payload["oss_object_key"] = cfg["oss_object_key"]
    else:
        _pin_dated_oss_object_key(payload)


def _apply_ownership_snapshot(
    payload: dict[str, Any],
    run: BenchmarkRun,
    cfg: dict[str, Any],
) -> None:
    """回放 dataset / task / project 归属（瘦 placeholder 可能没有）。"""
    for key in ("dataset_id", "eval_task_id"):
        if cfg.get(key):
            payload[key] = cfg[key]
    if run.dataset_name:
        payload["dataset_name"] = run.dataset_name
    if run.project_id:
        payload["project_id"] = run.project_id
    if cfg.get("mode") == "attribution_only":
        payload["mode"] = "attribution_only"


def _apply_rerun_suffix(payload: dict[str, Any], run_id: str, date_str: str | None) -> None:
    """原 run 有后缀时追加 ``_rerun``，避免与源 run_id 冲突。"""
    if not date_str:
        return
    after_date = run_id.split(f"-{date_str}-", 1)
    if len(after_date) == _RUN_ID_DATE_SPLIT_PARTS and after_date[1]:
        payload["run_id_suffix"] = after_date[1] + "_rerun"


def _build_rerun_payload(run: BenchmarkRun) -> dict[str, Any]:
    """从 run.config_json 重建 ExternalBenchmarkRequest payload。

    快照残缺（例如上次 rerun init 失败留下的瘦 placeholder）时字段可为空，
    由 :func:`_enrich_rerun_payload` 按 benchmark 能力 / 任务数据集补齐。
    """
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    payload: dict[str, Any] = {
        "benchmark_name": cfg.get("benchmark_services") or run.benchmark_name,
        "judge_class": cfg.get("judge_class"),
        "attribution_class": cfg.get("attribution_class"),
        "rubric_class": _snapshot_rubric_class(cfg),
        "generate_report": cfg.get("generate_report", True),
    }
    date_str = _date_from_run_id(run.run_id)
    if date_str:
        payload["date"] = date_str
    _apply_oss_snapshot(payload, cfg)
    _apply_ownership_snapshot(payload, run, cfg)
    _apply_profile_snapshot(payload, cfg)
    _apply_rerun_suffix(payload, run.run_id, date_str)
    return payload


async def _lookup_dataset_for_benchmark(
    benchmark_name: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """按可调度任务反查数据集 OSS 前缀（瘦 placeholder 无快照时的兜底）。

    Returns:
        ``(oss_prefix, dataset_id, dataset_name, eval_task_id)``；找不到则全 None。
    """
    from benchmark.domain.benchmark_console import (  # noqa: PLC0415
        find_agent_document_by_id,
        find_dataset_document,
        list_schedulable_tasks,
    )

    for task in await list_schedulable_tasks():
        agent = await find_agent_document_by_id(task.agent_id)
        if agent is None or agent.name != benchmark_name:
            continue
        dataset = await find_dataset_document(task.dataset_id)
        if dataset is None or dataset.source == "realtime":
            continue
        return (
            (dataset.oss_prefix or "").strip() or None,
            dataset.id,
            dataset.name,
            task.id,
        )
    return None, None, None, None


def _enrich_execution_aliases(payload: dict[str, Any], declared: object) -> None:
    """按 batch 能力链补齐 judge / attribution 别名。"""
    aliases = batch_trigger_execution_aliases(declared)  # type: ignore[arg-type]
    for field in ("judge_class", "attribution_class"):
        if not payload.get(field) and aliases.get(field):
            payload[field] = aliases[field]
            print(f"  [enrich] {field} <- {aliases[field]}")


def _enrich_rubric_and_parse(payload: dict[str, Any], meta: object) -> None:
    """覆盖 component_rubric 占位，并补 parse_interceptor_ref。"""
    current_rubric = str(payload.get("rubric_class") or "").strip()
    if not current_rubric or current_rubric == "component_rubric":
        rubric = rubric_alias_for_ref(getattr(meta, "rubric_class", None))
        if rubric and rubric != "component_rubric":
            payload["rubric_class"] = rubric
            print(f"  [enrich] rubric_class <- {rubric}")
    if payload.get("parse_interceptor_ref"):
        return
    rubric_ref = getattr(meta, "rubric_class", None)
    if not (isinstance(rubric_ref, str) and ":" in rubric_ref):
        return
    try:
        from benchmark.plugins.class_loader import load_plugin_class  # noqa: PLC0415

        rubric_cls = load_plugin_class(rubric_ref, what="rubric_class")
        interceptor = getattr(rubric_cls, "parse_interceptor_ref", None)
        if interceptor:
            payload["parse_interceptor_ref"] = interceptor
            print(f"  [enrich] parse_interceptor_ref <- {interceptor}")
    except Exception as exc:
        print(f"  [enrich] skip parse_interceptor_ref: {exc}")


async def _enrich_oss_from_dataset(payload: dict[str, Any], benchmark_name: str) -> None:
    """缺 oss_prefix 时从可调度任务数据集回填。"""
    if payload.get("oss_prefix"):
        return
    oss_prefix, dataset_id, dataset_name, task_id = await _lookup_dataset_for_benchmark(
        benchmark_name,
    )
    if oss_prefix:
        payload["oss_prefix"] = oss_prefix
        print(f"  [enrich] oss_prefix <- {oss_prefix}")
    if dataset_id and not payload.get("dataset_id"):
        payload["dataset_id"] = dataset_id
    if dataset_name and not payload.get("dataset_name"):
        payload["dataset_name"] = dataset_name
    if task_id and not payload.get("eval_task_id"):
        payload["eval_task_id"] = task_id


def _finalize_oss_macros(payload: dict[str, Any]) -> None:
    """展开日期宏或钉死已含数据日的 object key。"""
    date_str = str(payload.get("date") or "")
    oss_prefix = str(payload.get("oss_prefix") or "")
    if date_str and oss_prefix and has_date_macro(oss_prefix):
        expanded = expand_date_macros(oss_prefix, base_date=date_str)
        payload["oss_prefix"] = expanded
        payload["oss_object_key"] = f"{expanded.rstrip('/')}/{DailyDataLoader.FILE_NAME}"
        print(f"  [enrich] oss_prefix macro -> {expanded}")
        return
    _pin_dated_oss_object_key(payload)


def _require_rerun_execution_fields(payload: dict[str, Any], benchmark_name: str) -> None:
    """补齐后仍缺执行类 / rubric 则失败，避免 silent 错跑。"""
    if not payload.get("judge_class") or not payload.get("attribution_class"):
        raise ValueError(
            f"run {benchmark_name} has no judge_class/attribution_class "
            "(snapshot empty and benchmark capabilities have no batch aliases)",
        )
    if not payload.get("rubric_class"):
        raise ValueError(
            f"run {benchmark_name} has no rubric_class "
            "(snapshot empty and metadata rubric_class has no alias)",
        )
    if payload.get("mode") != "attribution_only":
        payload.pop("rubric", None)


async def _enrich_rerun_payload(
    app: FastAPI,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """快照残缺时按 benchmark 能力链 / 任务数据集补齐触发字段。"""
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise SystemExit("runtime not initialized")
    benchmark_name = str(payload.get("benchmark_name") or "")
    try:
        benchmark_services = runtime.agent_manager.get(benchmark_name)
    except (KeyError, LookupError) as exc:
        raise SystemExit(
            f"benchmark_services not found: {benchmark_name} ({exc})",
        ) from exc

    meta = benchmark_services.get_metadata()
    declared = getattr(benchmark_services, "benchmark_capabilities", None)
    _enrich_execution_aliases(payload, declared)
    _enrich_rubric_and_parse(payload, meta)
    await _enrich_oss_from_dataset(payload, benchmark_name)
    _finalize_oss_macros(payload)
    _require_rerun_execution_fields(payload, benchmark_name)
    return payload


async def _retrigger(
    app: FastAPI,
    run_id: str,
    payload_dict: dict[str, Any],
    *,
    execute: bool,
) -> None:
    """直接调用 initialize_and_submit 重新触发（不走 HTTP）。"""
    runner = getattr(app.state, "external_benchmark_runner", None)
    if runner is None:
        raise SystemExit("external_benchmark_runner not initialized")
    benchmark_service = getattr(app.state, "benchmark_service", None)
    if benchmark_service is None:
        raise SystemExit("benchmark_service not initialized")
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise SystemExit("runtime not initialized")

    # 解析 benchmark_services
    benchmark_name = payload_dict.get("benchmark_name", "")
    try:
        benchmark_services = runtime.agent_manager.get(benchmark_name)
    except (KeyError, LookupError) as exc:
        raise SystemExit(
            f"benchmark_services not found: {benchmark_name} ({exc})",
        ) from exc

    # 构建 ExternalBenchmarkRequest
    payload = ExternalBenchmarkRequest(**payload_dict)

    # 归一化 effective 字段（对齐 API 端点逻辑）
    effective_oss_prefix = payload.oss_prefix or get_default_download_prefix()
    payload.effective_date = payload.date or ""
    payload.effective_oss_prefix = effective_oss_prefix
    # 末段已是 yyyymmdd 的 prefix：定死 object key，避免 loader 双拼日期
    if not payload.oss_object_key:
        pin: dict[str, Any] = {"oss_prefix": effective_oss_prefix}
        _pin_dated_oss_object_key(pin)
        if pin.get("oss_object_key"):
            payload.oss_object_key = str(pin["oss_object_key"])

    print(f"\n  benchmark_name={benchmark_name}")
    print(f"  run_id={run_id}")
    print(f"  date={payload.effective_date} oss_prefix={payload.effective_oss_prefix}")
    print(f"  oss_object_key={payload.oss_object_key}")
    if not execute:
        print("  [dry-run] skip initialize_and_submit")
        return

    # 创建占位 run：数据定位 + 执行类快照一并落盘，init 失败仍可再 rerun
    meta = benchmark_services.get_metadata()
    created = await benchmark_service.create_placeholder_run(
        run_id,
        BenchmarkRunSpec(
            benchmark_name=benchmark_name,
            service_name=meta.service_name,
            dataset_name=payload.dataset_name or "",
            config_json={
                "benchmark_services": benchmark_name,
                "judge_class": payload.judge_class,
                "attribution_class": payload.attribution_class,
                "rubric_class": payload.rubric_class,
                "attribution_provider": payload.attribution_provider,
                "generate_report": payload.generate_report,
                "data_date": payload.effective_date,
                "oss_prefix": payload.effective_oss_prefix,
                "oss_object_key": payload.oss_object_key,
                "dataset_id": payload.dataset_id,
                "eval_task_id": payload.eval_task_id,
            },
        ),
    )
    if not created:
        # 删建竞态 / 软删残留：若仍是 failed，CAS 拉回 initializing 再继续。
        existing = await BenchmarkRun.filter(run_id=run_id).first()
        status = str(
            getattr(getattr(existing, "status", None), "value", getattr(existing, "status", ""))
        )
        if existing is not None and status == "failed":
            reset = await benchmark_service.cas_reset_failed_run(run_id)
            if not reset:
                print(f"  WARN: cannot reset existing failed run: {run_id}")
                return
            print(f"  reset existing failed run → initializing: {run_id}")
        else:
            print(f"  WARN: placeholder run already exists for {run_id} status={status or '-'}")
            return

    # 直接调用 initialize_and_submit（不走 HTTP）
    print("  Calling initialize_and_submit...")
    try:
        total = await runner.initialize_and_submit(
            run_id=run_id,
            payload=payload,
            benchmark_services=benchmark_services,
        )
        print(f"  initialize_and_submit done: total_count={total}")
    except Exception as exc:
        print(f"  ERROR: initialize_and_submit failed: {exc}", file=sys.stderr)
        await benchmark_service.fail_run(
            run_id=run_id,
            reason=f"rerun init failed: {exc}",
        )
        raise
    await _sync_eval_task_after_rerun(run_id, payload_dict)


async def _sync_eval_task_after_rerun(run_id: str, payload_dict: dict[str, Any]) -> None:
    """CLI 同 run_id 重跑后回写任务指针，并纠回误写的 failed/complete 调度态。"""
    task_id = str(payload_dict.get("eval_task_id") or "").strip()
    if not task_id:
        print("  skip task pointer sync: no eval_task_id in payload")
        return
    run = await BenchmarkRun.filter(run_id=run_id, deleted_at__isnull=True).first()
    if run is None:
        print(f"  WARN: cannot sync task pointer, run missing: {run_id}")
        return
    document = await find_task_document(task_id)
    if document is None:
        print(f"  WARN: eval task not found for pointer sync: {task_id}")
        return
    updates: dict[str, Any] = {
        "manual_run_id": run_id,
        "manual_run_pk": str(run.id),
    }
    if document.trigger != "immediate" and document.status in ("failed", "complete"):
        updates["status"] = "active"
        updates["failure_reason"] = None
    await save_task(document.model_copy(update=updates))
    print(
        f"  synced eval task {task_id} → pk={run.id} "
        f"status={updates.get('status', document.status)}",
    )


async def _amain(args: argparse.Namespace) -> int:
    settings = Settings()
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        run_id = args.run_id

        # 1. 读取 run 信息
        run = await BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            raise SystemExit(f"run not found: {run_id}")
        print(
            f"Target: run_id={run.run_id} status={run.status} "
            f"evaluated={run.evaluated_items} failed={run.failed_items} "
            f"total={run.total_samples}",
        )
        print(f"  config_json keys: {list((run.config_json or {}).keys())}")

        # 构建重跑 payload（在任何删除操作之前，防止删除后丢失信息）；
        # 瘦 placeholder（上次 init 失败）缺执行类 / OSS 时按能力链与任务数据集补齐。
        payload = _build_rerun_payload(run)
        payload = await _enrich_rerun_payload(app, payload)
        print(f"\n  Rerun payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        qm = getattr(app.state, "eval_queue_manager", None)
        if qm is None:
            raise SystemExit("eval_queue_manager not initialized")

        # 2. Drain queues
        print(f"\n--- Drain queues for run_id={run_id} ---")
        stats = await _drain_run(qm, run_id, execute=args.execute)
        print(
            f"\nDrain: jobs_removed={stats['jobs_removed']} "
            f"slots_released={stats['slots_released']} "
            f"cluster_released={stats['cluster_released']}",
        )

        # 3. Clear inflight keys
        print("\n--- Clear inflight Redis keys ---")
        await _clear_inflight_keys(qm, run_id, execute=args.execute)

        # 4. Delete PG items
        print("\n--- Delete benchmark_run_items ---")
        await _delete_pg_items(run_id, execute=args.execute)

        # 5. Delete PG run
        print("\n--- Delete benchmark_runs row ---")
        await _delete_pg_run(run_id, execute=args.execute)

        # 6. Re-trigger（直接调用 initialize_and_submit，不走 HTTP）
        print("\n--- Re-trigger run ---")
        await _retrigger(app, run_id, payload, execute=args.execute)

        if not args.execute:
            print("\n[dry-run] no changes made. Add --execute to apply.")
        else:
            print("\n✅ Done.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun: delete all items + clean queues + re-trigger",
    )
    parser.add_argument("--run-id", type=str, required=True, help="target run_id")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually execute (default: dry-run)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
