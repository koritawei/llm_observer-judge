#!/usr/bin/env python3
"""运维：重跑 run 下所有失败 items — 仅删 failed items + 调整计数器 + 重新入队。

操作流程：
1. 从 PG 读取 run 信息 + config_json
2. 查询 benchmark_run_items 中 status='failed' 的 items
3. 从 OSS 重新加载原始数据，按 failed item_id 过滤
4. Drain 6 个队列中该 run 的 failed item jobs + 释放槽位
   （§8.3 队列归并：统一 run 的 judge job 驻留 benchmark-cases/running）
5. DEL 泄漏的 inflight Redis 键（per-node + cluster）
6. 从 benchmark_run_items 删除 failed items
7. 调整 run 计数器：
   - judge_received_count -= N
   - failed_items -= N
   - 移除 attribution_received_item_ids 中 failed item_ids（如有）
   - 重算 attribution_received_count
8. 重置 run status 为 running（重置 finished_at / final_score / duration_ms）
9. 重新入队 failed items：按 run config_json 的 queue_pair 路由
   （存量 → EXTERNAL_WAIT；统一 run → benchmark-cases，worker 按快照路由）

不影响 CAS：judge_received_count 减 N 后 CAS 条件不满足，
rerun judge 回调补齐后 CAS 自然触发 → 报告重新生成。

默认 dry-run；加 ``--execute`` 才真正执行。

用法（优先包装脚本，线上会拼 agentgraph PYTHONPATH）::

  bash scripts/rerun_failed.sh --run-id ieval_external-20260819
  bash scripts/rerun_failed.sh --run-id ieval_external-20260819 --execute
"""

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING, Any

import _bootstrap  # sys.path/.env/禁 MQ 副作用，须先于 app.* 执行
import redis.asyncio as aioredis
from benchmark.settings import Settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun, BenchmarkRunItem
from benchmark.libs.jobqueue.job_listing import list_queue_jobs
from benchmark.service.main import create_app
from benchmark.runs.common.parse_interceptor import DataParseContext
from benchmark.runs.common.profile_resolver import (
    default_rubric_ref,
    load_class_ref,
)
from benchmark.runs.common.utils import extract_ds
DailyDataLoader = None
from benchmark.runs.schema import FUSED_CRITERION_ID
from benchmark.queue.callback_next_submit import (
    enqueue_external_cases,
)
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
from benchmark.workers.external_runner import (
    UnifiedCaseSpec,
    build_unified_case_payload,
)
from fastapi import FastAPI

if TYPE_CHECKING:
    from benchmark.libs.storage import StorageClient

del _bootstrap  # 仅为副作用导入：已执行，释放模块名

_ALL_JOB_TYPES = ["waiting", "delayed", "active", "paused", "prioritized"]
_PREVIEW_COUNT = 10

_QUEUE_KIND_MAP = (
    (EXTERNAL_RUNNING, "judge"),
    (BENCHMARK_RUNNING, "judge"),
    (ATTRIBUTION_RUNNING, "attr"),
    (EXTERNAL_WAIT, ""),
    (BENCHMARK_CASES, ""),
    (ATTRIBUTION_WAIT, ""),
)


def _replay_parse_interceptor(
    run: BenchmarkRun,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """拦截器重放（§7.2/§7.5）：OSS 重载的全量 KV → 业务结构。

    loader 透传改造后 OSS items 只含 ``metadata.csv_row``；run 快照（或
    默认 rubric 映射，§9 迁移锚点）挂载拦截器时重放恢复业务结构，
    保证 re-enqueue 后 worker / build_record 消费的结构与首次一致。
    无拦截器配置（透传业务）→ 原样返回；重放异常降级原样（worker 侧
    csv_row 透传分支仍可提交 judge）。
    """
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    benchmark_name = str(cfg.get("benchmark_services") or run.benchmark_name)
    ref = cfg.get("parse_interceptor_ref")
    if not ref:
        rubric_ref = cfg.get("rubric_ref") or default_rubric_ref(benchmark_name)
        if rubric_ref:
            rubric_cls = load_class_ref(str(rubric_ref), what="rubric_ref")
            ref = getattr(rubric_cls, "parse_interceptor_ref", None)
    if not ref:
        return items
    try:
        interceptor = load_class_ref(str(ref), what="parse_interceptor_ref")()
        parsed = interceptor.parse(
            DataParseContext(
                benchmark_name=benchmark_name,
                rubric_ref=str(cfg.get("rubric_ref") or ""),
                raw_items=items,
                profile_snapshot=cfg,
            ),
        )
        print(f"  Parse interceptor replayed ({ref}): {len(items)} → {len(parsed)} items")
        return parsed
    except Exception as exc:
        print(
            f"  WARN: parse interceptor replay failed ({ref}): {exc}; "
            "falling back to raw passthrough",
            file=sys.stderr,
        )
        return items


async def _reenqueue_failed_cases(
    qm: EvalQueueRegistry,
    run: BenchmarkRun,
    items: list[dict[str, Any]],
    ds: str,
) -> int:
    """按 run 的 ``queue_pair`` 路由重新入队（§8.3 队列归并）。

    存量 run → ``EXTERNAL_WAIT``（现网语义不变）；统一 run →
    ``benchmark-cases``（payload 经 :func:`build_unified_case_payload`
    单点构建，worker 按 run config_json 快照路由 sync/async judge）。
    """
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    if cfg.get("queue_pair") != "benchmark":
        return await enqueue_external_cases(
            queue_manager=qm,
            run_id=run.run_id,
            items=items,
            ds=ds,
        )
    judge_cfg = cfg.get("judge") if isinstance(cfg.get("judge"), dict) else {}
    spec = UnifiedCaseSpec(
        run_id=run.run_id,
        criteria=[
            {"id": str(c.get("id") or ""), "evaluator_ref": str(c.get("evaluator_ref") or "")}
            for c in (cfg.get("rubric") or [])
            if isinstance(c, dict)
        ],
        benchmark_name=str(cfg.get("benchmark_services") or run.benchmark_name),
        service_name=run.service_name or "",
        dataset_name=run.dataset_name or "",
        project_id=run.project_id,
        judge_execution=str(judge_cfg.get("execution") or "sync"),
        ds=ds,
        generate_report=bool(cfg.get("generate_report", True)),
    )
    queue = qm.get_queue(BENCHMARK_CASES)
    job_opts = {"attempts": 1, "removeOnComplete": True, "removeOnFail": True}
    count = 0
    for item in items:
        payload = build_unified_case_payload(item, spec)
        if payload is None:
            continue
        await queue.add(
            "case",
            payload,
            {"jobId": f"{run.run_id}:{payload['item_id']}", **job_opts},
        )
        count += 1
    return count


async def _drain_failed_jobs(
    qm: EvalQueueRegistry,
    run_id: str,
    failed_ids: set[str],
    *,
    execute: bool,
) -> dict[str, int]:
    """扫描 4 个队列，删除该 run 中 failed item 的 job + 释放槽位。"""
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
            item_id = str(data.get("item_id") or "")
            if item_id not in failed_ids:
                continue
            slot_key = data.get("slot_key")
            kind = str(data.get("cluster_kind") or default_kind)
            print(
                f"  [{queue_name}] job_id={job.id} name={job.name} "
                f"item={item_id} slot_key={slot_key} kind={kind}",
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
    failed_ids: set[str],
    *,
    execute: bool,
) -> int:
    """直接 DEL 泄漏的 inflight Redis 键（per-node + cluster）。"""
    prefix = qm.prefix
    r = aioredis.from_url(qm.redis_url, decode_responses=True)
    keys: list[str] = []
    try:
        # cluster 级别键始终清理
        for kind in ("judge", "attr"):
            keys.append(f"{prefix}:inflight:cluster:{kind}")

        # per-node 键只清理 failed item 相关的
        for pattern in (
            f"{prefix}:inflight:judge:*:{run_id}",
            f"{prefix}:inflight:attr:*:{run_id}",
        ):
            async for key in r.scan_iter(match=pattern, count=50):
                # 检查 key 的值是否属于 failed item
                val = await r.get(key)
                if val is None:
                    if key not in keys:
                        keys.append(key)
                # inflight key 的值可能是 item_id 或 JSON
                elif any(fid in (val or "") for fid in failed_ids):
                    if key not in keys:
                        keys.append(key)

        print("Inflight keys (failed-related):")
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


async def _load_failed_items_from_oss(
    storage_client: "StorageClient",
    run: BenchmarkRun,
    failed_ids: set[str],
) -> list[dict[str, Any]]:
    """从 OSS 重新加载原始数据，按 failed item_id 过滤。"""
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    oss_prefix = cfg.get("oss_prefix", "")
    date = cfg.get("data_date", "")

    if not oss_prefix or not date:
        print(
            "  WARN: oss_prefix or data_date missing in config_json; "
            "falling back to DB reconstruction",
            file=sys.stderr,
        )
        return []

    loader = DailyDataLoader(storage_client)
    try:
        parse_result = await loader.load_daily(
            run_id=run.run_id,
            oss_prefix=oss_prefix,
            date=date,
        )
    except Exception as exc:
        print(
            f"  WARN: OSS load failed: {exc}; falling back to DB reconstruction",
            file=sys.stderr,
        )
        return []

    filtered = [
        item
        for item in parse_result.items
        if str(item.get("item_id") or item.get("id") or "") in failed_ids
    ]
    print(
        f"  OSS loaded {len(parse_result.items)} items, filtered {len(filtered)} failed items",
    )
    return filtered


async def _reconstruct_from_db(
    run_id: str,
    failed_ids: set[str],
) -> list[dict[str, Any]]:
    """从 DB 的 input/output 字段重建 item payload（OSS 不可用时回退）。"""
    items: list[dict[str, Any]] = []
    ds = extract_ds(run_id)
    for item_id in sorted(failed_ids):
        row = await BenchmarkRunItem.filter(
            run_id=run_id,
            item_id=item_id,
            criterion_id=FUSED_CRITERION_ID,
        ).first()
        if row is None:
            print(f"  WARN: item {item_id} not found in DB, skipping", file=sys.stderr)
            continue
        payload = {
            "run_id": run_id,
            "item_id": item_id,
            "id": item_id,
            "eval_type": "external",
            "retry_count": 0,
            "ds": ds,
            "input": row.input if isinstance(row.input, dict) else {},
            "output": row.output if isinstance(row.output, dict) else {},
            "metadata": {},
            "case_id": item_id,
        }
        items.append(payload)
    print(f"  DB reconstruction: {len(items)} items")
    return items


async def _delete_failed_items(
    run_id: str,
    failed_ids: set[str],
    *,
    execute: bool,
) -> int:
    """从 benchmark_run_items 删除 failed items（含 fused + per-criterion 全部行）。"""
    qs = BenchmarkRunItem.filter(
        run_id=run_id,
        item_id__in=list(failed_ids),
    )
    count = await qs.count()
    print(f"  benchmark_run_items (failed): {count} row(s)")
    if not execute:
        print("  [dry-run] skip DELETE items")
        return 0
    deleted = await qs.delete()
    print(f"  Deleted {deleted} item row(s)")
    return deleted


async def _adjust_run_counters(
    run_id: str,
    failed_ids: set[str],
    *,
    execute: bool,
) -> dict[str, Any]:
    """调整 run config_json 计数器 + 重置 status。

    - judge_received_count -= N
    - failed_items -= N
    - 移除 attribution_received_item_ids 中 failed item_ids
    - 重算 attribution_received_count
    - status → running; reset finished_at / final_score / duration_ms
    """
    run = await BenchmarkRun.filter(run_id=run_id).first()
    if run is None:
        raise SystemExit(f"run not found: {run_id}")

    config = dict(run.config_json) if isinstance(run.config_json, dict) else {}
    old_judge_rcv = int(config.get("judge_received_count", 0) or 0)
    old_failed = int(run.failed_items or 0)
    old_attr_rcv = int(config.get("attribution_received_count", 0) or 0)
    old_attr_ids = [str(x) for x in (config.get("attribution_received_item_ids") or [])]

    new_judge_rcv = max(0, old_judge_rcv - len(failed_ids))
    new_failed = max(0, old_failed - len(failed_ids))

    # 移除 failed item_ids
    new_attr_ids = [x for x in old_attr_ids if x not in failed_ids]
    new_attr_rcv = len(new_attr_ids)

    print(f"  judge_received_count: {old_judge_rcv} → {new_judge_rcv}")
    print(f"  failed_items: {old_failed} → {new_failed}")
    print(f"  attribution_received_count: {old_attr_rcv} → {new_attr_rcv}")
    print(f"  attribution_received_item_ids: {len(old_attr_ids)} → {len(new_attr_ids)}")
    print(f"  status: {run.status} → {BenchmarkStatus.RUNNING.value}")

    if not execute:
        print("  [dry-run] skip counter update")
        return {
            "judge_received_count": new_judge_rcv,
            "failed_items": new_failed,
            "attribution_received_count": new_attr_rcv,
        }

    config["judge_received_count"] = new_judge_rcv
    config["attribution_received_count"] = new_attr_rcv
    config["attribution_received_item_ids"] = new_attr_ids

    await BenchmarkRun.filter(run_id=run_id).update(
        config_json=config,
        failed_items=new_failed,
        status=BenchmarkStatus.RUNNING.value,
        finished_at=None,
        final_score=0.0,
        duration_ms=None,
        completed_evaluators=0,
    )
    print("  Counters updated, status reset to running")
    return {
        "judge_received_count": new_judge_rcv,
        "failed_items": new_failed,
        "attribution_received_count": new_attr_rcv,
    }


async def _load_rerun_context(
    app: FastAPI,
    run_id: str,
) -> tuple[BenchmarkRun, set[str], list[dict[str, Any]]]:
    """加载 rerun 上下文：run 记录、failed item 集合与原始 item 数据。

    原始数据优先从 OSS 重新加载（加载后重放 run 快照挂载的拦截器恢复
    业务结构，§7.2），不可用时回退 DB 重建；无 failed item 时 items
    返回空列表（调用方短路退出）。
    """
    # 1. 读取 run 信息
    run = await BenchmarkRun.filter(run_id=run_id).first()
    if run is None:
        raise SystemExit(f"run not found: {run_id}")
    print(
        f"Target: run_id={run.run_id} status={run.status} "
        f"evaluated={run.evaluated_items} failed={run.failed_items} "
        f"total={run.total_samples}",
    )
    config = run.config_json if isinstance(run.config_json, dict) else {}
    print(f"  config_json keys: {list(config.keys())}")
    print(
        f"  judge_received={config.get('judge_received_count', 0)} "
        f"judge_expected={config.get('judge_expected_count', 0)} "
        f"attr_received={config.get('attribution_received_count', 0)} "
        f"attr_expected={config.get('attribution_expected_count', 0)}",
    )

    # 2. 查询 failed items（distinct item_id，对齐 external_runner 统计模式）
    failed_ids = set(
        str(x)
        for x in await BenchmarkRunItem.filter(
            run_id=run_id,
            status="failed",
        )
        .distinct()
        .values_list("item_id", flat=True)
    )
    _preview = ", ".join(sorted(failed_ids)[:_PREVIEW_COUNT])
    _suffix = "..." if len(failed_ids) > _PREVIEW_COUNT else ""
    print(f"\nFailed items: {len(failed_ids)} ({_preview}{_suffix})")

    if not failed_ids:
        print("\nNo failed items found. Nothing to rerun.")
        return run, failed_ids, []

    # 3. 从 OSS 加载原始数据（回退 DB 重建）
    print("\n--- Load original item data ---")
    runner = getattr(app.state, "external_benchmark_runner", None)
    storage_client = getattr(runner, "_storage_client", None) if runner else None

    items: list[dict[str, Any]] = []
    oss_loaded = False
    if storage_client is not None:
        items = await _load_failed_items_from_oss(
            storage_client,
            run,
            failed_ids,
        )
        oss_loaded = bool(items)
    if not items:
        print("  Falling back to DB reconstruction")
        items = await _reconstruct_from_db(run_id, failed_ids)
    # OSS 重载的全量 KV 需重放拦截器恢复业务结构（§7.2；DB 重建路径已含
    # input/output，无需重放）
    if oss_loaded:
        items = _replay_parse_interceptor(run, items)

    if not items:
        raise SystemExit(
            "No item data available (OSS + DB both empty) — cannot rerun",
        )

    print(f"  Rerun items: {len(items)}")
    return run, failed_ids, items


async def _amain(args: argparse.Namespace) -> int:
    settings = Settings()
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        # 1-3. 加载 run 信息 / failed items / 原始数据（OSS → DB 回退）
        run_id = args.run_id
        run, failed_ids, items = await _load_rerun_context(app, run_id)
        if not failed_ids:
            return 0

        qm = getattr(app.state, "eval_queue_manager", None)
        if qm is None:
            raise SystemExit("eval_queue_manager not initialized")

        # 4. Drain queues for failed items
        print(f"\n--- Drain queues for {len(failed_ids)} failed items ---")
        stats = await _drain_failed_jobs(qm, run_id, failed_ids, execute=args.execute)
        print(
            f"\nDrain: jobs_removed={stats['jobs_removed']} "
            f"slots_released={stats['slots_released']} "
            f"cluster_released={stats['cluster_released']}",
        )

        # 5. Clear inflight keys
        print("\n--- Clear inflight Redis keys ---")
        await _clear_inflight_keys(qm, run_id, failed_ids, execute=args.execute)

        # 6. Delete failed items from DB
        print("\n--- Delete failed benchmark_run_items ---")
        await _delete_failed_items(run_id, failed_ids, execute=args.execute)

        # 7. Adjust run counters + reset status
        print("\n--- Adjust run counters + reset status ---")
        await _adjust_run_counters(run_id, failed_ids, execute=args.execute)

        # 8. Re-enqueue failed items（按 run 的 queue_pair 路由，§8.3）
        print("\n--- Re-enqueue failed items (routed by queue_pair) ---")
        ds = extract_ds(run_id)
        print(f"  ds={ds} items={len(items)}")
        if not args.execute:
            print("  [dry-run] skip enqueue")
        else:
            enqueued = await _reenqueue_failed_cases(qm, run, items, ds)
            print(f"  Enqueued {enqueued} item(s)")

        if not args.execute:
            print("\n[dry-run] no changes made. Add --execute to apply.")
        else:
            print("\n✅ Done. Failed items re-enqueued; CAS will fire when reruns complete.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun failed items: delete failed items + adjust counters + re-enqueue",
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
