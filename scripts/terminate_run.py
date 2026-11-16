#!/usr/bin/env python3
"""运维：终止 run 并释放 judge + attribution 全部槽位。

操作：
1. 扫描 6 个队列（EXTERNAL_RUNNING / BENCHMARK_RUNNING /
   ATTRIBUTION_RUNNING / EXTERNAL_WAIT / BENCHMARK_CASES /
   ATTRIBUTION_WAIT）删除该 run 的所有 job
   （§8.3 队列归并：统一 run 的 judge job 驻留 benchmark-cases/running）
2. 释放每个 job 持有的 per-node slot + cluster slot（judge / attr）
3. 直接 DEL 泄漏的 inflight Redis 键（per-node + cluster）
4. PG fail_run

默认 dry-run；加 ``--execute`` 才真正执行。

用法（优先包装脚本，线上会拼 agentgraph PYTHONPATH）::

  bash scripts/terminate_run.sh --run-id ieval_external-20260812-34
  bash scripts/terminate_run.sh --run-id ieval_external-20260812-34 --execute
"""

import argparse
import asyncio
import sys
from typing import Any

import _bootstrap  # sys.path/.env/禁 MQ 副作用，须先于 app.* 执行
import redis.asyncio as aioredis
from benchmark.settings import Settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun
from benchmark.libs.jobqueue.job_listing import list_queue_jobs
from benchmark.service.main import create_app
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
)

del _bootstrap  # 仅为副作用导入：已执行，释放模块名

#: 包含 active，确保 drain 时清理正在处理的 job
_ALL_JOB_TYPES = ["waiting", "delayed", "active", "paused", "prioritized"]

#: (队列名, 槽位 kind)；kind 为空表示 wait 队列不持槽
#: §8.3 队列归并：统一 run 的 judge job 驻留 benchmark-cases/running
_QUEUE_KIND_MAP = (
    (EXTERNAL_RUNNING, "judge"),
    (BENCHMARK_RUNNING, "judge"),
    (ATTRIBUTION_RUNNING, "attr"),
    (EXTERNAL_WAIT, ""),
    (BENCHMARK_CASES, ""),
    (ATTRIBUTION_WAIT, ""),
)

_TERMINAL_STATUSES = frozenset(
    {
        BenchmarkStatus.COMPLETED.value,
        BenchmarkStatus.FAILED.value,
    }
)


async def _drain_run(
    qm: Any,
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
    qm: Any,
    run_id: str,
    *,
    execute: bool,
) -> int:
    """直接 DEL 泄漏的 inflight Redis 键（per-node + cluster）。"""
    prefix = qm.prefix
    r = aioredis.from_url(qm.redis_url, decode_responses=True)
    keys: list[str] = []
    try:
        # cluster slot counters
        for kind in ("judge", "attr"):
            keys.append(f"{prefix}:inflight:cluster:{kind}")
        # per-node slot keys for this run (any node, any kind)
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


async def _amain(args: argparse.Namespace) -> int:
    settings = Settings()
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        run_id = args.run_id

        # 1. Show run status
        run = await BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            raise SystemExit(f"run not found: {run_id}")
        print(
            f"Target: run_id={run.run_id} status={run.status} "
            f"evaluated={run.evaluated_items} failed={run.failed_items} "
            f"total={run.total_samples}",
        )
        if run.status in _TERMINAL_STATUSES:
            print(f"  (run already {run.status}, will still clean queues/slots)")

        qm = getattr(app.state, "eval_queue_manager", None)
        if qm is None:
            raise SystemExit("eval_queue_manager not initialized")
        svc = getattr(app.state, "benchmark_service", None)
        if svc is None:
            raise SystemExit("benchmark_service not initialized")

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

        # 4. PG fail_run
        print("\n--- PG fail_run ---")
        if run.status in _TERMINAL_STATUSES:
            print(f"  skip fail_run (run already {run.status})")
        elif args.execute:
            await svc.fail_run(run_id=run_id, reason=args.reason)
            after = await BenchmarkRun.filter(run_id=run_id).first()
            print(f"After: status={after.status if after else None}")
        else:
            print(f"[dry-run] skip fail_run reason={args.reason!r}")

        if not args.execute:
            print("\n[dry-run] no changes made. Add --execute to apply.")
        else:
            print("\n✅ Done.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminate run and release all judge+attr slots",
    )
    parser.add_argument("--run-id", type=str, required=True, help="target run_id")
    parser.add_argument(
        "--reason",
        type=str,
        default="ops: manual terminate + slot release",
        help="fail reason",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually execute (default: dry-run)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
