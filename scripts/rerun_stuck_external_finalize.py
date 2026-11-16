#!/usr/bin/env python3
"""运维：重跑卡在 aggregating / generating_report 的 ieval external finalize。

背景：旧版 finalize 回调错误调用 ``complete_run(run_id=...)``（缺参 TypeError），
CAS 已把 ``benchmark_runs.status`` 推到 ``aggregating`` 后收尾失败，状态卡死。

本脚本启动与线上一致的 app lifespan，调用 ``ExternalBenchmarkRunner._finalize_run``
补做：aggregating → generating_report → 聚合 → 报告 → completed。

用法（优先用包装脚本，线上会拼 agentgraph PYTHONPATH）::

  bash scripts/rerun_stuck_external_finalize.sh --list
  bash scripts/rerun_stuck_external_finalize.sh \\
    --run-id ieval_external-20260812-03 --finalize --dry-run
  bash scripts/rerun_stuck_external_finalize.sh \\
    --run-id ieval_external-20260812-03 --finalize
  bash scripts/rerun_stuck_external_finalize.sh \\
    --run-id ieval_external-20260812-03 --finalize --force
  bash scripts/rerun_stuck_external_finalize.sh \\
    --run-id ieval_external-20260812-03 --mark-failed

线上无新包装脚本时，手工对齐 bazel_docker_server.sh::

  export BAZEL_WORKSPACE_HOME=/home/admin/bazel-workspace
  export PYTHONPATH="\\
  $BAZEL_WORKSPACE_HOME/libs/agent_graph/src:\\
  $BAZEL_WORKSPACE_HOME/libs/data_models/py:\\
  $BAZEL_WORKSPACE_HOME/libs:\\
  $BAZEL_WORKSPACE_HOME/libs/storage/src:\\
  $BAZEL_WORKSPACE_HOME/services/agent-eval:\\
  ${HOME}/.local/lib/python3.*/site-packages"
  cd /home/admin/agent-eval-service   # 或 $BAZEL_WORKSPACE_HOME/services/agent-eval
  python3 scripts/rerun_stuck_external_finalize.py --run-id ... --mark-failed

可选：清理泄漏的 inflight 槽位（slot_busy 死循环）::

  bash scripts/rerun_stuck_external_finalize.sh \\
    --run-id ieval_external-20260812-03 --clear-inflight-slots --dry-run
"""

import argparse
import asyncio
import json
import socket
import sys
from typing import Any

import _bootstrap  # sys.path/.env/禁 MQ 副作用，须先于 app.* 执行
import redis.asyncio as aioredis
from benchmark.settings import Settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun
from benchmark.service.main import create_app
from fastapi import FastAPI

del _bootstrap  # 仅为副作用导入：已执行，释放模块名

_STUCK_STATUSES = (
    BenchmarkStatus.AGGREGATING.value,
    BenchmarkStatus.GENERATING_REPORT.value,
)


def _is_external(run: BenchmarkRun) -> bool:
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    return bool(cfg.get("external")) or str(run.run_id).startswith("ieval_external")


def _config_counters(run: BenchmarkRun) -> dict[str, Any]:
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    return {
        "judge_received": cfg.get("judge_received_count"),
        "judge_expected": cfg.get("judge_expected_count"),
        "attr_received": cfg.get("attribution_received_count"),
        "attr_expected": cfg.get("attribution_expected_count"),
        "attribution_provider": cfg.get("attribution_provider"),
        "external": cfg.get("external"),
    }


async def _list_stuck(*, limit: int) -> list[BenchmarkRun]:
    rows = (
        await BenchmarkRun.filter(status__in=list(_STUCK_STATUSES))
        .order_by(
            "-updated_at",
        )
        .limit(limit)
    )
    return [r for r in rows if _is_external(r)]


def _print_run(run: BenchmarkRun) -> None:
    counters = _config_counters(run)
    print(
        f"  run_id={run.run_id} status={run.status} "
        f"evaluated={run.evaluated_items} failed={run.failed_items} "
        f"total={run.total_samples} score={run.final_score} "
        f"counters={json.dumps(counters, ensure_ascii=False)}",
    )


async def _cmd_list(*, limit: int) -> int:
    rows = await _list_stuck(limit=limit)
    if not rows:
        print("No stuck external runs in aggregating/generating_report.")
        return 0
    print(f"Stuck external runs ({len(rows)}):")
    for run in rows:
        _print_run(run)
    return 0


async def _ensure_aggregating(run_id: str, *, force: bool) -> BenchmarkRun:
    run = await BenchmarkRun.filter(run_id=run_id).first()
    if run is None:
        raise SystemExit(f"run not found: {run_id}")
    if not _is_external(run):
        raise SystemExit(f"run_id={run_id} does not look like external/ieval")

    status = str(run.status)
    if status == BenchmarkStatus.COMPLETED.value:
        raise SystemExit(f"run already completed: {run_id}")
    if status == BenchmarkStatus.FAILED.value:
        raise SystemExit(f"run already failed: {run_id}")

    if status == BenchmarkStatus.AGGREGATING.value:
        return run

    if status == BenchmarkStatus.GENERATING_REPORT.value:
        if not force:
            raise SystemExit(
                f"run_id={run_id} status=generating_report; "
                "pass --force to reset to aggregating then finalize",
            )
        n = await BenchmarkRun.filter(
            run_id=run_id,
            status=BenchmarkStatus.GENERATING_REPORT.value,
        ).update(status=BenchmarkStatus.AGGREGATING.value)
        if n == 0:
            raise SystemExit(f"failed to reset generating_report → aggregating: {run_id}")
        print(f"[force] reset {run_id}: generating_report → aggregating")
        refreshed = await BenchmarkRun.filter(run_id=run_id).first()
        assert refreshed is not None
        return refreshed

    if status == BenchmarkStatus.RUNNING.value and force:
        # 计数已齐但未 CAS 时，允许强制走 finalize（仍由 _finalize_run 做 CAS）
        print("[force] status=running; will call _finalize_run (CAS Phase1+2)")
        return run

    raise SystemExit(
        f"unsupported status for finalize: {status} "
        f"(expect aggregating|generating_report[--force])",
    )


async def _cmd_finalize(
    *,
    run_id: str,
    force: bool,
    dry_run: bool,
    app: FastAPI,
) -> int:
    run = await _ensure_aggregating(run_id, force=force)
    print("Target run:")
    _print_run(run)
    if dry_run:
        print("[dry-run] skip ExternalBenchmarkRunner._finalize_run")
        return 0

    runner = getattr(app.state, "external_benchmark_runner", None)
    if runner is None:
        raise SystemExit("external_benchmark_runner not initialized on app.state")

    print(f"Calling ExternalBenchmarkRunner._finalize_run({run_id!r}) ...")
    await runner._finalize_run(run_id)
    after = await BenchmarkRun.filter(run_id=run_id).first()
    if after is None:
        print("WARN: run disappeared after finalize")
        return 1
    print(f"After: status={after.status} final_score={after.final_score}")
    if after.status != BenchmarkStatus.COMPLETED.value:
        print(
            "WARN: not completed — check logs for external.finalize_failed / "
            "finalize_election_lost",
            file=sys.stderr,
        )
        return 2
    print("OK: completed")
    return 0


async def _cmd_mark_failed(
    *,
    run_id: str,
    reason: str,
    dry_run: bool,
    app: FastAPI,
) -> int:
    run = await BenchmarkRun.filter(run_id=run_id).first()
    if run is None:
        raise SystemExit(f"run not found: {run_id}")
    print("Target run:")
    _print_run(run)
    if dry_run:
        print(f"[dry-run] skip fail_run reason={reason!r}")
        return 0
    svc = getattr(app.state, "benchmark_service", None)
    if svc is None:
        raise SystemExit("benchmark_service not initialized")
    await svc.fail_run(run_id=run_id, reason=reason)
    after = await BenchmarkRun.filter(run_id=run_id).first()
    print(f"After: status={after.status if after else None}")
    return 0


async def _cmd_clear_inflight(
    *,
    run_id: str,
    dry_run: bool,
    app: FastAPI,
) -> int:
    """删除 per-run / cluster judge inflight 计数键（确认无合法 pending 后使用）。"""
    qm = getattr(app.state, "eval_queue_manager", None)
    if qm is None:
        raise SystemExit("eval_queue_manager not initialized")
    prefix = qm.prefix
    host = socket.gethostname()
    keys = [
        f"{prefix}:inflight:cluster:judge",
        f"{prefix}:inflight:judge:{host}:{run_id}",
    ]
    # 扫描该 run 在其它节点上的 per-node key
    pattern = f"{prefix}:inflight:judge:*:{run_id}"
    r = aioredis.from_url(qm.redis_url, decode_responses=True)
    try:
        async for key in r.scan_iter(match=pattern, count=50):
            if key not in keys:
                keys.append(key)
        print("Inflight keys to clear:")
        for k in keys:
            val = await r.get(k)
            print(f"  {k} = {val!r}")
        if dry_run:
            print("[dry-run] skip DEL")
            return 0
        if keys:
            n = await r.delete(*keys)
            print(f"Deleted {n} key(s)")
        return 0
    finally:
        await r.aclose()


async def _amain(args: argparse.Namespace) -> int:
    settings = Settings()
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        if args.list:
            return await _cmd_list(limit=args.limit)
        if args.clear_inflight_slots:
            if not args.run_id:
                raise SystemExit("--clear-inflight-slots requires --run-id")
            return await _cmd_clear_inflight(
                run_id=args.run_id,
                dry_run=args.dry_run,
                app=app,
            )
        if args.mark_failed:
            if not args.run_id:
                raise SystemExit("--mark-failed requires --run-id")
            return await _cmd_mark_failed(
                run_id=args.run_id,
                reason=args.reason,
                dry_run=args.dry_run,
                app=app,
            )
        if args.finalize:
            if not args.run_id:
                raise SystemExit("--finalize requires --run-id")
            return await _cmd_finalize(
                run_id=args.run_id,
                force=args.force,
                dry_run=args.dry_run,
                app=app,
            )
        raise SystemExit("specify --list | --finalize | --mark-failed | --clear-inflight-slots")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun stuck ieval external finalize (aggregating)",
    )
    parser.add_argument("--list", action="store_true", help="list stuck external runs")
    parser.add_argument("--limit", type=int, default=50, help="--list max rows")
    parser.add_argument("--run-id", type=str, default=None, help="target run_id")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="call ExternalBenchmarkRunner._finalize_run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow generating_report→aggregating reset before finalize",
    )
    parser.add_argument(
        "--mark-failed",
        action="store_true",
        help="mark run failed instead of finalize",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="ops: stuck aggregating reaped",
        help="fail reason for --mark-failed",
    )
    parser.add_argument(
        "--clear-inflight-slots",
        action="store_true",
        help="DEL leaked Redis inflight keys for run (and cluster:judge)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print actions only")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
