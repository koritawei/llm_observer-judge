#!/usr/bin/env python3
"""Ops: overwrite the same-day official run_id with the -a1 run.

1. Delete rows in ``benchmark_run_items`` / ``benchmark_runs`` whose
   ``run_id = ieval_external-20260921``.
2. Rename every row whose ``run_id = ieval_external-20260921-a1`` to
   ``ieval_external-20260921``.

Both steps are one transaction: if the source run does not exist the whole change rolls back, and the target is not deleted alone.
Only the ``run_id`` column changes; old ids embedded in JSON are left alone, and Redis / OSS are not touched.

Dry-run by default; it writes the database only with ``--execute``.

Usage (prefer the wrapper script; production prepends agentgraph to PYTHONPATH)::

  bash scripts/reassign_benchmark_run.sh
  bash scripts/reassign_benchmark_run.sh --execute
  bash scripts/reassign_benchmark_run.sh \\
    --source ieval_external-20260921-a1 \\
    --target ieval_external-20260921 --execute
"""

import argparse
import asyncio
import sys

import _bootstrap  # sys.path/.env/disable-MQ side effects must run before app.*
from benchmark.settings import Settings
from benchmark.records.models import BenchmarkRun, BenchmarkRunItem
from benchmark.service.main import create_app
from tortoise.transactions import in_transaction

del _bootstrap

_SOURCE_RUN_ID = "ieval_external-20260921-a1"
_TARGET_RUN_ID = "ieval_external-20260921"


async def _counts(run_id: str) -> tuple[int, int]:
    """Return ``(run row count, item row count)``."""
    runs = await BenchmarkRun.filter(run_id=run_id).count()
    items = await BenchmarkRunItem.filter(run_id=run_id).count()
    return runs, items


def _print_counts(label: str, run_id: str, runs: int, items: int) -> None:
    print(f"  {label} run_id={run_id} runs={runs} items={items}")


async def _apply(source: str, target: str) -> tuple[int, int, int, int]:
    """Delete target, then rename source to target. Return the row counts affected by the four steps."""
    async with in_transaction():
        deleted_items = await BenchmarkRunItem.filter(run_id=target).delete()
        deleted_runs = await BenchmarkRun.filter(run_id=target).delete()
        renamed_items = await BenchmarkRunItem.filter(run_id=source).update(run_id=target)
        renamed_runs = await BenchmarkRun.filter(run_id=source).update(run_id=target)
    return deleted_items, deleted_runs, renamed_items, renamed_runs


async def _amain(args: argparse.Namespace) -> int:
    source = args.source
    target = args.target
    if source == target:
        print("source 与 target 相同，拒绝执行", file=sys.stderr)
        return 2

    settings = Settings()
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        src_runs, src_items = await _counts(source)
        dst_runs, dst_items = await _counts(target)
        print("Before:")
        _print_counts("source", source, src_runs, src_items)
        _print_counts("target", target, dst_runs, dst_items)

        if src_runs == 0:
            print(f"\nsource run 不存在: {source}，未做任何删除", file=sys.stderr)
            return 1

        print("\nPlan:")
        print(f"  DELETE items/runs where run_id={target} ({dst_items}/{dst_runs})")
        print(f"  UPDATE items/runs {source} -> {target} ({src_items}/{src_runs})")

        if not args.execute:
            print("\n[dry-run] no changes made. Add --execute to apply.")
            return 0

        deleted_items, deleted_runs, renamed_items, renamed_runs = await _apply(
            source,
            target,
        )
        print("\nApplied:")
        print(f"  deleted target items={deleted_items} runs={deleted_runs}")
        print(f"  renamed source items={renamed_items} runs={renamed_runs}")

        left_src_runs, left_src_items = await _counts(source)
        now_runs, now_items = await _counts(target)
        print("After:")
        _print_counts("source", source, left_src_runs, left_src_items)
        _print_counts("target", target, now_runs, now_items)
        if left_src_runs or left_src_items or now_runs != src_runs or now_items != src_items:
            print("结果与预期不一致，请人工核对", file=sys.stderr)
            return 1
        print("\nDone.")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete target run_id rows, then rename source run_id onto it",
    )
    parser.add_argument("--source", default=_SOURCE_RUN_ID, help="被改名的 run_id")
    parser.add_argument("--target", default=_TARGET_RUN_ID, help="先清空、再写入的 run_id")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正写库（默认只打印计划）",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
