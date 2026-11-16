#!/usr/bin/env python3
"""Ops: delete runs / run_items data for a given run_id (two PG tables, no re-trigger).

Steps:
1. Read the run from PG (exit with an error if it does not exist)
2. Count benchmark_run_items rows (filtered by run_id)
3. With --execute, delete item rows, then delete the runs row (both WHERE run_id)

Difference from rerun.sh: PG delete only, **does not clear the queue / Redis / inflight slots, and does not re-trigger**.
If the run is still active (initializing/accepting/running/aggregating/generating_report),
an in-flight worker callback may recreate the row after the delete — terminate first, then delete:

  bash scripts/terminate_run.sh --run-id <run_id> --execute

Guards:
- DB-RDB-003: every DELETE must have a WHERE (this script filters by run_id);
  a single operation must touch < 10000 rows, and at the limit it refuses to run (batch it or use a DMS change ticket).
- Dry-run by default; it really runs only with ``--execute``.

Usage (prefer the wrapper script; production prepends agentgraph to PYTHONPATH)::

  bash scripts/delete_run_data.sh --run-id ieval_external-20260819
  bash scripts/delete_run_data.sh --run-id ieval_external-20260819 --execute
  bash scripts/delete_run_data.sh --run-id ieval_external-20260819 --execute --force
"""

import argparse
import asyncio
import sys

import _bootstrap  # sys.path/.env/disable-MQ side effects must run before app.*
from benchmark.settings import Settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models import BenchmarkRun, BenchmarkRunItem
from benchmark.service.main import create_app

del _bootstrap  # imported only for side effects: already executed, release the module name

#: DB-RDB-003: a single operation must touch fewer than 10000 rows; at the limit, refuse the delete.
_MAX_DELETE_ROWS = 10000

#: Non-terminal states: deleting in these states races in-flight worker callbacks (rows may be recreated).
_ACTIVE_STATUSES = (
    BenchmarkStatus.INITIALIZING.value,
    BenchmarkStatus.ACCEPTING.value,
    BenchmarkStatus.RUNNING.value,
    BenchmarkStatus.AGGREGATING.value,
    BenchmarkStatus.GENERATING_REPORT.value,
)


async def _delete_run_data(run_id: str, *, execute: bool, force: bool) -> int:
    """Delete benchmark_run_items + benchmark_runs rows by run_id."""
    run = await BenchmarkRun.filter(run_id=run_id).first()
    if run is None:
        print(f"ERROR: run not found: {run_id}", file=sys.stderr)
        return 1

    items_count = await BenchmarkRunItem.filter(run_id=run_id).count()
    print(
        f"Target: run_id={run.run_id} status={run.status} "
        f"evaluated={run.evaluated_items} failed={run.failed_items} "
        f"total={run.total_samples}",
    )
    print(f"  benchmark_run_items: {items_count} row(s) to delete")
    print("  benchmark_runs:      1 row(s) to delete")

    if run.status in _ACTIVE_STATUSES and not force:
        print(
            f"ERROR: run status={run.status!r} 为活动态，删除后在途回调可能重建行。\n"
            f"  建议先终止: bash scripts/terminate_run.sh --run-id {run_id} --execute\n"
            "  确认仍要删除请加 --force",
            file=sys.stderr,
        )
        return 1

    if items_count >= _MAX_DELETE_ROWS:
        print(
            f"ERROR: items 行数 {items_count} >= {_MAX_DELETE_ROWS}"
            "（DB-RDB-003 单次操作行数上限），请分批删除或走 DMS 变更单",
            file=sys.stderr,
        )
        return 1

    if not execute:
        print("\n[dry-run] no changes made. Add --execute to apply.")
        return 0

    deleted_items = await BenchmarkRunItem.filter(run_id=run_id).delete()
    print(f"  Deleted {deleted_items} item row(s) from benchmark_run_items")
    deleted_run = await BenchmarkRun.filter(run_id=run_id).delete()
    print(f"  Deleted {deleted_run} run row(s) from benchmark_runs")
    print("\nDone.")
    return 0


async def _amain(args: argparse.Namespace) -> int:
    settings = Settings()
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        return await _delete_run_data(
            args.run_id,
            execute=args.execute,
            force=args.force,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete benchmark_runs / benchmark_run_items "
        "rows for a run_id (PG only; no queue/Redis cleanup, no re-trigger)",
    )
    parser.add_argument("--run-id", type=str, required=True, help="target run_id")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually execute (default: dry-run)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow deleting a non-terminal (active) run",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
