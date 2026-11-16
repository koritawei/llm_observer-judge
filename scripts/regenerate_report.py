#!/usr/bin/env python3
"""运维：为所有 case 已跑完的 external run 生成/重生成日报（静态模板主路径）。

背景：``_finalize_run`` 的 Phase 5 报告生成可能因 webdev-agent 超时
或网络抖动失败；或 run 因 CAS 选举/状态机卡死停留在
``running`` / ``aggregating`` / ``generating_report``，但所有 case
实际已跑完（计数器满足）。此时无论当前 status，只要：

  judge_received_count >= judge_expected_count
  attribution_received_count >= attribution_expected_count

即可触发 report 过程，并在报告后将 status 更新为 ``completed``、
写入 ``report_meta_json``（走 ``complete_run`` 的 merge 语义）。

本脚本跳过 CAS 状态机（直接面向终态收尾），流程：
1. 从 PG 读取 run 记录 + config_json
2. 校验所有 case 已跑完（计数器满足）
3. ``_ensure_stat_dimensions_snapshot``（缺快照则 rubric_ref 回填）
4. 调用 ``ExternalBenchmarkRunner.publish_external_report``
   （与线上 ``_finalize_run`` Phase 4-6 **同一实现**：聚合 + 报告 + complete）

用法::

  # 列出所有 case 已跑完但无报告的 external run（不限 status）
  python3 scripts/regenerate_report.py --list

  # 为单个 run 生成/重生成报告（不限 status，case 跑完即可）
  python3 scripts/regenerate_report.py --run-id ieval_external-20260814-03

  # dry-run（只聚合，不生成报告 / complete_run）
  python3 scripts/regenerate_report.py --run-id ieval_external-20260814-03 --dry-run

线上运行需对齐 PYTHONPATH（同 rerun_stuck_external_finalize.sh）。
本地请用 ``services/agent-eval/.venv``（tortoise-orm>=1.1.7）；
系统/其它 venv 若装成 0.25.x 会缺 ``tortoise.context``，脚本无法启动。

报告生成与线上一致：默认静态 ``benchmark_report.html``；
仅模板加载失败或 ``AGENT_EVAL_FORCE_WEBDEV_REPORT=1`` 时回退 webdev skill。
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from typing import Any

import _bootstrap_quiet  # sys.path/.env/禁 MQ+healer+日调度，须先于 app.* 执行
from benchmark.settings import Settings
from benchmark.records.models import BenchmarkRun, BenchmarkRunItem
from benchmark.service.main import create_app
from benchmark.runs.common.config_helpers import (
    load_aggregation_from_config,
    load_rubric_from_config,
)
from benchmark.workers.external_runner import ExternalReportPublish
from fastapi import FastAPI

del _bootstrap_quiet  # 仅为副作用导入：已执行，释放模块名


def _is_external(run: BenchmarkRun) -> bool:
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    return bool(cfg.get("external")) or str(run.run_id).startswith("ieval_external")


def _has_report(run: BenchmarkRun) -> bool:
    meta = run.report_meta_json if isinstance(run.report_meta_json, dict) else {}
    return bool(meta.get("report_url"))


def _config_counters(run: BenchmarkRun) -> dict[str, Any]:
    """提取 config_json 中的完成计数器（对齐 try_finalize_external CAS）。"""
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    return {
        "judge_received": int(cfg.get("judge_received_count") or 0),
        "judge_expected": int(cfg.get("judge_expected_count") or 0),
        "attr_received": int(cfg.get("attribution_received_count") or 0),
        "attr_expected": int(cfg.get("attribution_expected_count") or 0),
    }


def _counters_done(run: BenchmarkRun) -> bool:
    """计数器口径：与 ``try_finalize_external`` CAS 条件一致。

    judge_received >= judge_expected 且 attribution_received >=
    attribution_expected；``attribution_expected_count`` 缺省 0
    （无归因任务时不阻塞完成）。
    """
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    judge_received = int(cfg.get("judge_received_count") or 0)
    judge_expected = int(cfg.get("judge_expected_count") or 0)
    attr_received = int(cfg.get("attribution_received_count") or 0)
    attr_expected = int(cfg.get("attribution_expected_count") or 0)
    return judge_received >= judge_expected and attr_received >= attr_expected


async def _all_cases_done(run: BenchmarkRun) -> bool:
    """检查所有 case 是否跑完：计数器满足，或数据侧兜底。

    兜底口径（存量 run 救援）：旧代码的 sync judge 无 worker 代位递增
    ``judge_received_count``（如 search_eval-20260906-b/c/d，计数器=0 但
    fused 行数据完整）——此时改查 PG 实际数据：终态行（success/failed）
    去重 item_id 数 >= judge_expected_count 即数据完整，可重生成报告。
    """
    if _counters_done(run):
        return True
    cfg = run.config_json if isinstance(run.config_json, dict) else {}
    judge_expected = int(cfg.get("judge_expected_count") or 0)
    if judge_expected <= 0:
        return False
    done_item_ids = (
        await BenchmarkRunItem.filter(
            run_id=run.run_id,
            status__in=["success", "failed"],
        )
        .distinct()
        .values_list("item_id", flat=True)
    )
    return len(done_item_ids) >= judge_expected


def _print_run(run: BenchmarkRun) -> None:
    meta = run.report_meta_json if isinstance(run.report_meta_json, dict) else {}
    counters = _config_counters(run)
    print(
        f"  run_id={run.run_id} status={run.status} "
        f"score={run.final_score} "
        f"report_url={meta.get('report_url', '(none)')} "
        f"counters={counters}",
    )


async def _list_missing_reports(*, limit: int) -> int:
    """列出所有 case 已跑完但无 report_url 的 external run（不限 status）。

    扫描非 failed 的 external run，筛选计数器或数据侧判定跑完
    （``_all_cases_done``）且尚未生成报告的记录。failed run 不纳入
    （避免把超时/崩溃的 run 误判为可补报告）。
    """
    rows = (
        await BenchmarkRun.exclude(
            status="failed",
        )
        .order_by("-updated_at")
        .limit(limit)
    )
    missing = []
    for r in rows:
        if not _is_external(r) or _has_report(r):
            continue
        if await _all_cases_done(r):
            missing.append(r)
    if not missing:
        print("No external runs with all cases done missing reports.")
        return 0
    print(f"External runs (all cases done) without report ({len(missing)}):")
    for run in missing:
        _print_run(run)
    return 0


async def _prepare_regenerate_run(
    run_id: str,
    app: FastAPI,
) -> tuple[BenchmarkRun, Any] | int:
    """加载并校验补报告目标：run 存在 / external / case 跑完 / runner 就绪。

    返回 ``(run, runner)`` 或错误退出码（错误信息已打印到 stderr）。
    """
    run = await BenchmarkRun.filter(run_id=run_id).first()
    if run is None:
        print(f"ERROR: run not found: {run_id}", file=sys.stderr)
        return 1
    if not _is_external(run):
        print(f"ERROR: {run_id} is not an external run", file=sys.stderr)
        return 1
    if not await _all_cases_done(run):
        print(
            f"ERROR: not all cases done for {run_id} "
            f"counters={_config_counters(run)} — wait for callbacks "
            f"or check stuck judge/attribution tasks.",
            file=sys.stderr,
        )
        return 1
    runner = getattr(app.state, "external_benchmark_runner", None)
    if runner is None:
        print("ERROR: external_benchmark_runner not initialized", file=sys.stderr)
        return 1
    config = run.config_json if isinstance(run.config_json, dict) else {}
    if not config:
        print("ERROR: config_json is empty — cannot rebuild rubric", file=sys.stderr)
        return 1
    return run, runner


async def _regenerate_report(
    *,
    run_id: str,
    dry_run: bool,
    app: FastAPI,
) -> int:
    """为单个 run 生成/重生成报告（不限 status，case 跑完即可）。

    无论当前 status（running / aggregating / generating_report / completed），
    只要 case 跑完（计数器满足或数据侧兜底，``_all_cases_done``）就触发：
    聚合 → 报告 → ``complete_run``（status=completed + report_meta_json merge）。
    """
    prepared = await _prepare_regenerate_run(run_id, app)
    if isinstance(prepared, int):
        return prepared
    run, runner = prepared

    print("Target run:")
    _print_run(run)

    config = run.config_json if isinstance(run.config_json, dict) else {}

    # 1. 重新聚合指标（从已有 benchmark_run_items 读取，不重跑评测）
    rubric = load_rubric_from_config(config)
    aggregation = load_aggregation_from_config(config)

    print(f"  rubric: {len(rubric)} criteria, method={aggregation.method}")
    print("  ensuring stat_dimensions snapshot + aggregating ...")

    config = await runner._ensure_stat_dimensions_snapshot(run_id, config)

    if dry_run:
        final_score, metrics = await runner._aggregate_external_metrics(
            run_id,
            rubric,
            aggregation,
        )
        print(f"  final_score={final_score:.4f}")
        print("[dry-run] skipping report generation and complete_run")
        crit_keys = list(getattr(metrics, "criterion_metrics", {}) or {})
        attr_keys = list(getattr(metrics, "attribution_metrics", {}) or {})
        print(f"  criterion_metrics keys: {crit_keys}")
        print(f"  attribution_metrics keys: {attr_keys}")
        return 0

    # 2-3. 与线上 finalize Phase4-6 共用 publish_external_report
    print("  calling publish_external_report (static template primary; webdev skill fallback) ...")
    report_url = await runner.publish_external_report(
        ExternalReportPublish(
            run_id=run_id,
            config=config,
            rubric=rubric,
            aggregation=aggregation,
        ),
    )

    if report_url is None:
        print(
            "WARN: report generation returned no report_url; "
            "run marked completed without report (report_meta_json preserved)",
            file=sys.stderr,
        )
        return 2

    # regenerate 额外打 regenerated_at（merge 进 report_meta）
    now_iso = datetime.now(UTC).isoformat()
    await runner._benchmark_service.merge_report_meta(
        run_id,
        patch={"regenerated_at": now_iso},
    )

    print(f"OK: report_url={report_url}")
    print(f"     status=completed, report_meta_json updated for run_id={run_id}")
    return 0


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate/regenerate benchmark report for an external run "
        "whose all cases are done (regardless of status), then mark completed.",
    )
    parser.add_argument("--list", action="store_true", help="List runs done but missing reports")
    parser.add_argument("--run-id", type=str, help="Run ID to generate/regenerate report for")
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip report generation + complete_run"
    )
    parser.add_argument("--limit", type=int, default=50, help="Max runs to list (default 50)")
    args = parser.parse_args()

    if not args.list and not args.run_id:
        parser.error("either --list or --run-id is required")

    # Initialize app lifespan (same pattern as rerun_stuck_external_finalize.py)
    app = create_app(settings=Settings())
    async with app.router.lifespan_context(app):
        if args.list:
            return await _list_missing_reports(limit=args.limit)
        return await _regenerate_report(
            run_id=args.run_id,
            dry_run=args.dry_run,
            app=app,
        )


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
