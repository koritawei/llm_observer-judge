#!/usr/bin/env python3
"""运维/验证：trace 日级报表（T+1）独立链路 dry-run 测试脚本。

链路覆盖（与线上 ``TraceDailyReportScheduler._run_item`` 同一实现）：
RemoteConfig 配置 → rubric 解析（``rubric_class`` 别名优先 / ``benchmark_name``
组件元数据回退）→ T-1 窗口采样 ``trace_pipeline_records`` → per-criterion
聚合 + 归因分布 → （--execute 时）占位 run + 公用 report 出报 + complete_run。

用法（services/agent-eval 目录下，.venv 解释器）::

    # dry-run 全部 RemoteConfig 报表项（默认）：打印 rubric / 采样数 / 聚合摘要，
    # 不建 run、不出报——验证配置与数据是否就绪的安全姿势
    python3 scripts/trace_daily_report_dryrun.py

    # 指定单个报表项 + 指定数据日期（默认昨日，按 item 时区）
    python3 scripts/trace_daily_report_dryrun.py --name amap_agent_daily --date 20260922

    # 本地配置文件替代 RemoteConfig（验收新配置前先行验证）
    python3 scripts/trace_daily_report_dryrun.py \
      --config-json docs/remote_config-trace-daily-report-config-example.json

    # 真实执行：建 run + 出报 + complete_run（幂等——同名 run 存在即跳过，
    # 与调度器线上行为一致；failed / 需重出报时加 --force 先删再跑）
    python3 scripts/trace_daily_report_dryrun.py --name amap_agent_daily --execute
    python3 scripts/trace_daily_report_dryrun.py \
      --name amap_agent_daily --date 20260929 --execute --force

线上运行需对齐 PYTHONPATH（同 regenerate_report.py）；本地请用
``services/agent-eval/.venv``（tortoise-orm>=1.1.7）。
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _bootstrap_quiet  # sys.path/.env/禁 MQ+healer+日调度，须先于 app.* 执行
from benchmark.settings import Settings
from benchmark.records.models import BenchmarkRun, BenchmarkRunItem
from benchmark.domain.trace_report_config import (
    TraceReportConfig,
    TraceReportItemConfig,
    fetch_trace_report_config,
)
from benchmark.service.main import create_app
from benchmark.runs.run_helpers import TraceRecordSampleQuery
from benchmark.workers.trace_report_scheduler import (
    TraceDailyReportScheduler,
    TraceReportSchedulerDeps,
)

if TYPE_CHECKING:
    from datetime import date as date_t
    from datetime import tzinfo

del _bootstrap_quiet  # 仅为副作用导入：已执行，释放模块名


def _load_config(config_json: str | None) -> TraceReportConfig:
    """--config-json 指定时从本地文件加载；否则拉 RemoteConfig（带缓存）。"""
    if not config_json:
        return fetch_trace_report_config()
    raw = json.loads(Path(config_json).read_text(encoding="utf-8"))
    return TraceReportConfig.model_validate(raw)


def _resolve_data_date(
    scheduler: TraceDailyReportScheduler,
    item: TraceReportItemConfig,
    date_arg: str | None,
) -> "tuple[date_t, tzinfo]":
    """数据日期：--date 指定（yyyymmdd）或昨日（item 时区 T-1，与线上一致）。"""
    tz = scheduler._item_tz(item)
    if date_arg:
        parsed = datetime.strptime(date_arg, "%Y%m%d").date()  # noqa: DTZ007 — 脚本入参
        return parsed, tz
    return datetime.now(tz).date() - timedelta(days=1), tz


async def _force_delete_run(run_id: str) -> None:
    """``--force``：删除同名 run / items，打破幂等以便重出报。

    日报链路通常不写 ``benchmark_run_items``；仍按 run_id 清 items，
    与 ``delete_run_data.py`` 口径一致。无 run 时仅打印提示。
    """
    run = await BenchmarkRun.filter(run_id=run_id).first()
    if run is None:
        print(f"  force: no existing run {run_id!r} (nothing to delete)")
        return
    deleted_items = await BenchmarkRunItem.filter(run_id=run_id).delete()
    deleted_run = await BenchmarkRun.filter(run_id=run_id).delete()
    print(
        f"  force: deleted run_id={run_id} status={run.status} "
        f"items={deleted_items} runs={deleted_run}",
    )


async def _dry_run_item(
    scheduler: TraceDailyReportScheduler,
    item: TraceReportItemConfig,
    *,
    data_date: "date_t",
    tz: "tzinfo",
) -> int:
    """dry-run：rubric 解析 + 采样 + 聚合摘要打印；不建 run、不出报。"""
    service = scheduler._benchmark_service

    # 1. rubric 解析（rubric_class 优先 / benchmark_name 回退，与线上一致）
    resolved = scheduler._resolve_rubric_aggregation(item)
    if resolved is None:
        print(f"ERROR: rubric resolve failed for report {item.name!r}", file=sys.stderr)
        return 1
    rubric, aggregation, rubric_ref, service_name = resolved
    print(
        f"[{item.name}] rubric via {'rubric_class=' + item.rubric_class if rubric_ref else 'benchmark_name=' + item.benchmark_name}"
    )
    if rubric_ref:
        print(f"  rubric_ref: {rubric_ref}")
    print(f"  service_name: {service_name}  pass_threshold: {aggregation.pass_threshold}")
    for c in rubric:
        print(
            f"  criterion: id={c.id} name={c.name} weight={c.weight} evaluator_ref={c.evaluator_ref}"
        )

    # 2. T-1 窗口采样
    window_start = datetime(data_date.year, data_date.month, data_date.day, tzinfo=tz)
    window_end = window_start + timedelta(days=1)
    trace_ids = await service.sample_trace_pipeline_records(
        TraceRecordSampleQuery(
            date_start=window_start,
            date_end=window_end,
            evaluator_names=[c.evaluator_ref for c in rubric],
            source=item.source,
            agent_name=item.agent_name,
            project_id=item.project_id,
            sample_size=item.sample_size,
        ),
    )
    print(
        f"  window: [{window_start.isoformat()}, {window_end.isoformat()}) "
        f"source={item.source} agent={item.agent_name} project={item.project_id}",
    )
    print(f"  sampled traces: {len(trace_ids)}")
    if not trace_ids:
        print("  (no data — 线上此时记 warning 跳过，不建 run)")
        return 0

    # 3. 聚合 + 归因分布摘要
    entries = await service.load_kafka_criterion_entries(
        trace_ids=trace_ids,
        rubric=rubric,
        pass_threshold=aggregation.pass_threshold,
    )
    final_score, _per, _metrics = await scheduler._aggregation.aggregate(
        aggregation,
        rubric,
        entries,
    )
    print(f"  final_score: {final_score:.4f}")
    for entry in entries:
        stats = entry.criterion_stats_json or {}
        print(
            f"  criterion {entry.criterion_id}: n={entry.sample_count} "
            f"success={entry.success_count} failed={entry.failed_count} "
            f"mean={stats.get('mean', 0):.4f} pass_rate={stats.get('pass_rate', 0):.4f}",
        )
    attr_by_cid = await service.aggregate_kafka_attribution(
        trace_ids=trace_ids,
        rubric=rubric,
    )
    for cid, attr in attr_by_cid.items():
        total = attr.get("total_attributed", 0)
        distributions: dict[str, Any] = attr.get("field_distributions", {})
        print(f"  attribution[{cid}]: total={total}")
        for field, dist in distributions.items():
            top = sorted(dist.items(), key=lambda kv: -kv[1])[:3]
            print(f"    {field}: {top}")
    print("  dry-run OK（未建 run / 未出报；--execute 走完整链路）")
    return 0


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="trace 日级报表（T+1）链路 dry-run：rubric 解析 + 采样 + 聚合摘要；"
        "--execute 走完整链路（建 run + 出报，幂等）；"
        "--force 需配合 --execute，先删同名 run 再重跑。",
    )
    parser.add_argument("--name", type=str, help="只跑指定 name 的报表项（默认全部）")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="数据日期 yyyymmdd（默认昨日，按 item 时区 T-1）",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="本地配置文件替代 RemoteConfig（如 docs/remote_config-trace-daily-report-config-example.json）",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真实执行（建 run + 出报 + complete_run；幂等）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="配合 --execute：删除同名 run（及 items）后再出报，用于重跑已完成日报",
    )
    args = parser.parse_args()

    if args.force and not args.execute:
        print("ERROR: --force requires --execute", file=sys.stderr)
        return 1

    cfg = _load_config(args.config_json)
    if not cfg.reports:
        print(
            "ERROR: no reports configured (RemoteConfig trace-daily-report empty or disabled)",
            file=sys.stderr,
        )
        return 1
    items = [it for it in cfg.reports if not args.name or it.name == args.name]
    if not items:
        names = ", ".join(it.name for it in cfg.reports)
        print(f"ERROR: report {args.name!r} not in config (available: {names})", file=sys.stderr)
        return 1

    app = create_app(settings=Settings())
    async with app.router.lifespan_context(app):
        scheduler = TraceDailyReportScheduler(
            TraceReportSchedulerDeps(
                benchmark_service=app.state.benchmark_service,
                aggregation_calculator=app.state.benchmark_runner._aggregation,
                webdev=app.state.benchmark_runner._webdev if args.execute else None,
                pages_publisher=(
                    app.state.benchmark_runner._pages_publisher if args.execute else None
                ),
                config=cfg,
            ),
        )
        rc = 0
        for item in items:
            data_date, tz = _resolve_data_date(scheduler, item, args.date)
            run_id = f"{item.name}-{data_date.strftime('%Y%m%d')}"
            mode = "EXECUTE"
            if args.execute and args.force:
                mode = "EXECUTE+FORCE"
            elif not args.execute:
                mode = "dry-run"
            print(f"== report {run_id} ({mode}) ==")
            if args.execute:
                if args.force:
                    await _force_delete_run(run_id)
                await scheduler._run_item(item, data_date=data_date, tz=tz)
            else:
                rc |= await _dry_run_item(scheduler, item, data_date=data_date, tz=tz)
        return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
