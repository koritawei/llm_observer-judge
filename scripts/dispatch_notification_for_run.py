#!/usr/bin/env python3
"""运维：对指定 run 已完成归因的 case 批量补推 Aone 工单。

背景：Aone 工单派发挂在归因回调链路内（``schedule_notification_dispatch``），
可能因当时派单配置缺失（assignee 空 / unknown 拦截等历史规则）、
回调异常或派发侧故障而漏派。本脚本**不重跑 judge / 归因**，直接读取
``benchmark_run_items`` 已落盘的归因结果（fused 行
``attribution_detail_json``），复用线上同一条派发编排补推工单：

  分数门槛判定（should_dispatch_attribution）
    → 映射（map_attribution_to_result）
    → create_aone_task（a1 CLI）
    → 结果写回 ``notification_dispatch_json`` bucket

幂等：已有 ``notification_dispatch_json.notification_ref``（工单已建）的行跳过，
防重复建单；只有 ``notification_skip_reason``（如 score_threshold / 历史
拦截原因）的行重新参与判定 — 配置补齐后即可补推。

用法::

  # dry-run（复现完整判定链：门槛 + profiles + RemoteConfig 目标预检，不调 a1 CLI）
  python3 scripts/dispatch_notification_for_run.py --run-id ieval_external-20260820-01 --dry-run

  # 真补推
  python3 scripts/dispatch_notification_for_run.py --run-id ieval_external-20260820-01

线上运行需 a1 CLI 登录态（同 rerun 系列）；PYTHONPATH 对齐走
dispatch_aone_for_run.sh。
"""

import argparse
import asyncio
import sys
from collections import Counter
from typing import Any

import _bootstrap_quiet  # sys.path/.env/禁 MQ+healer+日调度，须先于 app.* 执行
from benchmark.settings import Settings
from benchmark.records.models import BenchmarkRun, BenchmarkRunItem
from benchmark.service.main import create_app
from benchmark.attribution.badcase.dispatch_policy import (
    resolve_dispatch_policy,
)
BENCHMARK_DISPATCH_BUSINESS = None
class DispatchDecision:
    pass
DispatchDecisionInput = None
DispatchRequest = None
dispatch_notification_for_attribution = None
parse_notification_dispatch_config = None
resolve_dispatch_decision = None
from benchmark.runs.run_service import BenchmarkService
from benchmark.runs.schema import (
    FUSED_CRITERION_ID,
    AttributionResult,
)
from fastapi import FastAPI

del _bootstrap_quiet  # 仅为副作用导入：已执行，释放模块名


def _as_dict(value: object) -> dict[str, Any]:
    """JSONField 列 → dict（None / 非法值归一空 dict）。"""
    return value if isinstance(value, dict) else {}


async def _dry_run_dispatch(
    *,
    benchmark_service: BenchmarkService,
    run_id: str,
    config: dict[str, Any],
    rows: list[BenchmarkRunItem],
) -> int:
    """dry-run：复现完整判定链，逐条展示 verdict，不调用 a1 CLI。

    判定与线上编排同源（resolve_dispatch_decision，含 profiles / RemoteConfig
    目标解析）；BLOCKED 预检（project / assignee 缺失）对齐
    create_aone_task 校验 — verdict 与真补推结果一致。
    """
    options, minor_dispatch = parse_notification_dispatch_config(config)
    service_name = await benchmark_service.get_service_name(run_id)
    policy = resolve_dispatch_policy(service_name)

    verdicts: Counter[str] = Counter()
    target: DispatchDecision | None = None
    for row in rows:
        fused_score = await benchmark_service.get_fused_score(run_id, row.item_id)
        # 误判 case（attribution_detail_json.is_eval_misjudge=true）不补推，
        # 与线上判定同源（should_dispatch_attribution 误判拦截）。
        misjudge = bool(_as_dict(row.attribution_detail_json).get("is_eval_misjudge"))
        decision = await resolve_dispatch_decision(
            DispatchDecisionInput(
                options=options,
                business=BENCHMARK_DISPATCH_BUSINESS,
                minor_dispatch=minor_dispatch,
                fused_score=fused_score,
                policy=policy,
                is_eval_misjudge=misjudge,
                # external 补推路径 map 后 should_dispatch 恒 True
                attribution_should_dispatch=True,
                config=config,
            )
        )
        if not decision.should:
            verdict = f"skip({decision.skip_reason})"
        elif not decision.project:
            verdict = "blocked(project_missing)"
        elif not decision.assignee:
            verdict = "blocked(assignee_missing)"
        else:
            verdict = "dispatch"
            target = decision
        verdicts[verdict] += 1
        print(f"  item={row.item_id} score={fused_score} misjudge={misjudge} → {verdict}")

    # 过门槛行的推送目标（profiles → RemoteConfig 查表，与线上同源）
    if target is not None:
        print(
            f"[dry-run] target profile={target.profile_key} "
            f"project={target.project} assignee={target.assignee} "
            f"participants={list(target.participants)}"
        )
    print(f"[dry-run] summary: {dict(verdicts)}")
    print("[dry-run] no workitem created; rerun without --dry-run to dispatch")
    return 0


async def _dispatch_for_run(
    *,
    run_id: str,
    dry_run: bool,
    app: FastAPI,
) -> int:
    """补推入口：扫描 fused 归因行 → 已建单跳过 → 其余走线上派发编排。"""
    error: str | None = None
    run = await BenchmarkRun.filter(run_id=run_id).first()
    if run is None:
        error = f"run not found: {run_id}"
    runner = getattr(app.state, "external_benchmark_runner", None)
    benchmark_service = getattr(runner, "_benchmark_service", None) if runner else None
    if error is None and benchmark_service is None:
        error = "external_benchmark_runner not initialized"

    config = _as_dict(run.config_json) if run else {}
    if error is None and not config:
        error = "config_json is empty — cannot resolve aone dispatch config"
    if error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    rows = await BenchmarkRunItem.filter(
        run_id=run_id,
        criterion_id=FUSED_CRITERION_ID,
    ).order_by("item_id")

    # 只处理归因已落盘的行（attribution_detail_json 非空 = 归因回调已完成）
    candidates = [r for r in rows if _as_dict(r.attribution_detail_json)]
    if not candidates:
        print(
            f"ERROR: no fused attribution rows found for {run_id} — "
            "attribution must complete before dispatching",
            file=sys.stderr,
        )
        return 1

    already = [r for r in candidates if _as_dict(r.notification_dispatch_json).get("notification_ref")]
    pending = [r for r in candidates if not _as_dict(r.notification_dispatch_json).get("notification_ref")]

    print(f"Target run: run_id={run_id} status={run.status}")
    print(f"  fused rows={len(rows)} with_attribution={len(candidates)}")
    print(f"  already_dispatched={len(already)} pending={len(pending)}")

    if not pending:
        print("OK: nothing to dispatch (all attribution rows already have workitems)")
        return 0

    if dry_run:
        return await _dry_run_dispatch(
            benchmark_service=benchmark_service,
            run_id=run_id,
            config=config,
            rows=pending,
        )

    stats: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in pending:
        data = AttributionResult.model_validate(_as_dict(row.attribution_detail_json))
        await dispatch_notification_for_attribution(
            DispatchRequest(
                store=benchmark_service,
                run_id=run_id,
                item_id=row.item_id,
                source_trace_id=row.source_trace_id or row.trace_id,
                config=config,
            ),
            data=data,
        )
        # dispatch_aone_for_attribution 内部全捕获异常并落 bucket；
        # 重读行拿最终结果（CREATED → notification_ref；其余 → notification_skip_reason）。
        fresh = await BenchmarkRunItem.filter(
            run_id=run_id,
            item_id=row.item_id,
            criterion_id=FUSED_CRITERION_ID,
        ).first()
        bucket = _as_dict(fresh.notification_dispatch_json if fresh else None)
        if bucket.get("notification_ref"):
            stats["created"] += 1
            print(f"  [created] item={row.item_id} notification_ref={bucket['notification_ref']}")
        elif bucket.get("notification_skip_reason"):
            stats["blocked"] += 1
            reason = str(bucket["notification_skip_reason"])
            reasons[reason] += 1
            print(f"  [blocked] item={row.item_id} reason={reason}")
        else:
            stats["unknown"] += 1
            print(f"  [unknown] item={row.item_id} — no bucket after dispatch")

    print(
        f"OK: run_id={run_id} created={stats['created']} blocked={stats['blocked']} "
        f"already_dispatched={len(already)} unknown={stats['unknown']}"
    )
    if reasons:
        print(f"     block reasons: {dict(reasons)}")
    return 0


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Dispatch Aone workitems for a run's completed attribution rows "
        "(reads fused rows from PG; never re-runs judge/attribution).",
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="Run ID to dispatch workitems for"
    )
    parser.add_argument("--dry-run", action="store_true", help="Replay gating only; no a1 CLI call")
    args = parser.parse_args()

    # Initialize app lifespan (same pattern as regenerate_report.py)
    app = create_app(settings=Settings())
    async with app.router.lifespan_context(app):
        return await _dispatch_for_run(run_id=args.run_id, dry_run=args.dry_run, app=app)


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
