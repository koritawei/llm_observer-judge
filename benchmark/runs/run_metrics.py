"""指标 merge / criterion 聚合 / 日报趋势

Mixin for :class:`~benchmark.runs.run_service.BenchmarkService`.
Symbols resolve through :mod:`run_helpers` so tests can patch
``benchmark.runs.run_helpers.*``.
"""

from datetime import date, datetime, timedelta
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Any

from tortoise.expressions import Q

from benchmark.records.enums import BenchmarkStatus
from benchmark.domain.eval_config import benchmark_run_id_stem
from benchmark.runs import run_helpers as rh
from benchmark.runs.common.dimension_stats import (
    collect_dimension_scores,
    resolve_run_stat_dimensions,
)
from business.evaluators.example.grading import (
    GRADE_ACCEPTABLE,
    GRADE_EXCELLENT,
    GRADE_FAIL,
    GRADE_GOOD,
    GRADE_POOR,
    rubric_v2_grade,
)
from benchmark.runs.schema import FUSED_CRITERION_ID
from benchmark.runs.trace_daily_trend import attach_trace_daily_trend
from benchmark.reporting.framework.aggregation_calculator import EvaluatorResultEntry
from benchmark.reporting.framework.performance_metrics import (
    PERF_TREND_KEYS,
    performance_trend_point,
)
from benchmark.reporting.framework.schema import CriterionStats

if TYPE_CHECKING:
    from benchmark.reporting.framework.schema import Criterion

#: run_id 中日期段 ``yyyymmdd`` 长度
_YMD_LEN = 8


def _trend_run_id_stem(benchmark_name: str) -> str:
    """趋势序列 run_id 前缀：剥离 ``@registration``，兼容新旧列名。"""
    return benchmark_run_id_stem(benchmark_name) or benchmark_name.strip()


def _compatible_benchmark_name_q(stem: str) -> Q:
    """裸茎与 ``stem@registrationId`` 互认（旧列 / 新列同一趋势序列）。"""
    return Q(benchmark_name=stem) | Q(benchmark_name__startswith=f"{stem}@")


class BenchmarkRunMetrics:
    """指标 merge / criterion 聚合 / 日报趋势."""

    async def load_criterion_entries_from_pipeline(
        self,
        *,
        run_id: str,
        rubric: "list[Criterion]",
        pass_threshold: float,
    ) -> list[EvaluatorResultEntry]:
        """按 criterion 聚合本 job 全部 case → per-criterion entries（ORM + Python 统计）。

        优先从 fused 行的 ``judge_json.criterion_scores`` 解析 per-dimension
        分数（外部触发模式，不写 per-criterion 行）；若 fused 行无
        ``criterion_scores``（inline/multi-node 模式），回退查 per-criterion 行。
        统计（mean / median / p95 / std / pass_rate）由 ``rh._criterion_stats``
        纯 Python 完成。只读明细表，不写回；结果供
        :class:`AggregationCalculator` 算 ``final_score``，再写入
        ``benchmark_runs.metrics_json``。

        【仅 stop-run】读 ``config_json.stop_requested_at`` 判定阶段性报告，
        为真时跳过 running 预写；正常 completed/failed 出报不过滤。
        """
        # 单次查询，Python 分流 fused / per-criterion
        all_rows: list[Any] = []
        async for rec in rh.BenchmarkRunItem.filter(run_id=run_id):
            all_rows.append(rec)

        run = await rh.BenchmarkRun.filter(run_id=run_id).first()
        cfg = run.config_json if run is not None and isinstance(run.config_json, dict) else {}
        stop_partial = rh._is_partial_stop_config(cfg)

        by_cid = rh._collect_fused_criterion_scores(
            all_rows,
            rubric,
            partial_stop=stop_partial,
        )

        # 回退：从 per-criterion 行读取（inline/multi-node 模式）
        if not by_cid:
            for rec in all_rows:
                if stop_partial and not rh._is_terminal_item_status(
                    getattr(rec, "status", None),
                ):
                    continue
                if rec.criterion_id == "fused":
                    continue
                by_cid.setdefault(rec.criterion_id, []).append(
                    (rec.score, rec.status == "failed"),
                )

        entries: list[EvaluatorResultEntry] = []
        for criterion in rubric:
            stats = rh._criterion_stats(
                by_cid.get(criterion.id, []),
                pass_threshold=pass_threshold,
            )
            entries.append(
                EvaluatorResultEntry(
                    criterion_id=criterion.id,
                    evaluator_ref=criterion.evaluator_ref,
                    sample_count=stats["sample_count"],
                    success_count=stats["success_count"],
                    failed_count=stats["failed_count"],
                    criterion_stats_json=stats["criterion_stats_json"],
                )
            )
        return entries

    async def expand_dimension_metrics(
        self,
        spec: rh.ExpandDimensionMetricsSpec,
    ) -> bool:
        """按 benchmark 注册的统计维度展开 sub_scores → 并入 criterion_metrics。

        插件式维度统计（§8 扩展点）：holistic judge（单次调用产出多维
        sub_scores，如 search_ai_card）的维度统计不进 rubric criteria
        （避免 worker 按 criteria 重复评测），而由业务经
        ``register_stat_dimensions`` 注册，此处从 fused 行
        ``judge_json.criterion_scores[*].sub_scores`` 展开 per-维度统计。

        维度来源优先级：``resolve_run_stat_dimensions``（config 快照 →
        进程注册表 → rubric_ref / 默认 Rubric），与报告 criteria /
        case 展开同源。未解析到维度的业务（ieval 等）直接返回 False，
        零开销、零行为变化。
        分数口径：sub_scores 原始分直读；pass_rate 即满分率
        （threshold = full_score）。
        """
        run_id = spec.run_id
        benchmark_name = spec.benchmark_name
        metrics = spec.metrics
        config = spec.config
        dimensions = resolve_run_stat_dimensions(config, benchmark_name)
        if not dimensions:
            return False
        # 【仅 stop-run】过滤 running 预写；正常出报保持全量 fused 行。
        raw_fused = list(
            await rh.BenchmarkRunItem.filter(
                run_id=run_id,
                criterion_id=FUSED_CRITERION_ID,
            ),
        )
        fused_rows = (
            [row for row in raw_fused if rh._is_terminal_item_status(getattr(row, "status", None))]
            if rh._is_partial_stop_config(config)
            else raw_fused
        )
        by_key = collect_dimension_scores(fused_rows, dimensions)
        if not by_key:
            return False

        for dim in dimensions:
            stats = rh._criterion_stats(
                by_key.get(dim.key, []),
                pass_threshold=dim.full_score,
            )
            stats_json = stats["criterion_stats_json"]
            metrics.criterion_metrics[dim.key] = CriterionStats(
                criterion_id=dim.key,
                evaluator_ref="sub_scores",
                sample_count=stats["sample_count"],
                success_count=stats["success_count"],
                failed_count=stats["failed_count"],
                mean=stats_json.get("mean", 0.0),
                median=stats_json.get("median", 0.0),
                p95=stats_json.get("p95", 0.0),
                min=stats_json.get("min", 0.0),
                max=stats_json.get("max", 0.0),
                std=stats_json.get("std", 0.0),
                pass_rate=stats_json.get("pass_rate", 0.0),
            )
        rh.logger.info(
            "dimension_stats.expanded run_id=%s benchmark=%s dims=%d",
            run_id,
            benchmark_name,
            len(dimensions),
        )
        return True

    async def attach_daily_report_context(
        self,
        spec: dict[str, Any],
        *,
        days: int = rh._DAILY_TREND_DAYS,
    ) -> None:
        """为报告 spec 装配天级跟踪日报上下文（近 N 天趋势 + 昨日维度均值）。

        近 N 天趋势按 **run_id 数据日** 分日（上海时区日历），不用出报
        ``now()`` 作右端——延迟 finalize 时否则会把本 run 分数贴到日历今天、
        数据日落空（``ieval_external-20261031`` 于 11 月出报即此病）。

        run_id 序列规则（严格隔离，互不混入）：
        - 当前 run **不带** suffix：只看精确 ``{stem}-{yyyymmdd}``；
          该日无 completed / 失败 → 该日空白，绝不混入带 suffix 的 run；
        - 当前 run **带** suffix：只看同日带任意 suffix 的 completed run
          （``{stem}-{yyyymmdd}-*``），不看无 suffix 主跑。

        ``stem`` 由 ``benchmark_run_id_stem`` 从逻辑名 / ``stem@reg`` 解析；
        列 ``benchmark_name`` 新旧形态（裸茎与 ``stem@registrationId``）
        互认，保证迁移前后趋势连续。

        逐日统计 fused 均值 / 优良率 / 不合格率 / 评测成功率；**数据日**点
        直接取当前 spec 现值（当前 run 状态尚未落库 completed，避免自查询
        竞态）。昨日各 criterion / 统计维度均值相对 **数据日 - 1**。

        幂等：spec 已含非空 ``trend_7d`` 时跳过。查询异常整体降级为
        warning——趋势与雷达对比是增强信息，缺失时报告按渲染约定
        省略对应栏位，不影响主报告生成。

        ``trace_daily_report`` 走独立分支：按 run_id 数据日回放
        ``trace_pipeline_records``，日点缓存进 ``benchmark_runs``。
        不读 ``benchmark_run_items``，也不进入下面的 run 序列逻辑。
        """
        if spec.get("trend_7d"):
            return
        benchmark_name = str(spec.get("benchmark_name") or "").strip()
        if not benchmark_name:
            return
        if spec.get("trace_daily_report"):
            # 趋势按报表命名空间归组（与 run_id 前缀一致）；列 benchmark_name
            # 存内部 agent 名，不能当 run_id 前缀解析。
            report_ns = str(spec.get("trace_report_namespace") or "").strip() or benchmark_name
            try:
                await attach_trace_daily_trend(
                    self,
                    spec,
                    benchmark_name=report_ns,
                    days=days,
                )
            except Exception:
                rh.logger.warning(
                    "benchmark.attach_trace_daily_trend_failed benchmark=%s "
                    "— 日报省略趋势/雷达对比栏位",
                    report_ns,
                    exc_info=True,
                )
            return
        run_id = str(spec.get("run_id") or "")
        # run_id_suffix 从 spec.run_id 解析（``{stem}-{yyyymmdd}[-{suffix}]``）：
        # 有/无 suffix 两套序列严格隔离，互不回退混入；茎兼容 stem@reg。
        run_id_suffix = rh._parse_run_id_suffix(run_id, benchmark_name)
        anchor_date = rh.parse_benchmark_run_data_date(run_id, benchmark_name)
        if anchor_date is None:
            # 回退日历今天（语义不变）；仅日志分级：日跑形态解析失败 → WARNING，
            # 实时流 / inline 等预期无数据日 → debug，避免盖掉真异常。
            anchor_date = datetime.now(rh._UTC_PLUS_8).date()
            looks_like_daily = rh.looks_like_daily_run_id(run_id, benchmark_name)
            log_fn = rh.logger.warning if looks_like_daily else rh.logger.debug
            log_fn(
                "benchmark.trend_anchor_fallback_now run_id=%s benchmark=%s — %s，趋势退回日历今天",
                run_id or "-",
                benchmark_name,
                ("日跑形态解析不出数据日" if looks_like_daily else "非 {stem}-{yyyymmdd} 形态"),
            )
        today_point = self._build_today_trend_point(spec, anchor_date=anchor_date)
        try:
            trend = await self.load_daily_trend(
                rh.DailyTrendQuery(
                    benchmark_name=benchmark_name,
                    today_point=today_point,
                    days=days,
                    run_id_suffix=run_id_suffix,
                    anchor_date=anchor_date,
                ),
            )
            if trend:
                spec["trend_7d"] = trend
            y_means = await self.load_yesterday_criterion_means(
                benchmark_name=benchmark_name,
                run_id_suffix=run_id_suffix,
                anchor_date=anchor_date,
            )
            if y_means:
                spec["yesterday_criterion_means"] = y_means
            y_grades = await self.load_yesterday_grade_distribution(
                benchmark_name=benchmark_name,
                run_id_suffix=run_id_suffix,
                anchor_date=anchor_date,
            )
            if y_grades:
                spec["yesterday_grade_distribution"] = (
                    BenchmarkRunMetrics.normalize_grade_distribution(y_grades)
                )
            rh.logger.info(
                "benchmark.attach_daily_report_context benchmark=%s stem=%s "
                "data_date=%s trend_points=%s yesterday_criteria=%d yesterday_grades=%d",
                benchmark_name,
                _trend_run_id_stem(benchmark_name),
                anchor_date.isoformat(),
                len(trend["dates"]) if trend else 0,
                len(y_means),
                sum(y_grades.values()) if y_grades else 0,
            )
        except Exception:
            rh.logger.warning(
                "benchmark.attach_daily_report_context_failed "
                "benchmark=%s — 报告省略趋势/雷达对比栏位",
                benchmark_name,
                exc_info=True,
            )

    @staticmethod
    def _build_today_trend_point(
        spec: dict[str, Any],
        *,
        anchor_date: date | None = None,
    ) -> dict[str, Any] | None:
        """从当前 spec 现值构造数据日趋势点（无有效 fused 分时返回 None）。

        ``anchor_date`` 缺省东八区今天，仅兼容旧测试；出报路径应传入
        run_id 解析出的数据日。
        """
        final_score = spec.get("final_score")
        if isinstance(final_score, bool):
            return None
        if isinstance(final_score, str):
            try:
                final_score = float(final_score.strip())
            except ValueError:
                return None
        if not isinstance(final_score, int | float):
            return None
        graded: dict[str, int] = {}
        for case in spec.get("case_results") or []:
            if not isinstance(case, dict):
                continue
            grade = case.get("grade")
            if isinstance(grade, str) and grade:
                graded[grade] = graded.get(grade, 0) + 1
        graded_total = sum(graded.values())
        eg = graded.get(GRADE_EXCELLENT, 0) + graded.get(GRADE_GOOD, 0)
        pf = graded.get(GRADE_POOR, 0) + graded.get(GRADE_FAIL, 0)
        ega = eg + graded.get(GRADE_ACCEPTABLE, 0)  # 任务满足率 (E+G+A)/graded
        day = anchor_date or datetime.now(rh._UTC_PLUS_8).date()
        point: dict[str, Any] = {
            "date": day.strftime("%m-%d"),
            "fused_avg": round(float(final_score), 4),
        }
        if graded_total:
            point["excellent_good_rate"] = round(eg / graded_total * 100, 1)
            point["poor_fail_rate"] = round(pf / graded_total * 100, 1)
            point["task_satisfaction_rate"] = round(ega / graded_total * 100, 1)
        total_samples = spec.get("total_samples")
        total_failed = spec.get("total_failed")
        if isinstance(total_samples, int) and total_samples > 0 and isinstance(total_failed, int):
            point["eval_success_rate"] = round(
                max(total_samples - total_failed, 0) / total_samples * 100,
                1,
            )
        # 性能 & 效率：spec 现值（attach_performance_metrics 已在本次出报算好）
        point.update(performance_trend_point(spec.get("performance_metrics")))
        return point

    async def load_daily_trend(
        self,
        query: rh.DailyTrendQuery,
    ) -> dict[str, Any] | None:
        """查询近 N 天同 benchmark 已完成 runs，按日聚合趋势序列。

        run_id_suffix 语义（严格隔离）：
        - 不带 suffix：只取精确 ``{stem}-{yyyymmdd}`` 的
          completed run；缺失 / 未完成 → 该日 null，不回退带 suffix；
        - 带 suffix：只统计带**任意** suffix 的 run（``{stem}
          -{yyyymmdd}-*``），不看无 suffix 主跑；同日多个取
          ``total_samples`` 最大（并列取最早创建）。

        ``stem`` 兼容裸名与 ``stem@registrationId``；列名新旧互认。
        窗口右端为 ``query.anchor_date``（数据日）；缺省东八区今天。
        run_id 中的日期即数据日期（补跑历史日期归历史日，而非
        created_at / 出报日）。返回 ``{dates: [MM-DD...], fused_avg,
        excellent_good_rate, poor_fail_rate, eval_success_rate,
        task_satisfaction_rate}``（序列等长，缺失点为 null）。除数据日点外
        每天取当日同口径的 completed run。无有效数据点时返回
        None（报告省略趋势栏）。
        """
        benchmark_name, today_point, days, run_id_suffix = (
            query.benchmark_name,
            query.today_point,
            query.days,
            query.run_id_suffix,
        )
        stem = _trend_run_id_stem(benchmark_name)
        anchor = query.anchor_date or datetime.now(rh._UTC_PLUS_8).date()
        # 每日期望 run_id：无 suffix = ``{stem}-{yyyymmdd}``；
        # 有 suffix = ``{stem}-{yyyymmdd}-*``（带 suffix 序列）
        day_specs: list[tuple[str, str]] = [
            (
                (day_cn := anchor - timedelta(days=days - 1 - offset)).isoformat(),
                day_cn.strftime("%Y%m%d"),
            )
            for offset in range(days)
        ]
        all_ymds = [ymd for _, ymd in day_specs]
        if run_id_suffix:
            # 带 suffix：同日任意 suffix 均计入（run_id 前缀匹配），
            # 同日多个取 total_samples 最大的「主跑」；不混入无 suffix。
            run_by_id = await self._preferred_suffixed_run_by_day(stem, all_ymds)
        else:
            # 无 suffix 主序列：只认精确 ``{stem}-{yyyymmdd}``。
            # ``{stem}-{yyyymmdd}-mxxx`` / ``-sxx`` / ``-a2`` 等带后缀 run
            # 一律不进（SQL 用 run_id__in 精确列表，禁止 startswith 回退）。
            exact_ids = [f"{stem}-{ymd}" for ymd in all_ymds]
            runs = await rh.BenchmarkRun.filter(
                _compatible_benchmark_name_q(stem),
                status=BenchmarkStatus.COMPLETED,
                run_id__in=exact_ids,
            ).order_by("created_at")
            # run_id 业务唯一（无 suffix 每日至多一条）；升序遍历覆盖
            # 仅作同 id 多行防御性兜底。缺失日保持空白，禁止回退带 suffix
            # （曾混入 smoke 小样本 → 优良率 100% 污染趋势图）。
            run_by_id = {run.run_id: run for run in runs if run.run_id in exact_ids}

        today_key = anchor.isoformat()
        metric_keys = (
            "fused_avg",
            "excellent_good_rate",
            "poor_fail_rate",
            "eval_success_rate",
            "task_satisfaction_rate",
            *PERF_TREND_KEYS,
        )
        series: dict[str, list[Any]] = {"dates": [], **{k: [] for k in metric_keys}}
        has_any = today_point is not None
        for day, ymd in day_specs:
            series["dates"].append(day[5:])
            if day == today_key and today_point is not None:
                for key in metric_keys:
                    series[key].append(today_point.get(key))
                continue
            run = run_by_id.get(f"{stem}-{ymd}")
            if run is None:
                for key in metric_keys:
                    series[key].append(None)
                continue
            has_any = True
            day_point = await self._load_run_trend_point(run)
            for key in metric_keys:
                series[key].append(day_point.get(key))
        if not has_any:
            return None
        return series

    @staticmethod
    async def _completed_run_for_day(
        benchmark_name: str,
        ymd: str,
        *,
        suffixed_only: bool,
    ) -> "rh.BenchmarkRun | None":
        """某日同口径 completed run；有/无 suffix 序列严格隔离。

        - ``suffixed_only=False``：只取精确 ``{stem}-{ymd}``；缺失 → None，
          绝不混入带 suffix 的 run；
        - ``suffixed_only=True``：只看 ``{stem}-{ymd}-*``，取
          ``total_samples`` 最大（并列最早创建），不看无 suffix 主跑。

        列名兼容裸茎与 ``stem@registrationId``。
        """
        stem = _trend_run_id_stem(benchmark_name)
        if not suffixed_only:
            return (
                await rh.BenchmarkRun.filter(
                    _compatible_benchmark_name_q(stem),
                    status=BenchmarkStatus.COMPLETED,
                    run_id=f"{stem}-{ymd}",
                )
                .order_by("-created_at")
                .first()
            )
        preferred = await BenchmarkRunMetrics._preferred_suffixed_run_by_day(
            stem,
            [ymd],
        )
        return preferred.get(f"{stem}-{ymd}")

    @staticmethod
    def _run_sample_size(run: rh.BenchmarkRun) -> int:
        """趋势选跑用的样本量；缺失 / 非法按 0（让位给有样本的主跑）。"""
        n = getattr(run, "total_samples", None)
        return n if isinstance(n, int) and n > 0 else 0

    @staticmethod
    async def _preferred_suffixed_run_by_day(
        benchmark_name: str,
        ymds: list[str],
    ) -> dict[str, rh.BenchmarkRun]:
        """``{stem}-{yyyymmdd}-*`` → 每日 ``total_samples`` 最大的一条。

        并列时保留最早创建（``created_at`` 升序先写入）。返回键为无 suffix
        形态 ``{stem}-{yyyymmdd}``，与精确匹配口径对齐。

        ieval_external-20260916：同日先有 smoke（样本极少、全优 → 优良率
        100%）再有正式大盘；若取最早一条会把 100% 画进趋势图。
        """
        if not ymds:
            return {}
        stem = _trend_run_id_stem(benchmark_name)
        prefix_query = reduce(
            or_,
            (Q(run_id__startswith=f"{stem}-{ymd}-") for ymd in ymds),
        )
        runs = await rh.BenchmarkRun.filter(
            prefix_query,
            _compatible_benchmark_name_q(stem),
            status=BenchmarkStatus.COMPLETED,
        ).order_by("created_at")
        ymd_start = len(stem) + 1
        ymd_end = ymd_start + _YMD_LEN
        run_by_id: dict[str, rh.BenchmarkRun] = {}
        for run in runs:
            ymd_part = run.run_id[ymd_start:ymd_end]
            key = f"{stem}-{ymd_part}"
            prev = run_by_id.get(key)
            if prev is None or (
                BenchmarkRunMetrics._run_sample_size(run)
                > BenchmarkRunMetrics._run_sample_size(prev)
            ):
                run_by_id[key] = run
        return run_by_id

    # 兼容旧测试 / 调用方别名
    _earliest_suffixed_run_by_day = _preferred_suffixed_run_by_day

    async def _load_run_trend_point(
        self,
        run: rh.BenchmarkRun,
    ) -> dict[str, Any]:
        """从单个 run 的 fused 行统计当日趋势点。

        优良 / 不合格 / 任务满足判定与报告 grade 口径一致
        （``rubric_v2_grade``）；评测成功率 = success fused 行 /
        全部 fused 行（含 failed）。

        性能 & 效率（延迟 / TTFT / Token / ReAct）直读该 run 出报时落库的
        ``metrics_json.performance_metrics``，不重扫 items 明细；历史 run
        无该键时对应趋势点留空。
        """
        rows = await rh.BenchmarkRunItem.filter(
            run_id=run.run_id,
            criterion_id=FUSED_CRITERION_ID,
        )
        graded = [rec.score for rec in rows if rec.status == "success" and rec.score is not None]
        point: dict[str, Any] = {}
        if graded:
            point["fused_avg"] = round(sum(graded) / len(graded), 4)
            grades = [rubric_v2_grade(s) for s in graded]
            eg = sum(1 for g in grades if g in {GRADE_EXCELLENT, GRADE_GOOD})
            pf = sum(1 for g in grades if g in {GRADE_POOR, GRADE_FAIL})
            ega = eg + sum(1 for g in grades if g == GRADE_ACCEPTABLE)
            point["excellent_good_rate"] = round(eg / len(graded) * 100, 1)
            point["poor_fail_rate"] = round(pf / len(graded) * 100, 1)
            point["task_satisfaction_rate"] = round(ega / len(graded) * 100, 1)
        if rows:
            point["eval_success_rate"] = round(len(graded) / len(rows) * 100, 1)
        metrics_json = getattr(run, "metrics_json", None)
        if isinstance(metrics_json, dict):
            point.update(performance_trend_point(metrics_json.get("performance_metrics")))
        return point

    async def load_yesterday_grade_distribution(
        self,
        *,
        benchmark_name: str,
        run_id_suffix: str | None = None,
        anchor_date: date | None = None,
    ) -> dict[str, int]:
        """相对 ``anchor_date`` 前一日同口径 completed run 的档位计数。

        供报告「档位分布」相对昨日变化量 / 变化比例；口径与
        ``_load_run_trend_point`` / ``rubric_v2_grade`` 一致。无昨日 run
        或无有效 fused 分时返回空 dict（模板显示 ``-``）。
        ``anchor_date`` 缺省东八区今天。
        """
        anchor = anchor_date or datetime.now(rh._UTC_PLUS_8).date()
        yesterday_ymd = (anchor - timedelta(days=1)).strftime("%Y%m%d")
        run = await self._completed_run_for_day(
            benchmark_name,
            yesterday_ymd,
            suffixed_only=bool(run_id_suffix),
        )
        if run is None:
            return {}
        rows = await rh.BenchmarkRunItem.filter(
            run_id=run.run_id,
            criterion_id=FUSED_CRITERION_ID,
        )
        dist: dict[str, int] = {}
        for rec in rows:
            if rec.status != "success" or rec.score is None:
                continue
            grade = rubric_v2_grade(float(rec.score))
            dist[grade] = dist.get(grade, 0) + 1
        return dist

    @staticmethod
    def normalize_grade_distribution(raw: object) -> dict[str, int]:
        """档位计数字典规范化：兼容 float（``10.0``）/ 字符串数字，非法值丢弃。"""
        if not isinstance(raw, dict):
            return {}
        out: dict[str, int] = {}
        for key, val in raw.items():
            label = str(key or "").strip()
            if not label:
                continue
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                out[label] = int(round(float(val)))
                continue
            if isinstance(val, str):
                try:
                    out[label] = int(round(float(val.strip())))
                except ValueError:
                    continue
        return out

    async def load_yesterday_criterion_means(
        self,
        *,
        benchmark_name: str,
        run_id_suffix: str | None = None,
        anchor_date: date | None = None,
    ) -> dict[str, float]:
        """相对 ``anchor_date`` 前一日同口径 completed run 的各维度均值。

        供报告维度雷达图「今日 vs 昨日」对比；与 ``load_daily_trend``
        同口径严格隔离——无 suffix 只取精确 ``{stem}-{yyyymmdd}``
        （缺失空白）；带 suffix 只看 ``{stem}-{yyyymmdd}-*``；
        列名兼容裸茎与 ``stem@registrationId``。``anchor_date`` 缺省东八区今天。

        分数提取：
        1. 若 run 注册了 ``stat_dimensions``（如 search_eval）：从 fused 行
           ``sub_scores`` 按维度 key 聚合——与今日 ``criterion_metrics`` /
           雷达轴 id **同 key**，避免只抽 holistic ``criterion_scores``
           导致昨日为空、雷达只画今日；
        2. 否则与 ``load_case_results`` 同口径：有 per-criterion 行走行级，
           否则从 fused 行 ``judge_json.criterion_scores`` 提取（ieval）。

        无昨日 run 或无有效评分时返回空 dict（雷达只画今日）。
        """
        anchor = anchor_date or datetime.now(rh._UTC_PLUS_8).date()
        yesterday_ymd = (anchor - timedelta(days=1)).strftime("%Y%m%d")
        run = await self._completed_run_for_day(
            benchmark_name,
            yesterday_ymd,
            suffixed_only=bool(run_id_suffix),
        )
        if run is None:
            return {}
        rows = await rh.BenchmarkRunItem.filter(run_id=run.run_id)

        # search_eval 等：按统计维度 sub_scores 聚合，对齐今日雷达轴
        config = getattr(run, "config_json", None)
        config = config if isinstance(config, dict) else {}
        dimensions = resolve_run_stat_dimensions(config, benchmark_name)
        if dimensions:
            fused_rows = [r for r in rows if r.criterion_id == FUSED_CRITERION_ID]
            by_key = collect_dimension_scores(fused_rows, dimensions)
            means: dict[str, float] = {}
            for dim in dimensions:
                scores = [s for s, _failed in by_key.get(dim.key, []) if s is not None]
                if scores:
                    means[dim.key] = round(sum(scores) / len(scores), 4)
            if means:
                return means

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}

        def _add(cid: str, score: float) -> None:
            totals[cid] = totals.get(cid, 0.0) + score
            counts[cid] = counts.get(cid, 0) + 1

        # 模式互斥（同 load_case_results）：有 per-criterion 行走行级
        # 读取，否则从 fused 行 judge_json.criterion_scores 提取
        has_per_criterion = any(r.criterion_id != FUSED_CRITERION_ID for r in rows)
        for rec in rows:
            if has_per_criterion:
                if (
                    rec.criterion_id != FUSED_CRITERION_ID
                    and rec.status == "success"
                    and rec.score is not None
                ):
                    _add(rec.criterion_id, rec.score)
            elif rec.criterion_id == FUSED_CRITERION_ID:
                for cid, score in rh._extract_criterion_scores(rec.judge_json).items():
                    _add(cid, score)
        return {cid: round(totals[cid] / counts[cid], 4) for cid in totals if counts[cid]}

    async def merge_criterion_metrics(
        self,
        run_id: str,
        *,
        criterion_id: str,
        criterion_payload: dict[str, Any],
    ) -> None:
        """增量写入 ``metrics_json.criterion_metrics[criterion_id]``。

        中途轮询可见已完成 criterion；不触碰 attribution / final_score。
        """
        await self._patch_metrics(
            run_id,
            {"criterion_metrics": {criterion_id: criterion_payload}},
        )

    async def merge_attribution_metrics(
        self,
        run_id: str,
        *,
        attribution_by_id: dict[str, dict[str, Any]],
    ) -> None:
        """增量写入 ``metrics_json.attribution_metrics``。

        键为真实 ``criterion_id``（per_eval）或字面量 ``fused``（fused 模式）。
        与已有 attribution / criterion 分桶共存。
        """
        if not attribution_by_id:
            return
        await self._patch_metrics(
            run_id,
            {"attribution_metrics": attribution_by_id},
        )

    async def merge_performance_metrics(
        self,
        run_id: str,
        *,
        performance_metrics: dict[str, Any],
    ) -> None:
        """写入 ``metrics_json.performance_metrics``（性能 & 效率指标）。

        出报时由 ``attach_performance_metrics`` 调用：当日聚合结果落库后，
        近 7 天趋势与「vs 昨日」环比直接读 DB，无需回扫 items 明细。
        按子键 merge（见 ``_METRICS_NESTED_DICT_KEYS``），出报重试幂等。
        """
        if not performance_metrics:
            return
        await self._patch_metrics(
            run_id,
            {"performance_metrics": performance_metrics},
        )

    async def merge_report_meta(
        self,
        run_id: str,
        *,
        patch: dict[str, Any],
    ) -> None:
        """浅合并 ``report_meta_json``（可在 complete 前后分步补 url / size）。"""
        run = await rh.BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            rh.logger.error(
                "benchmark.merge_report_meta_missing_run run_id=%s",
                run_id,
            )
            return
        merged = rh.merge_report_meta_json(
            run.report_meta_json if isinstance(run.report_meta_json, dict) else None,
            patch,
        )
        await rh.BenchmarkRun.filter(run_id=run_id).update(report_meta_json=merged)

    async def merge_config(
        self,
        run_id: str,
        *,
        patch: dict[str, Any],
    ) -> None:
        """浅合并 ``config_json``（运行中补实际 sample_size 等）。"""
        run = await rh.BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            rh.logger.error(
                "benchmark.merge_config_missing_run run_id=%s",
                run_id,
            )
            return
        merged = rh.merge_config_json(
            run.config_json if isinstance(run.config_json, dict) else None,
            patch,
        )
        await rh.BenchmarkRun.filter(run_id=run_id).update(config_json=merged)

    async def _patch_metrics(
        self,
        run_id: str,
        patch: dict[str, Any],
    ) -> None:
        run = await rh.BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            rh.logger.error(
                "benchmark.merge_metrics_missing_run run_id=%s",
                run_id,
            )
            return
        merged = rh.merge_metrics_json(
            run.metrics_json if isinstance(run.metrics_json, dict) else None,
            patch,
        )
        await rh.BenchmarkRun.filter(run_id=run_id).update(metrics_json=merged)

    # ═══════════════════════════════════════════════════════════════════════
    # Complete / fail
    # ═══════════════════════════════════════════════════════════════════════

    async def increment_config_counter(
        self,
        run_id: str,
        key: str,
    ) -> tuple[int, dict[str, Any]]:
        """原子递增 config_json 中的计数器（§12.1）。

        使用 SELECT FOR UPDATE 行锁消除 TOCTOU 竞态。
        返回 (new_count, config) 元组：config 是行锁保护下的最新完整 config_json。
        """
        async with rh.in_transaction() as conn:
            run = (
                await rh.BenchmarkRun.filter(
                    run_id=run_id,
                )
                .select_for_update()
                .using_db(conn)
                .first()
            )
            if run is None:
                raise RuntimeError(f"run not found: {run_id}")
            config = dict(run.config_json) if isinstance(run.config_json, dict) else {}
            new_count = config.get(key, 0) + 1
            config[key] = new_count
            run.config_json = rh._stamp_bucket(config)
            await run.save(using_db=conn, update_fields=["config_json"])
            return new_count, config

    async def merge_config_json(
        self,
        run_id: str,
        patch: dict[str, Any],
    ) -> None:
        """浅合并 config_json（回调中收集低分 item 等，§11）。"""
        run = await rh.BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            rh.logger.error(
                "benchmark.merge_config_json_missing run_id=%s",
                run_id,
            )
            return
        existing = run.config_json if isinstance(run.config_json, dict) else {}
        merged = dict(existing)
        merged.update(patch)
        run.config_json = rh._stamp_bucket(merged)
        await run.save(update_fields=["config_json"])


__all__ = ["BenchmarkRunMetrics"]
