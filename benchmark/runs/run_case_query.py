"""Case 明细查询 / Kafka 抽样聚合

Mixin for :class:`~benchmark.runs.run_service.BenchmarkService`.
Symbols resolve through :mod:`run_helpers` so tests can patch
``benchmark.runs.run_helpers.*``.
"""

from typing import TYPE_CHECKING, Any

from benchmark.domain.benchmark_console import (
    _JUDGE_CONTENT_VERSION_PATHS,
    content_version_from_bucket,
)
from benchmark.domain.enums import PipelineSource, is_dataset_pipeline_source
from benchmark.domain.eval_config import (
    eval_name_to_criterion_ids,
    parse_eval_identity_key,
)
from benchmark.runs import run_helpers as rh
from benchmark.runs.schema import FUSED_CRITERION_ID
from benchmark.runs.trace_daily_report_adapter import (
    TraceDailyDecorFields,
    decorate_trace_daily_case,
)
from benchmark.reporting.framework.aggregation_calculator import EvaluatorResultEntry
from benchmark.services.trace_writer import (
    DEFAULT_ATTRIBUTION_DIST_FIELDS,
    build_attribution_value_labels,
    count_attribution_field_values,
)

if TYPE_CHECKING:
    from benchmark.reporting.framework.schema import Criterion


def _eval_to_criteria(
    rubric: "list[Criterion]",
    *,
    registration_id: str | None = None,
    stream_instance_id: str | None = None,
) -> dict[str, list[str]]:
    """``evaluator_name``（茎 / ``stem@reg[@stream]`` / 裸 reg）→ criterion id。"""
    refs: dict[str, list[str]] = {}
    for c in rubric:
        refs.setdefault(c.evaluator_ref, []).append(c.id)
    return eval_name_to_criterion_ids(
        rubric_refs_to_ids=refs,
        registration_id=registration_id,
        stream_instance_id=stream_instance_id,
    )


def _resolve_eval_criterion_ids(
    eval_to_criteria: dict[str, list[str]],
    eval_name: str,
    *,
    fallback_raw: bool = False,
) -> list[str]:
    """解析落库 ``evaluator_name`` → criterion id。

    优先精确键（含 ``stem@reg@stream``）；未命中时回退茎名，避免实时流
    复合键无法映射到 ``amap_agent_overall`` 导致报告缺 ``scores.fused``。
    """
    name = (eval_name or "").strip()
    if not name:
        return []
    cids = eval_to_criteria.get(name)
    if cids:
        return list(cids)
    stem, _reg, _stream = parse_eval_identity_key(name)
    if stem and stem != name:
        cids = eval_to_criteria.get(stem)
        if cids:
            return list(cids)
    if fallback_raw:
        return [name]
    return []


def _eval_name_override_rank(
    eval_name: str,
    *,
    registration_id: str | None = None,
    stream_instance_id: str | None = None,
) -> int:
    """同 trace 多 evaluator 行时覆盖优先级：复合键 > 裸 reg > 茎名。"""
    name = str(eval_name or "")
    reg = (registration_id or "").strip()
    stream = (stream_instance_id or "").strip()
    if not name:
        return 0
    if reg and stream:
        # 任意茎的 stem@reg@stream
        if name.endswith(f"@{reg}@{stream}"):
            return 3
    if stream and name.endswith(f"@{stream}"):
        return 2
    if reg and name == reg:
        return 2
    if reg and name.endswith(f"@{reg}"):
        return 1
    return 0


class BenchmarkRunCaseQuery:
    """Case 明细查询 / Kafka 抽样聚合."""

    async def load_case_results(
        self,
        *,
        run_id: str,
        benchmark_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """从 ``benchmark_run_items`` 查全量采样 case，组装 ``case_results``。

        优先从 fused 行的 ``judge_json.criterion_scores`` 提取 per-dimension
        分数（外部触发模式，不写 per-criterion 行）；若 fused 行无
        ``criterion_scores``（inline/multi-node 模式），回退从 per-criterion
        行读取。

        ``input`` / ``output`` 从结构化列直接读取。fused 模式的 attribution
        附在 case 级别的 ``attribution`` 字段。

        ``config``：run ``config_json``（可选）；传入时 case 六维展开与报告
        维度解析同源（含 rubric_ref 回退）。缺省时仅靠进程注册表。
        """
        by_case: dict[str, dict[str, Any]] = {}
        stats = rh._CaseRowStats()

        # 轻量级模式检测（避免全量加载到内存）
        non_fused_count = (
            await rh.BenchmarkRunItem.filter(
                run_id=run_id,
            )
            .exclude(criterion_id=FUSED_CRITERION_ID)
            .count()
        )
        is_external_only = non_fused_count == 0

        # 仅有 fused 行 → 外部触发路径；存在 per-criterion 行 → inline 回退。
        # 不额外探测 fused.criterion_scores：有 per-criterion 行必为 inline、
        # 仅有 fused 行必为外部路径，该探测无法改变分支结果。
        process_fused = is_external_only

        # 分页加载（避免大批量数据内存风险）
        offset = 0
        while True:
            page_rows: list[rh.BenchmarkRunItem] = (
                await rh.BenchmarkRunItem.filter(
                    run_id=run_id,
                )
                .order_by("item_id", "criterion_id")
                .offset(offset)
                .limit(
                    rh._CASE_RESULTS_PAGE_SIZE,
                )
            )

            if not page_rows:
                break

            # 【仅 stop-run】跳过 running 预写；正常 completed/failed 出报保持全量投影。
            partial_stop = rh._is_partial_stop_config(config)
            for rec in page_rows:
                stats.total_rows += 1
                if partial_stop and not rh._is_terminal_item_status(
                    getattr(rec, "status", None),
                ):
                    continue
                if rec.criterion_id == FUSED_CRITERION_ID:
                    stats.fused_count += 1
                    if not rec.item_id:
                        continue
                    if not process_fused:
                        # inline 模式：fused 行只贡献 case 级 attribution，分数走 per-criterion
                        rh._attach_inline_fused_attribution(by_case, rec, stats)
                        continue
                    rh._process_external_fused_row(
                        by_case,
                        rh.ExternalFusedRowProcess(
                            rec=rec,
                            stats=stats,
                            benchmark_name=benchmark_name,
                            config=config,
                        ),
                    )
                    continue

                # per-criterion 行（inline 模式）
                if process_fused:
                    continue
                if not rec.item_id:
                    continue
                rh._process_criterion_row(by_case, rec, stats)

            offset += rh._CASE_RESULTS_PAGE_SIZE

        result = list(by_case.values())
        rh._decorate_case_rows(
            result,
            pass_threshold=rh._pass_threshold_from_config(config),
        )
        rh.logger.info(
            "benchmark.load_case_results run_id=%s "
            "total_rows=%d scored_rows=%d fused_count=%d "
            "failed_rows=%d unique_cases=%d "
            "null(score=%d/input=%d/output=%d/judge=%d/attr=%d) "
            "fused_with_attr=%d result_count=%d "
            "pages_loaded=%d",
            run_id,
            stats.total_rows,
            stats.scored_rows,
            stats.fused_count,
            stats.failed_rows,
            len(by_case),
            stats.null_score_rows,
            stats.null_input_rows,
            stats.null_output_rows,
            stats.null_judge_rows,
            stats.null_attr_rows,
            stats.fused_with_attr,
            len(result),
            (offset // rh._CASE_RESULTS_PAGE_SIZE),
        )
        if not result:
            rh.logger.warning(
                "benchmark.load_case_results_empty run_id=%s "
                "no scored rows found in benchmark_run_items",
                run_id,
            )
        return result

    async def get_low_score_items(
        self,
        run_id: str,
        pass_threshold: float,
    ) -> dict[str, float]:
        """Query ``benchmark_run_items`` for items with score below pass_threshold.

        Replaces the old ``_low_score_items`` config_json merge approach —
        reading directly from DB eliminates the ``rh.merge_config_json`` race
        condition and keeps low-score collection atomic with score writing.

        Returns:
            ``{item_id: score}`` mapping for items below threshold.
        """
        result: dict[str, float] = {}
        async for rec in (
            rh.BenchmarkRunItem.filter(
                run_id=run_id,
                criterion_id=FUSED_CRITERION_ID,
            )
            .exclude(status="running")
            .filter(score__lt=pass_threshold)
        ):
            if rec.item_id:
                result[rec.item_id] = rec.score if rec.score is not None else 0.0
        rh.logger.info(
            "benchmark.low_score_items run_id=%s threshold=%s count=%d",
            run_id,
            pass_threshold,
            len(result),
        )
        return result

    async def load_fused_score_average(
        self,
        run_id: str,
    ) -> float | None:
        """Compute the average of per-item fused scores for a run.

        Queries ``benchmark_run_items`` rows with ``criterion_id="fused"``
        and returns the arithmetic mean of their ``score`` values.  Used to
        override the run-level ``final_score`` when item-level gate rules
        (via ``fusion_fn_ref``) make the per-criterion weighted average
        inaccurate — e.g. D0=Fail zeroes the fused score but leaves
        per-criterion means unchanged.

        Returns:
            Average fused score, or ``None`` if no fused rows exist.
        """
        scores: list[float] = []
        async for rec in rh.BenchmarkRunItem.filter(
            run_id=run_id,
            criterion_id=FUSED_CRITERION_ID,
        ).exclude(status="running"):
            if rec.score is not None:
                scores.append(rec.score)
        if not scores:
            return None
        return sum(scores) / len(scores)

    async def load_kafka_case_results(
        self,
        query: "rh.KafkaRubricQuery | None" = None,
        **legacy: object,
    ) -> list[dict[str, Any]]:
        """从 Kafka 来源的 trace_pipeline_records 加载 case_results。

        与 :meth:`load_case_results` 不同，本方法按真实 ``trace_id`` 查询
        （非合成 ``benchmark:run_id:...`` trace_id），并用 ``evaluator_name``
        映射到 ``criterion.id``（经 rubric ``evaluator_ref`` / registrationId
        / streamInstanceId 桥接）。返回格式与 :meth:`load_case_results` 完全
        一致，供 webdev-agent benchmark-report skill 在 Case 详情 modal 展示。
        """
        q = rh.resolve_kafka_rubric_query(query, legacy)
        trace_ids = q.trace_ids
        rubric = q.rubric
        registration_id = q.registration_id
        stream_instance_id = q.stream_instance_id
        if not trace_ids:
            return []

        # Multiple criteria may share the same evaluator_ref (e.g. both
        # use ``evaluator_ref="example"``); map each evaluator_ref / 注册 id
        # / stream 复合键 to a *list* of criterion IDs so the score is
        # assigned to all.
        eval_to_criteria = _eval_to_criteria(
            rubric,
            registration_id=registration_id,
            stream_instance_id=stream_instance_id,
        )

        records = rh.TraceRecord.filter(trace_id__in=trace_ids)

        rows: list[dict[str, Any]] = []
        async for rec in records:
            judge = rec.judge_json
            if not rh._judge_has_score(judge):
                continue
            meta = rec.trace_meta_json or {}
            rows.append(
                {
                    "case_id": rec.trace_id,
                    "criterion_id": rec.evaluator_name,
                    "score": rh._judge_score(judge),
                    "reason": judge.get("reason"),
                    "eval_failed": rh._judge_eval_failed(judge),
                    "sub_scores": judge.get("sub_scores"),
                    "input": meta.get("input"),
                    "output": meta.get("output"),
                    "judge_json": judge,
                    "attribution_detail_json": rec.attribution_detail_json,
                }
            )
        # 茎名先、复合键后：同 case 双行时更具体的行覆盖旧行。
        rows.sort(
            key=lambda r: (
                r["case_id"],
                _eval_name_override_rank(
                    str(r["criterion_id"]),
                    registration_id=registration_id,
                    stream_instance_id=stream_instance_id,
                ),
                r["criterion_id"],
            )
        )

        by_case: dict[str, dict[str, Any]] = {}
        for row in rows:
            case_id = str(row.get("case_id") or "")
            if not case_id:
                continue
            eval_name = str(row.get("criterion_id") or "")
            cids = [
                c
                for c in _resolve_eval_criterion_ids(
                    eval_to_criteria,
                    eval_name,
                    fallback_raw=True,
                )
                if c and c != "fused"
            ]
            if not cids:
                continue
            case = by_case.setdefault(
                case_id,
                {"case_id": case_id, "input": "", "output": "", "scores": {}},
            )
            # Assign the same verdict to every criterion sharing this evaluator.
            for cid in cids:
                rh._merge_case_row(case, cid, row)

        result = list(by_case.values())
        for case in result:
            if not case.get("trace_id") and case.get("case_id"):
                case["trace_id"] = case["case_id"]
            case["display_case_id"] = rh._resolve_report_case_id(case)
            case["display_input"] = rh._extract_display_input(case.get("input"))
            case["display_output"] = rh._extract_display_output(case.get("output"))
        return result

    async def load_trace_daily_case_results(
        self,
        query: "rh.KafkaRubricQuery | None" = None,
        **legacy: object,
    ) -> list[dict[str, Any]]:
        """Trace Daily Report 专用 case_results（不影响其他 kafka / benchmark 报告）。

        基于 :meth:`load_kafka_case_results` 同源查询，额外：
        - ``trace_url`` ← ``trace_meta_json.langfuse_url``（缺省直拼回填）
        - ``grade`` ← ``triage_level`` 兼容映射（非 TraceStatus）；
          ``attribution_overridden`` 取 ``attribution_detail_json
          .corrected_triage_level`` 二次映射
        - ``scores.fused`` ← 「综合质量」分值
        - ``root_cause_display`` ← 标量 ``root_cause``
        """
        q = rh.resolve_kafka_rubric_query(query, legacy)
        trace_ids = q.trace_ids
        rubric = q.rubric
        pass_threshold = q.pass_threshold
        registration_id = q.registration_id
        stream_instance_id = q.stream_instance_id
        if not trace_ids:
            return []

        eval_to_criteria = _eval_to_criteria(
            rubric,
            registration_id=registration_id,
            stream_instance_id=stream_instance_id,
        )

        records = rh.TraceRecord.filter(trace_id__in=trace_ids)

        rows: list[dict[str, Any]] = []
        # trace_id → 日报装饰元数据（多 evaluator 行取首次非空）
        extras: dict[str, TraceDailyDecorFields] = {}
        async for rec in records:
            judge = rec.judge_json
            if not rh._judge_has_score(judge):
                continue
            meta = rec.trace_meta_json or {}
            tid = str(rec.trace_id or "")
            if tid and tid not in extras:
                extras[tid] = TraceDailyDecorFields(
                    triage_level=(
                        rec.triage_level.value
                        if hasattr(rec.triage_level, "value")
                        else rec.triage_level
                    ),
                    root_cause=rec.root_cause,
                    pipeline_status=(
                        rec.status.value if hasattr(rec.status, "value") else rec.status
                    ),
                    project_id=rec.project_id,
                    meta=meta,
                    # attribution_overridden 标量与 corrected_triage_level
                    # bucket 同一次 UPDATE 写入，首行取值即可一致。
                    attribution_detail=rec.attribution_detail_json,
                    # 各维度评分展开源：judge_json（首行即综合 judge 行）。
                    judge_json=judge,
                )
            rows.append(
                {
                    "case_id": rec.trace_id,
                    "criterion_id": rec.evaluator_name,
                    "score": rh._judge_score(judge),
                    "reason": judge.get("reason"),
                    "eval_failed": rh._judge_eval_failed(judge),
                    "sub_scores": judge.get("sub_scores"),
                    "input": meta.get("input"),
                    "output": meta.get("output"),
                    "judge_json": judge,
                    "attribution_detail_json": rec.attribution_detail_json,
                }
            )
        rows.sort(
            key=lambda r: (
                r["case_id"],
                _eval_name_override_rank(
                    str(r["criterion_id"]),
                    registration_id=registration_id,
                    stream_instance_id=stream_instance_id,
                ),
                r["criterion_id"],
            )
        )

        by_case: dict[str, dict[str, Any]] = {}
        for row in rows:
            case_id = str(row.get("case_id") or "")
            if not case_id:
                continue
            eval_name = str(row.get("criterion_id") or "")
            cids = [
                c
                for c in _resolve_eval_criterion_ids(
                    eval_to_criteria,
                    eval_name,
                    fallback_raw=True,
                )
                if c and c != "fused"
            ]
            if not cids:
                continue
            case = by_case.setdefault(
                case_id,
                {"case_id": case_id, "input": "", "output": "", "scores": {}},
            )
            for cid in cids:
                rh._merge_case_row(case, cid, row)

        result = list(by_case.values())
        for case in result:
            if not case.get("trace_id") and case.get("case_id"):
                case["trace_id"] = case["case_id"]
            case["display_case_id"] = rh._resolve_report_case_id(case)
            case["display_input"] = rh._extract_display_input(case.get("input"))
            case["display_output"] = rh._extract_display_output(case.get("output"))
            tid = str(case.get("trace_id") or "")
            decorate_trace_daily_case(
                case,
                rubric=rubric,
                pass_threshold=pass_threshold,
                fields=extras.get(tid) or TraceDailyDecorFields(),
            )
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Pipeline benchmark: Kafka record sampling + aggregation
    # ═══════════════════════════════════════════════════════════════════════

    async def sample_kafka_pipeline_records(
        self,
        query: rh.KafkaSampleQuery,
    ) -> list[str]:
        """Scan Kafka-sourced ``trace_pipeline_records`` by date range.

        Returns a list of ``trace_id`` values matching the criteria.
        When ``sample_size`` is set, returns the most recent N records
        (``ORDER BY created_at DESC`` 后取前 N 条，对齐原 SQL ``LIMIT``）。

        ``source='kafka'`` 与 ``judge_json`` 含 ``score`` 键两个行级谓词在
        Python 侧判定（JSON 路径操作无法下推 ORM），与原 SQL WHERE 语义一致。
        """
        date_start = query.date_start
        date_end = query.date_end
        evaluator_names = query.evaluator_names
        sample_size = query.sample_size
        project_id = query.project_id
        qs = rh.TraceRecord.filter(
            created_at__gte=date_start,
            created_at__lte=date_end,
            evaluator_name__in=evaluator_names,
        )
        if project_id is not None:
            qs = qs.filter(project_id=project_id)

        trace_ids: list[str] = []
        async for rec in qs.order_by("-created_at"):
            meta = rec.trace_meta_json or {}
            if meta.get("source") != "kafka":
                continue
            if not rh._judge_has_score(rec.judge_json):
                continue
            trace_ids.append(rec.trace_id)
            if sample_size is not None and len(trace_ids) >= sample_size:
                break
        return trace_ids

    async def sample_trace_pipeline_records(  # noqa: PLR0912 — 分页/过滤/采样分支既有复杂度
        self,
        query: rh.TraceRecordSampleQuery,
    ) -> list[str]:
        """Scan ``trace_pipeline_records`` by date range + configurable source.

        :meth:`sample_kafka_pipeline_records` 的泛化版，供 trace 日级报表
        （T+1）独立链路使用；kafka 采样路径行为不变。

        - 时间窗为半开区间 ``[date_start, date_end)``（``created_at__gte`` /
          ``created_at__lt``），与 kafka 版的闭区间（``__lte``）语义不同：
          调用方应传「次日零点」而非「当天 23:59:59」，避免跨天边界
          重复/漏计。
        - ``source``：固定通道（``kafka``/``metaq``/``http``/``sidecar``/
          ``benchmark``）精确匹配 ``trace_meta_json.source``；``dataset``
          匹配复合前缀（:func:`is_dataset_pipeline_source`）；空白值
          fail-fast（该值来自报表配置，静默放行会误采所有缺失 source
          的 trace）。
        - ``agent_name`` / ``project_id``：标量列过滤，可下推 ORM。
        - ``judge_content_version``：按 judge lineage ``content_version``
          行级过滤（与注册页 / 调用次数版本隔离对齐）；``None`` = 不过滤。
        - ``source`` 与 ``judge_json`` 含 ``score`` 键两个行级谓词仍在
          Python 侧判定（JSON 路径操作无法下推 ORM）。
        - 分页迭代（对齐 :meth:`load_case_results` 的分页约定）：
          ``async for`` 会把整窗结果集缓存进 Tortoise ``_result_cache``，
          日级窗口全量载入有内存风险；``sample_size`` 命中即提前终止。

        Returns a list of ``trace_id`` values; ``sample_size`` 设置时取窗口内
        最新 N 条（``ORDER BY created_at DESC``）。
        """
        wanted = query.source.strip().lower()
        if not wanted:
            raise ValueError(
                "TraceRecordSampleQuery.source 不能为空：source 来自报表配置，"
                "空白值会静默匹配所有缺失 source 的 trace，请检查配置"
            )
        want_cv = (query.judge_content_version or "").strip() or None

        base_qs = rh.TraceRecord.filter(
            created_at__gte=query.date_start,
            created_at__lt=query.date_end,
            evaluator_name__in=query.evaluator_names,
        )
        if query.project_id is not None:
            base_qs = base_qs.filter(project_id=query.project_id)
        if query.agent_name is not None:
            base_qs = base_qs.filter(agent_name=query.agent_name)

        trace_ids: list[str] = []
        offset = 0
        while True:
            page_rows: list[rh.TraceRecord] = (
                await base_qs.order_by("-created_at")
                .offset(offset)
                .limit(rh._TRACE_SAMPLE_PAGE_SIZE)
            )
            if not page_rows:
                break
            for rec in page_rows:
                meta = rec.trace_meta_json or {}
                src = str(meta.get("source") or "")
                if wanted == PipelineSource.DATASET.value:
                    if not is_dataset_pipeline_source(src):
                        continue
                elif src.lower() != wanted:
                    continue
                if not rh._judge_has_score(rec.judge_json):
                    continue
                if want_cv is not None:
                    row_cv = content_version_from_bucket(
                        rec.judge_json,
                        _JUDGE_CONTENT_VERSION_PATHS,
                    )
                    if row_cv != want_cv:
                        continue
                trace_ids.append(rec.trace_id)
                if query.sample_size is not None and len(trace_ids) >= query.sample_size:
                    return trace_ids
            offset += rh._TRACE_SAMPLE_PAGE_SIZE
        return trace_ids

    async def load_kafka_criterion_entries(
        self,
        query: "rh.KafkaRubricQuery | None" = None,
        **legacy: object,
    ) -> list[EvaluatorResultEntry]:
        """Aggregate Kafka-sourced scores per ``evaluator_name`` → entries.

        Unlike :meth:`load_criterion_entries_from_pipeline` (which groups by
        ``criterion_id`` from synthetic ``benchmark:`` trace IDs), this method
        groups by ``evaluator_name`` and maps it to ``criterion.id`` via
        ``criterion.evaluator_ref`` / ``registration_id`` / stream 别名
        （复合键未命中时回退茎名）。
        """
        q = rh.resolve_kafka_rubric_query(query, legacy)
        trace_ids = q.trace_ids
        rubric = q.rubric
        pass_threshold = q.pass_threshold
        registration_id = q.registration_id
        stream_instance_id = q.stream_instance_id

        if not trace_ids:
            return [
                EvaluatorResultEntry(
                    criterion_id=c.id,
                    evaluator_ref=c.evaluator_ref,
                    sample_count=0,
                    success_count=0,
                    failed_count=0,
                    criterion_stats_json={
                        "mean": 0.0,
                        "median": 0.0,
                        "p95": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "std": 0.0,
                        "pass_rate": 0.0,
                    },
                )
                for c in rubric
            ]

        records = rh.TraceRecord.filter(trace_id__in=trace_ids)
        eval_to_criteria = _eval_to_criteria(
            rubric,
            registration_id=registration_id,
            stream_instance_id=stream_instance_id,
        )

        # (criterion_id → trace_id → score)；同 trace 多行时优先更具体的复合键。
        by_criterion_trace: dict[str, dict[str, tuple[float | None, bool]]] = {}
        by_criterion_rank: dict[str, dict[str, int]] = {}
        async for rec in records:
            judge = rec.judge_json
            if not rh._judge_has_score(judge):
                continue
            eval_name = str(rec.evaluator_name or "")
            cids = _resolve_eval_criterion_ids(eval_to_criteria, eval_name)
            if not cids:
                continue
            score_pair = (rh._judge_score(judge), rh._judge_eval_failed(judge))
            tid = str(rec.trace_id)
            rank = _eval_name_override_rank(
                eval_name,
                registration_id=registration_id,
                stream_instance_id=stream_instance_id,
            )
            for cid in cids:
                bucket = by_criterion_trace.setdefault(cid, {})
                ranks = by_criterion_rank.setdefault(cid, {})
                if tid not in bucket or rank >= ranks.get(tid, -1):
                    bucket[tid] = score_pair
                    ranks[tid] = rank

        entries: list[EvaluatorResultEntry] = []
        for criterion in rubric:
            stats = rh._criterion_stats(
                list(by_criterion_trace.get(criterion.id, {}).values()),
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
                ),
            )
        return entries

    async def aggregate_kafka_attribution(
        self,
        query: "rh.KafkaRubricQuery | None" = None,
        **legacy: object,
    ) -> dict[str, dict[str, Any]]:
        """Aggregate ``attribution_detail_json`` from Kafka-sourced records.

        Unlike :meth:`BenchmarkTraceRecordWriter.aggregate_attribution`
        (which reads from ``benchmark_run_items`` for regular benchmarks),
        this reads directly from ``trace_pipeline_records`` by real
        trace_ids — pipeline benchmark does not write detail rows.

        Groups by ``evaluator_name`` → ``criterion.id`` via rubric
        ``evaluator_ref`` / registrationId / stream（复合键回退茎名）。
        Records whose ``evaluator_name`` does not match any rubric entry
        (e.g. fused attribution logic name ``attribution_*``) are grouped
        under ``"fused"``.

        按通用字段分布聚合（默认 :data:`DEFAULT_ATTRIBUTION_DIST_FIELDS`，
        即 ``primary_issue_category``）；调用方可传任意归因字段。
        """
        q = rh.resolve_kafka_rubric_query(query, legacy)
        trace_ids = q.trace_ids
        rubric = q.rubric
        fields = q.fields
        registration_id = q.registration_id
        stream_instance_id = q.stream_instance_id

        if not trace_ids:
            return {}

        dist_fields = fields if fields is not None else DEFAULT_ATTRIBUTION_DIST_FIELDS

        eval_to_criteria = _eval_to_criteria(
            rubric,
            registration_id=registration_id,
            stream_instance_id=stream_instance_id,
        )

        records = rh.TraceRecord.filter(trace_id__in=trace_ids)

        by_key: dict[str, list[dict[str, Any]]] = {}
        async for rec in records:
            attr = rec.attribution_detail_json
            if not attr:
                continue
            eval_name = str(rec.evaluator_name or "")
            cids = _resolve_eval_criterion_ids(eval_to_criteria, eval_name)
            if cids:
                for cid in cids:
                    by_key.setdefault(cid, []).append(attr)
            else:
                # Fused attribution logic name or unknown evaluator
                by_key.setdefault("fused", []).append(attr)

        result: dict[str, dict[str, Any]] = {}
        for key, attrs in by_key.items():
            result[key] = {
                "field_distributions": count_attribution_field_values(
                    attrs,
                    dist_fields,
                ),
                "value_labels": build_attribution_value_labels(
                    attrs,
                    dist_fields,
                ),
                "total_attributed": len(attrs),
            }
        return result

    async def aggregate_trace_daily_root_cause(
        self,
        query: "rh.KafkaRubricQuery | None" = None,
        **legacy: object,
    ) -> dict[str, dict[str, Any]]:
        """Trace Daily Report：按标量 ``root_cause`` 聚合不合格归因。

        与 :meth:`aggregate_kafka_attribution`（读 ``attribution_detail_json``）
        隔离；仅日报调度器调用，其他 kafka / pipeline report 不受影响。
        """
        q = rh.resolve_kafka_rubric_query(query, legacy)
        trace_ids = q.trace_ids
        rubric = q.rubric
        if not trace_ids:
            return {}

        eval_to_criteria = _eval_to_criteria(
            rubric,
            registration_id=q.registration_id,
            stream_instance_id=q.stream_instance_id,
        )

        records = rh.TraceRecord.filter(trace_id__in=trace_ids)
        by_key: dict[str, list[str]] = {}
        async for rec in records:
            rc = rec.root_cause
            if not isinstance(rc, str) or not rc.strip():
                continue
            rc = rc.strip()
            eval_name = str(rec.evaluator_name or "")
            cids = _resolve_eval_criterion_ids(eval_to_criteria, eval_name)
            if cids:
                for cid in cids:
                    by_key.setdefault(cid, []).append(rc)
            else:
                by_key.setdefault("fused", []).append(rc)

        result: dict[str, dict[str, Any]] = {}
        for key, values in by_key.items():
            dist: dict[str, int] = {}
            for v in values:
                dist[v] = dist.get(v, 0) + 1
            result[key] = {
                "field_distributions": {"root_cause": dist},
                "value_labels": {"root_cause": {k: k for k in dist}},
                "total_attributed": len(values),
            }
        return result


__all__ = ["BenchmarkRunCaseQuery"]
