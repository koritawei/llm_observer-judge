"""TraceRecordWriter — trace 记录写入抽象层。

通过 Protocol 抽象 ``EvaluatorService.after_eval()`` 的评分写入与
``TriageAttributionService.run()`` 的 triage/归因结果写入，使日常评测和
benchmark 各自实现写库逻辑而互不干扰。

- :class:`DailyTraceRecordWriter` — 委托 ``TraceRecordService``，写入
  ``trace_pipeline_records``（日常评测路径）。
- :class:`BenchmarkTraceRecordWriter` — 写入 ``benchmark_run_items`` 专表，
  结构化列存储分数 / 理由 / 归因，以 ``(run_id, item_id, criterion_id)``
  唯一键隔离 run / criterion；``benchmark_runs`` 仅存汇总。

术语约定：Protocol 层统一使用 ``evaluator_ref``（与
``Criterion.evaluator_ref`` 一致）。日常侧 ``TraceRecordService`` 既有签名
使用 ``evaluator_name`` 参数，由 ``DailyTraceRecordWriter`` 桥接映射。

所有写方法均为 upsert 语义（行不存在则插入）。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.enums import TraceStatus, TriageLevel
from benchmark.records.models import BenchmarkRunItem, TraceRecord
from benchmark.services.trace_record_service import AttributionUpdate, TraceRecordKey

if TYPE_CHECKING:
    from benchmark.services.trace_record_service import TraceRecordKey, TraceRecordService

logger = get_logger("agent_eval.trace_writer")

#: 归因分布默认聚合字段 —— report 归因分析按这些字段的取值统计个数。
#: 机制不限定字段集合：调用方可传任意归因字段（如 root_cause / severity /
#: improvement_priority），见 :meth:`BenchmarkTraceRecordWriter.aggregate_attribution`。
DEFAULT_ATTRIBUTION_DIST_FIELDS: tuple[str, ...] = ("primary_issue_category",)

#: 统计字段 → 展示前缀字段 映射 —— 统计键不变（仍按统计字段取值计数），
#: 仅展示时把取值标签拼为 ``前缀字段值/统计字段值``（如
#: ``tool/tool_timeout``）。前缀字段值缺失时回退原取值。
ATTRIBUTION_LABEL_PREFIX_FIELDS: dict[str, str] = {
    "primary_issue_category": "primary_issue_layer",
}


@dataclass
class RunMeta:
    """Writer 绑定 run 的展示元数据（构造方透传；写入逻辑不读取）。"""

    benchmark_name: str = ""
    service_name: str = ""
    dataset_name: str = ""
    project_id: str = ""


@dataclass
class TriageWrite:
    """triage 写入内容三件套（分级 / 根因 / 归因明细 bucket）。

    与行定位键（``trace_id`` / ``evaluator_ref`` / ``dataset_item_id``）
    正交：本对象只描述“写什么”，定位参数描述“写哪行”。
    """

    triage_level: str
    root_cause: str | None
    attribution_detail_json: dict[str, Any]


@dataclass
class ScoreWrite:
    """``upsert_score`` 写入定位键 + 内容（FUNC-PARAMS-001 参数收敛）。

    定位键：``trace_id`` / ``evaluator_ref`` / ``dataset_item_id``；
    内容：``score`` / ``judge_json``。
    """

    trace_id: str | None
    evaluator_ref: str
    score: float
    judge_json: dict[str, Any]
    dataset_item_id: str | None = None


@dataclass
class TriageUpsert:
    """``upsert_triage`` 写入定位键 + 内容（FUNC-PARAMS-001 参数收敛）。

    定位键：``trace_id`` / ``evaluator_ref`` / ``dataset_item_id``；
    内容：``triage``（:class:`TriageWrite` 三件套）。
    """

    trace_id: str | None
    evaluator_ref: str
    triage: TriageWrite
    dataset_item_id: str | None = None


def count_attribution_field_values(
    attrs: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> dict[str, dict[str, int]]:
    """按字段名统计归因取值分布（通用机制，不限定字段集合）。

    非空字符串取值按原值统计；None / 空串 / 字段缺失映射为
    ``"unknown"``；非标量值（如 ``affected_dims`` 列表或
    ``confidence`` 浮点）跳过——不适合做分类统计键。

    Args:
        attrs: ``attribution_detail_json`` 列表。
        fields: 需要统计的字段名列表。

    Returns:
        ``{field: {value: count}}``，无取值的字段不出现在结果中。
    """
    field_distributions: dict[str, dict[str, int]] = {}
    for field in fields:
        dist: dict[str, int] = {}
        for attr in attrs:
            value = attr.get(field)
            if isinstance(value, str) and value:
                dist[value] = dist.get(value, 0) + 1
            elif value is None or isinstance(value, str):
                # None / 空串 / 字段缺失 → "unknown"
                dist["unknown"] = dist.get("unknown", 0) + 1
            # else: 非标量（int/float/list/dict）→ 跳过
        if dist:
            field_distributions[field] = dist
    return field_distributions


def build_attribution_value_labels(
    attrs: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> dict[str, dict[str, str]]:
    """构建统计取值的展示标签（统计键不变，仅展示层拼接前缀）。

    对配置了前缀字段的统计字段（见 :data:`ATTRIBUTION_LABEL_PREFIX_FIELDS`），
    生成 ``{field: {value: "prefix/value"}}`` —— 报告归因分析条形图用该标签
    替代原始取值展示（如 ``primary_issue_category=tool_timeout`` 展示为
    ``tool/tool_timeout``），统计数量与排序不变。

    同一取值出现多个前缀时取首次出现的非空前缀（layer → category 语义上
    应唯一）；前缀值为空 / 缺失的行不生成标签（回退原取值）。
    未配置前缀字段的统计字段不出现在结果中。
    """
    labels: dict[str, dict[str, str]] = {}
    for field in fields:
        prefix_field = ATTRIBUTION_LABEL_PREFIX_FIELDS.get(field)
        if not prefix_field:
            continue
        field_labels: dict[str, str] = {}
        for attr in attrs:
            value = attr.get(field)
            if not (isinstance(value, str) and value):
                continue
            prefix = attr.get(prefix_field)
            if isinstance(prefix, str) and prefix and value not in field_labels:
                field_labels[value] = f"{prefix}/{value}"
        if field_labels:
            labels[field] = field_labels
    return labels


@runtime_checkable
class TraceRecordWriter(Protocol):
    """trace 记录写入接口 — 日常评测与 benchmark 各自实现。

    所有方法均为 upsert 语义：行不存在则插入，存在则更新。
    """

    async def upsert_score(
        self,
        write: ScoreWrite,
        **kwargs: object,
    ) -> object:
        """写入评分结果（upsert：行不存在则插入）。

        参数:
            write: 定位键 + 评分内容（:class:`ScoreWrite`）。
            **kwargs: 额外字段（project_id, agent_name, scene 等）。

        返回:
            写入的记录对象（或 None）。
        """
        ...

    async def upsert_triage(
        self,
        write: TriageUpsert,
        **kwargs: object,
    ) -> object:
        """写入 triage + 归因结果（upsert：行不存在则插入）。

        参数:
            write: 定位键 + triage 内容（:class:`TriageUpsert`）。
            **kwargs: 额外字段。

        返回:
            写入的记录对象（或 None）；benchmark 实现返回
            ``BenchmarkRunItem``，故返回类型放宽为 ``object``（调用方不消费
            返回值）。
        """
        ...


# ════════════════════════════════════════════════════════════════════════════
# Daily — 委托 TraceRecordService
# ════════════════════════════════════════════════════════════════════════════


class DailyTraceRecordWriter:
    """封装 ``TraceRecordService``，实现 ``TraceRecordWriter``。

    桥接映射：``evaluator_ref`` → ``TraceRecordService.upsert(evaluator_name=...)``；
    ``dataset_item_id`` 日常表无此列，忽略。

    写入目标：``trace_pipeline_records`` 表。
    """

    def __init__(self, service: "TraceRecordService") -> None:
        self._service = service

    async def upsert_score(
        self,
        write: ScoreWrite,
        **kwargs: object,
    ) -> TraceRecord:
        """委托 ``TraceRecordService.upsert()``，写入 ``trace_pipeline_records``。"""

        project_id = str(kwargs.get("project_id") or "")
        agent_name = str(kwargs.get("agent_name") or "")
        key = TraceRecordKey(
            trace_id=write.trace_id or "",
            project_id=project_id,
            agent_name=agent_name,
            evaluator_name=write.evaluator_ref,
        )
        upsert_kwargs: dict[str, Any] = {
            "status": TraceStatus.SCORED.value,
            "scene": kwargs.get("scene", "non_voice"),
            "judge_json": write.judge_json,
        }
        # Dataset-eval (and other callers) may pass trace_meta_json via kwargs.
        if "trace_meta_json" in kwargs and kwargs["trace_meta_json"] is not None:
            upsert_kwargs["trace_meta_json"] = kwargs["trace_meta_json"]
        return await self._service.upsert(key, **upsert_kwargs)

    async def upsert_triage(
        self,
        write: TriageUpsert,
        **kwargs: object,
    ) -> None:
        """委托 ``TraceRecordService`` 写入 triage + 归因结果。

        先按业务键查找记录获取 ``record_id``，再调 ``update_triage_level``
        和 ``update_attribution``。
        """

        triage_level = write.triage.triage_level
        root_cause = write.triage.root_cause
        attribution_detail_json = write.triage.attribution_detail_json
        project_id = str(kwargs.get("project_id") or "")
        agent_name = str(kwargs.get("agent_name") or "")

        record = await self._service.get_by_trace_id(
            TraceRecordKey(
                trace_id=write.trace_id or "",
                project_id=project_id,
                agent_name=agent_name,
                evaluator_name=write.evaluator_ref,
            ),
        )
        if record is None:
            logger.warning(
                "daily_writer.record_not_found trace_id=%s evaluator_ref=%s",
                write.trace_id,
                write.evaluator_ref,
            )
            return

        record_id = str(record.id)

        if triage_level:
            try:
                await self._service.update_triage_level(
                    record_id=record_id,
                    triage_level=TriageLevel(triage_level),
                )
            except Exception:
                logger.warning(
                    "daily_writer.update_triage_failed record_id=%s level=%s",
                    record_id,
                    triage_level,
                    exc_info=True,
                )

        if root_cause:
            try:
                await self._service.update_attribution(
                    record_id=record_id,
                    update=AttributionUpdate(
                        root_cause=root_cause,
                        owner_domain=attribution_detail_json.get("owner_domain"),
                        owner_group=attribution_detail_json.get("owner_group"),
                        confidence=attribution_detail_json.get("confidence"),
                        attribution_raw=attribution_detail_json.get("attribution_raw"),
                        corrected_triage_level=attribution_detail_json.get(
                            "corrected_triage_level",
                        ),
                    ),
                )
            except Exception:
                logger.warning(
                    "daily_writer.update_attribution_failed record_id=%s",
                    record_id,
                    exc_info=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# Benchmark — 写入 benchmark_run_items；汇总仍在 benchmark_runs
# ════════════════════════════════════════════════════════════════════════════


class BenchmarkTraceRecordWriter:
    """将单条评估 / triage 明细写入 ``benchmark_run_items``。

    构造时绑定 ``(run_id, criterion_id)``：
    - per-criterion 评估时 ``criterion_id=criterion.id``
    - fused triage 时 ``criterion_id="fused"``

    以 ``(run_id, item_id, criterion_id)`` 唯一键 upsert，保证：
    - 不同 run 互不覆盖
    - 同 run 内多 criterion 复用同一 ``evaluator_ref`` 也不冲突

    ``clear_run`` 为兼容 API 的空操作（明细已落盘，不在 finally 删除）。
    清理请走 ``BenchmarkService.delete_run`` / ``delete_pipeline_records``。
    """

    def __init__(
        self,
        *,
        run_id: str,
        criterion_id: str,
        meta: RunMeta | None = None,
    ) -> None:
        m = meta or RunMeta()
        self._run_id = run_id
        self._criterion_id = criterion_id
        self._benchmark_name = m.benchmark_name
        self._service_name = m.service_name
        self._dataset_name = m.dataset_name
        self._project_id = m.project_id

    @staticmethod
    def _extract_io(kwargs: dict[str, Any]) -> tuple[Any, Any]:
        """从 ``kwargs["trace_meta_json"]`` 提取 ``input`` / ``output``。"""
        meta = kwargs.get("trace_meta_json")
        if not isinstance(meta, dict):
            return None, None
        return meta.get("input"), meta.get("output")

    async def upsert_score(
        self,
        write: ScoreWrite,
        **kwargs: object,
    ) -> BenchmarkRunItem:
        """upsert ``benchmark_run_items``，状态 success / failed。

        对齐 ``DatasetEvalRunner._pg_record_item_results`` 的语义：失败 stub
        不覆写已有成功行。若行已存在且 ``status=success``，仅追加
        ``error_message``，保留 score/reason/input/output/judge_json 不变。
        """

        bucket = dict(write.judge_json)
        if "score" in bucket and not isinstance(bucket["score"], str):
            bucket["score"] = str(bucket["score"])
        bucket.setdefault("score", str(write.score))
        bucket["benchmark_run_id"] = self._run_id
        bucket["criterion_id"] = self._criterion_id

        eval_failed = (
            bucket.get("eval_failed") is True
            or str(bucket.get("eval_failed", "")).lower() == "true"
        )
        status = "failed" if eval_failed else "success"

        item_id = write.dataset_item_id or ""

        if eval_failed:
            existing = await BenchmarkRunItem.filter(
                run_id=self._run_id,
                item_id=item_id,
                criterion_id=self._criterion_id,
            ).first()
            if existing is not None and existing.status == "success":
                existing.error_message = str(bucket.get("reason", ""))[:2000]
                await existing.save(update_fields=["error_message"])
                return existing

        input_val, output_val = self._extract_io(kwargs)

        defaults: dict[str, Any] = {
            "evaluator_ref": write.evaluator_ref,
            "status": status,
            "score": write.score,
            "reason": bucket.get("reason"),
            "sub_scores": bucket.get("sub_scores"),
            "judge_json": bucket,
            "source_trace_id": write.trace_id,
            "input": input_val,
            "output": output_val,
            "error_message": str(bucket.get("reason", ""))[:2000] if eval_failed else None,
        }

        record, _created = await BenchmarkRunItem.update_or_create(
            run_id=self._run_id,
            item_id=item_id,
            criterion_id=self._criterion_id,
            defaults=defaults,
        )
        return record

    async def upsert_triage(
        self,
        write: TriageUpsert,
        **kwargs: object,
    ) -> BenchmarkRunItem | None:
        """upsert ``benchmark_run_items``，追加归因详情。

        空归因（before-hook 预写 triage_level 但未走归因 agent 的情况）
        不写 ``attribution_detail_json``，保持 None 以确保
        ``aggregate_attribution`` 只统计真正走过归因 agent 的行。
        无 existing 行时（``upsert_score`` 未先调用）跳过写入，
        避免产生有归因无 judge 的脏行。
        """

        triage_level = write.triage.triage_level
        root_cause = write.triage.root_cause
        attribution_detail_json = write.triage.attribution_detail_json
        has_real_attribution = bool(attribution_detail_json)

        item_id = write.dataset_item_id or ""
        existing = await BenchmarkRunItem.filter(
            run_id=self._run_id,
            item_id=item_id,
            criterion_id=self._criterion_id,
        ).first()

        if existing is None:
            logger.warning(
                "benchmark.upsert_triage.no_existing_row "
                "run_id=%s item_id=%s criterion_id=%s "
                "has_real_attribution=%s — upsert_score was not called first, "
                "skipping triage write",
                self._run_id,
                item_id,
                self._criterion_id,
                has_real_attribution,
            )
            return None

        defaults: dict[str, Any] = {
            "evaluator_ref": write.evaluator_ref,
        }

        if has_real_attribution:
            bucket = dict(attribution_detail_json)
            bucket["triage_level"] = triage_level
            if root_cause:
                bucket["root_cause"] = root_cause
            bucket["benchmark_run_id"] = self._run_id
            bucket["criterion_id"] = self._criterion_id
            defaults["attribution_detail_json"] = bucket
            # Attribution reasoning overwrites judge's reason when present:
            # upsert_score wrote judge_json["reason"]; upsert_triage with a
            # non-empty ``reasoning`` (from AttributionResult.reasoning) replaces it.
            reasoning = bucket.get("reasoning")
            if reasoning:
                defaults["reason"] = reasoning
        # else: before-hook empty write — only update evaluator_ref,
        # leave attribution_detail_json as None so aggregate_attribution
        # correctly excludes non-badcase rows.

        if "judge_json" in kwargs and kwargs["judge_json"] is not None:
            defaults["judge_json"] = kwargs["judge_json"]
        # Preserve IO from the prior upsert_score row: when triage is called
        # without trace_meta_json in kwargs (the common case), keep the
        # existing input/output intact so the report still shows the user's
        # question and answer.
        if defaults.get("input") is None:
            defaults["input"] = existing.input
        if defaults.get("output") is None:
            defaults["output"] = existing.output
        if defaults.get("score") is None:
            defaults["score"] = existing.score
        if defaults.get("status") is None:
            defaults["status"] = existing.status
        if defaults.get("source_trace_id") is None:
            defaults["source_trace_id"] = existing.source_trace_id

        record, _created = await BenchmarkRunItem.update_or_create(
            run_id=self._run_id,
            item_id=item_id,
            criterion_id=self._criterion_id,
            defaults=defaults,
        )
        return record

    @staticmethod
    def clear_run(run_id: str) -> None:
        """No-op — rows are durable; use ``delete_pipeline_records``."""

    @staticmethod
    async def delete_pipeline_records(run_id: str) -> int:
        """Delete all ``benchmark_run_items`` belonging to a benchmark run."""

        deleted = await BenchmarkRunItem.filter(
            run_id=run_id,
        ).delete()
        return int(deleted)

    @staticmethod
    async def aggregate_attribution(
        run_id: str,
        *,
        fields: Sequence[str] = DEFAULT_ATTRIBUTION_DIST_FIELDS,
    ) -> dict[str, dict[str, Any]]:
        """Aggregate attribution from ``benchmark_run_items`` for a run.

        按调用方指定的字段列表统计每个字段取值的个数（通用机制，
        不限定字段集合）：默认仅 ``primary_issue_category``；其他业务
        可传 ``root_cause`` 等任意 ``attribution_detail_json`` 内的
        字段名。仅统计非空字符串取值（跳过 None / 空串 / 非标量）。

        Args:
            run_id: benchmark run ID。
            fields: 聚合字段列表，默认 :data:`DEFAULT_ATTRIBUTION_DIST_FIELDS`。

        Returns:
            ``{criterion_id: {field_distributions: {field: {value: count}},
            total_attributed}}``.  Fused rows stay under ``"fused"``.
        """

        rows = await BenchmarkRunItem.filter(
            run_id=run_id,
        ).exclude(attribution_detail_json=None)

        # 按 (criterion_id, item_id) 去重，确保 total_attributed 与
        # load_case_results 的 case 数一致（后者按 item_id 去重构建 case
        # 列表；此处同步去重避免归因统计虚高）。
        by_criterion: dict[str, dict[str, dict[str, Any]]] = {}
        for record in rows:
            attr = record.attribution_detail_json
            if not attr:
                continue
            if not record.item_id:
                continue
            cid = record.criterion_id or attr.get("criterion_id", "")
            by_criterion.setdefault(cid, {})[record.item_id] = attr

        result: dict[str, dict[str, Any]] = {}
        for cid, item_attrs in by_criterion.items():
            attrs = list(item_attrs.values())
            result[cid] = {
                "field_distributions": count_attribution_field_values(
                    attrs,
                    fields,
                ),
                "value_labels": build_attribution_value_labels(attrs, fields),
                "total_attributed": len(item_attrs),
            }
        return result


__all__ = [
    "ATTRIBUTION_LABEL_PREFIX_FIELDS",
    "DEFAULT_ATTRIBUTION_DIST_FIELDS",
    "RunMeta",
    "ScoreWrite",
    "TraceRecordWriter",
    "TriageUpsert",
    "TriageWrite",
    "DailyTraceRecordWriter",
    "BenchmarkTraceRecordWriter",
    "build_attribution_value_labels",
    "count_attribution_field_values",
]
