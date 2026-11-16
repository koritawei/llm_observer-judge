"""BenchmarkRun ORM model — the benchmark_runs table.

One BenchmarkRun row is one complete benchmark execution (dataset sampling →
multi-criterion evaluation → aggregation → report generation).

Single-table design: per-criterion stats and the attribution distribution are written to
``metrics_json``. There is no separate ``benchmark_evaluator_results`` / ``benchmark_trace_records`` table.

Usage:
    from benchmark.records.models import BenchmarkRun
"""

from benchmark.libs.models import BaseModelWithSoftDelete
from benchmark.libs.models.fields import fields
from benchmark.libs.models.mixin import ObserverMixin

from benchmark.records.enums import BenchmarkStatus


class BenchmarkRun(BaseModelWithSoftDelete, ObserverMixin):
    """Summary row of one benchmark run (table: benchmark_runs).

    Scalar columns follow the spec:
    - identity: run_id (readable job id: ``bm_{name}_{yyyyMMddHHmmss}_{6hex}``)
    - dimensions: benchmark_name, service_name, dataset_name, project_id
    - status: status (BenchmarkStatus), owner, heartbeat_at
    - progress: total_samples, total_evaluators, completed_evaluators,
      evaluated_items, failed_items
    - result: final_score, error_message
    - time: started_at, finished_at, duration_ms

    Buckets (JSON objects) — **incrementally merged** by
    :class:`~benchmark.runs.run_service.BenchmarkService` (aligned with TraceRecord ``trace_meta_json``):
    - metrics_json: incremented by criterion_id while running; on completion merge the final
      draft plus attribution / react (nested dicts merge by sub-key; unmentioned sub-keys stay)
    - config_json: snapshot at create time; later patches are shallow-merged
    - report_meta_json: shallow merge of report metadata (url/size/generated_at may be written in steps)

    Soft-delete column migration (avoid a bad rollout order): ``BaseModelWithSoftDelete`` adds
    ``is_deleted`` / ``deleted_at`` to the existing benchmark_runs table. The DDL ships with this
    MR (the ALTER TABLE section at the end of ``sql/schema.sql``, idempotent and re-runnable).
    **Run the ALTER before releasing the model code**, or SELECT/INSERT on benchmark_runs fails
    with Unknown column. Later changes to existing columns go through ``scripts/table_ops.py``
    (shipped with the ops-script MR, not with this file).
    """

    # ── Identity ──────────────────────────────────────────────────────────
    run_id = fields.CharField(
        max_length=128,
        description=(
            "benchmark job id（bm_{name}_{yyyyMMddHHmmss}_{6hex}）；"
            "可据此查 pipeline 明细并聚合 metrics_json"
        ),
    )

    # ── Dimensions ────────────────────────────────────────────────────────
    benchmark_name = fields.CharField(
        max_length=128,
        description="内部 benchmark agent 注册名（@benchmark_component / "
        "agent 文档 name；对外不感知，用于数据隔离统计）",
    )
    service_name = fields.CharField(
        max_length=128,
        description="被评估的业务 service 名称（如 amap_chat；"
        "写入 pipeline 时映射到 agent_name 列）",
    )
    dataset_name = fields.CharField(
        max_length=256,
        description="数据集注册页面名称（与 WebUI dataset 文档 name 一致；"
        "T+1 日报不再写 trace_pipeline:*）",
    )
    project_id = fields.CharField(
        max_length=128,
        null=True,
        description="Langfuse project ID",
    )

    # ── Status ────────────────────────────────────────────────────────────
    status = fields.CharEnumField(
        BenchmarkStatus,
        description="运行状态（initializing/running/aggregating/generating_report/completed/failed）",
    )
    owner = fields.CharField(
        max_length=256,
        null=True,
        description="执行节点标识（hostname:pid），recover_stale_runs 判 stale 用",
    )
    heartbeat_at = fields.DatetimeField(
        null=True,
        description="心跳时间，runner 周期性更新；超时未更新视为 stale",
    )

    # ── Progress ──────────────────────────────────────────────────────────
    total_samples = fields.IntField(
        description="评测集总量（落定后恒定；stop 禁止改写为已处理数）",
    )
    total_evaluators = fields.IntField(
        description="criterion 总数",
    )
    completed_evaluators = fields.IntField(
        default=0,
        description="已完成 criterion 数（仅进度展示）",
    )
    evaluated_items = fields.IntField(
        default=0,
        description="已完成评测的 case 数（item 级进度，对齐 D12 三阶段写入）",
    )
    failed_items = fields.IntField(
        default=0,
        description="失败评测的 case 数（item 级进度）",
    )

    # ── Result ────────────────────────────────────────────────────────────
    final_score = fields.FloatField(
        null=True,
        description="最终分数 0.0~1.0",
    )
    error_message = fields.TextField(
        null=True,
        description="失败原因（status=failed 时写入；GET 可读）",
    )
    # Soft-delete fields (is_deleted / deleted_at) and soft_delete()/restore()
    # come from BaseModelWithSoftDelete: deleting one evaluation on the page only marks the row,
    # and page-facing queries skip marked rows (deleted_at is not null).

    # ── Timestamps ────────────────────────────────────────────────────────
    started_at = fields.DatetimeField(
        null=True,
        description="开始时间",
    )
    finished_at = fields.DatetimeField(
        null=True,
        description="完成时间",
    )
    duration_ms = fields.IntField(
        null=True,
        description="总耗时（毫秒）",
    )

    # ── Bucket ────────────────────────────────────────────────────────
    metrics_json = fields.JSONField(
        null=True,
        description="聚合指标 bucket — 按 criterion_id 分桶的 CriterionStats + AttributionStats",
    )
    config_json = fields.JSONField(
        null=True,
        description="配置快照 bucket — 含 rubric + AggregationConfig + 实际生效的 sample_size",
    )
    report_meta_json = fields.JSONField(
        null=True,
        description="报告元信息 bucket — url/size/generated_at",
    )

    async def save(self, *args: object, **kwargs: object) -> object:
        """Override save(): after the write, fire registered lifecycle observers.

        ``BenchmarkRunObserver.on_save`` only records an observation log. Paths that bypass save,
        such as ``filter().update()``, do not fire it.
        """
        result = await super().save(*args, **kwargs)
        observer = type(self).get_observer()
        if observer is not None:
            await observer.on_save(self)
        return result

    class Meta:
        table = "benchmark_runs"
        unique_together = (("run_id",),)
        indexes = (
            ("benchmark_name", "status"),
            ("service_name", "status"),
            ("status", "created_at"),
        )


__all__ = ["BenchmarkRun"]
