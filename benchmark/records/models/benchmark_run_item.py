"""BenchmarkRunItem ORM model — the benchmark_run_items table.

Evaluation-result rows: one row per item × criterion in a run, storing the score,
reason, and attribution detail. Deleted in cascade with ``benchmark_runs`` (the
caller is ``BenchmarkService.delete_run``).

Same shape as ``DatasetEvalRunItem``. The differences: the shard key is ``criterion_id``
(benchmark shards by criterion) rather than ``evaluator_name``, and the full ``judge_json``
bucket is kept for the report Case detail view.

Usage:
    from benchmark.records.models import BenchmarkRunItem
"""

from benchmark.libs.models import BaseModel
from benchmark.libs.models.fields import fields

from benchmark.records.models.validators import validate_notification_dispatch_json


class BenchmarkRunItem(BaseModel):
    """Evaluation-result row (table: benchmark_run_items).

    Each row is the result of one (item, criterion). Lifecycle (three-phase write, D12):
    1. INSERT a base row when judge is submitted (``status=running``, including input/output);
    2. UPDATE score fields when judge finishes (``status=success`` + score/reason);
    3. UPDATE attribution fields when attribution finishes (``attribution_detail_json``).
    An evaluation error writes ``status=failed`` + score=0 + error_message.
    per_eval attribution is appended to ``attribution_detail_json`` by ``upsert_triage``;
    a fused-mode attribution row has ``criterion_id="fused"`` and no score.

    State machine: ``running → success | failed`` (aligned with ``RUNNING`` on ``benchmark_runs``).
    """

    run_id = fields.CharField(
        max_length=128,
        description="FK → benchmark_runs.run_id（delete_run 时显式删除）",
    )
    item_id = fields.CharField(
        max_length=128,
        description="case_id 或 dataset_item_id（与 writer 的 dataset_item_id 参数一致）",
    )
    criterion_id = fields.CharField(
        max_length=128,
        description="criterion.id 或 fused（fused triage 行）",
    )
    evaluator_ref = fields.CharField(
        max_length=128,
        description="evaluator 实现引用（Criterion.evaluator_ref）",
    )
    status = fields.CharField(
        max_length=32,
        default="running",
        description="评测状态（running/success/failed）",
    )
    score = fields.FloatField(
        null=True,
        description="评测分数（success 时写入；failed 时为 0.0）",
    )
    reason = fields.TextField(
        null=True,
        description="评分理由（judge 输出）",
    )
    sub_scores = fields.JSONField(
        null=True,
        description="子分数 bucket（多维度 evaluator 的明细分）",
    )
    judge_json = fields.JSONField(
        null=True,
        description="完整 judge bucket（report Case 详情展示用，含 triage_level 等）",
    )
    attribution_detail_json = fields.JSONField(
        null=True,
        description="per_eval 归因详情（upsert_triage 写入）",
    )
    notification_dispatch_json = fields.JSONField(
        null=True,
        description="Aone 工单派发信息 bucket（notification_ref/notification_title/"
        "notification_dispatched_at/notification_skip_reason，归因回调推送后写入）",
        validators=[validate_notification_dispatch_json],
    )
    trace_id = fields.CharField(
        max_length=256,
        null=True,
        description="数据源原始 trace_id（judge 回调写入；与 source_trace_id 区分）",
    )
    source_trace_id = fields.CharField(
        max_length=256,
        null=True,
        description="Langfuse 原始 trace_id（source_trace_id）",
    )
    trace_url = fields.CharField(
        max_length=2048,
        null=True,
        description="数据源 trace 页面链接（judge 回调写入；report Case 跳转用）",
    )
    attribution_job_id = fields.CharField(
        max_length=256,
        null=True,
        description="异步归因 task_id（provider 返回，回调匹配用）",
    )
    input = fields.JSONField(
        null=True,
        description="case 输入（从 trace_meta_json 提取）",
    )
    output = fields.JSONField(
        null=True,
        description="case 输出（从 trace_meta_json 提取）",
    )
    error_message = fields.TextField(
        null=True,
        description="失败原因（status=failed 时写入）",
    )

    class Meta:
        table = "benchmark_run_items"
        unique_together = (("run_id", "item_id", "criterion_id"),)
        indexes = (
            ("run_id", "criterion_id"),
            ("item_id",),
            ("attribution_job_id",),
        )


__all__ = ["BenchmarkRunItem"]
