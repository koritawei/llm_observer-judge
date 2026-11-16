"""BenchmarkKafkaTask ORM — the benchmark_kafka_tasks table (streaming task declaration mirror).

Role (§5.3 revision): a **declaration mirror / audit snapshot**, not the source of truth.
RemoteConfig ``agent-eval.benchmark.streaming`` remains the only source of truth; after each
successful reconcile the reconciler upserts declared tasks into this table (concurrent upserts
from multiple nodes are idempotent) for task-list queries, audit, and future P6 agent-park
page reuse. Soft-delete audit lives in RemoteConfig publish history (§5.3); this table keeps
only the current declaration (removing an entry deletes the row).

Composing ``ObserverMixin`` activates ``BenchmarkKafkaTaskObserver``. Note that the UPDATE
branch of ``update_or_create`` does not pass through the ``save()`` lifecycle (only the INSERT
branch goes through ``create()`` and fires ``on_save``), and delete goes through ``row.delete()``
and fires ``on_delete`` — observers only emit observation logs (constitution §4.6).

Usage::

    from benchmark.records.models import BenchmarkKafkaTask
"""

from benchmark.libs.models import BaseModel
from benchmark.libs.models.fields import fields
from benchmark.libs.models.mixin import ObserverMixin


class BenchmarkKafkaTask(BaseModel, ObserverMixin):
    """Declaration mirror of a streaming realtime evaluation task (table: benchmark_kafka_tasks).

    ``task_id`` is the business unique key (same-name convention as the streaming task ``name`` /
    trace-daily-report item ``name``). ``spec_fingerprint`` is a fingerprint of the declared
    content (first 16 hex chars of SHA256), used for change detection and cross-node reconcile.
    """

    task_id = fields.CharField(
        max_length=128,
        unique=True,
        description="streaming 任务名（注册表主键；= RemoteConfig tasks[].name）",
    )
    source = fields.CharField(
        max_length=32,
        default="remote_config",
        description="配置来源：remote_config（声明式）| env（冷启动兜底任务）",
    )
    state = fields.CharField(
        max_length=16,
        default="running",
        description="声明态：running / paused / stopped（删除 = 行移除）",
    )
    topics = fields.JSONField(
        default=list,
        description="订阅 topic 集（元素粒度 diff 增删）",
    )
    group_id = fields.CharField(
        max_length=256,
        description="Kafka 消费组（解析后值；shadow 期带 -shadow 后缀）",
    )
    judge_class = fields.CharField(
        max_length=256,
        null=True,
        description="streaming judge 别名（STREAMING_JUDGE_CLASS_REFS）",
    )
    attribution_class = fields.CharField(
        max_length=256,
        null=True,
        description="归因别名（null=同步 triage 链 / no_attribution=关闭）",
    )
    rubric_class = fields.CharField(
        max_length=256,
        null=True,
        description="Rubric 类别名（RUBRIC_CLASS_REFS；关联校验键）",
    )
    sampling = fields.FloatField(
        null=True,
        description="采样率覆盖 (0,1]；null = 以 RemoteConfig eval config 为准",
    )
    daily_report_enabled = fields.BooleanField(
        default=False,
        description="日报关联（对应 v4 t1_enabled；仅关联校验，不驱动调度）",
    )
    shadow = fields.BooleanField(
        default=False,
        description="shadow 对账模式（S8 起短路，不执行 / 不落库 / 不归因）",
    )
    spec_fingerprint = fields.CharField(
        max_length=64,
        null=True,
        description="声明内容指纹（SHA256 前 16 hex；变更检测用）",
    )
    last_reconciled_at = fields.DatetimeField(
        null=True,
        description="最近一次对账落库时间（UTC）",
    )

    class Meta:
        table = "benchmark_kafka_tasks"


__all__ = ["BenchmarkKafkaTask"]
