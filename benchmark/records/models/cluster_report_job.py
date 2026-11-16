"""Persisted job model for Cluster Report."""

from uuid import uuid4

from benchmark.libs.models import BaseModel
from benchmark.libs.models.fields import fields


class ClusterReportJob(BaseModel):
    """Cluster-report job (table: ``cluster_report_jobs``).

    ``source_batch_tags`` and each JSON bucket use JSONField so local SQLite
    and production PostgreSQL share one ORM contract; on PostgreSQL the column is JSONB.
    """

    job_id = fields.UUIDField(default=uuid4, unique=True, description="对外 job 标识")
    environment = fields.CharField(max_length=32, description="运行环境")
    evaluator_name = fields.CharField(max_length=128, description="评估器名称")
    job_type = fields.CharField(max_length=32, description="normal/stability/performance")
    source_batch_tags = fields.JSONField(description="数据来源 run tag 数组")
    idempotency_key = fields.CharField(max_length=128, description="请求幂等键")

    status = fields.CharField(max_length=16, default="pending", description="任务状态")
    phase = fields.CharField(max_length=32, default="", description="当前执行阶段")
    attempt = fields.IntField(default=0, description="租约成功执行次数")
    lease_owner = fields.CharField(max_length=128, default="", description="当前执行实例")
    lease_expire_at = fields.DatetimeField(null=True, description="租约截止时间")

    job_request = fields.JSONField(default=dict, description="完整请求快照")
    artifacts = fields.JSONField(default=dict, description="报告产物位置")
    result_summary = fields.JSONField(default=dict, description="轻量结果摘要")
    dispatch = fields.JSONField(default=dict, description="Aone 派发结果")
    error = fields.TextField(default="", description="最后一次错误")

    class Meta:
        table = "cluster_report_jobs"
        unique_together = (("environment", "evaluator_name", "job_type", "idempotency_key"),)
        indexes = (("status", "lease_expire_at"), ("evaluator_name", "job_type"))


__all__ = ["ClusterReportJob"]
