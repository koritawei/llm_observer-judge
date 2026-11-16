"""BenchmarkReportProvider ORM — registry of external report services per business.

Each ``benchmark_name`` uniquely registers one external report HTTP service. When
publishing a report, if ``enabled`` and ``provider=external_http``, the platform POSTs
a ReportSpec to ``endpoint_url`` and the business returns ``report_url`` and/or ``html``.

Usage::

    from benchmark.records.models import BenchmarkReportProvider
"""

from benchmark.libs.models import BaseModel
from benchmark.libs.models.fields import fields
from benchmark.libs.models.mixin import ObserverMixin


class BenchmarkReportProvider(BaseModel, ObserverMixin):
    """External report-service registration for a business (table: benchmark_report_providers).

    ``benchmark_name`` aligns with the ``@benchmark_component`` / run-external registration name.
    ``provider`` values:
    - ``platform``: still uses webdev-agent (equivalent to unregistered / disabled)
    - ``external_http``: calls ``endpoint_url``
    """

    benchmark_name = fields.CharField(
        max_length=128,
        unique=True,
        description="benchmark 注册名（与 run.benchmark_name 对齐）",
    )
    provider = fields.CharField(
        max_length=32,
        default="external_http",
        description="platform | external_http",
    )
    endpoint_url = fields.CharField(
        max_length=1024,
        description="外部报告服务 URL（POST ReportSpec）",
    )
    bearer_token = fields.CharField(
        max_length=1024,
        null=True,
        description="可选 Authorization Bearer token（勿入日志）",
    )
    timeout_ms = fields.IntField(
        default=120_000,
        description="HTTP 超时毫秒",
    )
    fallback_to_platform = fields.BooleanField(
        default=True,
        description="外部失败时是否降级 webdev-agent",
    )
    enabled = fields.BooleanField(
        default=True,
        description="是否启用；false 时等同未注册",
    )
    description = fields.CharField(
        max_length=512,
        null=True,
        description="备注（业务方 / 负责人）",
    )

    class Meta:
        table = "benchmark_report_providers"


__all__ = ["BenchmarkReportProvider"]
