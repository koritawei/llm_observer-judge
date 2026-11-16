"""Remote ReportPayload plugin adapter."""

from typing import Any

from benchmark.domain.plugin_binding import PluginBinding
from benchmark.reporting.framework.unified_report.payload import CaseColumnSpec, ReportPayload
from benchmark.reporting.framework.unified_report.plugin import ReportPayloadContext
from benchmark.plugins.remote_client import RemotePluginClient, RemotePluginError


class RemoteReportPayloadPlugin:
    """Assemble ReportPayload over HTTP."""

    name = "remote_report_payload"

    def __init__(self, binding: PluginBinding) -> None:
        self._binding = binding
        self._client = RemotePluginClient()
        self.name = binding.name or self.name

    def case_columns(self) -> list[CaseColumnSpec]:
        return []

    def fused_attr_field(self) -> str:
        return "primary_issue_category"

    async def build_payload(self, ctx: ReportPayloadContext) -> ReportPayload:
        job = ctx.job
        body = await self._client.invoke(
            binding=self._binding,
            kind="report_payload",
            action="build_payload",
            payload={
                "run_id": job.run_id,
                "benchmark_name": (ctx.spec or {}).get("benchmark_name"),
                "spec": ctx.spec,
            },
        )
        if not isinstance(body, dict):
            raise RemotePluginError(f"report_payload remote invalid body type={type(body)}")
        raw: Any = body.get("payload", body)
        if not isinstance(raw, dict):
            raise RemotePluginError("report_payload remote missing payload object")
        return ReportPayload.model_validate(raw)


__all__ = ["RemoteReportPayloadPlugin"]
