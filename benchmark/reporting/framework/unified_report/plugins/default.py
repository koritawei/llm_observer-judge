"""Default ReportPayload plugin: reuse assemble_report_spec plus webdev precomputation."""

from typing import Any

from benchmark.reporting.framework.report_spec import assemble_report_spec
from benchmark.reporting.framework.unified_report.payload import CaseColumnSpec, ReportPayload
from benchmark.reporting.framework.unified_report.plugin import ReportPayloadContext
from benchmark.reporting.framework.webdev_client import WebdevReportClient

#: Default score-key priority when the generic template reads score reasons (business plugins may override).
DEFAULT_JUDGE_REASON_PREFERRED_KEYS: tuple[str, ...] = ("fused", "综合质量")


class DefaultReportPayloadPlugin:
    """Standard fill path (fallback for every business).

    Precomputation shares ``WebdevReportClient.enrich_report_spec`` with the LLM path, so the
    static template fill fields (grade / insight / waterfall / heatmap / deltas) are complete.
    """

    name = "default"

    def case_columns(self) -> list[CaseColumnSpec]:
        return []

    def fused_attr_field(self) -> str:
        return "primary_issue_category"

    def judge_reason_preferred_keys(self) -> tuple[str, ...]:
        """``scores`` key order preferred when the report detail reads "score reason"."""
        return DEFAULT_JUDGE_REASON_PREFERRED_KEYS

    def _enrich_spec(self, spec: dict[str, Any]) -> None:
        """Apply the same precomputation as the LLM path (grade / insight / enhance)."""
        spec["judge_reason_preferred_keys"] = list(self.judge_reason_preferred_keys())
        WebdevReportClient.enrich_report_spec(spec)

    async def build_payload(self, ctx: ReportPayloadContext) -> ReportPayload:
        spec = ctx.spec
        if spec is None:
            spec = await assemble_report_spec(ctx.job)
        else:
            spec = dict(spec)
        self._enrich_spec(spec)
        return ReportPayload.from_report_spec(
            spec,
            fused_attr_field=self.fused_attr_field(),
            case_columns=self.case_columns(),
        )


__all__ = ["DEFAULT_JUDGE_REASON_PREFERRED_KEYS", "DefaultReportPayloadPlugin"]
