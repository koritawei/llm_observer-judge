"""Unified daily-report package: static template (primary) + ReportPayload + business plugin fill."""

from benchmark.reporting.framework.unified_report.filler import fill_report_template
from benchmark.reporting.framework.unified_report.payload import CaseColumnSpec, ReportPayload
from benchmark.reporting.framework.unified_report.pipeline import (
    force_webdev_report,
    generate_unified_report_from_spec,
    generate_unified_report_html,
    unified_report_enabled,
)
from benchmark.reporting.framework.unified_report.plugin import (
    ReportPayloadContext,
    ReportPayloadPlugin,
    clear_report_payload_plugins,
    register_report_payload_plugin,
    resolve_report_payload_plugin,
)
from benchmark.reporting.framework.unified_report.template_loader import UnifiedTemplateError

__all__ = [
    "CaseColumnSpec",
    "ReportPayload",
    "ReportPayloadContext",
    "ReportPayloadPlugin",
    "UnifiedTemplateError",
    "clear_report_payload_plugins",
    "fill_report_template",
    "force_webdev_report",
    "generate_unified_report_from_spec",
    "generate_unified_report_html",
    "register_report_payload_plugin",
    "resolve_report_payload_plugin",
    "unified_report_enabled",
]
