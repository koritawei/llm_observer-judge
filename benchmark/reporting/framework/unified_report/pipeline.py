"""Unified daily report: the static template is the primary path; the webdev skill is only a fallback when template loading fails."""

import asyncio
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.setting import is_force_webdev_report_runtime
from benchmark.reporting.framework.report_spec import ReportJobInput, assemble_report_spec
from benchmark.reporting.framework.schema import AggregatedMetrics
from benchmark.reporting.framework.unified_report.filler import fill_report_template
from benchmark.reporting.framework.unified_report.payload import ReportPayload
from benchmark.reporting.framework.unified_report.plugin import (
    ReportPayloadContext,
    resolve_report_payload_plugin,
)
from benchmark.reporting.framework.unified_report.report_parser import prepare_report_plugin
from benchmark.reporting.framework.unified_report.template_loader import (
    UnifiedTemplateError,
    load_unified_template,
)

logger = get_logger("agent_eval.unified_report.pipeline")


class _SpecOnlyBenchmarkService:
    """Placeholder benchmark_service when only an assembled spec is held (no DB access)."""

    def __init__(self, case_results: list[Any]) -> None:
        self._case_results = list(case_results)

    async def get_run(self, _run_id: str) -> None:
        return None

    async def load_case_results(self, **_kwargs: object) -> list[Any]:
        return list(self._case_results)

    async def attach_daily_report_context(self, _spec: dict[str, Any]) -> None:
        return None

    async def merge_performance_metrics(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def merge_report_meta(self, *_args: object, **_kwargs: object) -> None:
        return None


def force_webdev_report() -> bool:
    """Whether to force LLM/webdev skill report generation (skip the static template)."""
    return is_force_webdev_report_runtime()


def unified_report_enabled() -> bool:
    """Whether the static unified template is the primary path (default True)."""
    return not force_webdev_report()


def _apply_static_postprocess(html: str, spec: dict[str, Any]) -> str:
    """Light post-processing on the static template path.

    The template already includes interaction and charts; this only does data hygiene and
    replaces the scoring-rule HTML slot. WebdevReportClient is imported lazily to avoid a
    webdev_client → pipeline import cycle.
    """
    from benchmark.reporting.framework.webdev_client import WebdevReportClient  # noqa: PLC0415

    html = WebdevReportClient._inject_report_data(html, spec)
    html = WebdevReportClient._fix_case_data(html, spec)
    html = WebdevReportClient._fill_link_slots(html, spec.get("rule_link_url"))
    html = WebdevReportClient._ensure_charset(html)
    return html


def _template_name_from(
    spec: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Report spec → run config → business plugin template_name → generic template."""
    for source in (spec, config):
        if not isinstance(source, dict):
            continue
        raw = source.get("report_html")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    name = str((spec or {}).get("benchmark_name") or "")
    plugin = resolve_report_payload_plugin(name or None)
    return getattr(plugin, "template_name", None)


def _job_from_spec(spec: dict[str, Any]) -> ReportJobInput:
    """Build a minimal ReportJobInput from an assembled spec (for plugin enrichment)."""
    return ReportJobInput(
        benchmark_service=_SpecOnlyBenchmarkService(  # type: ignore[arg-type]
            list(spec.get("case_results") or []),
        ),
        run_id=str(spec.get("run_id") or ""),
        metrics=AggregatedMetrics(
            total_samples=int(spec.get("total_samples") or 0),
            aggregation_method=str(spec.get("aggregation_method") or "weighted_avg"),
            total_duration_ms=int(spec.get("total_duration_ms") or 0),
        ),
        final_score=float(spec.get("final_score") or 0),
    )


async def _payload_via_plugin(
    *,
    spec: dict[str, Any],
    job: ReportJobInput | None = None,
) -> tuple[ReportPayload, dict[str, Any]]:
    plugin, prepared = prepare_report_plugin(dict(spec))
    if job is None:
        job = _job_from_spec(prepared)
    payload = await plugin.build_payload(ReportPayloadContext(job=job, spec=prepared))
    filled = payload.to_report_spec()
    filled.setdefault("fused_attr_field", plugin.fused_attr_field())
    return payload, filled


async def generate_unified_report_html(
    job: ReportJobInput,
    *,
    spec: dict[str, Any] | None = None,
) -> str:
    """Load the template and fill it.

    When ``spec`` is already assembled, the template name is taken from that spec and
    runs in parallel with the plugin payload. When ``spec`` is empty, ``assemble_report_spec``
    runs first: the template name and the spec share the assembled config (falling back to
    ``run_config`` on the run when ``job.config`` is empty). Parallelism with assembly is
    dropped in that case, so an empty ``job.config`` does not select the default template
    while the spec has already written ``report_html``.

    Raises:
        UnifiedTemplateError: the static template is unavailable (the caller should fall back
        to the webdev skill).
    """
    base = spec if spec is not None else await assemble_report_spec(job)
    template_name = _template_name_from(base, job.config)

    async def _build() -> tuple[ReportPayload, dict[str, Any]]:
        return await _payload_via_plugin(spec=base, job=job)

    template, (payload, filled_spec) = await asyncio.gather(
        load_unified_template(template_name),
        _build(),
    )
    html = fill_report_template(template, payload)
    html = _apply_static_postprocess(html, filled_spec)
    logger.info(
        "unified_report.generated run_id=%s benchmark=%s html_len=%d cases=%d",
        job.run_id,
        filled_spec.get("benchmark_name"),
        len(html),
        len(payload.cases_json()),
    )
    return html


async def generate_unified_report_from_spec(spec: dict[str, Any]) -> str:
    """Unified template path when only an assembled ReportSpec is held.

    Raises:
        UnifiedTemplateError: the static template is unavailable (the caller should fall back
        to the webdev skill).
    """
    template, (payload, filled_spec) = await asyncio.gather(
        load_unified_template(_template_name_from(spec)),
        _payload_via_plugin(spec=spec),
    )
    html = fill_report_template(template, payload)
    html = _apply_static_postprocess(html, filled_spec)
    logger.info(
        "unified_report.from_spec run_id=%s benchmark=%s html_len=%d",
        spec.get("run_id"),
        filled_spec.get("benchmark_name"),
        len(html),
    )
    return html


__all__ = [
    "UnifiedTemplateError",
    "force_webdev_report",
    "generate_unified_report_from_spec",
    "generate_unified_report_html",
    "unified_report_enabled",
]
