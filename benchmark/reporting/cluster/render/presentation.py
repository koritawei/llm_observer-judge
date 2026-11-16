"""Private presentation adapter for Cluster Report: shared base-page data and scene Fragments.

Vehicle/VUI dimensions both use 2 as the full score. Task completion rate is counted by
whether a Judge score exists; business tiers are shown separately. Attribution shares use
the full attributed count, and items outside the Top 10 are counted as "other". Overall and
dimension reasons are passed through to ``scores[*].reason``, which the shared template reads.
"""

import json
from html import escape
from statistics import mean
from typing import Any

from benchmark.reporting.cluster.schema import BatchCaseRecord, ClusterReport
from benchmark.reporting.cluster.unified_payload import _root_cause_label, build_vui_metrics
from benchmark.runs.common.trace_url_builder import TRACE_TRACE_URL_BASE
from benchmark.reporting.cluster_service.evaluator_profile import resolve_evaluator_report_profile
from benchmark.reporting.framework.unified_report.payload import ReportPayload


def build_base_payload(report: ClusterReport) -> ReportPayload:
    """Assemble only shared ``ReportPayload`` fields; do not pass scene extension fields to the shared template."""

    profile = resolve_evaluator_report_profile(report.evaluator_name or "vui_agent")
    dimensions = [criterion.id for criterion in profile.rubric.criteria()]
    cases = [_case(record, dimensions) for record in report.cases]
    criterion_metrics = _criterion_metrics(report.cases, dimensions)
    criterion_means = {
        dimension: float(metric["mean"])
        for dimension, metric in criterion_metrics.items()
        if isinstance(metric.get("mean"), int | float)
    }
    distribution = _mapping(report.stats.effectiveness.get("score_distribution"))
    return ReportPayload(
        run_id=report.batch_tag,
        benchmark_name=profile.benchmark_name,
        service_name=profile.service_name,
        dataset_name=str(report.batch_meta.get("dataset_name") or profile.dataset_name),
        final_score=min(criterion_means.values()) if criterion_means else None,
        aggregation_method="min_dimension_mean",
        total_samples=int(report.stats.effectiveness.get("total") or len(report.cases)),
        total_success=sum(record.judge_score is not None for record in report.cases),
        total_failed=sum(record.judge_score is None for record in report.cases),
        criteria=[
            {
                "id": criterion.id,
                "name": f"{criterion.id} · {criterion.name}",
                "weight": criterion.weight,
                "full_score": _FULL_SCORE,
            }
            for criterion in profile.rubric.criteria()
        ],
        criterion_metrics=criterion_metrics,
        today_criterion_means=criterion_means,
        attribution_metrics=_attribution_metrics(report),
        performance_metrics=_performance_metrics(report),
        case_results=cases,
        grade_distribution={
            "EXCELLENT": int(distribution.get("2") or 0),
            "GOOD": 0,
            "ACCEPTABLE": int(distribution.get("1") or 0),
            "POOR": 0,
            "FAIL": int(distribution.get("0") or 0),
        },
        insight_text=str(report.llm_summary.get("headline") or "").strip() or None,
        top_problems=[
            {
                "name": item.get("title"),
                "affected_cases": item.get("affected_count"),
                "typical_case_desc": item.get("summary"),
                "chain": [item["evidence"]] if item.get("evidence") else [],
            }
            for item in _dict_list(report.llm_summary.get("clusters"))
        ],
        report_date=_display_data_window(report),
        score_loss_waterfall=_mapping(report.stats.enhancements.get("score_loss_waterfall"))
        or None,
        root_cause_dim_matrix=_mapping(report.stats.enhancements.get("root_cause_dim_matrix"))
        or None,
    )


def _display_data_window(report: ClusterReport) -> str:
    """Time-window reports prefer the sample range, so the generation date is not mistaken for the data date."""

    start = report.data_window.get("start")
    end = report.data_window.get("end")
    if isinstance(start, str) and isinstance(end, str) and start and end:
        return f"数据范围：{start[:16].replace('T', ' ')} ～ {end[:16].replace('T', ' ')}"
    return (report.generated_at or report.batch_tag)[:10]


def render_private_fragment(report: ClusterReport) -> str:
    """Return a scene-diagnosis Fragment injectable into the base page; all business text is HTML-escaped."""

    if (report.evaluator_name or "vui_agent") == "vui_agent":
        return _render_vui_fragment(report)
    return _render_vehicle_fragment(report)


def inject_private_fragment(html: str, fragment: str) -> tuple[str, bool]:
    """Insert the Fragment before Case; if the upgraded anchor is missing, append it to the end of the main content."""

    anchor = '<div id="case-section"'
    if anchor in html:
        return html.replace(anchor, f"{fragment}\n{anchor}", 1), True
    closing_tag = "</main>"
    if closing_tag in html:
        return html.replace(closing_tag, f"{fragment}\n{closing_tag}", 1), False
    return f"{html}\n{fragment}", False


def render_fallback_html(report: ClusterReport) -> str:
    """When the shared template is unavailable, provide scene-neutral minimal HTML so the report is still readable besides Markdown/JSON."""

    profile = resolve_evaluator_report_profile(report.evaluator_name or "vui_agent")
    fragment = render_private_fragment(report)
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title><style>
body{{font:14px/1.6 sans-serif;margin:24px;color:#1f2430}}
.card{{border:1px solid #e6e9ef;border-radius:12px;padding:16px;margin:12px 0}}
.sec{{font-weight:700;margin-top:24px}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #e6e9ef;padding:8px;text-align:left}}
</style></head><body><h1>{title}</h1><p>批次：{batch}；生成时间：{generated}</p>
<div class="card">样本数：{total}；平均分：{score}</div>{fragment}</body></html>""".format(
        title=escape(f"{profile.service_name} Cluster Report"),
        batch=escape(report.batch_tag),
        generated=escape(report.generated_at),
        total=len(report.cases),
        score=escape(str(report.stats.effectiveness.get("avg_score") or "-")),
        fragment=fragment,
    )


def _case(record: BatchCaseRecord, dimensions: list[str]) -> dict[str, Any]:
    scores = {
        str(item["dimension"]): {
            "score": item.get("score_hint"),
            "status": item.get("status"),
            "reason": item.get("reason") or "",
        }
        for item in record.dimension_results
        if isinstance(item, dict) and item.get("dimension") in dimensions
    }
    score = record.effective_score
    return {
        "trace_id": record.trace_id,
        "case_id": record.case_key.case_id,
        "display_case_id": record.case_key.case_id or "实时请求",
        "display_input": _query(record),
        "display_output": _display_output(
            record.trace_meta.get("output") or record.trace_meta.get("response")
        ),
        "trace_url": _original_trace_url(record),
        "grade": _grade(score),
        "scores": {**scores, "fused": {"score": score, "reason": record.reason}},
        "root_cause_display": _root_cause_label(
            record.attribution.root_cause_code
            or record.attribution.root_cause
            or record.reason
            or "未归因"
        ),
        "attribution": {
            "primary_issue_category": record.attribution.root_cause_code,
            "root_cause": _root_cause_label(
                record.attribution.root_cause_code or record.attribution.root_cause
            ),
            "reasoning_raw": record.attribution.root_cause or record.reason,
            "action_hint": record.attribution.action_hint,
            "confidence": record.attribution.confidence,
        },
        "reason": record.reason,
    }


def _criterion_metrics(
    records: list[BatchCaseRecord], dimensions: list[str]
) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for dimension in dimensions:
        values = [
            float(item["score_hint"])
            for record in records
            for item in record.dimension_results
            if isinstance(item, dict)
            and item.get("dimension") == dimension
            and item.get("status") not in {"not_applicable", "unknown"}
            and isinstance(item.get("score_hint"), int | float)
        ]
        metrics[dimension] = {
            "mean": round(mean(values), 4) if values else None,
            "count": len(values),
        }
    return metrics


def _attribution_metrics(report: ClusterReport) -> dict[str, Any]:
    roots = _dict_list(report.stats.attribution.get("root_cause_top"))
    if not roots:
        return {}
    total = int(report.stats.attribution.get("attributed") or 0)
    counts = {str(item.get("code") or "未归因"): int(item.get("count") or 0) for item in roots}
    remaining = total - sum(counts.values())
    label = "根因"
    if remaining > 0:
        counts["其他"] = counts.get("其他", 0) + remaining
        label = "根因（Top10，其余合并为其他）"
    return {
        "fused": {
            "total_attributed": total,
            "field_distributions": {label: counts},
        }
    }


def _performance_metrics(report: ClusterReport) -> dict[str, Any]:
    performance = report.stats.performance
    e2e = _mapping(performance.get("e2e_ms"))
    first_text = _mapping(performance.get("first_text_ms"))
    tokens = _mapping(performance.get("total_tokens_per_case"))
    collected = int(performance.get("collected") or 0)
    return {
        "latency": {
            "sample_count": int(e2e.get("count") or collected),
            "p50_ms": e2e.get("p50"),
            "p99_ms": e2e.get("p99"),
        },
        "ttft": {
            "sample_count": int(first_text.get("count") or 0),
            "p50_ms": first_text.get("p50"),
            "p99_ms": first_text.get("p99"),
        },
        "tokens": {
            "sample_count": int(tokens.get("count") or collected),
            "avg_total": tokens.get("avg"),
            "avg_input": None,
            "avg_output": None,
        },
    }


def _render_vui_fragment(report: ClusterReport) -> str:
    """Render run observations and low-score problem clusters in the VUI private Fragment."""

    metrics_json = json.dumps(build_vui_metrics(report), ensure_ascii=False).replace("</", "<\\/")
    return f"""<script>
document.addEventListener("DOMContentLoaded", function () {{
  var perfSection = document.getElementById("perf-section");
  if (!perfSection || !perfSection.parentNode) return;
  try {{
    var metrics = {metrics_json};
    var section = document.createElement("section");
    section.id = "vui-cluster-metrics";
    perfSection.parentNode.insertBefore(section, perfSection);

    var routes = metrics.routes && metrics.routes.modes || {{}};
    var labels = {{lite_sdm: "Lite-SDM", lite_dag: "Lite-DAG", standard: "Standard"}};
    var routeRows = Object.keys(labels).map(function (key) {{
      var item = routes[key] || {{}};
      return "<tr><td>" + labels[key] + "</td><td>" + Number(item.count || 0)
        + "</td><td>" + (Number(item.share || 0) * 100).toFixed(1) + "%</td></tr>";
    }}).join("");
    var models = metrics.models || {{}};
    var modelRows = Object.keys(models).map(function (name) {{
      var item = models[name] || {{}};
      var calls = item.calls || {{}};
      var tokens = item.total_tokens || {{}};
      var latency = item.latency_ms || {{}};
      return "<tr><td>" + escHtml(name) + "</td><td>" + Number(item.cases || 0)
        + "</td><td>" + fmtNum(calls.avg, 2) + "</td><td>" + fmtNum(tokens.avg, 0)
        + "</td><td>" + fmtDurationMs(latency.avg) + "</td></tr>";
    }}).join("");
    var html = '<div class="sec">VUI 运行观测</div>';
    html += '<div class="card"><div class="attr-title">VUI 路由分布</div>'
      + '<table class="attr-tbl"><thead><tr><th>链路</th><th>case 数</th><th>比例</th>'
      + '</tr></thead><tbody>' + routeRows + '</tbody></table></div>';
    if (modelRows) {{
      html += '<div class="card"><div class="attr-title">真实模型与 Token 消耗</div>'
        + '<table class="attr-tbl"><thead><tr><th>模型</th><th>命中 case</th><th>平均调用数</th>'
        + '<th>平均 Token / case</th><th>平均单调用耗时</th></tr></thead><tbody>'
        + modelRows + '</tbody></table></div>';
    }}
    var total = Number(metrics.low_score_total || 0);
    if (total > 0) {{
      var clusters = Array.isArray(metrics.clusters) ? metrics.clusters : [];
      var rows = clusters.map(function (cluster) {{
        var cases = Array.isArray(cluster.cases) ? cluster.cases.slice(0, 3) : [];
        var casesHtml = cases.length
          ? '<details><summary>查看 ' + cases.length + ' 条</summary><ul>'
            + cases.map(function (item) {{
              return '<li><b>' + escHtml(item.id || '实时请求') + '</b>：'
                + escHtml(item.query || '未采集')
                + '（原始评测 ' + fmtNum(item.score, 2) + ' 分）</li>';
            }}).join('') + '</ul></details>'
          : '未关联 case';
        var evidence = cluster.evidence
          ? '<br><span class="muted">依据：' + escHtml(cluster.evidence) + '</span>'
          : '';
        return '<tr><td>' + escHtml(cluster.name || '未命名问题簇') + '</td><td>'
          + Number(cluster.count || 0) + '</td><td>'
          + (Number(cluster.count || 0) / total * 100).toFixed(1) + '%</td><td>'
          + casesHtml + '</td><td><b>共性：</b>'
          + escHtml(cluster.summary || '未生成共性分析') + evidence + '</td></tr>';
      }}).join('');
      html += '<div class="sec">低分问题聚类</div>'
        + '<div class="muted" style="margin:-8px 0 10px">'
        + '仅统计原始评测分低于 2.0 的 ' + total
        + ' 条样本；仅基于评测原因与维度扣分生成问题簇。</div>';
      var tableHead = '<thead><tr><th>问题类别</th><th>样本数</th><th>低分样本占比</th>'
        + '<th>典型 case</th><th>问题分析</th></tr></thead><tbody>';
      html += rows
        ? '<div class="card"><table class="attr-tbl">' + tableHead
          + rows + '</tbody></table></div>'
        : '<div class="card muted">本批低分样本尚未生成评测分析问题簇。</div>';
    }}
    section.innerHTML = html;
  }} catch (_) {{}}
}});
</script>"""


def _render_vehicle_fragment(report: ClusterReport) -> str:
    clusters = _dict_list(report.llm_summary.get("clusters"))
    rows = (
        "".join(
            "<tr><td>{title}</td><td>{count}</td><td>{summary}</td></tr>".format(
                title=escape(str(item.get("title") or "未命名问题簇")),
                count=escape(str(item.get("affected_count") or 0)),
                summary=escape(str(item.get("summary") or "")),
            )
            for item in clusters
        )
        or '<tr><td colspan="3">暂无路线规划或工具调用失败簇</td></tr>'
    )
    return f"""<section class="vehicle-cluster-fragment">
<div class="sec">Vehicle 运行诊断</div><div class="card"><table class="attr-tbl">
<thead><tr><th>问题簇</th><th>影响 case</th><th>诊断摘要</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>"""


def _original_trace_url(record: BatchCaseRecord) -> str:
    """Return the link to the original business Trace, not the evaluation-process Trace."""

    persisted = str(record.trace_meta.get("trace_url") or "").strip()
    if persisted:
        return persisted
    project_id = record.project_id.strip()
    trace_id = record.trace_id.strip()
    if not project_id or not trace_id:
        return ""
    return f"{TRACE_TRACE_URL_BASE}/project/{project_id}/traces/{trace_id}?traceId={trace_id}"


def _query(record: BatchCaseRecord) -> str:
    if record.case_key.query:
        return record.case_key.query
    for source in (record.case_meta, record.trace_meta):
        query = _find_query(source)
        if query:
            return query
    return ""


def _find_query(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                return _find_query(json.loads(text))
            except (TypeError, ValueError):
                return text
    if not isinstance(value, dict):
        return ""
    for key in ("query", "user_query", "query_text"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    for key in ("input", "input_snapshot", "request", "request_body", "payload"):
        query = _find_query(value.get(key))
        if query:
            return query
    return ""


def _display_output(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError):
                return text
            return _display_output(decoded) or text
    if not isinstance(value, dict):
        return ""
    component = value.get("component")
    if isinstance(component, dict):
        for key in ("displayText", "text", "content"):
            text = component.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    for key in ("displayText", "text", "content", "answer"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


_FULL_SCORE = 2


def _grade(score: float | None) -> str:
    if score is None:
        return ""
    if score >= _FULL_SCORE:
        return "EXCELLENT"
    if score >= 1:
        return "ACCEPTABLE"
    return "FAIL"


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_list(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


__all__ = [
    "build_base_payload",
    "inject_private_fragment",
    "render_fallback_html",
    "render_private_fragment",
]
