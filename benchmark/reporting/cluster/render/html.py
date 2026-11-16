"""Self-contained VUI HTML renderer for Cluster Report."""

from collections import Counter
from html import escape
from statistics import mean
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.reporting.cluster.constants import DIMENSION_LABELS, DIMENSIONS, PASS_SCORE
from benchmark.reporting.cluster.render.presentation import (
    build_base_payload,
    inject_private_fragment,
    render_fallback_html,
    render_private_fragment,
)
from benchmark.reporting.cluster.schema import BatchCaseRecord, ClusterReport
from benchmark.reporting.framework.unified_report.filler import fill_report_template
from benchmark.reporting.framework.unified_report.template_loader import load_unified_template_sync

logger = get_logger(__name__)

_CASES_PER_PAGE = 20
_FAILURE_STATUSES = frozenset({"partial", "fail", "ambiguous", "unknown"})


def render_legacy_html(report: ClusterReport) -> str:
    """Legacy VUI static page, used only as the fallback when the unified template fails to load."""

    effectiveness = report.stats.effectiveness
    score_distribution = _mapping(effectiveness.get("score_distribution"))
    scored_cases = [case for case in report.cases if case.effective_score is not None]
    low_score_cases = [case for case in scored_cases if case.effective_score < PASS_SCORE]
    dimension_rows = _dimension_rows(report.cases)
    cluster_rows = _cluster_rows(report, len(low_score_cases))
    performance = _mapping(report.stats.performance)
    enhancements = _mapping(report.stats.enhancements)

    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
:root {{ color-scheme:light; --ink:#1f2430; --ink-3:#687386; --line:#e6e9ef; --brand:#2563eb; --good:#16a34a; --warn:#f59e0b; --bad:#dc2626; --surface:#fff; --surface-2:#f8fafc; --bg:#f4f7fb; --r-md:12px; --r-lg:16px; --shadow:0 3px 12px rgba(15,23,42,.06); }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif }} main {{ max-width:1180px; margin:32px auto; padding:0 22px }} h1 {{ font-size:28px; margin:0 0 4px }} h2 {{ font-size:18px; margin:30px 0 10px }} .sub,.note {{ color:var(--ink-3) }} .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:20px 0 }} .card,.panel {{ background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); padding:16px; box-shadow:var(--shadow) }} .card {{ position:relative; overflow:hidden }} .card::before {{ content:""; position:absolute; top:0; left:0; right:0; height:3px; background:var(--brand) }} .card b {{ display:block; font-size:25px; margin-top:3px }} .card.fail::before {{ background:var(--bad) }} .card.fail b {{ color:var(--bad) }} .card .label {{ color:var(--ink-3); font-size:12px }} .insight-banner {{ border:1px solid #fde68a; border-left:3px solid var(--warn); background:#fffbeb; color:#78350f; border-radius:var(--r-md); padding:13px 16px; margin:18px 0 }} .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:var(--r-lg); background:#fff }} table {{ width:100%; border-collapse:collapse }} th {{ position:sticky; top:0; z-index:1; background:#f1f5f9; text-align:left; font-weight:600 }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top }} tbody tr:hover {{ background:#f8fbff }} tr:last-child td {{ border-bottom:0 }} small {{ color:var(--ink-3) }} .warning {{ border-left:4px solid var(--warn); padding:10px 12px; background:#fff8e1; border-radius:4px }} .waterfall {{ display:flex; align-items:end; gap:9px; min-height:180px; padding:10px 2px }} .wf-step {{ flex:1; min-width:72px; text-align:center }} .wf-bar {{ min-height:5px; background:linear-gradient(180deg,#fb7185,var(--bad)); border-radius:6px 6px 0 0 }} .wf-start {{ background:linear-gradient(180deg,#60a5fa,var(--brand)) }} .wf-value {{ font-weight:600; font-variant-numeric:tabular-nums }} .wf-label {{ font-size:11px; overflow-wrap:anywhere; color:var(--ink-3) }} .heatmap td {{ text-align:center; font-variant-numeric:tabular-nums }} .pagination {{ display:flex; align-items:center; gap:12px; margin-top:12px }} .pagination button {{ border:1px solid var(--line); border-radius:6px; background:white; padding:6px 10px; cursor:pointer }} .pagination button:disabled {{ color:var(--ink-3); cursor:not-allowed }}
@media (max-width:768px) {{ main {{ margin:16px auto; padding:0 10px }} .cards {{ grid-template-columns:repeat(2,1fr) }} table {{ font-size:12px }} }}
@media (max-width:480px) {{ .cards {{ grid-template-columns:1fr }} }}
</style></head><body><main>
<h1>{title}</h1>
<p class="sub">批次：{batch_tag} · 生成时间：{generated_at} · 评测器：{evaluator_name}</p>
<div class="cards"><div class="card"><span class="label">样本数</span><b>{total}</b></div><div class="card"><span class="label">评测成功</span><b>{scored}</b></div><div class="card"><span class="label">平均分</span><b>{average} / 2.00</b></div><div class="card fail"><span class="label">低于满分</span><b>{low_score}</b></div></div>
{insight_banner}<div class="panel"><b>批次口径</b><ul><li>已召回 {judged} 条 Judge 记录，期望样本数 {expected_total} 条。</li><li>综合分分布：0 分 {score_zero}、1 分 {score_one}、2 分 {score_two}。</li><li>归因状态：已完成 {attributed} 条，待归因 {pending} 条。</li></ul></div>
<h2>效果概览</h2><p class="note">单维度均分 = 该维度所有可评分样本的 <code>score_hint</code> 之和 ÷ 可评分样本数；单条评分为 0/1/2 分。<code>not_applicable</code> 与 <code>unknown</code> 不计入均分；Benchmark run 级汇总分取各适用维度均分中的最低值。</p>{effectiveness_cards}
<h2>性能概览</h2>{performance_cards}<h2>VUI 路由分布</h2>{route_table}<h2>模型 Token 消耗</h2>{performance_table}
<h2>维度汇总</h2>{dimension_table}
<h2>分数损失瀑布</h2>{waterfall}
<h2>根因 × 维度失分矩阵</h2>{root_cause_matrix}
<h2>低分问题聚类</h2><p class="note">仅统计综合分低于 2.0 的 {low_score} 条样本；仅展示评测分析生成的 L2 问题簇，不使用归因结果。</p>{cluster_table}
<h2>归因收敛情况</h2>{coverage_table}
<h2>失败案例</h2>{case_table}
<p class="warning"><b>解读限制：</b>本报告基于已持久化的 Judge 与归因记录聚合；性能或归因未收敛时会在批次口径和收敛情况中明确标识。</p>
</main></body></html>""".format(  # noqa: E501 - preserve embedded HTML/CSS literal
        title=escape(f"VUI 聚类评测报告 · {report.batch_tag}"),
        batch_tag=escape(report.batch_tag or "未记录"),
        generated_at=escape(report.generated_at or "未记录"),
        evaluator_name=escape(
            report.evaluator_name or str(report.batch_meta.get("evaluator_name") or "未记录")
        ),
        total=effectiveness.get("total", len(report.cases)),
        scored=len(scored_cases),
        average=_number(effectiveness.get("avg_score")),
        low_score=len(low_score_cases),
        judged=report.coverage.judged,
        expected_total=report.coverage.expected_total,
        score_zero=score_distribution.get("0", 0),
        score_one=score_distribution.get("1", 0),
        score_two=score_distribution.get("2", 0),
        attributed=report.coverage.attributed,
        pending=report.coverage.pending,
        insight_banner=_insight_banner(report, enhancements),
        effectiveness_cards=_dimension_average_cards(dimension_rows),
        performance_cards=_performance_cards(performance),
        performance_table=_performance_table(performance),
        route_table=_route_table(performance),
        dimension_table=_table(
            [
                "维度（检查内容）",
                "均分",
                "可评分样本",
                "不适用（规则未生效，不参与评分）",
                "存在扣分",
            ],
            dimension_rows,
        ),
        waterfall=_waterfall(enhancements.get("score_loss_waterfall")),
        root_cause_matrix=_root_cause_matrix(enhancements.get("root_cause_dim_matrix")),
        cluster_table=_table(
            ["问题类别", "样本数", "低分样本占比", "典型 case", "问题分析"],
            cluster_rows,
        ),
        coverage_table=_coverage_table(report),
        case_table=_case_table(report),
    )


def _insight_banner(report: ClusterReport, enhancements: dict[str, Any]) -> str:
    """Reuse the daily-report insight banner semantics and prefer the L2-verified batch summary."""

    headline = str(report.llm_summary.get("headline") or "").strip()
    waterfall = _mapping(enhancements.get("score_loss_waterfall"))
    steps = _dict_list(waterfall.get("steps"))
    if not headline and steps:
        headline = f"主要失分来源：{steps[0].get('label') or '未归因'}。"
    return f'<div class="insight-banner">💡 {escape(headline)}</div>' if headline else ""


def _waterfall(value: object) -> str:
    """Render the shared score-loss waterfall; degrade explicitly when it is missing or every score is full."""

    waterfall = _mapping(value)
    steps = _dict_list(waterfall.get("steps"))
    start = waterfall.get("start")
    if not isinstance(start, int | float) or not steps:
        return '<p class="note">无可分摊的失分数据（可能全部满分或无有效评分）。</p>'
    columns = [
        (
            '<div class="wf-step"><div class="wf-'
            'value">'
            f"{float(start):.2f}"
            '</div><div class="wf-bar wf-start" s'
            'tyle="height:100%"></div><div class='
            '"wf-label">满分</div></div>'
        )
    ]
    for step in steps:
        loss = step.get("loss")
        if not isinstance(loss, int | float):
            continue
        height = min(100.0, max(4.0, float(loss) / float(start) * 100.0))
        columns.append(
            (
                '<div class="wf-step"><div class="wf-'
                'value">-{loss:.2f}</div><div class="'
                'wf-bar" style="height:{height:.1f}%"'
                '></div><div class="wf-label">{label}'
                "<br><small>{count} 条</small></div></"
                "div>"
            ).format(
                loss=float(loss),
                height=height,
                label=escape(str(step.get("label") or "未归因")),
                count=int(step.get("count") or 0),
            )
        )
    return (
        '<div class="panel"><div class="water'
        'fall">{}</div><p class="note">柱高按综合分'
        "落差分摊；未归因失分单列，便于判断归因覆盖盲区。</p></div>"
    ).format("".join(columns))


def _root_cause_matrix(value: object) -> str:
    """Render the root-cause × dimension mean score-loss heatmap."""

    matrix = _mapping(value)
    rows = [str(row) for row in matrix.get("rows") or [] if row]
    columns = _dict_list(matrix.get("cols"))
    cells = _mapping(matrix.get("cells"))
    if not rows or not columns:
        return '<p class="note">暂无可计算的根因与维度交集。</p>'
    header = "".join(f"<th>{escape(str(column.get('id') or ''))}</th>" for column in columns)
    body: list[str] = []
    for row in rows:
        row_cells = _mapping(cells.get(row))
        cells_html = []
        for column in columns:
            loss = row_cells.get(str(column.get("id") or ""))
            if not isinstance(loss, int | float):
                cells_html.append("<td>—</td>")
                continue
            opacity = min(0.85, max(0.08, float(loss) / 2.0))
            cells_html.append(
                f'<td style="background:rgba(220,38,38,{opacity:.2f})">{float(loss):.2f}</td>'
            )
        body.append(f"<tr><td>{escape(row)}</td>{''.join(cells_html)}</tr>")
    return (
        '<div class="table-wrap"><table class'
        '="heatmap"><thead><tr><th>根因</th>{}<'
        "/tr></thead><tbody>{}</tbody></table"
        '></div><p class="note">单元格为该根因下对应维度的'
        "平均失分；满分 case 以 0 计入分母。</p>"
    ).format(header, "".join(body))


def _dimension_rows(cases: list[BatchCaseRecord]) -> list[list[str]]:
    rows: list[list[str]] = []
    for dimension in DIMENSIONS:
        items = [
            item
            for case in cases
            for item in case.dimension_results
            if isinstance(item, dict) and item.get("dimension") == dimension
        ]
        evaluable = [item for item in items if item.get("status") != "not_applicable"]
        scores = [
            float(item["score_hint"])
            for item in evaluable
            if isinstance(item.get("score_hint"), int | float)
        ]
        statuses = Counter(str(item.get("status") or "unknown") for item in items)
        rows.append(
            [
                f"<b>{escape(dimension)}</b><br><small>{escape(DIMENSION_LABELS[dimension])}</small>",
                "—" if not scores else f"{mean(scores):.2f} / 2.00",
                str(len(evaluable)),
                str(statuses["not_applicable"]),
                str(
                    sum(count for status, count in statuses.items() if status in _FAILURE_STATUSES)
                ),
            ]
        )
    return rows


def _dimension_average_cards(dimension_rows: list[list[str]]) -> str:
    """Show per-dimension mean scores for D0-D6 as cards."""

    cards = "".join(
        '<div class="card"><span class="label">'
        f"{escape(dimension)} · {escape(DIMENSION_LABELS[dimension])}</span>"
        f"<b>{row[1]}</b></div>"
        for dimension, row in zip(DIMENSIONS, dimension_rows, strict=True)
    )
    return f'<div class="cards">{cards}</div>'


def _cluster_rows(report: ClusterReport, low_score_count: int) -> list[list[str]]:
    clusters = _dict_list(report.llm_summary.get("clusters"))
    if clusters:
        return [
            [
                escape(str(cluster.get("title") or "未命名问题簇")),
                str(cluster.get("affected_count") or 0),
                _share(cluster.get("affected_count"), low_score_count),
                _related_case_details(report, cluster.get("related_cases")),
                _cluster_analysis(cluster),
            ]
            for cluster in clusters
        ]

    return []


def _cluster_analysis(cluster: dict[str, Any]) -> str:
    """Render the commonality analysis and evidence summary of L2 clusters."""

    parts = [
        f"<b>共性：</b>{escape(str(cluster['summary']))}"
        for key in ("summary",)
        if cluster.get(key)
    ]
    if cluster.get("evidence"):
        parts.append(f"<br><small>依据：{escape(str(cluster['evidence']))}</small>")
    return "".join(parts) or "未生成分析"


def _related_case_details(report: ClusterReport, related_cases: object) -> str:
    """Show at most three reviewable typical cases by the trace ids returned from L2."""

    trace_ids = (
        [str(trace_id) for trace_id in related_cases if trace_id]
        if isinstance(related_cases, list)
        else []
    )
    cases_by_trace = {case.trace_id: case for case in report.cases}
    examples = [cases_by_trace[trace_id] for trace_id in trace_ids if trace_id in cases_by_trace][
        :3
    ]
    if not examples:
        return "未关联 case"
    items = "".join(
        "<li>"
        f"<b>{escape(case.case_key.case_id or '实时请求')}</b>："
        f"{escape(case.case_key.query or '未采集')}"
        f"（{escape(_number(case.effective_score))} 分）</li>"
        for case in examples
    )
    return f"<details><summary>查看 {len(examples)} 条</summary><ul>{items}</ul></details>"


def _root_cause_cases(report: ClusterReport, code: str) -> str:
    """On L1 fallback, show low-score samples of the same root cause so there is reviewable evidence, not only counts."""

    examples = [
        case
        for case in report.cases
        if case.effective_score is not None
        and case.effective_score < PASS_SCORE
        and code in case.attribution.all_root_cause_codes
    ][:3]
    if not examples:
        return "未关联 case"
    items = "".join(
        "<li>"
        f"<b>{escape(case.case_key.case_id or '实时请求')}</b>："
        f"{escape(case.case_key.query or '未采集')}"
        f"（{escape(_number(case.effective_score))} 分）</li>"
        for case in examples
    )
    return f"<details><summary>查看 {len(examples)} 条</summary><ul>{items}</ul></details>"


def _coverage_table(report: ClusterReport) -> str:
    """Show attribution convergence in one line, focused on the completion ratio of samples that should be attributed."""

    coverage = report.coverage
    attributed_share = (
        coverage.attributed / coverage.should_attribute if coverage.should_attribute else 1.0
    )
    notes = "；".join(item for item in (coverage.watcher_note, coverage.perf_note) if item)
    summary = (
        f"归因完成：<b>{coverage.attributed} / {coverage.should_attribute} "
        f"（{attributed_share:.1%}）</b>；待归因 {coverage.pending} 条；"
        f"已判 {coverage.judged} / {coverage.expected_total} 条。"
    )
    return (
        f'<div class="panel">{summary}<p class="note">{escape(notes or "无收敛告警。")}</p></div>'
    )


def _performance_cards(performance: dict[str, Any]) -> str:
    e2e = _mapping(performance.get("e2e_ms"))
    tokens = _mapping(performance.get("total_tokens_per_case"))
    collected = performance.get("collected", 0)
    missing = performance.get("missing", 0)
    return (
        '<div class="cards"><div class="card"'
        '><span class="label">平均端到端延迟</span><'
        "b>"
        f"{_number(e2e.get('avg'))}"
        ' ms</b></div><div class="card"><span'
        ' class="label">P95 端到端延迟</span><b>'
        f"{_number(e2e.get('p95'))}"
        ' ms</b></div><div class="card"><span'
        ' class="label">平均总 Token / case</spa'
        "n><b>"
        f"{_number(tokens.get('avg'))}"
        '</b></div><div class="card"><span cl'
        'ass="label">性能采集覆盖</span><b>'
        f"{collected}"
        " / "
        f"{collected + missing}"
        "</b></div></div>"
    )


def _performance_table(performance: dict[str, Any]) -> str:
    """Show performance by the actual Langfuse model, without exposing call-stage buckets."""

    models = _mapping(performance.get("models"))
    rows = []
    for model, value in models.items():
        stats = _mapping(value)
        if not stats.get("cases"):
            continue
        token_stats = _mapping(stats.get("total_tokens"))
        latency_stats = _mapping(stats.get("latency_ms"))
        rows.append(
            [
                escape(str(model)),
                str(stats.get("cases", 0)),
                _number(_mapping(stats.get("calls")).get("avg")),
                _number(token_stats.get("avg")),
                _number(latency_stats.get("avg")),
            ]
        )
    return _table(
        ["模型", "命中 case", "平均调用数", "平均 Token / case", "平均单调用耗时(ms)"],
        rows,
    )


def _route_table(performance: dict[str, Any]) -> str:
    route_modes = _mapping(performance.get("route_modes"))
    modes = _mapping(route_modes.get("modes"))
    labels = (("lite_sdm", "Lite-SDM"), ("lite_dag", "Lite-DAG"), ("standard", "Standard"))
    rows = [
        [
            label,
            str(_mapping(modes.get(mode)).get("count", 0)),
            _percent(_mapping(modes.get(mode)).get("share")),
        ]
        for mode, label in labels
    ]
    rows.append(["未分类", str(route_modes.get("unclassified", 0)), "不计入有效比例分母"])
    return _table(["链路", "case 数", "比例"], rows)


def _case_table(report: ClusterReport) -> str:
    failed = [
        case
        for case in report.cases
        if case.effective_score is not None and case.effective_score < PASS_SCORE
    ]
    rows = [
        [
            escape(case.trace_id),
            escape(case.case_key.case_id or "实时请求"),
            escape(case.case_key.query or "未采集"),
            escape(_number(case.effective_score)),
            escape(case.attribution.root_cause or case.reason or "未归因"),
        ]
        for case in failed
    ]
    table = _table(
        ["Trace", "Case", "Query", "综合分", "原因"],
        rows,
        table_id="failed-cases",
    )
    if len(rows) <= _CASES_PER_PAGE:
        return table
    return table + (
        '<div class="pagination"><button type'
        '="button" id="failed-cases-prev">上一页'
        '</button><span id="failed-cases-page'
        '"></span><button type="button" id="f'
        'ailed-cases-next">下一页</button></div>'
        "\n<script>(() => {\n  const rows = [.."
        ".document.querySelectorAll('#failed-"
        "cases tbody tr')];\n  const pageSize "
        "= "
        f"{_CASES_PER_PAGE}"
        ";\n  const pageLabel = document.getEl"
        "ementById('failed-cases-page');\n  co"
        "nst previous = document.getElementBy"
        "Id('failed-cases-prev');\n  const nex"
        "t = document.getElementById('failed-"
        "cases-next');\n  let page = 0;\n  cons"
        "t render = () => {\n    const pageCou"
        "nt = Math.ceil(rows.length / pageSiz"
        "e);\n    rows.forEach((row, index) =>"
        " { row.hidden = Math.floor(index / p"
        "ageSize) !== page; });\n    pageLabel"
        ".textContent = `第 ${page + 1} / ${pa"
        "geCount} 页，共 ${rows.length} 条`;\n    "
        "previous.disabled = page === 0;\n    "
        "next.disabled = page === pageCount -"
        " 1;\n  };\n  previous.addEventListener"
        "('click', () => { page -= 1; render("
        "); });\n  next.addEventListener('clic"
        "k', () => { page += 1; render(); });"
        "\n  render();\n})();</script>"
    )


def _table(headers: list[str], rows: list[list[str]], *, table_id: str = "") -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    table_id_attr = f' id="{escape(table_id)}"' if table_id else ""
    return (
        '<div class="table-wrap"><table'
        f"{table_id_attr}"
        "><thead><tr>"
        f"{head}"
        "</tr></thead><tbody>"
        f"{body or _empty_row(len(headers))}"
        "</tbody></table></div>"
    )


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_list(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _number(value: object) -> str:
    return f"{float(value):.2f}" if isinstance(value, int | float) else "-"


def _percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, int | float) else "-"


def _share(value: object, total: int) -> str:
    return f"{int(value) / total:.1%}" if isinstance(value, int | float) and total else "—"


def _empty_row(columns: int) -> str:
    return f'<tr><td class="note" colspan="{columns}">暂无数据</td></tr>'


def render_html(report: ClusterReport) -> str:
    """Build the shared base page and inject the Cluster private scene section; on failure, degrade to a scene-neutral page."""

    try:
        base_html = fill_report_template(load_unified_template_sync(), build_base_payload(report))
        html, anchor_found = inject_private_fragment(base_html, render_private_fragment(report))
        if not anchor_found:
            logger.warning(
                "cluster_report_html_anchor_missing evaluator=%s",
                report.evaluator_name,
            )
        return html
    except Exception:  # noqa: BLE001 - failure of the HTML artifact alone must not block the main report path
        logger.warning(
            "cluster_report_html_template_fallback evaluator=%s",
            report.evaluator_name,
            exc_info=True,
        )
        return render_fallback_html(report)


__all__ = ["render_html", "render_legacy_html"]
