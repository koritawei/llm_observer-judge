"""T32 Markdown / JSON rendering — a human summary and the full machine-readable payload.

- **JSON**: the full structured artifact (`ClusterReport` serialized directly), including
  `schema_version` + `batch_meta`. It is the only authoritative format for cross-batch
  comparison and downstream consumption.
- **Markdown**: a human summary, L1 numbers + L2 semantic clusters. All numbers come from
  L1; L2 contributes text only.
"""

import json
from collections.abc import Callable
from typing import Any

from benchmark.reporting.cluster.constants import BUCKET_LABELS, MODEL_BUCKETS
from benchmark.reporting.cluster.schema import ClusterReport
from benchmark.reporting.cluster.stats.stability import stability_group_sort_key
mode_name = None


def render_json(report: ClusterReport, output_path: str) -> str:
    """Write the full machine-readable JSON."""

    payload = report.model_dump(mode="json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


def render_markdown(
    report: ClusterReport,
    *,
    cluster_links: dict[int, str] | None = None,
    cluster_dispatch_statuses: dict[int, str] | None = None,
) -> str:
    """Render the human-summary Markdown.

    ``cluster_links`` and ``cluster_dispatch_statuses`` are injected only when writing back
    the Aone parent ticket body, to add formal cluster links and dispatch-failure status.
    Artifacts written to disk pass empty maps before Aone child tickets exist, so they stay
    independently readable.
    """

    job_type = str(report.batch_meta.get("report_job_type") or "normal")
    title = {
        "normal": "聚合根因报告",
        "stability": "稳定性聚合报告",
        "performance": "性能报告",
    }.get(job_type, "批次评测聚合总报告")
    evaluator_name = report.evaluator_name or str(
        report.batch_meta.get("evaluator_name") or "未记录"
    )
    lines: list[str] = [
        f"# {title} · {report.batch_tag}",
        "",
        f"评测器：{evaluator_name}",
        "",
    ]
    if job_type == "stability":
        groups = [
            group
            for group in report.stats.observations.get("stability") or []
            if isinstance(group, dict)
        ]
        lines.extend(_stability_conclusion_lines(report, groups))
        lines.extend(_stability_priority_lines(report.llm_summary or {}, groups))
        lines.extend(_stability_distribution_lines(groups))
        lines.extend(["", "## 评测结果证据"])
        lines.extend(_coverage_lines(report, include_perf_note=False))
        lines.extend(_effectiveness_lines(report.stats.effectiveness))
        lines.extend(_stability_index_lines(groups))
        lines.extend(_stability_evidence_lines(report.llm_summary or {}, groups))
        lines.extend(_stability_caveat_lines(report.llm_summary or {}))
    else:
        lines.extend(
            _llm_summary_lines(
                report.llm_summary,
                candidate_fallback=report.stats.attribution,
                cluster_links=cluster_links,
                cluster_dispatch_statuses=cluster_dispatch_statuses,
            )
        )
        if job_type != "performance":
            lines.extend(["", "## 评测结论"])
            lines.extend(_coverage_lines(report, include_perf_note=False))
            lines.extend(_effectiveness_lines(report.stats.effectiveness))
            lines.extend(["", "## 归因结论"])
            lines.extend(_attribution_lines(report.stats.attribution))
            lines.extend(_crosstab_lines(report.stats.crosstab))
            if report.stats.performance:
                lines.extend(["", "## 性能与路由"])
                lines.extend(_coverage_lines(report, include_perf_note=True))
                lines.extend(_performance_lines(report.stats.performance))
        if job_type == "performance":
            lines.extend(["", "## 性能与运行观测"])
            lines.extend(_coverage_lines(report, include_perf_note=True))
            lines.extend(_performance_lines(report.stats.performance))
            lines.extend(_observation_lines(report.stats.observations))
        elif report.stats.observations:
            lines.extend(["", "## 运行观测"])
            lines.extend(_observation_lines(report.stats.observations))
    lines.extend(
        [
            "",
            "## 附录",
            "",
            "### 批次信息",
            "",
            f"- 批次: {report.batch_tag}",
            f"- schema: `{report.schema_version}`",
            f"- 生成时间: {report.generated_at}",
        ]
    )
    lines.extend(_batch_meta_lines(report.batch_meta))
    return "\n".join(_number_headings(lines)) + "\n"


def _number_headings(lines: list[str]) -> list[str]:
    """Append hierarchical numbers (1. / 1.1) to ## / ### headings so a long report is easier to locate in.

    Only heading lines produced by the renderer (leading `## `/`### `) are handled. #### and
    deeper levels are left alone — formal cluster titles (####) are matched as-is by the Aone
    write-back logic, and numbering would break that contract.
    """

    numbered: list[str] = []
    top = 0
    sub = 0
    for line in lines:
        if line.startswith("## "):
            top += 1
            sub = 0
            numbered.append(f"## {top}. {line[3:]}")
        elif line.startswith("### "):
            sub += 1
            numbered.append(f"### {top}.{sub} {line[4:]}")
        else:
            numbered.append(line)
    return numbered


_STABILITY_LABELS = {
    "stable": "稳定",
    "context_diff": "输入上下文差异",
    "expected_diversity": "合理多样性",
    "tool_data_variance": "工具或数据波动",
    "nondeterminism": "执行不确定性",
    "quality_regression": "质量退化",
    "evaluation_inconsistency": "评测判定不一致",
    "evaluation_comparability_risk": "评测不可比",
    "insufficient_evidence": "证据不足",
}
_STABILITY_DETAIL_LIMIT = 5


def _stability_conclusion_lines(report: ClusterReport, groups: list[dict[str, Any]]) -> list[str]:
    """Emit the program-determined conclusion first so the reader is not buried in per-query evidence."""

    summary = report.llm_summary or {}
    overview = report.stats.observations.get("stability_overview") or {}
    total = int(overview.get("total_queries") or len(groups))
    stable = int(overview.get("stable_queries") or 0)
    stable_rate = float(overview.get("stable_rate") or 0.0)
    high_risk = int(overview.get("high_risk_queries") or 0)
    comparable_risk = int(overview.get("comparable_risk_queries") or 0)
    insufficient = int(overview.get("insufficient_evidence_queries") or 0)
    governance_count = int(overview.get("governance_count") or max(total - stable, 0))
    lines = ["## 稳定性结论", ""]
    if summary.get("skipped"):
        lines.append(f"- L2 文字解释未生成：{summary['skipped']}；以下结论均来自 L1 证据。")
    if summary.get("headline"):
        lines.extend([f"**{summary['headline']}**", ""])
    total_runs = sum(int(group.get("run_count") or 0) for group in groups)
    lines.extend(
        [
            f"- 覆盖范围：{total} 个场景 query，共 {total_runs} 次运行。",
            f"- 稳定情况：{stable}/{total} 个 query 稳定（{stable_rate:.1%}）。",
            f"- 高优先级治理：{high_risk} 项；全部待治理项：{governance_count} 项。",
            f"- 评测不可比：{comparable_risk} 项；证据不足：{insufficient} 项。",
        ]
    )
    if high_risk == 0 and governance_count:
        lines.append(
            "- 当前无 Agent 高风险，但仍需处理下方评测输入、验收准则、评测规则或可观测性治理项。"
        )
    return lines


def _stability_priority_lines(summary: dict[str, Any], groups: list[dict[str, Any]]) -> list[str]:
    groups_to_govern = _stability_governance_groups(groups)
    lines = ["", "## 优先治理项", ""]
    if not groups_to_govern:
        return [*lines, "- 所有已覆盖 query 均稳定，无待治理项。"]
    finding_by_key = _finding_by_group(summary)
    lines.extend(
        [
            "| 优先级 | 分类 | 场景 / Query | 运行次数 | 证据 | 差异原因 | 责任域 | 治理动作 |",
            "| --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    row_template = (
        "| {priority} | {label} | {target} | {runs} | {strength} | {reason} | {domain} | {action} |"
    )
    for group in groups_to_govern[:_STABILITY_DETAIL_LIMIT]:
        finding = finding_by_key.get(_stability_key(group), {})
        governance = _group_governance(group)
        lines.append(
            row_template.format(
                priority=group.get("priority", "P2"),
                label=_stability_label(group),
                target=_table_cell(f"{group.get('scene_tag', '-')} / {group.get('query', '-')}"),
                runs=group.get("run_count", 0),
                strength=group.get("evidence_strength", "weak"),
                reason=_table_cell(str(finding.get("summary") or governance["reason"])),
                domain=_table_cell(str(governance["domain"])),
                action=_table_cell(str(governance["action"])),
            )
        )
    return lines


def _stability_distribution_lines(groups: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## 差异原因与风险分布", ""]
    if not groups:
        return [*lines, "- 未找到满足至少两次运行的场景 tag + query 组合。"]
    by_observation: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        by_observation.setdefault(
            str(group.get("agent_execution_observation") or "insufficient_evidence"), []
        ).append(group)
    lines.extend(
        [
            "| 分类 | 数量 | 占比 | 中文原因 | 责任域 | 建议动作 |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for _observation, classified_groups in sorted(
        by_observation.items(),
        key=lambda item: stability_group_sort_key(item[1][0]),
    ):
        governance = _group_governance(classified_groups[0])
        lines.append(
            "| {label} | {count} | {share:.1%} | {reason} | {domain} | {action} |".format(
                label=_stability_label(classified_groups[0]),
                count=len(classified_groups),
                share=len(classified_groups) / len(groups),
                reason=_table_cell(str(governance["reason"])),
                domain=_table_cell(str(governance["domain"])),
                action=_table_cell(str(governance["action"])),
            )
        )
    return lines


def _stability_index_lines(groups: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## 场景索引", ""]
    if not groups:
        return [*lines, "- 无可展示的场景索引。"]
    lines.extend(
        [
            "| 场景 | Query | 运行次数 | judge 0/1/2 | 观察 | 证据 | 优先级 | 可比性 |",
            "| --- | --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for group in sorted(groups, key=stability_group_sort_key):
        distribution = group.get("judge_score_distribution") or {}
        score_counts = "/".join(str(distribution.get(score, 0)) for score in ("0", "1", "2"))
        comparable = group.get("evaluation_comparability") or {}
        comparability = f"{comparable.get('status', 'risk')}（{comparable.get('reason', '')}）"
        row_template = (
            "| {scene} | {query} | {runs} | {scores} | {observation} | "
            "{strength} | {priority} | {comparability} |"
        )
        lines.append(
            row_template.format(
                scene=_table_cell(str(group.get("scene_tag") or "-")),
                query=_table_cell(str(group.get("query") or "-")),
                runs=group.get("run_count", 0),
                scores=score_counts,
                observation=_stability_label(group),
                strength=group.get("evidence_strength", "weak"),
                priority=group.get("priority", "P2"),
                comparability=_table_cell(comparability),
            )
        )
    return lines


def _stability_evidence_lines(summary: dict[str, Any], groups: list[dict[str, Any]]) -> list[str]:
    groups_to_govern = _stability_governance_groups(groups)[:_STABILITY_DETAIL_LIMIT]
    if not groups_to_govern:
        return []
    finding_by_key = _finding_by_group(summary)
    lines = ["", "## 重点证据下钻", ""]
    for group in groups_to_govern:
        finding = finding_by_key.get(_stability_key(group), {})
        governance = _group_governance(group)
        heading = (
            f"### {_stability_label(group)} · {group.get('scene_tag', '-')} / "
            f"{group.get('query', '-')}"
        )
        lines.extend(
            [
                heading,
                "",
                f"- 原因：{finding.get('summary') or governance['reason']}",
                f"- 责任域：{governance['domain']}；治理动作：{governance['action']}",
            ]
        )
        for run in group.get("representative_runs") or []:
            if not isinstance(run, dict):
                continue
            context = run.get("context") or {}
            output = run.get("output_evidence") or {}
            lines.append(
                f"- trace `{run.get('trace_id', '')}`：score={run.get('judge_score', '-')}; "
                f"context={context.get('fingerprint', '-')}; "
                f"output={output.get('fingerprint', '-')}"
            )
        lines.append("")
    return lines


def _stability_caveat_lines(summary: dict[str, Any]) -> list[str]:
    caveats = _text_items(summary.get("caveats"))
    if not caveats:
        return []
    return ["", "## 结论限制", "", *(f"- {item}" for item in caveats)]


def _stability_governance_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        group
        for group in sorted(groups, key=stability_group_sort_key)
        if str(group.get("agent_execution_observation") or "") != "stable"
    ]


def _finding_by_group(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        _stability_key(finding): finding
        for finding in summary.get("findings") or []
        if isinstance(finding, dict)
    }


def _stability_key(item: dict[str, Any]) -> tuple[str, str]:
    return (
        str(item.get("scene_tag") or ""),
        " ".join(str(item.get("query") or "").split()).casefold(),
    )


def _stability_label(group: dict[str, Any]) -> str:
    observation = str(group.get("agent_execution_observation") or "insufficient_evidence")
    return _STABILITY_LABELS.get(observation, observation)


def _group_governance(group: dict[str, Any]) -> dict[str, Any]:
    governance = group.get("governance")
    if isinstance(governance, dict):
        return {
            "reason": str(governance.get("reason") or "需要核查差异来源。"),
            "domain": str(governance.get("domain") or "可观测性"),
            "action": str(governance.get("action") or "补齐证据后复核。"),
        }
    return {"reason": "需要核查差异来源。", "domain": "可观测性", "action": "补齐证据后复核。"}


def _text_items(value: object) -> list[str]:
    """A bad type on a historical or external summary must not interrupt rendering."""
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _batch_meta_lines(batch_meta: dict[str, Any]) -> list[str]:
    if not batch_meta:
        return []
    lines = ["", "| 字段 | 值 |", "| --- | --- |"]
    lines.extend(
        f"| {_table_cell(str(key))} | {_table_cell(_fmt(value))} |"
        for key, value in batch_meta.items()
    )
    return lines


def _coverage_lines(report: ClusterReport, *, include_perf_note: bool) -> list[str]:
    coverage = report.coverage
    lines = [
        "",
        "### 覆盖与收口",
        "",
        f"- 期望 case: {coverage.expected_total}，实际召回 judge: {coverage.judged}"
        f"（覆盖率 {_pct(coverage.eval_rate)}）",
        f"- 应归因: {coverage.should_attribute}，已完成: {coverage.attributed}"
        f"（覆盖率 {_pct(coverage.attribution_rate)}）",
    ]
    if coverage.pending:
        lines.append(f"- ⚠️ 归因未完成 {coverage.pending} 条（watcher 收口时仍在处理或已停滞）")
    if coverage.watcher_note:
        lines.append(f"- ⚠️ 收口情况：{coverage.watcher_note}")
    if include_perf_note and coverage.perf_note:
        lines.append(f"- ⚠️ 性能块：{coverage.perf_note}")
    return lines


def _effectiveness_lines(stats: dict[str, Any]) -> list[str]:
    if not stats:
        return []
    dist = stats.get("score_distribution") or {}
    lines = [
        "",
        "### 评测结果",
        "",
        f"- 已判分 {stats.get('scored', 0)} / {stats.get('total', 0)}，"
        f"平均分 {stats.get('avg_score', 0)}",
        f"- 通过 {dist.get('2', 0)}（{_pct(stats.get('pass_rate', 0))}）· "
        f"部分正确 {dist.get('1', 0)}（{_pct(stats.get('partial_rate', 0))}）· "
        f"失败 {dist.get('0', 0)}（{_pct(stats.get('fail_rate', 0))}）",
        f"- 硬规则命中率 {_pct(stats.get('hard_rule_hit_rate', 0))}",
    ]
    score_views = stats.get("score_views") or {}
    if score_views:
        lines.extend(
            [
                "",
                "### 分数口径",
                "",
                "| 分数字段 | 含义 | 样本 | 均值 | 0分 | 1分 | 2分 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        labels = {
            "judge_score": "原评测分",
            "attribution_score": "归因纠正分",
            "effective_score": "聚合最终采用分",
        }
        for field, label in labels.items():
            summary = score_views.get(field) or {}
            distribution = summary.get("distribution") or {}
            lines.append(
                f"| `{field}` | {label} | {summary.get('scored', 0)} | "
                f"{summary.get('avg_score', 0)} | {distribution.get('0', 0)} | "
                f"{distribution.get('1', 0)} | {distribution.get('2', 0)} |"
            )
    dimension = stats.get("dimension_fail_rate") or {}
    if dimension:
        lines.extend(
            [
                "",
                "### 维度失败率",
                "",
                "| 维度 | 判定数 | 失败数 | 失败率 |",
                "| --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {_table_cell(str(name))} | {item.get('judged', 0)} | {item.get('failed', 0)} | "
            f"{_pct(item.get('fail_rate', 0))} |"
            for name, item in dimension.items()
        )
    lines.extend(_top_table("失败模式 Top", stats.get("failure_modes") or {}, name_fn=mode_name))
    lines.extend(_top_table("checker 失败命中", stats.get("checker_fail_hits") or {}))
    return lines


def _attribution_lines(stats: dict[str, Any]) -> list[str]:
    if not stats:
        return []
    overturn = stats.get("overturn") or {}
    confidence = stats.get("confidence") or {}
    evidence = stats.get("evidence") or {}
    lines = [
        "",
        "### 归因概览",
        "",
        f"- 应归因 {stats.get('should_attribute', 0)}，已完成 {stats.get('attributed', 0)}"
        f"，进行中 {stats.get('pending', 0)}",
        f"- 平均置信度 {confidence.get('avg', 0)}"
        f"（低置信占比 {_pct(confidence.get('low_rate', 0))}）",
        f"- 证据充分率 {_pct(evidence.get('evidence_rate', 0))}",
        f"- owner 兜底率 {_pct(stats.get('owner_fallback_rate', 0))}"
        "（偏高说明 RemoteConfig owner_mapping 缺配）",
    ]
    if overturn.get("overturn_count"):
        lines.append(
            f"- **推翻评测结论 {overturn['overturn_count']} 条"
            f"（{_pct(overturn.get('overturn_rate', 0))}）**："
            + "、".join(
                f"{name} {count}" for name, count in (overturn.get("by_type") or {}).items()
            )
        )
    root_causes = stats.get("root_cause_top") or []
    if root_causes:
        lines.extend(
            [
                "",
                "### 主根因 Top",
                "",
                "| 根因 | 数量 | 占比 |",
                "| --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {_table_cell(str(mode_name(item.get('code', ''))))} | {item.get('count', 0)} | "
            f"{_pct(item.get('share', 0))} |"
            for item in root_causes
        )
    all_root_causes = stats.get("all_root_cause_top") or []
    if all_root_causes:
        lines.extend(
            [
                "",
                "### 全部根因 Top",
                "",
                "一个 case 命中多个根因时全部计入（计数可大于 case 数）；派单仅按主根因。",
                "",
                "| 根因 | 数量 | 占比 |",
                "| --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {_table_cell(str(mode_name(item.get('code', ''))))} | {item.get('count', 0)} | "
            f"{_pct(item.get('share', 0))} |"
            for item in all_root_causes
        )
    breakdown = stats.get("unclassified_breakdown") or {}
    if breakdown.get("count"):
        lines.extend(
            [
                "",
                "### 未分类原因分布",
                "",
                "未分类指无正式根因且非推翻/客观无责结论的 case；桶名来自归因门禁的拒绝原因，",
                "聚类与人工应据此定位缺口性质，不得笼统归因为「规则覆盖不足」。",
                "",
                "| 拒绝原因 | 数量 |",
                "| --- | --- |",
            ]
        )
        lines.extend(
            f"| {_table_cell(str(bucket))} | {count} |"
            for bucket, count in (breakdown.get("buckets") or {}).items()
        )
    lines.extend(_top_table("责任团队分布", stats.get("owner_group_distribution") or {}))
    human_review = stats.get("human_review") or {}
    if human_review.get("count"):
        lines.append("")
        lines.append(f"### 需人工复核（{human_review['count']} 条）")
        lines.append("")
        lines.extend(
            f"- `{case.get('trace_id', '')}` {case.get('suite', '')} "
            f"{case.get('case_id', '')} — {case.get('root_cause', '')}"
            for case in human_review.get("cases") or []
        )
    backlog = stats.get("pending_backlog") or {}
    if backlog.get("count"):
        lines.append("")
        lines.append(f"### 归因待补（{backlog['count']} 条）")
        lines.append("")
        lines.extend(
            f"- `{case.get('trace_id', '')}` {case.get('suite', '')} "
            f"{case.get('case_id', '')}（{case.get('triage_level', '')}）"
            for case in backlog.get("cases") or []
        )
    return lines


def _other_candidate_lines(attribution: dict[str, Any]) -> list[str]:
    """Render the standalone `other:` candidate section, separate from the formal root-cause Top."""

    stats = attribution.get("other_candidate_stats") or {}
    candidates = stats.get("candidates") or []
    if not candidates:
        return []
    lines = [
        "",
        "### 新问题候选簇（`other:`，待治理）",
        "",
        "候选仅供 scope 内人工查看，不进入 owner 路由、正式根因 Top 或 Aone 子工单。",
        "确认需升级时，直接新增正式 failure mode/taxonomy 与对应代码。",
    ]
    for candidate in candidates:
        lines.extend(
            [
                "",
                f"#### {candidate.get('title') or '未命名候选'}"
                f"（{candidate.get('count', 0)} 条 · {candidate.get('agent_scope') or '-'}）",
                f"- 候选: `{candidate.get('candidate_key') or '-'}`",
                f"- 证据完整率: {_pct(candidate.get('evidence_complete_rate', 0))}",
            ]
        )
        trace_ids = [
            str(case.get("trace_id") or "")
            for case in candidate.get("cases") or []
            if isinstance(case, dict) and case.get("trace_id")
        ]
        if trace_ids:
            lines.append(f"- 关联 trace: {', '.join(f'`{tid}`' for tid in trace_ids)}")
    return lines


def _performance_lines(stats: dict[str, Any]) -> list[str]:
    if not stats:
        return []
    lines = ["", "### 性能指标", ""]
    if stats.get("missing"):
        lines.append(
            f"- ⚠️ {stats['missing']} 条性能采集缺失："
            + "、".join(
                f"{reason} {count}"
                for reason, count in (stats.get("missing_reasons") or {}).items()
            )
        )
    lines.extend(
        [
            "| 指标 | 均值 | P50 | P95 | P99 | 样本 |",
            "| --- | --- | --- | --- | --- | --- |",
            _latency_row("端到端耗时(ms)", stats.get("e2e_ms") or {}),
            _latency_row("首正文耗时(ms)", stats.get("first_text_ms") or {}),
        ]
    )
    lines.extend(
        [
            "",
            "### 三桶明细",
            "",
            "| 桶 | case | 调用(均) | 单调用耗时均值 | 耗时P95 | "
            "输入Tokens(次均) | 输出Tokens(次均) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    buckets = stats.get("buckets") or {}
    for bucket in (bucket for bucket in MODEL_BUCKETS if bucket != "verify"):
        item = buckets.get(bucket) or {}
        if not item.get("cases"):
            lines.append(f"| {BUCKET_LABELS[bucket]} | 0 | - | - | - | - | - | - |")
            continue
        latency = item.get("latency_ms") or {}
        lines.append(
            f"| {BUCKET_LABELS[bucket]} | {item.get('cases', 0)} | "
            f"{(item.get('calls') or {}).get('avg', 0)} | {latency.get('avg', 0)} | "
            f"{latency.get('p95', 0)} | "
            f"{(item.get('prompt_tokens') or {}).get('avg_per_call', 0)} | "
            f"{(item.get('completion_tokens') or {}).get('avg_per_call', 0)} |"
        )
    lines.append("")
    total_tokens = stats.get("total_tokens_per_case") or {}
    lines.append(
        f"- 总 tokens {stats.get('total_tokens', 0)}，平均总 Token / case "
        f"{total_tokens.get('avg', 0)}，总成本 {stats.get('total_cost', 0)}"
    )
    route_modes = stats.get("route_modes") or {}
    modes = route_modes.get("modes") or {}
    lines.extend(
        [
            "",
            "### VUI 路由分布",
            "",
            "| 链路 | case 数 | 有效比例 |",
            "| --- | ---: | ---: |",
            f"| Lite-SDM | {(modes.get('lite_sdm') or {}).get('count', 0)} | "
            f"{_pct((modes.get('lite_sdm') or {}).get('share', 0))} |",
            f"| Lite-DAG | {(modes.get('lite_dag') or {}).get('count', 0)} | "
            f"{_pct((modes.get('lite_dag') or {}).get('share', 0))} |",
            f"| Standard | {(modes.get('standard') or {}).get('count', 0)} | "
            f"{_pct((modes.get('standard') or {}).get('share', 0))} |",
            f"| 未分类 | {route_modes.get('unclassified', 0)} | 不计入有效比例分母 |",
        ]
    )
    defects = stats.get("aone_defects") or {}
    lines.extend(
        [
            "",
            "### Aone 缺陷工单",
            "",
            f"- 去重后的已派发工单：{defects.get('unique_workitems', 0)} 个",
            f"- 关联 case：{defects.get('related_cases', 0)} 条",
        ]
    )
    return lines


def _crosstab_lines(stats: dict[str, Any]) -> list[str]:
    if not stats:
        return []
    lines: list[str] = []
    gap = [row for row in (stats.get("checker_root_cause_gap") or []) if row.get("unmapped")]
    suites = [row for row in (stats.get("suite_overturn") or []) if row.get("overturned")]
    if gap or suites:
        lines.extend(["", "### 交叉分析"])
    if gap:
        lines.extend(
            [
                "",
                "### checker → 根因映射缺口",
                "",
                "| checker | 命中 | 未映射 | 未映射率 |",
                "| --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {_table_cell(str(row.get('checker', '')))} | {row.get('hits', 0)} | "
            f"{row.get('unmapped', 0)} | {_pct(row.get('unmapped_rate', 0))} |"
            for row in gap
        )
    if suites:
        lines.extend(
            [
                "",
                "### 各 suite 推翻情况",
                "",
                "| suite | 总数 | 低分 | 已归因 | 推翻 | 推翻率 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {_table_cell(str(row.get('suite', '')))} | {row.get('total', 0)} | "
            f"{row.get('low_score', 0)} | "
            f"{row.get('attributed', 0)} | {row.get('overturned', 0)} | "
            f"{_pct(row.get('overturn_rate', 0))} |"
            for row in suites
        )
    return lines


def _observation_lines(stats: dict[str, Any]) -> list[str]:
    """Independent observation slices of the unified envelope (Metric / Diagnostic / business, scene, Profile).

    Display only; they do not participate in effectiveness scoring (design §5.3). For
    historical non-envelope batches this block is empty, skipped as a whole, and the output
    matches the baseline.
    """

    if not stats:
        return []
    lines = ["", "### 运行观测（不影响效果评分）"]
    metrics = stats.get("metric_observations") or {}
    if metrics:
        lines.extend(
            [
                "",
                "### 指标超限",
                "",
                "| 指标 | 观测数 | 超限数 | 超限率 |",
                "| --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {_table_cell(str(key))} | {item.get('observed', 0)} | {item.get('exceeded', 0)} | "
            f"{_pct(item.get('exceeded_rate', 0))} |"
            for key, item in metrics.items()
        )
    diagnostics = stats.get("diagnostic_findings") or {}
    if diagnostics:
        lines.extend(
            [
                "",
                "### 诊断发现",
                "",
                "| 发现 | 命中数 |",
                "| --- | --- |",
            ]
        )
        lines.extend(
            f"| {_table_cell(str(key))} | {count} |"
            for key, count in (diagnostics.get("by_finding_key") or {}).items()
        )
        severity = diagnostics.get("by_severity") or {}
        if severity:
            severity_text = "、".join(f"{name} {count}" for name, count in severity.items())
            lines.extend(["", f"严重度：{severity_text}"])
    profiles = stats.get("profiles") or {}
    if profiles:
        lines.extend(
            [
                "",
                "### 业务 / 场景 / Profile 分布",
                "",
                "| 切分键 | case 数 |",
                "| --- | --- |",
            ]
        )
        lines.extend(
            f"| {_table_cell(str(key))} | {count} |"
            for key, count in (profiles.get("profiles") or {}).items()
        )
        business_text = "、".join(
            f"{name} {count}" for name, count in (profiles.get("business_ids") or {}).items()
        )
        scenario_text = "、".join(
            f"{name} {count}" for name, count in (profiles.get("scenarios") or {}).items()
        )
        lines.extend(["", f"- 业务：{business_text}", f"- 场景：{scenario_text}"])
    return lines


def _llm_summary_lines(  # noqa: PLR0912, PLR0915
    summary: dict[str, Any],
    *,
    candidate_fallback: dict[str, Any],
    cluster_links: dict[int, str] | None,
    cluster_dispatch_statuses: dict[int, str] | None,
) -> list[str]:
    """Render formal L2 clusters and the isolated candidate new-problem clusters."""
    lines = ["## 语义聚类", ""]
    if not summary:
        lines.extend(_other_candidate_lines(candidate_fallback))
        return lines
    if summary.get("skipped"):
        lines.append(f"- 未生成：{summary['skipped']}（报告主体为 L1 确定性统计）")
    if summary.get("parse_error"):
        lines.append(f"- ⚠️ LLM 输出解析失败：{summary['parse_error']}")
    if summary.get("headline"):
        lines.extend([f"**{summary['headline']}**", ""])

    links = cluster_links or {}
    dispatch_statuses = cluster_dispatch_statuses or {}
    formal_clusters = [item for item in summary.get("clusters") or [] if isinstance(item, dict)]
    if formal_clusters:
        lines.extend(["### 正式问题簇", ""])
    for index, cluster in enumerate(formal_clusters, start=1):
        title = str(cluster.get("title") or "未命名簇")
        url = links.get(index)
        display_title = f"[{title}]({url})" if url else title
        lines.append(
            f"#### {display_title}"
            f"（{cluster.get('affected_count', 0)} 条 · {cluster.get('owner_group', '-')}）"
        )
        lines.append("")
        if cluster.get("root_cause_codes"):
            names = ", ".join(mode_name(code) for code in cluster["root_cause_codes"])
            lines.append(f"- 根因: {names}")
        if cluster.get("summary"):
            lines.append(f"- 共性: {cluster['summary']}")
        if cluster.get("evidence"):
            lines.append(f"- 依据: {cluster['evidence']}")
        if cluster.get("related_cases"):
            lines.append(
                f"- 关联 trace: {', '.join(f'`{tid}`' for tid in cluster['related_cases'])}"
            )
        if status := dispatch_statuses.get(index):
            lines.append(f"- 子工单状态: {status}")
        lines.append("")

    candidate_clusters = [
        item for item in summary.get("candidate_clusters") or [] if isinstance(item, dict)
    ]
    if candidate_clusters:
        lines.extend(["### 新问题候选簇（`other:`，待治理）", ""])
        for cluster in candidate_clusters:
            lines.append(
                f"#### {cluster.get('title', '未命名候选簇')}"
                f"（{cluster.get('affected_count', 0)} 条 · {cluster.get('agent_scope', '-')}）"
            )
            lines.append("")
            lines.append(f"- 候选: `{cluster.get('candidate_key', '-')}`")
            if cluster.get("summary"):
                lines.append(f"- 共性: {cluster['summary']}")
            if cluster.get("evidence"):
                lines.append(f"- 依据: {cluster['evidence']}")
            if cluster.get("related_cases"):
                lines.append(
                    f"- 关联 trace: {', '.join(f'`{tid}`' for tid in cluster['related_cases'])}"
                )
            lines.append("- 治理状态: 待人工确认，不自动创建正式修复工单")
            lines.append("")
    else:
        lines.extend(_other_candidate_lines(candidate_fallback))

    suggestions = summary.get("feedback_suggestions") or []
    if suggestions:
        lines.extend(["### 回流建议", ""])
        lines.extend(
            f"- **{item.get('target', '')}**: {item.get('action', '')}"
            f"（依据: {item.get('rationale', '')}）"
            for item in suggestions
            if isinstance(item, dict)
        )
    if summary.get("caveats"):
        lines.extend(["", "### 结论限制", ""])
        lines.extend(f"- {item}" for item in _text_items(summary["caveats"]))
    if summary.get("number_warnings"):
        lines.extend(
            [
                "",
                "> ⚠️ 以下数字未能在 L1 统计中匹配，可能是 LLM 复述偏差，请以 L1 为准："
                + "、".join(str(item) for item in summary["number_warnings"][:10]),
            ]
        )
    return lines


def _top_table(
    title: str,
    counts: dict[str, int],
    limit: int = 10,
    *,
    name_fn: Callable[[str], str] | None = None,
) -> list[str]:
    """``name_fn`` is only a display transform (for example code → display name) and does not
    participate in aggregation. Aggregation keys must stay the original code; otherwise
    counts are silently overwritten when several codes map to the same display name.
    """

    if not counts:
        return []
    lines = ["", f"### {title}", "", "| 项 | 数量 |", "| --- | --- |"]
    lines.extend(
        f"| {_table_cell(str(name_fn(name) if name_fn else name))} | {count} |"
        for name, count in list(counts.items())[:limit]
    )
    return lines


def _latency_row(label: str, stats: dict[str, Any]) -> str:
    if not stats.get("count"):
        return f"| {_table_cell(str(label))} | - | - | - | - | 0 |"
    return (
        f"| {_table_cell(str(label))} | {stats.get('avg', 0)} | {stats.get('p50', 0)} | "
        f"{stats.get('p95', 0)} | {stats.get('p99', 0)} | {stats['count']} |"
    )


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


__all__ = ["render_json", "render_markdown"]
