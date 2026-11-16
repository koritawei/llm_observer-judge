"""T32 xlsx rendering — one sheet per suite, with column names and positions strictly aligned to eval-compare.py.

**Hard compatibility constraints** (inferred from eval-compare.py; if they fail, that tool reads no data):
1. The sheet name must hit ``SHEET_CONFIGS``, or it is skipped
2. The tag column / query column positions must match ``SHEET_CONFIGS`` (multi-turn sheets are (3, 4), the rest (4, 5))
3. **The header row must fall in rows 1-4, and the tag-column cell text must contain "intent"** —
   ``_find_header_row`` locates the header by that, otherwise it falls back to row 2 and misreads
4. Metric column names must equal ``[3.0] {label}`` exactly (``find_metric_columns`` is a full equality
   check, not a contains match). The label comes from ``BUCKET_LABELS``
5. Data rows are contiguous: ``read_sheet_data`` ``break``s on an empty tag cell, so no blank row in between
"""

from typing import TYPE_CHECKING, Any

from benchmark.reporting.cluster.constants import BUCKET_LABELS, MODEL_BUCKETS, UNGROUPED_SHEET
from benchmark.reporting.cluster.schema import BatchCaseRecord, CasePerf, ClusterReport
mode_name = None

if TYPE_CHECKING:  # pragma: no cover - needed only during type checking; openpyxl is an optional runtime dependency
    from openpyxl.workbook import Workbook

#: Column-position contract of eval-compare.py SHEET_CONFIGS: sheet name → (tag_col, query_col).
#: Tuple order matches ``SHEET_CONFIGS = [(name, tag_col, query_col)]`` in eval-compare.py.
#: Strictly: the tag(intent) column comes before query. A multi-turn sheet has one fewer leading
#: column than the other sheets, so positions must line up one by one.
SHEET_COLUMN_LAYOUT: dict[str, tuple[int, int]] = {
    "车机-Lite": (4, 5),
    "车机-Standard": (4, 5),
    "车机-Max": (4, 5),
    "车机-多轮": (3, 4),
    "车机-Standard-核心": (4, 5),
    "车机-Max-核心": (4, 5),
}

#: A suite that misses SHEET_CONFIGS uses the default positions (eval-compare.py skips such sheets,
#: but the report itself must stay readable, so the layout matches the mainstream sheets).
DEFAULT_COLUMN_LAYOUT = (4, 5)

#: Version prefix of column names. 3.0 = the current in-vehicle agent version (eval-compare detect_version prefers 3.0).
VERSION_PREFIX = "3.0"

#: Header row number. Must be ≤ 4 (eval-compare._find_header_row scans only rows 1-4).
HEADER_ROW = 2

#: Header text of the tag column. Must contain "intent" (the anchor _find_header_row locates).
TAG_HEADER = "intent"


def build_sheet_rows(records: list[BatchCaseRecord]) -> dict[str, list[BatchCaseRecord]]:
    """Group by ``case_meta.suite`` (under golden this matches the testset sheet structure).

    A missing suite (black-box/white-box or sampled cases) lands on ``UNGROUPED_SHEET``.
    The data is not dropped, but eval-compare.py will not read that sheet (it is not in SHEET_CONFIGS).
    """

    grouped: dict[str, list[BatchCaseRecord]] = {}
    for record in records:
        grouped.setdefault(record.suite or UNGROUPED_SHEET, []).append(record)
    return grouped


def build_header(
    sheet_name: str = "",
    *,
    version: str = VERSION_PREFIX,
) -> list[str]:
    """Build the header-row text list (1-based column position is index + 1).

    **The number of leading columns must vary by sheet**: eval-compare SHEET_CONFIGS defines
    (query=3, tag=4) for multi-turn sheets and (query=4, tag=5) for the rest. A fixed layout
    shifts a multi-turn sheet by one column, so that tool reads the wrong columns as tag/query.
    """

    tag_col, query_col = column_layout(sheet_name)
    # Build by column position (1-based): col1/2 are fixed as case_id/turn_index, and tag(intent) and
    # query land on the positions declared by SHEET_COLUMN_LAYOUT (aligned with eval-compare). Gaps in between
    # are filled with a notes placeholder. This does not depend on whether tag comes before query.
    lead: dict[int, str] = {1: "case_id", 2: "turn_index", tag_col: TAG_HEADER, query_col: "query"}
    header = [lead.get(col, "备注") for col in range(1, max(lead) + 1)]

    header.extend(
        [
            f"[{version}] 评测打分",
            f"[{version}] 评测原因",
            "judge_score（原评测分）",
            "attribution_score（归因纠正分）",
            "effective_score（聚合最终采用分）",
            f"[{version}] Trace ID",
            f"[{version}] 首正文回复耗时(ms)",
            f"[{version}] 端到端耗时(ms)",
            f"[{version}] React轮次",
            f"[{version}] 执行模式",
            "VUI 路由结果",
            "Aone 缺陷工单 ID",
            "failure_mode",
            "root_cause_code",
            "归因结论",
            "归因类型",
            "owner_group",
            "归因状态",
        ]
    )
    for bucket in MODEL_BUCKETS:
        label = BUCKET_LABELS[bucket]
        header.extend(
            [
                f"[{version}] {label}调用次数",
                f"[{version}] {label}平均耗时(ms)",
                f"[{version}] {label}平均输入Tokens",
                f"[{version}] {label}平均输出Tokens",
            ]
        )
    return header


def build_row(record: BatchCaseRecord, sheet_name: str = "") -> list[Any]:
    """Build the data row of one case, with column order strictly matching :func:`build_header`."""

    tag_col, query_col = column_layout(sheet_name)
    perf = record.perf
    # Same column-position scale as build_header: tag/query each go to their declared column, and gaps are empty strings.
    lead: dict[int, Any] = {
        1: record.case_key.case_id,
        2: record.case_key.turn_index,
        tag_col: record.case_key.tag,
        query_col: record.case_key.query,
    }
    row: list[Any] = [lead.get(col, "") for col in range(1, max(lead) + 1)]

    row.extend(
        [
            record.score,
            record.reason,
            record.judge_score,
            _score_value(record.attribution_score),
            record.effective_score,
            record.trace_id,
            _perf_value(perf.first_text_ms if perf else None),
            _perf_value(perf.e2e_ms if perf else None),
            _react_rounds(perf),
            record.case_meta.get("agent_mode", ""),
            perf.route_mode if perf else "",
            record.notification_workitem_id,
            _join_modes(record.failure_modes),
            mode_name(record.attribution.root_cause_code),
            record.attribution.root_cause,
            record.attribution.conclusion_type,
            record.attribution.owner_group,
            record.attribution.status,
        ]
    )
    for bucket in MODEL_BUCKETS:
        entry = perf.buckets.get(bucket) if perf else None
        if entry is None or not entry.calls:
            row.extend(["", "", "", ""])
            continue
        row.extend(
            [
                entry.calls,
                round(entry.latency_ms / entry.calls),
                round(entry.prompt_tokens / entry.calls),
                round(entry.completion_tokens / entry.calls),
            ]
        )
    return row


def _score_value(value: float | None) -> float | str:
    """Write an empty string when the corrected score is missing, so it is not confused with a real 0."""

    return "" if value is None else value


def _perf_value(value: float | None) -> int | str:
    """Write an empty string rather than 0 when performance is missing — eval-compare would treat 0 as a real measurement in the mean."""

    return "" if value is None else round(value)


def _react_rounds(perf: CasePerf | None) -> int | str:
    """React rounds = the LLM call count of the react bucket.

    The scale is locked to generate-eval-report.py (React rounds are taken from llm_calls
    of the observation's react bucket): when the count is 0, **do not write it**, so
    eval-compare does not treat it as a real measured 0 rounds.
    """

    if perf is None:
        return ""
    entry = perf.buckets.get("react")
    if entry is None or not entry.calls:
        return ""
    return entry.calls


def _join_modes(failure_modes: list[dict[str, Any]]) -> str:
    modes = [
        mode_name(str(entry.get("mode") or ""))
        for entry in failure_modes
        if isinstance(entry, dict) and entry.get("mode")
    ]
    return ",".join(modes)


def column_layout(sheet_name: str) -> tuple[int, int]:
    """Return this sheet's (tag_col, query_col), used to check that column positions line up."""

    return SHEET_COLUMN_LAYOUT.get(sheet_name, DEFAULT_COLUMN_LAYOUT)


def render_xlsx(
    records: list[BatchCaseRecord],
    output_path: str,
    *,
    version: str = VERSION_PREFIX,
    report: ClusterReport | None = None,
) -> str:
    """Write eval_summary.xlsx. When openpyxl is missing, raise RuntimeError and let the caller degrade."""

    try:
        from openpyxl import Workbook  # noqa: PLC0415 - optional dependency; degrade when missing
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("openpyxl_not_installed") from exc

    workbook = Workbook()
    workbook.remove(workbook.active)
    if report is not None:
        _write_summary_sheet(workbook, report)

    for sheet_name, items in build_sheet_rows(records).items():
        # openpyxl sheet names are limited to 31 characters and disallow []:*?/\
        _write_sheet(
            workbook,
            sheet_name,
            items,
            version=version,
        )

    if not workbook.sheetnames:
        # Write the header even for an empty batch: an empty sheet alone is unreadable and hides the column structure
        _write_sheet(workbook, UNGROUPED_SHEET, [], version=version)
    _write_candidate_sheet(workbook, records)
    workbook.save(output_path)
    return output_path


def _write_summary_sheet(workbook: "Workbook", report: ClusterReport) -> None:
    """Write report-level KPIs; this sheet does not affect the column-position contract of the detail sheets."""

    performance = report.stats.performance
    e2e = performance.get("e2e_ms") or {}
    tokens = performance.get("total_tokens_per_case") or {}
    route_modes = performance.get("route_modes") or {}
    modes = route_modes.get("modes") or {}
    defects = performance.get("aone_defects") or {}
    rows = [
        ("指标", "值"),
        ("平均端到端延迟(ms)", e2e.get("avg", "")),
        ("P95 端到端延迟(ms)", e2e.get("p95", "")),
        ("平均总 Token / case", tokens.get("avg", "")),
        ("Lite-SDM case 数", (modes.get("lite_sdm") or {}).get("count", 0)),
        ("Lite-DAG case 数", (modes.get("lite_dag") or {}).get("count", 0)),
        ("Standard case 数", (modes.get("standard") or {}).get("count", 0)),
        ("未分类 case 数", route_modes.get("unclassified", 0)),
        ("去重 Aone 缺陷工单", defects.get("unique_workitems", 0)),
        ("关联 case 数", defects.get("related_cases", 0)),
    ]
    worksheet = workbook.create_sheet("报告汇总")
    for row_index, values in enumerate(rows, start=1):
        for column, value in enumerate(values, start=1):
            worksheet.cell(row=row_index, column=column, value=value)


def _write_candidate_sheet(workbook: "Workbook", records: list[BatchCaseRecord]) -> None:
    """Write a separate candidate-root-cause sheet, without affecting the business suite headers eval-compare depends on."""

    candidates = [
        (record, record.attribution.candidate_root_cause)
        for record in records
        if record.attribution.candidate_root_cause is not None
    ]
    if not candidates:
        return
    worksheet = workbook.create_sheet("候选根因")
    headers = ["agent_scope", "candidate_key", "raw_code", "title", "trace_id", "case_id"]
    for column, header in enumerate(headers, start=1):
        worksheet.cell(row=1, column=column, value=header)
    for row_index, (record, candidate) in enumerate(candidates, start=2):
        assert candidate is not None
        values = [
            candidate.agent_scope,
            candidate.candidate_key(),
            candidate.raw_code,
            candidate.title,
            record.trace_id,
            record.case_key.case_id,
        ]
        for column, value in enumerate(values, start=1):
            worksheet.cell(row=row_index, column=column, value=value)


def _write_sheet(
    workbook: "Workbook",
    sheet_name: str,
    items: list[BatchCaseRecord],
    *,
    version: str,
) -> None:
    """Write one sheet: a title on row 1, the header on row HEADER_ROW, then contiguous data rows."""

    worksheet = workbook.create_sheet(_safe_sheet_name(sheet_name))
    worksheet.cell(row=1, column=1, value=f"批次明细 · {sheet_name}")
    for col, text in enumerate(build_header(sheet_name, version=version), start=1):
        if text:
            worksheet.cell(row=HEADER_ROW, column=col, value=text)
    for offset, record in enumerate(items):
        for col, value in enumerate(build_row(record, sheet_name), start=1):
            worksheet.cell(row=HEADER_ROW + 1 + offset, column=col, value=value)


_INVALID_SHEET_CHARS = "[]:*?/\\"


def _safe_sheet_name(name: str) -> str:
    cleaned = "".join(ch for ch in name if ch not in _INVALID_SHEET_CHARS)
    return (cleaned or UNGROUPED_SHEET)[:31]


__all__ = [
    "DEFAULT_COLUMN_LAYOUT",
    "HEADER_ROW",
    "SHEET_COLUMN_LAYOUT",
    "TAG_HEADER",
    "VERSION_PREFIX",
    "build_header",
    "build_row",
    "build_sheet_rows",
    "column_layout",
    "render_xlsx",
]
