"""Cluster Report case-evaluation detail Excel for product manual review."""

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean
from typing import TYPE_CHECKING

from benchmark.reporting.cluster.constants import DIMENSION_LABELS, DIMENSIONS
from benchmark.reporting.cluster.schema import BatchCaseRecord

if TYPE_CHECKING:  # pragma: no cover - openpyxl is an optional runtime dependency
    from openpyxl.cell.cell import Cell
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.workbook import Workbook

_EXCEL_CONTROL_ESCAPES = {code: f"\\x{code:02x}" for code in range(32) if code not in (9, 10, 13)}

_DETAIL_HEADERS = [
    "case_index",
    "query",
    "langfuse_trace_id（原始 VUI Trace）",
    "评测状态",
    "自动评分",
    *(f"{dimension} {DIMENSION_LABELS[dimension]}" for dimension in DIMENSIONS),
    "自动理由",
    "query_classify(实际)",
    "response_source",
    "文本回复",
    "是否有 CMD",
    "CMD 摘要",
    "工具调用数",
    "Lite DAG",
    "工具调用名称（实际顺序）",
    "用户所在城市",
    "城市区划码",
    "评测错误",
    "原始评测响应",
    "人工质量标签(通过/失败)",
    *(f"人工预期{dimension} {DIMENSION_LABELS[dimension]}(0/1/2)" for dimension in DIMENSIONS),
    "人工问题类别",
    "人工根因",
    "人工证据/备注",
    "自动评分是否正确",
    "误判类型",
]
_HUMAN_CONTEXT_HEADERS = [
    "case_index",
    "query",
    "用户所在城市",
    "城市区划码",
    "query_classify(实际)",
    "response_source",
    "文本回复",
    "是否有 CMD",
    "CMD 摘要",
    "工具调用数",
    "Lite DAG",
    "工具调用名称（实际顺序）",
    "langfuse_trace_id（原始 VUI Trace）",
]
_HUMAN_FILL_HEADERS = [
    "人工质量标签(通过/失败)",
    *(f"人工预期{dimension} {DIMENSION_LABELS[dimension]}(0/1/2)" for dimension in DIMENSIONS),
    "人工问题类别",
    "人工根因",
    "人工证据/备注",
]


@dataclass(frozen=True)
class _DetailStyle:
    """Header fill, font, and cell-alignment builder for the evaluation detail sheet."""

    header_fill: "PatternFill"
    header_font: "Font"
    alignment: type["Alignment"]
    get_column_letter: Callable[[int], str]


@dataclass(frozen=True)
class _HumanSheetStyle:
    """Fill, font, alignment, and border builder for the manual evaluation sheet."""

    pattern_fill: type["PatternFill"]
    header_font: "Font"
    alignment: type["Alignment"]
    border: type["Border"]
    side: type["Side"]
    get_column_letter: Callable[[int], str]


@dataclass(frozen=True)
class _CaseDetailStyle:
    """Header style and column-width helpers for the case detail report."""

    header_fill: "PatternFill"
    header_font: "Font"
    alignment: type["Alignment"]
    get_column_letter: Callable[[int], str]


def render_case_eval_xlsx(records: list[BatchCaseRecord], output_path: str) -> str:
    """Write manual evaluation details in the same layout as the local VUI evaluation workbook."""

    try:
        from openpyxl import Workbook  # noqa: PLC0415 - optional dependency; the caller degrades when it is missing
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: PLC0415
        from openpyxl.utils import get_column_letter  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("openpyxl_not_installed") from exc

    workbook = Workbook()
    summary = workbook.active
    summary.title = "汇总"
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    _write_summary(summary, records, header_fill, header_font)

    detail = workbook.create_sheet("评测明细")
    _write_detail(
        detail,
        records,
        _DetailStyle(
            header_fill=header_fill,
            header_font=header_font,
            alignment=Alignment,
            get_column_letter=get_column_letter,
        ),
    )
    append_human_eval_sheet(
        workbook,
        detail,
        _HumanSheetStyle(
            pattern_fill=PatternFill,
            header_font=header_font,
            alignment=Alignment,
            border=Border,
            side=Side,
            get_column_letter=get_column_letter,
        ),
    )
    _write_case_detail_sheet(
        workbook,
        records,
        _CaseDetailStyle(
            header_fill=header_fill,
            header_font=header_font,
            alignment=Alignment,
            get_column_letter=get_column_letter,
        ),
    )
    workbook.save(output_path)
    return output_path


def _write_summary(
    summary: object,
    records: list[BatchCaseRecord],
    header_fill: "PatternFill",
    header_font: "Font",
) -> None:
    scores = [record.effective_score for record in records if record.effective_score is not None]
    distribution = Counter(str(score) for score in scores)
    rows: list[tuple[str, object]] = [
        ("批次标识", "Cluster Report"),
        ("总 Case 数", len(records)),
        ("评测成功数", len(scores)),
        ("评测失败数", len(records) - len(scores)),
        ("平均分", f"{mean(scores):.2f}" if scores else "N/A"),
        ("评分分布", json.dumps(dict(sorted(distribution.items())), ensure_ascii=False)),
        ("评测方式", "Cluster Report 已归档评测结果"),
        ("Trace 来源", "评测记录 + trace_meta 快照"),
    ]
    for dimension in DIMENSIONS:
        values = [
            float(item["score_hint"])
            for record in records
            for item in record.dimension_results
            if isinstance(item, dict)
            and item.get("dimension") == dimension
            and item.get("status") not in {"not_applicable", "unknown"}
            and isinstance(item.get("score_hint"), int | float)
        ]
        rows.append(
            (
                f"{dimension} {DIMENSION_LABELS[dimension]}",
                (
                    f"样本={len(values)}，平均分={mean(values):.2f}"
                    if values
                    else "样本=0，平均分=N/A"
                ),
            )
        )
    for row_index, (key, value) in enumerate(rows, start=1):
        summary.cell(row=row_index, column=1, value=key)
        summary.cell(row=row_index, column=2, value=value)
    for cell in summary[1]:
        cell.fill = header_fill
        cell.font = header_font
    summary.column_dimensions["A"].width = 32
    summary.column_dimensions["B"].width = 100


def _write_detail(
    sheet: object,
    records: list[BatchCaseRecord],
    inputs: _DetailStyle,
) -> None:
    header_fill = inputs.header_fill
    header_font = inputs.header_font
    alignment = inputs.alignment

    _append_excel_row(sheet, _DETAIL_HEADERS)
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = alignment(wrap_text=True, vertical="top")
    for index, record in enumerate(records, start=1):
        _append_excel_row(sheet, _detail_row(index, record))
    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = alignment(wrap_text=True, vertical="top")
    _set_detail_widths(sheet, inputs.get_column_letter)
    sheet.freeze_panes = "A2"


def _detail_row(index: int, record: BatchCaseRecord) -> list[object]:
    meta = record.case_meta
    details = _extract_case_details(record)
    dimensions = _dimension_map(record)
    perf = record.perf
    error = perf.error if perf is not None else ""
    return _excel_safe_row(
        [
            meta.get("case_index", index),
            record.case_key.query or details["query"] or meta.get("query", ""),
            record.trace_id,
            "success" if record.effective_score is not None else "error",
            record.effective_score,
            *(dimensions[dimension].get("score_hint", "") for dimension in DIMENSIONS),
            record.reason[:2000],
            details["query_classify_actual"],
            details["response_source"],
            details["component_text"][:2000],
            "是" if details["has_cmd"] else "否",
            details["command_summary"],
            details["tool_call_count"],
            details["lite_dag"],
            details["tool_call_names"],
            details["user_city"],
            details["user_city_code"],
            error,
            json.dumps(
                {
                    "dimension_results": record.dimension_results,
                    "checker_results": record.checker_results,
                    "hard_rule_hits": record.hard_rule_hits,
                },
                ensure_ascii=False,
            )[:3000],
            "",
            *("" for _ in DIMENSIONS),
            "",
            "",
            "",
            "",
            "",
        ]
    )


def _default_human_style() -> _HumanSheetStyle:
    """Load styles only when optional Excel support is actually used."""
    from openpyxl.styles import (  # noqa: PLC0415 - optional
        Alignment,
        Border,
        Font,
        PatternFill,
        Side,
    )
    from openpyxl.utils import get_column_letter  # noqa: PLC0415 - optional

    return _HumanSheetStyle(
        PatternFill, Font(color="FFFFFF", bold=True), Alignment, Border, Side, get_column_letter
    )


def append_human_eval_sheet(
    workbook: "Workbook",
    detail: object,
    inputs: _HumanSheetStyle | None = None,
) -> list[str]:
    """Append the shared D0-D6 review layout; return missing context columns."""
    if "人工评测" in workbook.sheetnames:
        return []
    if inputs is None:
        inputs = _default_human_style()
    pattern_fill = inputs.pattern_fill
    header_font = inputs.header_font
    alignment = inputs.alignment
    border = inputs.border
    side = inputs.side
    get_column_letter = inputs.get_column_letter

    sheet = workbook.create_sheet("人工评测")
    header_map = {cell.value: index for index, cell in enumerate(detail[1])}
    aliases = {
        "langfuse_trace_id（原始 VUI Trace）": "trace_id",
        "是否有 CMD": "是否有CMD",
        "CMD 摘要": "CMD摘要",
    }
    indices = {
        name: header_map.get(name, header_map.get(aliases.get(name, name)))
        for name in _HUMAN_CONTEXT_HEADERS
    }
    missing = [name for name, index in indices.items() if index is None]
    thin_border = border(
        left=side(style="thin"),
        right=side(style="thin"),
        top=side(style="thin"),
        bottom=side(style="thin"),
    )
    context_fill = pattern_fill("solid", fgColor="2E75B6")
    input_fill = pattern_fill("solid", fgColor="E2EFDA")
    input_row_fill = pattern_fill("solid", fgColor="F2F7ED")
    headers = [*_HUMAN_CONTEXT_HEADERS, *_HUMAN_FILL_HEADERS]
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=column, value=header)
        cell.fill = context_fill if column <= len(_HUMAN_CONTEXT_HEADERS) else input_fill
        cell.font = header_font
        cell.alignment = alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = thin_border
    for row_index, row in enumerate(detail.iter_rows(min_row=2, values_only=True), start=2):
        for column, header in enumerate(_HUMAN_CONTEXT_HEADERS, start=1):
            cell = sheet.cell(row=row_index, column=column)
            _set_excel_value(cell, row[indices[header]] if indices[header] is not None else "")
            cell.alignment = alignment(wrap_text=True, vertical="top")
            cell.border = thin_border
        for column in range(len(_HUMAN_CONTEXT_HEADERS) + 1, len(headers) + 1):
            cell = sheet.cell(row=row_index, column=column)
            cell.fill = input_row_fill
            cell.alignment = alignment(wrap_text=True, vertical="top")
            cell.border = thin_border
    for column in range(1, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(column)].width = 16
    for column in (2, 3, 9, 11, 13, 23):
        sheet.column_dimensions[get_column_letter(column)].width = 40
    sheet.freeze_panes = "A2"
    return missing


def _write_case_detail_sheet(
    workbook: "Workbook",
    records: list[BatchCaseRecord],
    inputs: _CaseDetailStyle,
) -> None:
    header_fill = inputs.header_fill
    header_font = inputs.header_font
    alignment = inputs.alignment
    get_column_letter = inputs.get_column_letter

    sheet = workbook.create_sheet("Case详细报告")
    headers = [
        "case_index",
        "trace_id",
        "query",
        "auto_score",
        "overall_reason",
        "checker_judgments",
        "root_cause",
        "归因状态",
        "trace_meta",
        *(
            item
            for dimension in DIMENSIONS
            for item in (
                f"{dimension}_score",
                f"{dimension}_status",
                f"{dimension}_reason",
            )
        ),
    ]
    _append_excel_row(sheet, headers)
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = alignment(wrap_text=True, vertical="top")
    for index, record in enumerate(records, start=1):
        dimensions = _dimension_map(record)
        checker_judgments = "\n".join(
            f"{item.get('source_name', '')}: {item.get('status', '')} - {item.get('reason', '')}"
            for item in record.checker_results
            if isinstance(item, dict)
        )
        _append_excel_row(
            sheet,
            _excel_safe_row(
                [
                    record.case_meta.get("case_index", index),
                    record.trace_id,
                    record.case_key.query
                    or _extract_case_details(record)["query"]
                    or record.case_meta.get("query", ""),
                    record.effective_score,
                    record.reason,
                    checker_judgments,
                    record.attribution.root_cause_code or record.attribution.root_cause,
                    record.attribution.status,
                    record.trace_meta,
                    *(
                        value
                        for dimension in DIMENSIONS
                        for value in (
                            dimensions[dimension].get("score_hint", ""),
                            dimensions[dimension].get("status", ""),
                            dimensions[dimension].get("reason", ""),
                        )
                    ),
                ]
            ),
        )
    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = alignment(wrap_text=True, vertical="top")
    for column in range(1, len(headers) + 1):
        sheet.column_dimensions[get_column_letter(column)].width = 20
    for column in (3, 5, 6, 9, *range(12, len(headers) + 1, 3)):
        sheet.column_dimensions[get_column_letter(column)].width = 42
    sheet.freeze_panes = "A2"


def _extract_case_details(record: BatchCaseRecord) -> dict[str, object]:
    """Project the VUI evidence needed for product manual review from persisted evaluator IO."""

    trace = record.trace_meta
    input_value = _as_dict(trace.get("input"))
    output_value = _as_dict(trace.get("output"))
    request_body = _as_dict(input_value.get("request_body"))
    geography = _as_dict(request_body.get("geography"))
    viewer_geography = _as_dict(_as_dict(input_value.get("viewer_context")).get("geography"))
    cmd = output_value.get("cmd") or _command_from_vui_actions(output_value.get("vui_action"))
    tool_names = _tool_names(output_value)
    component = _as_dict(output_value.get("component"))
    city_code = geography.get("city_code") or viewer_geography.get("city_code") or ""
    city_name = geography.get("address_l3") or viewer_geography.get("address_l3") or ""
    return {
        "query": request_body.get("query") or input_value.get("query") or "",
        "query_classify_actual": output_value.get("queryClassify") or "",
        "response_source": output_value.get("source") or "",
        "component_text": component.get("text") or component.get("displayText") or "",
        "has_cmd": bool(cmd),
        "command_summary": _command_summary(cmd),
        "tool_call_count": len(tool_names),
        "lite_dag": output_value.get("dag") or "",
        "tool_call_names": "\n".join(
            f"{index}. {name}" for index, name in enumerate(tool_names, start=1)
        ),
        "user_city": city_name or city_code,
        "user_city_code": str(city_code),
    }


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _command_from_vui_actions(value: object) -> object:
    if not isinstance(value, list):
        return None
    for action in value:
        if isinstance(action, dict) and str(action.get("type", "")).lower() == "cmd":
            return action.get("data") or action
    return None


def _command_summary(cmd: object) -> str:
    if not isinstance(cmd, dict):
        return str(cmd or "")[:500]
    poi_list = cmd.get("poi_list")
    poi_count = len(poi_list) if isinstance(poi_list, list) else 0
    return (
        f"operateParam={cmd.get('operateParam', cmd.get('param_operate', ''))}; "
        f"paramType={cmd.get('paramType', cmd.get('param_type', ''))}; "
        f"paramType1={str(cmd.get('paramType1', cmd.get('param_type1', '')))[:200]}; "
        f"paramType2={cmd.get('paramType2', cmd.get('param_type2', ''))}; poi_count={poi_count}"
    )


def _tool_names(output_value: dict[str, object]) -> list[str]:
    used_tools = output_value.get("used_tools")
    if not isinstance(used_tools, list):
        used_tools = _as_dict(output_value.get("content_desc")).get("used_tool")
    if isinstance(used_tools, list):
        return [str(item) for item in used_tools if item]
    tool_calls = output_value.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [
        str(item.get("name") or item.get("tool_name") or "")
        for item in tool_calls
        if isinstance(item, dict) and (item.get("name") or item.get("tool_name"))
    ]


def _dimension_map(record: BatchCaseRecord) -> dict[str, dict[str, object]]:
    result = {dimension: {} for dimension in DIMENSIONS}
    for item in record.dimension_results:
        if isinstance(item, dict) and item.get("dimension") in result:
            result[str(item["dimension"])] = item
    return result


def _set_excel_value(cell: "Cell", value: object) -> None:
    """Write evidence as literal text; keep numeric scores numeric."""
    cell.value = _excel_safe(value)
    if isinstance(cell.value, str):
        cell.data_type = "s"


def _append_excel_row(sheet: object, values: list[object]) -> None:
    sheet.append(_excel_safe_row(values))
    for cell in sheet[sheet.max_row]:
        if isinstance(cell.value, str):
            cell.data_type = "s"


def _excel_safe_row(values: list[object]) -> list[str | int | float | bool | None]:
    return [_excel_safe(value) for value in values]


def _excel_safe(value: object) -> str | int | float | bool | None:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)[:5000]
    if isinstance(value, str):
        # Visible hex escapes preserve the identity of XML-illegal control characters.
        return value.translate(_EXCEL_CONTROL_ESCAPES)[:5000]
    return value


def _set_detail_widths(sheet: object, get_column_letter: Callable[[int], str]) -> None:

    for column in range(1, len(_DETAIL_HEADERS) + 1):
        sheet.column_dimensions[get_column_letter(column)].width = 16
    for column in (2, 3, 13, 15, 18, 20, 21, 32):
        if column <= len(_DETAIL_HEADERS):
            sheet.column_dimensions[get_column_letter(column)].width = 40


__all__ = ["append_human_eval_sheet", "render_case_eval_xlsx"]
