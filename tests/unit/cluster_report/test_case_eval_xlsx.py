"""Product review workbook tests migrated with their renderer."""

import subprocess
import sys
from pathlib import Path

import pytest
from benchmark.reporting.cluster.render.case_eval_xlsx import render_case_eval_xlsx
from scripts.add_human_eval_sheet import add_human_eval_sheet

from tests.unit.cluster_report.test_html_payload import _record


class TestCaseEvalXlsx:
    def test_writes_product_review_workbook(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        record = _record(trace_id="vui-trace", query="附近的咖啡店", score=1.0)
        record.dimension_results = [
            {"dimension": "D0", "score_hint": 2.0, "status": "pass", "reason": "安全"},
            {"dimension": "D4", "score_hint": 0.0, "status": "fail", "reason": "未完成"},
        ]
        record.trace_meta = {
            "input": {
                "request_body": {
                    "query": "附近的咖啡店",
                    "geography": {"address_l3": "杭州市", "city_code": "330100"},
                },
            },
            "output": {
                "component": {"text": "已为你找到附近咖啡店"},
                "source": "lite_sdm",
                "queryClassify": "search_poi",
                "vui_action": [
                    {"type": "cmd", "data": {"operateParam": "openPageViaSchema"}},
                ],
                "dag": "search_place",
                "used_tools": ["search_place"],
            },
        }
        path = tmp_path / "case_eval_detail.xlsx"

        render_case_eval_xlsx([record], str(path))

        workbook = openpyxl.load_workbook(path)
        assert workbook.sheetnames == ["汇总", "评测明细", "人工评测", "Case详细报告"]
        detail = workbook["评测明细"]
        headers = [cell.value for cell in detail[1]]
        assert "D0 安全与合规" in headers
        assert "D6 信息充分性与决策帮助度" in headers
        assert "D3 信息真实性与可信度" in headers
        assert "人工预期D4 任务落地与执行闭环(0/1/2)" in headers
        assert "HTTP 状态" not in headers
        assert (
            detail.cell(
                row=2,
                column=headers.index("langfuse_trace_id（原始 VUI Trace）") + 1,
            ).value
            == "vui-trace"
        )
        assert detail.cell(row=2, column=headers.index("query") + 1).value == "附近的咖啡店"
        assert "对话历史" not in headers
        assert (
            detail.cell(row=2, column=headers.index("文本回复") + 1).value == "已为你找到附近咖啡店"
        )
        assert detail.cell(row=2, column=headers.index("是否有 CMD") + 1).value == "是"
        assert detail.cell(row=2, column=headers.index("工具调用数") + 1).value == 1
        assert detail.cell(row=2, column=headers.index("用户所在城市") + 1).value == "杭州市"
        assert workbook["人工评测"].max_row == 2
        assert workbook["Case详细报告"].max_row == 2


@pytest.mark.parametrize("driver", ["function", "script"])
@pytest.mark.parametrize(
    "trace_header", ["langfuse_trace_id（原始 VUI Trace）", "trace_id"]
)
def test_existing_workbook_gets_human_sheet_without_changing_automatic_scores(
    tmp_path: Path,
    trace_header: str,
    driver: str,
) -> None:
    def invoke() -> None:
        if driver == "script":
            script = Path(__file__).resolve().parents[3] / "scripts" / "add_human_eval_sheet.py"
            subprocess.run(
                [sys.executable, str(script), path.name],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )
        else:
            add_human_eval_sheet(str(path))

    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "review.xlsx"
    record = _record(trace_id="original-trace", query="附近咖啡", score=1.0)
    record.trace_meta["output"] = {"component": {"text": "已找到咖啡店"}}
    render_case_eval_xlsx([record], str(path))
    workbook = openpyxl.load_workbook(path)
    expected_human_headers = [cell.value for cell in workbook["人工评测"][1]]
    workbook.remove(workbook["人工评测"])
    for cell in workbook["评测明细"][1]:
        if cell.value == "langfuse_trace_id（原始 VUI Trace）":
            cell.value = trace_header
    original_detail = list(workbook["评测明细"].values)
    workbook.save(path)
    workbook.close()
    invoke()
    workbook = openpyxl.load_workbook(path)
    assert "人工评测" in workbook.sheetnames
    human = workbook["人工评测"]
    assert [cell.value for cell in human[1]] == expected_human_headers
    row = dict(
        zip([cell.value for cell in human[1]], [cell.value for cell in human[2]], strict=True)
    )
    assert row["query"] == "附近咖啡"
    assert row["文本回复"] == "已找到咖啡店"
    assert row["langfuse_trace_id（原始 VUI Trace）"] == "original-trace"
    assert list(workbook["评测明细"].values) == original_detail
    workbook.close()
    before = path.read_bytes()
    invoke()
    assert path.read_bytes() == before


def test_missing_context_column_reports_warning(tmp_path: Path, capsys) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "legacy.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "评测明细"
    workbook.active.append(["case_index", "query"])
    workbook.active.append([1, "附近咖啡"])
    workbook.save(path)
    workbook.close()
    add_human_eval_sheet(str(path))
    captured = capsys.readouterr()
    assert "缺少上下文列" in captured.err
    assert "文本回复" in captured.err


def test_collection_does_not_require_optional_openpyxl() -> None:
    test_file = Path(__file__).resolve()
    code = """
import sys, importlib.abc, runpy
class BlockExcel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'openpyxl' or fullname.startswith('openpyxl.'):
            raise ModuleNotFoundError('optional Excel unavailable', name='openpyxl')
sys.meta_path.insert(0, BlockExcel())
runpy.run_path(sys.argv[1], run_name='collection_only')
assert 'openpyxl' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code, str(test_file)], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize("evidence", ["=1+1", "#N/A", "ESC\x1b NUL\x00 end", "tab\tline\n"])
def test_evidence_is_saved_as_literal_excel_text(tmp_path: Path, evidence: str) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "literal.xlsx"
    record = _record(trace_id="literal-trace", query=evidence, score=1.0)
    record.trace_meta["output"] = {"component": {"text": evidence}}
    record.reason = evidence
    render_case_eval_xlsx([record], str(path))
    expected = evidence.replace("\x1b", r"\x1b").replace("\x00", r"\x00")
    workbook = openpyxl.load_workbook(path, data_only=False)
    for name, columns in [
        ("评测明细", ["query", "文本回复", "自动理由"]),
        ("人工评测", ["query", "文本回复"]),
        ("Case详细报告", ["query", "overall_reason"]),
    ]:
        sheet = workbook[name]
        headers = [cell.value for cell in sheet[1]]
        for column in columns:
            cell = sheet.cell(row=2, column=headers.index(column) + 1)
            assert cell.value == expected
            assert cell.data_type == "s"
    detail = workbook["评测明细"]
    headers = [cell.value for cell in detail[1]]
    score = detail.cell(row=2, column=headers.index("自动评分") + 1)
    assert score.value == 1.0
    assert score.data_type == "n"
    workbook.remove(workbook["人工评测"])
    workbook.save(path)
    workbook.close()
    add_human_eval_sheet(str(path))
    workbook = openpyxl.load_workbook(path, data_only=False)
    human = workbook["人工评测"]
    query = human.cell(row=2, column=2)
    assert query.value == expected
    assert query.data_type == "s"
    workbook.close()


@pytest.mark.parametrize("driver", ["script", "module"])
def test_cli_missing_workbook_reports_usage(tmp_path: Path, driver: str) -> None:
    service = Path(__file__).resolve().parents[3]
    command = (
        [str(service / "scripts" / "add_human_eval_sheet.py")]
        if driver == "script"
        else ["-m", "scripts.add_human_eval_sheet"]
    )
    result = subprocess.run(
        [sys.executable, *command],
        cwd=tmp_path if driver == "script" else service,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "-m scripts.add_human_eval_sheet report.xlsx" in result.stderr
    assert "Traceback" not in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("naming", ["camel", "snake", "both"])
@pytest.mark.parametrize("source", ["cmd", "vui_action"])
def test_command_summary_preserves_parameter_aliases(
    tmp_path: Path, naming: str, source: str
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    camel = {"operateParam": "navigate", "paramType": "1", "paramType1": "8", "paramType2": "3"}
    snake = {"param_operate": "navigate", "param_type": "1", "param_type1": "8", "param_type2": "3"}
    cmd = {"poi_list": [{"name": "目的地"}]}
    if naming in {"snake", "both"}:
        cmd.update(snake)
    if naming == "both":
        cmd.update({key: "legacy" for key in snake})
    if naming in {"camel", "both"}:
        cmd.update(camel)
    record = _record(trace_id="cmd-trace", query="导航", score=1.0)
    record.trace_meta["output"] = (
        {"cmd": cmd} if source == "cmd" else {"vui_action": [{"type": "cmd", "data": cmd}]}
    )
    path = tmp_path / "cmd.xlsx"
    render_case_eval_xlsx([record], str(path))
    workbook = openpyxl.load_workbook(path)
    for name in ["评测明细", "人工评测"]:
        sheet = workbook[name]
        headers = [cell.value for cell in sheet[1]]
        assert sheet.cell(row=2, column=headers.index("CMD 摘要") + 1).value == (
            "operateParam=navigate; paramType=1; paramType1=8; paramType2=3; poi_count=1"
        )
    workbook.close()
