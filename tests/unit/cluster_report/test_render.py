"""T32 unit tests: artifact rendering (md / json / xlsx).

The point of the xlsx part is not "it can write a file". It is that **eval-compare.py can read it correctly** —
the test reuses that script's parsers (`_find_header_row` / `detect_version` /
`find_metric_columns` / `read_sheet_data`) to check from the other direction, instead of asserting column-name strings itself.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
from benchmark.reporting.cluster.constants import BUCKET_LABELS, MODEL_BUCKETS, UNGROUPED_SHEET
from benchmark.reporting.cluster.render.markdown import render_json, render_markdown
from benchmark.reporting.cluster.render.xlsx import (
    HEADER_ROW,
    SHEET_COLUMN_LAYOUT,
    TAG_HEADER,
    build_header,
    build_row,
    build_sheet_rows,
    column_layout,
    render_xlsx,
)
from benchmark.reporting.cluster.schema import (
    AttributionView,
    BatchCaseRecord,
    BucketPerf,
    CaseKey,
    CasePerf,
    ClusterReport,
    Coverage,
    L1Stats,
)

_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[4]
    / "vehicle-agent"
    / "skills"
    / "vehicle-agent-eval"
    / "scripts"
)


def _load_script(name: str, module_name: str):
    path = _SCRIPTS_DIR / name
    if not path.is_file():
        pytest.skip(f"script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip("cannot load spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"script import failed (missing deps?): {name} {exc}")
    return module


@pytest.fixture(scope="module")
def compare_script():
    return _load_script("eval-compare.py", "_eval_compare_render")


def _record(  # noqa: PLR0913
    *,
    trace_id: str = "t-1",
    suite: str = "车机-Standard",
    case_id: str = "std-1",
    tag: str = "navigation.direct",
    query: str = "导航回家",
    turn_index: str = "",
    score: float | None = 1.0,
    reason: str = "缺少 CMD",
    perf: CasePerf | None = None,
    attribution: AttributionView | None = None,
    failure_modes: list[str] | None = None,
) -> BatchCaseRecord:
    return BatchCaseRecord(
        trace_id=trace_id,
        score=score,
        reason=reason,
        case_meta={"suite": suite, "case_id": case_id, "agent_mode": "standard"},
        case_key=CaseKey(
            suite=suite,
            case_id=case_id,
            tag=tag,
            query=query,
            turn_index=turn_index,
        ),
        failure_modes=[{"mode": m} for m in (failure_modes or [])],
        attribution=attribution or AttributionView(),
        perf=perf,
    )


def _perf() -> CasePerf:
    return CasePerf(
        buckets={
            "react": BucketPerf(
                calls=2,
                prompt_tokens=200,
                completion_tokens=40,
                total_tokens=240,
                latency_ms=3000.0,
                cost=0.01,
            ),
            "cgi": BucketPerf(
                calls=1,
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
                latency_ms=200.0,
            ),
        },
        e2e_ms=4200.0,
        first_text_ms=980.0,
    )


class TestSheetGrouping:
    def test_groups_by_suite(self) -> None:
        grouped = build_sheet_rows(
            [
                _record(trace_id="t-1", suite="车机-Standard"),
                _record(trace_id="t-2", suite="车机-Standard"),
                _record(trace_id="t-3", suite="车机-Max"),
            ]
        )
        assert set(grouped) == {"车机-Standard", "车机-Max"}
        assert len(grouped["车机-Standard"]) == 2

    def test_missing_suite_falls_into_ungrouped(self) -> None:
        """A case with no suite (black-box/white-box/sampled) is not dropped."""
        grouped = build_sheet_rows([_record(suite="")])
        assert UNGROUPED_SHEET in grouped


class TestColumnLayout:
    @pytest.mark.parametrize(
        ("sheet", "expected"),
        [
            ("车机-Standard", (4, 5)),
            ("车机-Max", (4, 5)),
            ("车机-Lite", (4, 5)),
            ("车机-多轮", (3, 4)),  # Multi-turn is missing one leading column
            ("车机-Standard-核心", (4, 5)),
        ],
    )
    def test_layout_matches_sheet_configs(self, sheet: str, expected: tuple) -> None:
        assert column_layout(sheet) == expected

    def test_unknown_sheet_uses_default(self) -> None:
        assert column_layout("未分组") == (4, 5)

    def test_layout_table_matches_compare_script(self, compare_script) -> None:
        """The column-position table must match SHEET_CONFIGS in eval-compare.py item by item.

        An eval-compare tuple is ``(name, tag_col, query_col)``; SHEET_COLUMN_LAYOUT is likewise
        ``(tag_col, query_col)``. Unpack and compare with the same meaning; both sides must not be reversed together.
        """
        script_layout = {
            name: (tag_col, query_col) for name, tag_col, query_col in compare_script.SHEET_CONFIGS
        }
        assert SHEET_COLUMN_LAYOUT == script_layout


class TestHeaderAndRowAlignment:
    def test_tag_and_query_land_on_declared_columns(self, compare_script) -> None:
        """intent/query on the header and data rows must land on the columns eval-compare declares.

        Column positions use eval-compare.SHEET_CONFIGS=``(name, tag_col, query_col)`` as the only
        source of truth, and are not inferred backward from the column_layout under test — otherwise both sides
        reversed together would hide the bug. Assert the real meaning: column tag_col holds intent, column query_col holds query.
        """
        configs = {
            name: (tag_col, query_col) for name, tag_col, query_col in compare_script.SHEET_CONFIGS
        }
        for sheet in ("车机-Standard", "车机-多轮"):
            tag_col, query_col = configs[sheet]
            header = build_header(sheet)
            row = build_row(_record(query="去机场", tag="navigation.airport"), sheet)

            assert "intent" in header[tag_col - 1], sheet
            assert header[query_col - 1] == "query", sheet
            assert row[tag_col - 1] == "navigation.airport", sheet
            assert row[query_col - 1] == "去机场", sheet

    def test_header_and_row_same_length(self) -> None:
        for sheet in ("车机-Standard", "车机-多轮", "未知sheet"):
            assert len(build_header(sheet)) == len(build_row(_record(), sheet)), sheet

    def test_bucket_columns_use_script_labels(self, compare_script) -> None:
        """The three bucket column names must use eval-compare's BUCKET_LABELS (full equality; an alias does not count)."""
        header = build_header("车机-Standard")
        for bucket in MODEL_BUCKETS:
            label = compare_script.BUCKET_LABELS[bucket]
            assert f"[3.0] {label}调用次数" in header
            assert f"[3.0] {label}平均耗时(ms)" in header
            assert f"[3.0] {label}平均输入Tokens" in header
            assert f"[3.0] {label}平均输出Tokens" in header
        assert BUCKET_LABELS == compare_script.BUCKET_LABELS

    def test_bucket_averages_are_per_call(self) -> None:
        """Bucket columns write the per-call mean (total/call count), matching the script's avg_* definition."""
        header = build_header("车机-Standard")
        row = build_row(_record(perf=_perf()), "车机-Standard")
        idx = header.index("[3.0] React调用次数")

        assert row[idx] == 2
        assert row[idx + 1] == 1500  # 3000ms / 2 calls
        assert row[idx + 2] == 100  # 200 tokens / 2
        assert row[idx + 3] == 20  # 40 tokens / 2

    def test_missing_perf_writes_blank_not_zero(self) -> None:
        """Missing performance is written as an empty string — writing 0 would be treated by eval-compare as a real measurement in the mean."""
        header = build_header("车机-Standard")
        row = build_row(_record(perf=None), "车机-Standard")

        assert row[header.index("[3.0] 端到端耗时(ms)")] == ""
        assert row[header.index("[3.0] 首正文回复耗时(ms)")] == ""
        assert row[header.index("[3.0] React调用次数")] == ""

    def test_three_score_columns(self) -> None:
        header = build_header("车机-Standard")
        row = build_row(
            _record(
                score=0.0,
                attribution=AttributionView(corrected_triage_level="good"),
            ),
            "车机-Standard",
        )

        assert row[header.index("[3.0] 评测打分")] == 0.0
        assert row[header.index("judge_score（原评测分）")] == 0.0
        assert row[header.index("attribution_score（归因纠正分）")] == 2.0
        assert row[header.index("effective_score（聚合最终采用分）")] == 2.0

    def test_attribution_columns(self) -> None:
        """failure_mode / root_cause_code share one enum (see failure_modes.NAME_OF).
        The render layer shows the Chinese name rather than the English code, so non-developers can read the report."""
        header = build_header("车机-Standard")
        row = build_row(
            _record(
                failure_modes=["tool_cmd_tag_missing", "cmd_param_destination"],
                attribution=AttributionView(
                    status="completed",
                    root_cause="CMD 缺失",
                    root_cause_code="cmd_param_destination",
                    conclusion_type="agent_badcase",
                    owner_group="prompt-team",
                ),
            ),
            "车机-Standard",
        )

        assert (
            row[header.index("failure_mode")]
            == "最终 Chat 未输出 tool_cmd 标记（断点②）,终点参数错误"
        )
        assert row[header.index("root_cause_code")] == "终点参数错误"
        assert row[header.index("归因结论")] == "CMD 缺失"
        assert row[header.index("owner_group")] == "prompt-team"
        assert row[header.index("归因状态")] == "completed"

    def test_unmapped_code_falls_back_to_raw_value(self) -> None:
        """A code not in FAILURE_MODES (old data / a custom value) is shown as-is and not silently dropped."""
        header = build_header("车机-Standard")
        row = build_row(
            _record(
                failure_modes=["legacy_custom_mode"],
                attribution=AttributionView(root_cause_code="legacy_custom_mode"),
            ),
            "车机-Standard",
        )

        assert row[header.index("failure_mode")] == "legacy_custom_mode"
        assert row[header.index("root_cause_code")] == "legacy_custom_mode"


class TestXlsxReadableByCompareScript:
    """End-to-end reverse check: the written xlsx can be read correctly by eval-compare.py's parsers."""

    @pytest.fixture()
    def written_workbook(self, tmp_path: Path, compare_script):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "eval_summary.xlsx"
        render_xlsx(
            [
                _record(
                    trace_id="t-1",
                    suite="车机-Standard",
                    case_id="std-1",
                    tag="navigation.direct",
                    query="导航回家",
                    score=2.0,
                    perf=_perf(),
                ),
                _record(
                    trace_id="t-2",
                    suite="车机-Standard",
                    case_id="std-2",
                    tag="search.poi",
                    query="附近加油站",
                    score=0.0,
                    perf=_perf(),
                ),
                _record(
                    trace_id="t-3",
                    suite="车机-多轮",
                    case_id="mt-1",
                    tag="multi-turn.s1",
                    query="取消导航",
                    turn_index="Q2",
                    score=1.0,
                ),
            ],
            str(path),
        )
        return openpyxl.load_workbook(str(path)), compare_script

    def test_header_row_is_discoverable(self, written_workbook) -> None:
        """_find_header_row scans only rows 1-4 and locates the header because the tag column contains 'intent'."""
        workbook, script = written_workbook
        for sheet in ("车机-Standard", "车机-多轮"):
            worksheet = workbook[sheet]
            tag_col, _ = column_layout(sheet)
            assert script._find_header_row(worksheet, tag_col) == HEADER_ROW, sheet

    def test_version_detected_as_3_0(self, written_workbook) -> None:
        workbook, script = written_workbook
        worksheet = workbook["车机-Standard"]
        assert script.detect_version(worksheet, HEADER_ROW) == "3.0"

    def test_all_metric_columns_found(self, written_workbook) -> None:
        """find_metric_columns compares for full equality — any column-name drift makes that metric unreadable.

        Regression anchor (found in review): the ``React轮次`` column was missing at first, so the comparison report's
        "React average rounds" row was permanently N/A. Assert every patterns key the script declares.
        """
        workbook, script = written_workbook
        worksheet = workbook["车机-Standard"]

        cols = script.find_metric_columns(worksheet, HEADER_ROW, "3.0")

        for key in (
            "score",
            "score_comment",
            "trace_id",
            "first_resp_ms",
            "e2e_ms",
            "react_rounds",
            "exec_mode",
        ):
            assert key in cols, f"missing column: {key}"
        for bucket in MODEL_BUCKETS:
            for suffix in ("calls", "latency", "input", "output"):
                assert f"{bucket}_{suffix}" in cols, f"missing {bucket}_{suffix}"

    def test_react_rounds_aggregated_by_script(self, written_workbook) -> None:
        """React rounds must be aggregable by the script (definition = llm_calls of the react bucket)."""
        workbook, script = written_workbook
        worksheet = workbook["车机-Standard"]
        tag_col, query_col = column_layout("车机-Standard")
        cols = script.find_metric_columns(worksheet, HEADER_ROW, "3.0")

        rows = script.read_sheet_data(worksheet, HEADER_ROW, tag_col, query_col, cols)
        metrics = script.aggregate_metrics(rows)

        assert rows[0]["react_rounds"] == 2  # _perf() react calls=2
        assert metrics.react_rounds_vals == [2.0, 2.0]

    def test_react_rounds_blank_when_no_perf(self, written_workbook) -> None:
        """When there is no performance data, do not write 0 (a written 0 would be treated as a real measurement of 0 rounds and pull the mean down)."""
        workbook, script = written_workbook
        worksheet = workbook["车机-多轮"]  # This case has perf=None
        tag_col, query_col = column_layout("车机-多轮")
        cols = script.find_metric_columns(worksheet, HEADER_ROW, "3.0")

        rows = script.read_sheet_data(worksheet, HEADER_ROW, tag_col, query_col, cols)
        metrics = script.aggregate_metrics(rows)

        assert rows[0]["react_rounds"] in (None, "")
        assert metrics.react_rounds_vals == []

    def test_data_rows_read_back_correctly(self, written_workbook) -> None:
        workbook, script = written_workbook
        worksheet = workbook["车机-Standard"]
        tag_col, query_col = column_layout("车机-Standard")
        cols = script.find_metric_columns(worksheet, HEADER_ROW, "3.0")

        rows = script.read_sheet_data(worksheet, HEADER_ROW, tag_col, query_col, cols)

        assert len(rows) == 2
        assert rows[0]["_tag"] == "navigation.direct"
        assert rows[0]["_query"] == "导航回家"
        assert rows[0]["score"] == 2.0
        assert rows[0]["trace_id"] == "t-1"
        assert rows[0]["e2e_ms"] == 4200
        assert rows[0]["react_calls"] == 2

    def test_multi_turn_sheet_read_with_its_own_layout(self, written_workbook) -> None:
        """The multi-turn sheet column positions are (3, 4): a fixed layout would shift the whole row by one column."""
        workbook, script = written_workbook
        worksheet = workbook["车机-多轮"]
        tag_col, query_col = column_layout("车机-多轮")
        cols = script.find_metric_columns(worksheet, HEADER_ROW, "3.0")

        rows = script.read_sheet_data(worksheet, HEADER_ROW, tag_col, query_col, cols)

        assert len(rows) == 1
        assert rows[0]["_tag"] == "multi-turn.s1"
        assert rows[0]["_query"] == "取消导航"

    def test_aggregate_metrics_matches_source(self, written_workbook) -> None:
        """Pass/fail counts aggregated by the script match the source data (the definition closes)."""
        workbook, script = written_workbook
        worksheet = workbook["车机-Standard"]
        tag_col, query_col = column_layout("车机-Standard")
        cols = script.find_metric_columns(worksheet, HEADER_ROW, "3.0")
        rows = script.read_sheet_data(worksheet, HEADER_ROW, tag_col, query_col, cols)

        metrics = script.aggregate_metrics(rows)

        assert metrics.case_count == 2
        assert metrics.pass_count == 1  # score=2.0
        assert metrics.partial_count == 0
        assert len(metrics.e2e_vals) == 2  # Both rows have end-to-end latency

    def test_sheets_created_per_suite(self, written_workbook) -> None:
        workbook, _ = written_workbook
        assert set(workbook.sheetnames) == {"车机-Standard", "车机-多轮"}

    def test_empty_records_produces_valid_file(self, tmp_path: Path) -> None:
        pytest.importorskip("openpyxl")
        path = tmp_path / "empty.xlsx"
        render_xlsx([], str(path))
        assert path.is_file()

    def test_empty_batch_still_writes_header(self, tmp_path: Path) -> None:
        """Regression anchor (found in smoke): an empty batch must not create only an empty sheet.

        An early implementation only called `create_sheet` when records was empty and wrote no header. The artifact opened blank,
        so you could not see the column structure or tell "no data" from "render is broken".
        """
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "empty.xlsx"
        render_xlsx([], str(path))

        worksheet = openpyxl.load_workbook(str(path))[UNGROUPED_SHEET]

        assert worksheet.cell(row=1, column=1).value == f"批次明细 · {UNGROUPED_SHEET}"
        header = [
            worksheet.cell(row=HEADER_ROW, column=c).value
            for c in range(1, len(build_header(UNGROUPED_SHEET)) + 1)
        ]
        assert "case_id" in header
        assert TAG_HEADER in header
        assert "[3.0] 评测打分" in header
        assert "[3.0] React轮次" in header


def _report() -> ClusterReport:
    return ClusterReport(
        batch_tag="run-20260729-153000",
        batch_meta={"model": "qwen-max", "agent_mode": "standard", "planned": 100},
        coverage=Coverage(
            expected_total=100,
            judged=95,
            should_attribute=20,
            attributed=18,
            pending=2,
            perf_collected=93,
            perf_note="2 条 Langfuse 拉取超时",
        ),
        stats=L1Stats(
            effectiveness={
                "total": 95,
                "scored": 95,
                "avg_score": 1.4,
                "pass_rate": 0.6,
                "partial_rate": 0.2,
                "fail_rate": 0.2,
                "score_distribution": {"0": 19, "1": 19, "2": 57},
                "score_views": {
                    "judge_score": {
                        "scored": 95,
                        "avg_score": 1.3,
                        "distribution": {"0": 24, "1": 18, "2": 53},
                    },
                    "attribution_score": {
                        "scored": 5,
                        "avg_score": 2.0,
                        "distribution": {"0": 0, "1": 0, "2": 5},
                    },
                    "effective_score": {
                        "scored": 95,
                        "avg_score": 1.4,
                        "distribution": {"0": 19, "1": 19, "2": 57},
                    },
                },
                "hard_rule_hit_rate": 0.12,
                "dimension_fail_rate": {"D4": {"judged": 80, "failed": 24, "fail_rate": 0.3}},
                "failure_modes": {"cmd_tag_missing": 9},
                "checker_fail_hits": {"cmd_checker": 9},
            },
            attribution={
                "should_attribute": 20,
                "attributed": 18,
                "pending": 2,
                "owner_fallback_rate": 0.1,
                "root_cause_top": [{"code": "cmd_tag_missing", "count": 9, "share": 0.5}],
                "owner_group_distribution": {"prompt-team": 12},
                "overturn": {
                    "overturn_count": 3,
                    "overturn_rate": 0.1667,
                    "by_type": {"eval_rule_issue": 2, "judge_false_positive": 1},
                    "cases": [],
                },
                "confidence": {"avg": 0.82, "low_rate": 0.05},
                "evidence": {"evidence_rate": 0.9},
                "human_review": {
                    "count": 1,
                    "cases": [
                        {
                            "trace_id": "t-9",
                            "suite": "车机-Max",
                            "case_id": "max-3",
                            "root_cause": "工具超时",
                        }
                    ],
                },
                "pending_backlog": {
                    "count": 2,
                    "cases": [
                        {
                            "trace_id": "t-7",
                            "suite": "车机-Max",
                            "case_id": "max-1",
                            "triage_level": "severe_bad",
                        }
                    ],
                },
            },
            performance={
                "collected": 93,
                "missing": 2,
                "missing_reasons": {"timeout": 2},
                "e2e_ms": {
                    "count": 93,
                    "avg": 3200.5,
                    "p50": 3000.0,
                    "p95": 8800.0,
                    "p99": 9900.0,
                    "max": 12000.0,
                },
                "first_text_ms": {
                    "count": 93,
                    "avg": 980.0,
                    "p50": 900.0,
                    "p95": 1800.0,
                    "p99": 2100.0,
                    "max": 2500.0,
                },
                "buckets": {
                    "react": {
                        "cases": 93,
                        "calls": {"avg": 2.1, "total": 195},
                        "latency_ms": {"count": 93, "avg": 1500.0, "p95": 3000.0},
                        "prompt_tokens": {"avg": 100.0, "avg_per_call": 47.6},
                        "completion_tokens": {"avg": 20.0, "avg_per_call": 9.5},
                    },
                    "cgi": {"cases": 0},
                },
                "total_tokens": 123456,
                "total_cost": 1.23,
            },
            crosstab={
                "checker_root_cause_gap": [
                    {"checker": "poi_checker", "hits": 4, "unmapped": 4, "unmapped_rate": 1.0},
                ],
                "suite_overturn": [
                    {
                        "suite": "车机-Max",
                        "total": 30,
                        "low_score": 12,
                        "attributed": 10,
                        "overturned": 3,
                        "overturn_rate": 0.3,
                    },
                ],
            },
        ),
        llm_summary={
            "headline": "CMD 缺失是本批次主要问题",
            "clusters": [
                {
                    "title": "CMD 标签缺失簇",
                    "root_cause_codes": ["cmd_tag_missing"],
                    "affected_count": 9,
                    "owner_group": "prompt-team",
                    "summary": "导航类请求未生成 CMD",
                    "evidence": "root_cause_top 首位 9 条",
                    "related_cases": ["t-1", "t-2"],
                }
            ],
            "feedback_suggestions": [
                {"target": "prompt", "action": "补充 CMD 生成指令", "rationale": "failure_modes"},
            ],
            "aone_suggestions": [
                {
                    "title": "修复 CMD 缺失",
                    "priority": "P1",
                    "owner_group": "prompt-team",
                    "acceptance": "该 failure_mode 归零",
                    "related_cases": ["t-1"],
                },
            ],
            "caveats": ["归因覆盖率 90%，结论存在盲区"],
        },
        generated_at="2026-07-29T16:00:00+08:00",
    )


def _unnumbered(text: str) -> str:
    """Strip the heading numbers the renderer appends (## 1. / ### 1.2) and assert against the heading text."""
    return re.sub(
        r"^(##|###) \d+(?:\.\d+)?\.? ", lambda m: f"{m.group(1)} ", text, flags=re.MULTILINE
    )


class TestMarkdownRendering:
    @pytest.fixture()
    def text(self) -> str:
        return _unnumbered(render_markdown(_report()))

    def test_header_and_meta(self, text: str) -> None:
        assert "# 聚合根因报告 · run-20260729-153000" in text
        assert "cluster_report.v2" in text
        assert "qwen-max" in text

    def test_coverage_flags_gaps(self, text: str) -> None:
        """A coverage gap must be labeled explicitly; numbers must not be dropped silently."""
        assert "95" in text and "100" in text
        assert "归因未完成 2 条" in text
        # perf_note (a fetch timeout is a performance problem, not an attribution gap) may appear in the performance section only as a "performance block" label,
        # and must not be mixed into the coverage subsection of the evaluation conclusion.
        assert "⚠️ 性能块：2 条 Langfuse 拉取超时" in text
        assert "Langfuse 拉取超时" not in text.split("## 归因结论", maxsplit=1)[0]

    def test_effectiveness_block(self, text: str) -> None:
        assert "## 评测结论" in text
        assert "### 评测结果" in text
        assert "60.0%" in text  # pass_rate
        assert "D4" in text
        assert "### 分数口径" in text
        assert "`judge_score` | 原评测分 | 95 | 1.3" in text
        assert "`attribution_score` | 归因纠正分 | 5 | 2.0" in text
        assert "`effective_score` | 聚合最终采用分 | 95 | 1.4" in text

    def test_attribution_block_highlights_overturn(self, text: str) -> None:
        assert "## 归因结论" in text
        assert "### 归因概览" in text
        assert "推翻评测结论 3 条" in text
        assert "eval_rule_issue 2" in text
        assert "cmd_tag_missing" in text
        assert "需人工复核（1 条）" in text
        assert "归因待补（2 条）" in text

    def test_performance_block_has_percentiles(self, text: str) -> None:
        assert "## 性能与运行观测" not in text
        report = _report()
        report.batch_meta["report_job_type"] = "performance"
        performance_text = _unnumbered(render_markdown(report))
        assert "## 性能与运行观测" in performance_text
        assert "### 性能指标" in performance_text
        assert "8800.0" in performance_text  # e2e P95
        assert "1800.0" in performance_text  # first_text P95
        assert "性能采集缺失" in performance_text

    def test_stability_report_prioritizes_conclusions_and_limits_evidence(self) -> None:
        report = _report()
        report.batch_meta["report_job_type"] = "stability"
        governance = {
            "priority": "P0",
            "reason": "相同输入与依赖下，执行输出或评测结果出现不一致。",
            "domain": "Agent 执行",
            "action": "排查模型参数、编排分支、缓存和调用链路。",
            "is_agent_defect": True,
        }
        groups = [
            {
                "scene_tag": "navigation.direct",
                "query": f"query-{index}",
                "run_count": 2,
                "judge_score_distribution": {"0": 1, "1": 0, "2": 1},
                "span": 2,
                "evaluation_comparability": {
                    "status": "comparable",
                    "reason": "same_known_version",
                },
                "agent_execution_observation": "nondeterminism",
                "evidence_strength": "strong",
                "priority": "P0",
                "governance": governance,
                "representative_runs": [
                    {
                        "trace_id": f"trace-{index}",
                        "judge_score": 0,
                        "context": {"fingerprint": "context"},
                        "output_evidence": {"fingerprint": "output"},
                    }
                ],
            }
            for index in range(6)
        ]
        report.stats.observations = {
            "stability": groups,
            "stability_overview": {
                "total_queries": 6,
                "stable_queries": 0,
                "stable_rate": 0.0,
                "high_risk_queries": 6,
                "comparable_risk_queries": 0,
                "insufficient_evidence_queries": 0,
                "governance_count": 6,
            },
        }
        report.llm_summary = {
            "headline": "同类请求存在执行不确定性。",
            "findings": [
                {
                    "scene_tag": "navigation.direct",
                    "query": "query-0",
                    "summary": "LLM 解释",
                }
            ],
        }

        stability_text = _unnumbered(render_markdown(report))
        detail_text = stability_text.split("## 重点证据下钻", maxsplit=1)[1]

        assert stability_text.index("## 稳定性结论") < stability_text.index("## 优先治理项")
        assert stability_text.index("## 优先治理项") < stability_text.index("## 差异原因与风险分布")
        assert stability_text.index("## 差异原因与风险分布") < stability_text.index("## 场景索引")
        assert "## Agent 执行稳定性观察" not in stability_text
        assert "## 场景下钻" not in stability_text
        assert "稳定情况：0/6 个 query 稳定（0.0%）。" in stability_text
        assert "执行不确定性" in stability_text
        assert "query-5" in stability_text
        assert "query-5" not in detail_text
        assert "query-4" in detail_text

    def test_performance_bucket_tokens_render_per_call_not_per_case(self, text: str) -> None:
        """The three-bucket detail table shows the per-call mean (avg_per_call), not the per-case total mean (avg)."""
        report = _report()
        report.batch_meta["report_job_type"] = "performance"
        performance_text = _unnumbered(render_markdown(report))
        row = next(line for line in performance_text.splitlines() if line.startswith("| React |"))
        assert "47.6" in row  # prompt_tokens.avg_per_call
        assert "9.5" in row  # completion_tokens.avg_per_call
        assert "100.0" not in row  # Not the per-case total mean
        assert "20.0" not in row

    def test_crosstab_block(self, text: str) -> None:
        assert "### 交叉分析" in text
        assert "checker → 根因映射缺口" in text
        assert "poi_checker" in text

    def test_observations_section_absent_for_legacy_batch(self, text: str) -> None:
        """A historical batch with no envelope observations: the independent-observation block is not rendered at all, and the output matches the baseline."""
        assert "### 运行观测（不影响效果评分）" not in text

    def test_observations_section_rendered_when_present(self) -> None:
        report = _report()
        report.stats.observations = {
            "metric_observations": {
                "first_text_ms": {"observed": 4, "exceeded": 1, "exceeded_rate": 0.25},
            },
            "diagnostic_findings": {
                "by_finding_key": {"llm_retry": 2},
                "by_severity": {"warning": 2},
            },
            "profiles": {
                "business_ids": {"vehicle_agent": 4},
                "scenarios": {"navigation": 4},
                "profiles": {"vehicle_agent.standard@v1": 4},
            },
        }

        text = _unnumbered(render_markdown(report))

        assert "### 运行观测（不影响效果评分）" in text
        assert "### 指标超限" in text
        assert "| first_text_ms | 4 | 1 | 25.0% |" in text
        assert "### 诊断发现" in text
        assert "| llm_retry | 2 |" in text
        assert "严重度：warning 2" in text
        assert "### 业务 / 场景 / Profile 分布" in text
        assert "| vehicle_agent.standard@v1 | 4 |" in text

    def test_failure_mode_x_root_cause_table_removed(self, text: str) -> None:
        """T5 removes the co-occurrence table: a Cartesian expansion on a small sample is low-information and is no longer rendered."""
        assert "失败模式 × 归因根因" not in text
        assert "共现数量" not in text

    def test_llm_summary_block(self, text: str) -> None:
        assert "## 语义聚类" in text
        assert "### 正式问题簇" in text
        assert "CMD 标签缺失簇" in text
        assert "关联 trace: `t-1`, `t-2`" in text
        assert "回流建议" in text
        assert "Aone 工作项建议" not in text
        assert "结论限制" in text

    def test_cluster_links_candidate_clusters_and_section_order(self) -> None:
        report = _report()
        report.llm_summary["candidate_clusters"] = [
            {
                "title": "工具返回字段缺失",
                "agent_scope": "vehicle",
                "candidate_key": "vehicle:tool_data_missing",
                "affected_count": 1,
                "summary": "工具成功但缺少约束字段",
                "evidence": "候选统计命中 1 条",
                "related_cases": ["candidate-trace"],
            }
        ]

        text = _unnumbered(
            render_markdown(report, cluster_links={1: "https://aone.example.com/#cluster-1"})
        )

        assert "[CMD 标签缺失簇](https://aone.example.com/#cluster-1)" in text
        assert "### 新问题候选簇（`other:`，待治理）" in text
        assert "vehicle:tool_data_missing" in text
        assert "待人工确认，不自动创建正式修复工单" in text
        assert "Aone 工作项建议" not in text
        assert text.index("## 语义聚类") < text.index("## 评测结论")
        assert text.index("## 评测结论") < text.index("## 归因结论")
        assert text.index("## 归因结论") < text.index("## 附录")
        assert "## 性能与运行观测" not in text

    def test_failed_cluster_stays_unlinked_with_dispatch_status(self) -> None:
        report = _report()

        text = render_markdown(
            report,
            cluster_dispatch_statuses={1: "创建失败，未生成链接"},
        )

        assert "[CMD 标签缺失簇](" not in text
        assert "- 子工单状态: 创建失败，未生成链接" in text

    def test_verify_bucket_is_not_rendered(self) -> None:
        report = _report()
        report.stats.performance["buckets"]["verify"] = {
            "cases": 9,
            "calls": {"avg": 1.0},
            "latency_ms": {"avg": 100.0, "p95": 120.0},
            "prompt_tokens": {"avg_per_call": 10.0},
            "completion_tokens": {"avg_per_call": 2.0},
        }

        assert "| Verify |" not in render_markdown(report)

    def test_llm_skipped_degrades_gracefully(self) -> None:
        report = _report()
        report.llm_summary = {"skipped": "no_model_configured", "clusters": []}

        text = _unnumbered(render_markdown(report))

        assert "未生成：no_model_configured" in text
        assert "## 评测结论" in text  # The L1 body is unaffected
        assert "## 性能与运行观测" not in text

    def test_legacy_stats_render_primary_top_only(self, text: str) -> None:
        """Old stats (no all_root_cause_top) render only the primary-root-cause Top, not an empty all-root-causes table."""
        assert "### 主根因 Top" in text
        assert "### 全部根因 Top" not in text

    def test_all_root_cause_top_section_with_multi_root_stats(self) -> None:
        report = _report()
        report.stats.attribution["all_root_cause_top"] = [
            {"code": "cmd_tag_missing", "count": 11, "share": 0.55},
            {"code": "tool_param_error", "count": 2, "share": 0.1},
        ]

        text = _unnumbered(render_markdown(report))

        assert "### 主根因 Top" in text
        assert "### 全部根因 Top" in text
        assert "| 11 |" in text  # After expanding multiple root causes, the count can exceed the primary-root-cause definition

    def test_number_warnings_surfaced(self) -> None:
        report = _report()
        report.llm_summary["number_warnings"] = ["4321", "99.9%"]

        text = _unnumbered(render_markdown(report))

        assert "未能在 L1 统计中匹配" in text
        assert "4321" in text

    def test_empty_report_does_not_crash(self) -> None:
        text = render_markdown(ClusterReport(batch_tag="run-20260101-000000"))
        assert "run-20260101-000000" in text






class TestJsonRendering:
    def test_writes_full_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        render_json(_report(), str(path))

        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["schema_version"] == "cluster_report.v2"
        assert payload["batch_tag"] == "run-20260729-153000"
        assert payload["batch_meta"]["model"] == "qwen-max"
        assert payload["coverage"]["judged"] == 95
        assert payload["stats"]["effectiveness"]["pass_rate"] == 0.6
        assert (
            payload["stats"]["effectiveness"]["score_views"]["effective_score"]["avg_score"] == 1.4
        )
        assert payload["stats"]["crosstab"]["suite_overturn"][0]["suite"] == "车机-Max"
        assert payload["llm_summary"]["headline"]

    def test_cases_serialized_with_case_key(self, tmp_path: Path) -> None:
        report = _report()
        report.cases = [_record(trace_id="t-1", perf=_perf())]
        path = tmp_path / "report.json"

        render_json(report, str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))

        case = payload["cases"][0]
        assert case["trace_id"] == "t-1"
        assert case["case_key"]["case_id"] == "std-1"
        assert case["case_meta"]["suite"] == "车机-Standard"
        assert case["perf"]["e2e_ms"] == 4200.0

    def test_chinese_not_escaped(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        render_json(_report(), str(path))
        assert "车机-Max" in path.read_text(encoding="utf-8")

    def test_observations_block_serialized(self, tmp_path: Path) -> None:
        """Envelope observation splits land with the full JSON artifact (a historical batch is an empty dict, and the key still exists)."""
        report = _report()
        report.stats.observations = {
            "metric_observations": {
                "first_text_ms": {"observed": 2, "exceeded": 1, "exceeded_rate": 0.5},
            },
        }
        path = tmp_path / "report.json"

        render_json(report, str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["stats"]["observations"]["metric_observations"]["first_text_ms"] == {
            "observed": 2,
            "exceeded": 1,
            "exceeded_rate": 0.5,
        }

    def test_legacy_batch_observations_is_empty_object(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        render_json(_report(), str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["stats"]["observations"] == {}
