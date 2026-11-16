"""MR6 render review: metadata escaping, tolerance of bad structure, and real JS rendering."""

import json
import subprocess
from pathlib import Path

import pytest
from benchmark.reporting.cluster.render.markdown import _batch_meta_lines, _llm_summary_lines, _top_table


def test_meta_table_escapes_keys_values_and_newlines() -> None:
    lines = _batch_meta_lines({"model": "A|B", "no|tes": "first\nsecond\rthird"})
    assert "| model | A\\|B |" in lines
    assert "| no\\|tes | first second third |" in lines


def test_top_table_escapes_dynamic_labels() -> None:
    assert "| A\\|B C | 2 |" in _top_table("test", {"A|B\nC": 2})


@pytest.mark.parametrize("caveats", [1, "not a list", [{}], None])
def test_invalid_historical_caveats_do_not_crash_rendering(caveats: object) -> None:
    lines = _llm_summary_lines(
        {"headline": "x", "clusters": [], "caveats": caveats},
        candidate_fallback={},
        cluster_links=None,
        cluster_dispatch_statuses=None,
    )
    assert not any(line.startswith("- ") for line in lines)


def test_shared_template_renders_markdown_at_start_and_lists() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "app/services/eval_framework/unified_report/templates/benchmark_report.html"
    )
    template = path.read_text()
    functions = []
    for name in ("escHtml", "looksLikeMarkdown", "renderMdInline", "renderMarkdown"):
        start = template.index("function " + name + "(")
        end = template.index("\nfunction ", start + 1)
        functions.append(template[start:end])
    cases = ["# 标题", "- 项目", "+ 项目", "1. 导航失败\n2. 搜索失败", "   - 缩进", "plain text"]
    script = (
        "\n".join(functions)
        + "\nprocess.stdout.write(JSON.stringify("
        + json.dumps(cases)
        + ".map(renderMarkdown)));"
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    rendered = json.loads(result.stdout)
    for markup, expected in zip(
        rendered, ["<h1", "<ul>", "<ul>", "<ol>", "<ul>", 'class="detail-text"'], strict=True
    ):
        assert expected in markup
