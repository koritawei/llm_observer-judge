"""Report parser classes: the page alias decides which data-filling implementation is used at report time, and the rule link."""

import pytest
from benchmark.reporting.framework.unified_report.report_parser import (
    prepare_report_plugin,
    resolve_report_parser,
)

_DEFAULT_PARSER = (
    "benchmark.reporting.framework.unified_report.report_parser:DefaultReportParser"
)


def test_explicit_doc_url_overrides_rubric_class_link() -> None:
    plugin, spec = prepare_report_plugin(
        {
            "benchmark_name": "example_benchmark",
            "report_parser_class": _DEFAULT_PARSER,
            "rubric_doc_url": "https://docs.example/custom",
        }
    )
    assert plugin.fused_attr_field() == "primary_issue_category"
    assert spec["rule_link_url"] == "https://docs.example/custom"


def test_default_parser_is_the_standard_payload() -> None:
    plugin, spec = prepare_report_plugin(
        {
            "report_parser_class": _DEFAULT_PARSER,
            "rubric_class": "benchmark.runs.common.rubric_classes:ComponentRubric",
            "rubric_doc_url": "",
        }
    )
    assert plugin.name == "default"
    assert plugin.fused_attr_field() == "primary_issue_category"
    assert "rule_link_url" not in spec


def test_missing_parser_keeps_benchmark_resolution() -> None:
    plugin, spec = prepare_report_plugin({"benchmark_name": "example_benchmark"})
    assert plugin.name == "example_benchmark"
    assert "report_parser_class" not in spec


def test_unknown_parser_is_rejected() -> None:
    with pytest.raises(ValueError, match="module:Class"):
        resolve_report_parser("missing")
