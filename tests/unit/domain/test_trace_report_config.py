"""Unit tests for parsing TraceReportConfig (RemoteConfig trace daily-report schedule config).

Locks: field normalization (source / sample_size / optional strings), tolerance of a top-level null,
and ``fetch_trace_report_config`` fetching RemoteConfig plus falling back to the default on a parse failure (no raise).
"""

from typing import Any
from unittest.mock import patch

from benchmark.domain import trace_report_config as cfg_mod
from benchmark.domain.trace_report_config import (
    TraceReportConfig,
    TraceReportItemConfig,
    fetch_trace_report_config,
    registered_dataset_name_from_report_name,
    versioned_trace_report_name,
)


def _fetch(raw: Any) -> TraceReportConfig:
    """Bypass RemoteConfig / the cache, and drive fetch parsing directly from the given raw value."""
    with patch.object(cfg_mod, "fetch_cached_settings", return_value=raw):
        return fetch_trace_report_config(use_cache=False)


def test_registered_dataset_name_from_report_name() -> None:
    assert registered_dataset_name_from_report_name("example-v56") == "example-v56"
    assert (
        registered_dataset_name_from_report_name(
            versioned_trace_report_name("example-v56", "reg-1"),
        )
        == "example-v56"
    )


class TestTraceReportItemConfig:
    def test_defaults(self) -> None:
        item = TraceReportItemConfig(name="r1", benchmark_name="example")
        assert item.source == "kafka"
        assert item.agent_name is None
        assert item.project_id is None
        assert item.sample_size is None
        assert item.generate_report is True
        assert item.rule_link_url is None
        assert item.report_html is None
        assert item.schedule_time is None
        assert item.schedule_timezone is None
        assert item.agent_registration_id is None
        assert item.judge_content_version is None

    def test_version_isolation_fields(self) -> None:
        item = TraceReportItemConfig.model_validate(
            {
                "name": "ds__reg__reg-v56",
                "benchmark_name": "example_benchmark",
                "agent_registration_id": "reg-v56",
                "judge_content_version": "56",
            },
        )
        assert item.agent_registration_id == "reg-v56"
        assert item.judge_content_version == "56"

    def test_full_parse(self) -> None:
        item = TraceReportItemConfig.model_validate(
            {
                "name": "globalvoice_daily",
                "benchmark_name": "example",
                "source": "Kafka",
                "agent_name": "amap_chat",
                "project_id": "cmpwln0qs04fo6j8eryfdgznb",
                "sample_size": 200,
                "generate_report": False,
                "rule_link_url": "https://alidocs.dingtalk.com/i/nodes/xxx",
                "schedule_time": "03:30",
                "schedule_timezone": "Asia/Shanghai",
                "unknown_field": "ignored",
            },
        )
        assert item.name == "globalvoice_daily"
        assert item.source == "kafka"  # normalize to lowercase
        assert item.sample_size == 200
        assert item.generate_report is False
        assert item.rule_link_url == "https://alidocs.dingtalk.com/i/nodes/xxx"
        assert item.report_html is None

    def test_report_html_accepts_filename(self) -> None:
        item = TraceReportItemConfig(
            name="amap_agent_daily",
            benchmark_name="",
            report_html="amap_agent_daily.html",
        )
        assert item.report_html == "amap_agent_daily.html"

    def test_report_html_rejects_path(self) -> None:
        item = TraceReportItemConfig(
            name="r",
            benchmark_name="b",
            report_html="../secrets.html",
        )
        assert item.report_html is None

    def test_invalid_source_fallback_kafka(self) -> None:
        item = TraceReportItemConfig(name="r", benchmark_name="b", source="mysql")
        assert item.source == "kafka"

    def test_dataset_source_accepted(self) -> None:
        item = TraceReportItemConfig(name="r", benchmark_name="b", source="dataset")
        assert item.source == "dataset"

    def test_blank_optional_str_normalized_to_none(self) -> None:
        item = TraceReportItemConfig(
            name="r",
            benchmark_name="b",
            agent_name="   ",
            project_id="",
            rule_link_url=" ",
        )
        assert item.agent_name is None
        assert item.project_id is None
        assert item.rule_link_url is None

    def test_invalid_sample_size_fallback_none(self) -> None:
        for bad in (0, -5, "abc"):
            item = TraceReportItemConfig(name="r", benchmark_name="b", sample_size=bad)
            assert item.sample_size is None

    def test_rubric_class_optional(self) -> None:
        item = TraceReportItemConfig(name="r", benchmark_name="b")
        assert item.rubric_class is None
        item = TraceReportItemConfig(
            name="r",
            benchmark_name="b",
            rubric_class=" amap_agent_kafka_rubric ",
        )
        assert item.rubric_class == "amap_agent_kafka_rubric"
        item = TraceReportItemConfig(name="r", benchmark_name="b", rubric_class="  ")
        assert item.rubric_class is None

    def test_benchmark_name_null_tolerated(self) -> None:
        """In rubric_class mode, benchmark_name may be written as null (a RemoteConfig placeholder habit)."""
        item = TraceReportItemConfig.model_validate(
            {"name": "r", "benchmark_name": None, "rubric_class": "amap_agent_kafka_rubric"},
        )
        assert item.benchmark_name == ""
        assert item.rubric_class == "amap_agent_kafka_rubric"


class TestTraceReportConfig:
    def test_defaults_disabled(self) -> None:
        cfg = TraceReportConfig()
        assert cfg.enabled is False
        assert cfg.schedule_time == "09:00"
        assert cfg.schedule_timezone == "Asia/Shanghai"
        assert cfg.reports == []

    def test_top_level_null_normalized(self) -> None:
        cfg = TraceReportConfig.model_validate(
            {"enabled": True, "schedule_time": None, "schedule_timezone": "  "},
        )
        assert cfg.enabled is True
        assert cfg.schedule_time == "09:00"
        assert cfg.schedule_timezone == "Asia/Shanghai"


class TestFetchTraceReportConfig:
    def test_fetch_full_config(self) -> None:
        raw = {
            "enabled": True,
            "schedule_time": "04:00",
            "reports": [
                {
                    "name": "globalvoice_daily",
                    "benchmark_name": "example",
                    "source": "kafka",
                    "project_id": "p1",
                },
            ],
        }
        cfg = _fetch(raw)
        assert cfg.enabled is True
        assert cfg.schedule_time == "04:00"
        assert len(cfg.reports) == 1
        assert cfg.reports[0].name == "globalvoice_daily"
        assert cfg.reports[0].project_id == "p1"

    def test_fetch_empty_returns_default(self) -> None:
        cfg = _fetch(None)
        assert cfg.enabled is False
        assert cfg.reports == []

    def test_fetch_invalid_json_returns_default_no_raise(self) -> None:
        cfg = _fetch({"enabled": "not-a-bool-xx", "reports": [{"name": 1}]})
        # A parse failure falls the whole thing back to the default (enabled=False, everything forbidden to run), and does not raise
        assert cfg.enabled is False
        assert cfg.reports == []
