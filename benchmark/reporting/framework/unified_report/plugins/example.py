"""example_benchmark business ReportPayload plugin (default fill path)."""

from benchmark.reporting.framework.unified_report.plugins.default import (
    DEFAULT_JUDGE_REASON_PREFERRED_KEYS,
    DefaultReportPayloadPlugin,
)


class ExampleBenchmarkReportPayloadPlugin(DefaultReportPayloadPlugin):
    """example_benchmark: default fill / default template; score reasons prefer the ``example`` key."""

    name = "example_benchmark"

    def judge_reason_preferred_keys(self) -> tuple[str, ...]:
        return ("example", *DEFAULT_JUDGE_REASON_PREFERRED_KEYS)


__all__ = ["ExampleBenchmarkReportPayloadPlugin"]
