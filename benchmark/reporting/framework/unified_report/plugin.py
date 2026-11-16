"""ReportPayload business plugins: protocol, registry, and resolution."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.reporting.framework.report_spec import ReportJobInput
from benchmark.reporting.framework.unified_report.payload import CaseColumnSpec, ReportPayload
from benchmark.plugins.host import PluginResolution, resolve_plugin

logger = get_logger("agent_eval.unified_report.plugin")

_CATEGORY = "report_payload_plugins"


@dataclass(slots=True)
class ReportPayloadContext:
    """Context for plugin payload filling."""

    job: ReportJobInput
    spec: dict[str, Any] | None = None


@runtime_checkable
class ReportPayloadPlugin(Protocol):
    """Assemble a ReportPayload per benchmark; optional template_name supplies the business default template."""

    name: str

    async def build_payload(self, ctx: ReportPayloadContext) -> ReportPayload:
        """Assemble the full fill payload from the DB / job."""
        ...

    def case_columns(self) -> list[CaseColumnSpec]:
        """Optional custom Case columns; an empty list by default means the standard 8 columns."""
        ...

    def fused_attr_field(self) -> str:
        """Primary distribution field name for overall attribution."""
        ...


_REGISTRY: dict[str, ReportPayloadPlugin] = {}


def register_report_payload_plugin(benchmark_name: str, plugin: ReportPayloadPlugin) -> None:
    key = (benchmark_name or "").strip()
    if not key:
        raise ValueError("benchmark_name required")
    _REGISTRY[key] = plugin
    logger.info(
        "report_payload.registered benchmark=%s plugin=%s",
        key,
        type(plugin).__name__,
    )


def clear_report_payload_plugins() -> None:
    _REGISTRY.clear()


def _remote_factory(binding: object) -> ReportPayloadPlugin:
    from benchmark.reporting.framework.unified_report.remote_plugin import (  # noqa: PLC0415
        RemoteReportPayloadPlugin,
    )

    return RemoteReportPayloadPlugin(binding)  # type: ignore[arg-type]


def resolve_report_payload_plugin(benchmark_name: str | None) -> ReportPayloadPlugin:
    """RemoteConfig → registry → Default."""
    from benchmark.reporting.framework.unified_report.plugins.default import (  # noqa: PLC0415
        DefaultReportPayloadPlugin,
    )

    default = DefaultReportPayloadPlugin()
    key = (benchmark_name or "").strip()
    if not key:
        return default
    if key == "example_benchmark" and key not in _REGISTRY:
        from benchmark.reporting.framework.unified_report.plugins.example import (  # noqa: PLC0415
            ExampleBenchmarkReportPayloadPlugin,
        )

        example_default = ExampleBenchmarkReportPayloadPlugin()
        return resolve_plugin(
            _CATEGORY,
            key,
            resolution=PluginResolution(
                registry=_REGISTRY,
                default=example_default,
                remote_factory=_remote_factory,
            ),
        )
    return resolve_plugin(
        _CATEGORY,
        key,
        resolution=PluginResolution(
            registry=_REGISTRY,
            default=default,
            remote_factory=_remote_factory,
        ),
    )


__all__ = [
    "ReportPayloadContext",
    "ReportPayloadPlugin",
    "clear_report_payload_plugins",
    "register_report_payload_plugin",
    "resolve_report_payload_plugin",
]
