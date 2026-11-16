"""Report parser class filled in on the page.

Registration stores ``module:Class``. After report generation is checked, the report is
loaded and called by that class name. When left blank, resolution still uses the benchmark
name. Judge / attribution execution does not go through here.
"""

import importlib
from typing import Any

from benchmark.runs.common.rubric_base import BaseRubric
from benchmark.reporting.framework.unified_report.plugin import (
    ReportPayloadPlugin,
    resolve_report_payload_plugin,
)
from benchmark.reporting.framework.unified_report.plugins.default import (
    DefaultReportPayloadPlugin,
)
from benchmark.plugins.class_loader import (
    ALLOWED_PLUGIN_MODULE_PREFIXES,
    PluginClassLoadError,
    load_plugin_class,
)

_REPORT_PARSER_PREFIXES: tuple[str, ...] = (
    *ALLOWED_PLUGIN_MODULE_PREFIXES,
    "benchmark.reporting.framework.unified_report.",
)


class ReportParserError(ValueError):
    """The report parser class alias does not exist."""


class RubricDocumentReportParser:
    """Prepare the report from the selected rubric class and document URL, then hand it to the matching fill plugin."""

    alias = ""
    label = ""
    description = ""

    def payload_plugin(self) -> ReportPayloadPlugin:
        """Subclasses return the actual fill implementation."""
        raise NotImplementedError

    def apply(self, spec: dict[str, Any]) -> dict[str, Any]:
        """A non-empty document URL overrides the rule link; blank keeps the link on the rubric class."""
        prepared = dict(spec)
        explicit = prepared.get("rubric_doc_url")
        if isinstance(explicit, str) and explicit.strip():
            prepared["rule_link_url"] = explicit.strip()
            return prepared
        alias = prepared.get("rubric_class")
        if isinstance(alias, str) and alias.strip():
            fallback = _rubric_rule_link(alias.strip())
            if fallback:
                prepared["rule_link_url"] = fallback
        return prepared


class DefaultReportParser(RubricDocumentReportParser):
    """Generic report parser."""

    alias = "default_report"
    label = "默认报表解析"
    description = "标准装数，规则链接取页面 URL 或 rubric 类文档"

    def payload_plugin(self) -> ReportPayloadPlugin:
        return DefaultReportPayloadPlugin()


def resolve_report_parser(class_ref: str) -> RubricDocumentReportParser:
    """Load the parser class by ``module:Class``. Built-in aliases are no longer accepted."""
    cls = load_report_parser_class(class_ref)
    if not issubclass(cls, RubricDocumentReportParser):
        msg = f"报表解析类 {class_ref} 需要继承 RubricDocumentReportParser"
        raise ReportParserError(msg)
    instance = cls()
    if not isinstance(instance, RubricDocumentReportParser):
        msg = f"报表解析类 {class_ref} 无法实例化"
        raise ReportParserError(msg)
    return instance


def prepare_report_plugin(spec: dict[str, Any]) -> tuple[ReportPayloadPlugin, dict[str, Any]]:
    """Load and call when a parser class name is present; otherwise keep the original resolution."""
    class_ref = str(spec.get("report_parser_class") or "").strip()
    if not class_ref:
        name = str(spec.get("benchmark_name") or "")
        return resolve_report_payload_plugin(name or None), spec
    parser = resolve_report_parser(class_ref)
    return parser.payload_plugin(), parser.apply(spec)


def load_report_parser_class(class_ref: str) -> type:
    """Load a report parser class. Evaluation-plugin prefixes and the unified report package are allowed."""
    module_path, sep, attr_name = class_ref.strip().partition(":")
    module_path, attr_name = module_path.strip(), attr_name.strip()
    if not sep or not module_path or not attr_name:
        msg = f"报表解析类需要 module:Class，收到 {class_ref!r}"
        raise ReportParserError(msg)
    if not module_path.startswith(_REPORT_PARSER_PREFIXES):
        msg = f"报表解析类模块不在白名单: {module_path}"
        raise ReportParserError(msg)
    if module_path.startswith(ALLOWED_PLUGIN_MODULE_PREFIXES):
        try:
            return load_plugin_class(class_ref.strip(), what="report_parser")
        except PluginClassLoadError as exc:
            raise ReportParserError(str(exc)) from exc
    module = importlib.import_module(module_path)
    loaded = getattr(module, attr_name, None)
    if not isinstance(loaded, type):
        msg = f"报表解析类不存在: {class_ref}"
        raise ReportParserError(msg)
    return loaded


def validate_rubric_class_ref(class_ref: str) -> None:
    """The rubric must be a loadable subclass of BaseRubric."""
    try:
        rubric_cls = load_plugin_class(class_ref, what="rubric_class")
    except PluginClassLoadError as exc:
        raise ValueError(str(exc)) from exc
    if getattr(rubric_cls, "uses_component_rubric", False):
        return
    if not issubclass(rubric_cls, BaseRubric):
        msg = "rubric 类需要继承 BaseRubric"
        raise ValueError(msg)


def _rubric_rule_link(class_ref: str) -> str | None:
    try:
        rubric_cls = load_plugin_class(class_ref, what="rubric_class")
    except PluginClassLoadError:
        return None
    if not isinstance(rubric_cls, type) or not issubclass(rubric_cls, BaseRubric):
        return None
    url = getattr(rubric_cls, "rule_link_url", None)
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


__all__ = [
    "DefaultReportParser",
    "ReportParserError",
    "load_report_parser_class",
    "prepare_report_plugin",
    "resolve_report_parser",
    "validate_rubric_class_ref",
]
