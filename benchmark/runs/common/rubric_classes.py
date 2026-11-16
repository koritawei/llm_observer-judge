"""Benchmark Rubric class-name registry.

Same shape as the Judge / Attribution execution classes: external config (RemoteConfig
external-daily / ``/run-external/v2``) exposes only the stable alias ``rubric_class``.
The concrete ``module:Class`` is maintained by a code allowlist; the resolved class
reference is frozen into the run snapshot ``config_json.rubric_ref``, and callback /
aggregation / report rebuild from that snapshot so multi-node and delayed-callback
semantics stay consistent.

A Rubric class (a :class:`BaseRubric` subclass) carries what to evaluate (``criteria``),
how to score (aggregation / fusion), how to parse data (``parse_interceptor_ref``),
which dimensions to stat (``stat_dimensions``), and the judge execution policy
(``judge_execution_policy``).
"""

from dataclasses import dataclass

from benchmark.runs.common.rubric_base import BaseRubric
from benchmark.plugins.class_loader import PluginClassLoadError, load_plugin_class


class ComponentRubric:
    """Explicitly use the ``@benchmark_component`` metadata rubric (new businesses that do not provide a Rubric class)."""

    uses_component_rubric = True


RUBRIC_CLASS_REFS: dict[str, str] = {
    "component_rubric": "benchmark.runs.common.rubric_classes:ComponentRubric",
}


class RubricClassResolutionError(ValueError):
    """The Rubric class name does not exist, or the implementation is invalid."""


@dataclass(frozen=True)
class RubricClassSpec:
    """Resolved Rubric alias. ``class_ref=None`` means use the component metadata rubric."""

    alias: str
    class_ref: str | None


def resolve_rubric_class(alias: str) -> RubricClassSpec:
    """Resolve a stable Rubric alias; the class must be a :class:`BaseRubric` subclass or :class:`ComponentRubric`."""
    normalized = alias.strip()
    class_ref = RUBRIC_CLASS_REFS.get(normalized)
    if class_ref is None:
        choices = ", ".join(sorted(RUBRIC_CLASS_REFS))
        raise RubricClassResolutionError(
            f"unknown rubric_class {alias!r}; available aliases: {choices}",
        )
    try:
        cls = load_plugin_class(class_ref, what="rubric_class")
    except PluginClassLoadError as exc:
        raise RubricClassResolutionError(str(exc)) from exc

    if getattr(cls, "uses_component_rubric", False):
        return RubricClassSpec(alias=normalized, class_ref=None)
    if not issubclass(cls, BaseRubric):
        raise RubricClassResolutionError(
            f"rubric_class {normalized!r} must subclass BaseRubric",
        )
    return RubricClassSpec(alias=normalized, class_ref=class_ref)


def rubric_alias_for_ref(class_ref: str | None) -> str | None:
    """Reverse-lookup an alias from ``rubric_ref`` (for rerunning a legacy run); ``None`` → ``component_rubric``."""
    if not class_ref:
        return "component_rubric"
    for alias, ref in RUBRIC_CLASS_REFS.items():
        if ref == class_ref:
            return alias
    return None


__all__ = [
    "RUBRIC_CLASS_REFS",
    "ComponentRubric",
    "RubricClassResolutionError",
    "RubricClassSpec",
    "resolve_rubric_class",
    "rubric_alias_for_ref",
]
