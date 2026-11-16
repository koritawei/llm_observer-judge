"""BaseRubric — configurable Rubric protocol base class (design doc §6.1).

A business provides rubric config as a **subclass**. Interface / schedule config passes
only a class-name reference (``module:Class``), not a large JSON blob — "config sinks into
code, and code carries the business".

Split of responsibility with ``@benchmark_component`` metadata (§6.2; they coexist, neither replaces the other):

- **Rubric class**: what to evaluate (``criteria``) / how to score (``fusion_fn_ref`` /
  ``aggregation_method`` / ``pass_threshold``) / how to parse data
  (``parse_interceptor_ref``) / which dimensions to stat (``stat_dimensions`` —
  the multi-dimension sub_scores stat declaration of a holistic judge, decoupled from evaluation criteria);
- **component metadata**: which service executes it (evaluator binding / service_name /
  registration-name resolution; the F9 mechanism is unchanged).
"""

from typing import TYPE_CHECKING

from benchmark.reporting.framework.schema import AggregationConfig, Criterion
from benchmark.plugins.class_loader import (
    PluginClassLoadError,
    load_plugin_class,
)

if TYPE_CHECKING:
    from benchmark.runs.common.dimension_stats import DimensionStatSpec

#: benchmark_name → default rubric_ref (§9 migration anchor).
#: Guarantee that v1 / the default entry keeps its behavior: once existing business semantics (rubric + parse
#: interceptor)
#: live on the Rubric class, the default entry mounts them through this map and matches production bit for bit;
#: a new business with no default mapping → default = meta.rubric + passthrough (§7.4).
_DEFAULT_RUBRIC_REFS: dict[str, str] = {
    "example_benchmark": "benchmark.runs.example.rubric:ExampleRubric",
}


def default_rubric_ref(benchmark_name: str) -> str | None:
    """benchmark registration name → default rubric_ref (§9 migration anchor; no mapping returns None)."""
    return _DEFAULT_RUBRIC_REFS.get(benchmark_name)


class BaseRubric:
    """Configurable Rubric protocol (a business provides a subclass; the interface passes only the class name).

    The subclass must implement :meth:`criteria`. Class attributes carry aggregation defaults and
    the parse-interceptor mount (a profile may override with an explicit ``parse_interceptor_ref``).
    """

    #: Item-level gate/cap fusion function ref (``module:function``,
    #: aligned with ``AggregationConfig.fusion_fn_ref``)
    fusion_fn_ref: str | None = None
    #: Run-level aggregation method (``AggregationConfig.method``)
    aggregation_method: str = "weighted_avg"
    #: Pass threshold (``AggregationConfig.pass_threshold``)
    pass_threshold: float = 1.0
    #: Data parse-interceptor ref (``module:Class``, §7.5)
    parse_interceptor_ref: str | None = None
    #: Scoring-rules document link (the report "Scoring rules" entry) — snapshotted via ResolvedProfile
    #: into run config_json, and resolved from config when the report slot is filled; ``None`` =
    #: this business has no rules document (the report renders plain-text "Scoring rules" with no link).
    rule_link_url: str | None = None
    #: Stat-dimension declaration (sub_scores dimensions of a holistic judge; the aggregation layer expands them;
    #: empty = no dimension expansion, legacy behavior). Evaluation criteria and stat dimensions are decoupled —
    #: dimensions do not enter criteria, so the worker does not call the evaluator once per criterion.
    stat_dimensions: "tuple[DimensionStatSpec, ...]" = ()
    #: Judge execution-policy guard (Resolver fail-fast): ``"local"`` means only
    #: local sync evaluation is allowed; explicit async, or async derived by default, is rejected with 400. For
    #: businesses with a dedicated local evaluator whose dimensions do not interoperate with the generic Aone pipeline
    #: (for example search_eval's holistic 0/1/2 single dimension + six-axis sub_scores; submitting them to the
    #: generic e2e pipeline is rejected by its ALL_DIMENSIONS allowlist and every case fails,
    #: the 20260923 production incident); ``None`` = no policy (production default; async is allowed).
    judge_execution_policy: str | None = None

    @classmethod
    def criteria(cls) -> list[Criterion]:
        """Return the scoring-dimension list (the subclass must implement this)."""
        raise NotImplementedError

    @classmethod
    def default_aggregation(cls) -> AggregationConfig:
        """Aggregation config carried by the Rubric class (a profile may override with an explicit ``aggregation``)."""
        return AggregationConfig(
            method=cls.aggregation_method,
            pass_threshold=cls.pass_threshold,
            fusion_fn_ref=cls.fusion_fn_ref,
        )


def load_rubric_class_ref(
    class_ref: str,
) -> tuple[list[Criterion], AggregationConfig, type[BaseRubric]]:
    """Load a Rubric class by ``module:Class`` and read its criteria / aggregation.

    Shared by dynamic benchmark component registration and the external agent catalog.
    An illegal reference raises ``ValueError`` (the caller fail-fasts, or tolerates it per scenario).
    """
    try:
        cls = load_plugin_class(class_ref, what="rubric_class")
    except PluginClassLoadError as exc:
        raise ValueError(str(exc)) from exc
    if not issubclass(cls, BaseRubric):
        raise ValueError(f"rubric_class {class_ref!r} must subclass BaseRubric")
    return cls.criteria(), cls.default_aggregation(), cls


__all__ = [
    "BaseRubric",
    "default_rubric_ref",
    "load_rubric_class_ref",
]
