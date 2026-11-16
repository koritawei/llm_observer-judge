"""Rubric evaluation framework schema — rubric, aggregation, dataset references, and stats models.

This module is the shared data contract of ``benchmark.reporting.framework``: decoration-time
configuration and aggregate-stats models for the declarative rubric evaluation framework
(shared by benchmark batch and realtime streams).

- Frozen dataclasses (``Criterion`` / ``AggregationConfig`` / ``DatasetRef``)
  are decoration-time config, declared statically at class definition.
- Pydantic models (``CriterionStats`` / ``AttributionStats`` /
  ``AggregatedMetrics``) are aggregate-stats results, produced by
  :class:`~benchmark.reporting.framework.aggregation_calculator.AggregationCalculator`.

Domain-specific request/response payloads are defined in each domain schema (for example
``benchmark.runs.schema``).
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field

logger = get_logger("agent_eval.eval_framework.schema")

# ════════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════════


class AggregationMethod(StrEnum):
    """How final score is calculated from per-criterion statistics."""

    WEIGHTED_AVG = "weighted_avg"
    ALL_PASS = "all_pass"
    MAJORITY_VOTE = "majority_vote"
    CUSTOM = "custom"


# ════════════════════════════════════════════════════════════════════════════
# Frozen dataclasses — decoration-time configuration
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Criterion:
    """A single scoring dimension in a benchmark rubric.

    Each criterion binds an evaluator implementation (via ``evaluator_ref``)
    and carries a weight for aggregation. The same evaluator can be reused by
    different criteria across benchmarks — just point ``evaluator_ref`` to the
    same ID.

    When ``description`` is omitted it is filled with ``name`` (reports/snapshots consume it
    read-only, so the domain need not write a long description).
    """

    id: str
    name: str
    weight: float
    evaluator_ref: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Criterion.id must not be empty")
        if not self.evaluator_ref:
            raise ValueError("Criterion.evaluator_ref must not be empty")
        if self.weight < 0:
            raise ValueError(f"Criterion.weight must be >= 0, got {self.weight}")
        # description defaults to name (filled via object.__setattr__ on the frozen instance)
        if not self.description:
            object.__setattr__(self, "description", self.name)


@dataclass(frozen=True)
class AggregationConfig:
    """How per-criterion scores are combined into a final score.

    - ``weighted_avg``: sum(score * weight) / sum(weights)
    - ``all_pass``: all criteria must meet ``pass_threshold`` → 1.0 else 0.0
    - ``majority_vote``: > half of criteria meet threshold → 1.0 else 0.0
    - ``custom``: delegate to ``custom_fn_ref`` (dotted path, importlib-resolved)

    ``fusion_fn_ref`` (optional) overrides item-level fusion in
    :class:`ScoreFuser` — e.g. to implement gate rules that cap or zero
    the fused score based on specific criterion values.  When set, it
    takes precedence over ``method`` at the item level; ``method`` still
    governs run-level aggregation via :class:`AggregationCalculator`.
    """

    method: str
    pass_threshold: float = 0.8
    custom_fn_ref: str | None = None
    fusion_fn_ref: str | None = None

    def __post_init__(self) -> None:
        if self.method not in {m.value for m in AggregationMethod}:
            raise ValueError(f"Unknown aggregation method: {self.method}")
        # Thresholds share the business score scale (not a normalized assumption): ieval D max is 2,
        # search_eval raw is a 0/1/2 scale with pass=2.0 — only non-negative is required, no upper bound.
        if self.pass_threshold < 0.0:
            raise ValueError(f"pass_threshold must be >= 0, got {self.pass_threshold}")
        if self.method == AggregationMethod.CUSTOM.value and not self.custom_fn_ref:
            raise ValueError("custom_fn_ref is required when method='custom'")
        # Fail fast on malformed refs at config-load time — otherwise a missing
        # ':' separator only surfaces as a ValueError deep inside the run
        # (fusion per item / final-score aggregation), wasting the whole run.
        for ref_name, ref in (
            ("custom_fn_ref", self.custom_fn_ref),
            ("fusion_fn_ref", self.fusion_fn_ref),
        ):
            if ref is not None and ":" not in ref:
                raise ValueError(f"{ref_name} must be in 'module:function' format, got {ref!r}")


@dataclass(frozen=True)
class DatasetRef:
    """Reference to a Langfuse dataset used as the benchmark data source.

    - ``dataset_name``: Langfuse dataset name (SDK ``get_dataset()`` key)
    - ``project_id``: Langfuse project ID (for credential resolution)
    - ``sample_size``: max items to evaluate (``None`` = all)
    - ``sample_seed``: random seed for reproducible sampling
    """

    dataset_name: str
    project_id: str
    sample_size: int | None = None
    sample_seed: int = 42


# ════════════════════════════════════════════════════════════════════════════
# Pydantic models — aggregation statistics
# ════════════════════════════════════════════════════════════════════════════


class CriterionStats(BaseModel):
    """Statistics for a single criterion across all evaluated items.

    Failed evaluations (EvalFailure) are counted with score=0.0, so ``mean``
    reflects the penalty. ``failed_count`` is tracked separately for visibility.
    """

    criterion_id: str = Field(description="Criterion ID from rubric")
    evaluator_ref: str = Field(description="Evaluator implementation ID")
    sample_count: int = Field(ge=0, description="Total items evaluated")
    success_count: int = Field(ge=0, description="Items successfully evaluated")
    failed_count: int = Field(ge=0, description="Items that failed evaluation")
    mean: float = Field(description="Mean score (failures counted as 0.0)")
    median: float = Field(description="Median score")
    p95: float = Field(description="95th percentile score")
    min: float = Field(description="Minimum score")
    max: float = Field(description="Maximum score")
    std: float = Field(description="Standard deviation of scores")
    pass_rate: float = Field(
        description="Fraction of items with score >= pass_threshold",
    )


class AttributionStats(BaseModel):
    """Attribution distribution across all items for a criterion or fused run.

    Generic structure: value distributions are aggregated by attribution field name
    (``field_distributions``). Which fields to aggregate is chosen by the caller
    (default ``primary_issue_category``; other domains may pass ``root_cause`` and so on).
    The field set is not fixed.
    """

    field_distributions: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description=(
            "Count by attribution field value — "
            "{field_name: {value: count}} (e.g. "
            "{primary_issue_category: {intent_error: 3, ...}})"
        ),
    )
    value_labels: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description=(
            "Display labels for field_distributions values — "
            "{field_name: {value: label}} (e.g. "
            "{primary_issue_category: {tool_timeout: 'tool/tool_timeout'}}). "
            "Statistics keys/counts unchanged; report renders the label "
            "instead of the raw value when present."
        ),
    )
    total_attributed: int = Field(
        default=0,
        ge=0,
        description="Total rows that went through attribution",
    )
    # Legacy fields (pre field_distributions era) — kept so historical
    # metrics_json rows still deserialize; new aggregation no longer fills them.
    triage_level_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="[legacy] Count by triage level (e.g. {severe_bad: 3, good: 7})",
    )
    root_cause_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="[legacy] Count by root_cause category",
    )


class AggregatedMetrics(BaseModel):
    """Full aggregated metrics for a completed benchmark run.

    Indexed by ``criterion_id`` — matching the rubric's ``Criterion.id`` field.
    The ``fused`` key (if present) holds the fused-triage attribution stats.
    """

    criterion_metrics: dict[str, CriterionStats] = Field(
        default_factory=dict,
        description="Per-criterion statistics, keyed by criterion_id",
    )
    attribution_metrics: dict[str, AttributionStats] = Field(
        default_factory=dict,
        description=(
            "Attribution stats keyed by criterion_id (per_eval) or "
            "FUSED_CRITERION_ID (fused mode). Never broadcast fused onto "
            "every criterion_id."
        ),
    )
    react_rounds_summary: dict[int, int] = Field(
        default_factory=dict,
        description="ReAct rounds distribution (e.g. {1: 5, 2: 3, 3: 2})",
    )
    aggregation_method: str = Field(description="Aggregation method used")
    total_samples: int = Field(ge=0, description="Total samples evaluated")
    total_duration_ms: int = Field(ge=0, description="Total run duration in ms")


def aggregated_metrics_from_json(
    metrics_json: dict[str, Any] | None,
) -> AggregatedMetrics | None:
    """Rebuild aggregate metrics from persisted ``metrics_json`` (the DB source of truth for report data).

    Report generation must trust only the DB: under multi-node deployment the in-memory metrics
    passed by the caller may cover only items handled by this node (or a healer recovery node
    may have no in-memory state at all). Every finalize path persists the full aggregate via
    ``update_run_status(GENERATING_REPORT, metrics_json=...)`` before generating the report,
    so reading it back here yields a complete, cross-node-consistent result.

    Returns:
        The rebuilt ``AggregatedMetrics``. Returns ``None`` when ``metrics_json`` is missing,
        does not contain criterion_metrics (for example an incremental merge fragment mid-run),
        or fails to deserialize — the caller then falls back to the in-memory metrics it passed in.
    """
    if not isinstance(metrics_json, dict):
        return None
    if not metrics_json.get("criterion_metrics"):
        return None
    try:
        return AggregatedMetrics.model_validate(metrics_json)
    except Exception:
        logger.warning(
            "eval_framework.metrics_json_rebuild_failed — fallback to in-memory metrics",
            exc_info=True,
        )
        return None


__all__ = [
    # Enums
    "AggregationMethod",
    # Frozen dataclasses
    "Criterion",
    "AggregationConfig",
    "DatasetRef",
    # Pydantic models
    "CriterionStats",
    "AttributionStats",
    "AggregatedMetrics",
    "aggregated_metrics_from_json",
]
