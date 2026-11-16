"""AggregationCalculator — run-level score aggregation across criteria.

Consumes per-criterion :class:`EvaluatorResultEntry` objects (typically loaded
from ``benchmark_run_items`` via ORM in
:meth:`BenchmarkService.load_criterion_entries_from_pipeline`)
and computes the final benchmark score, per-criterion ``CriterionStats``, and
``AggregatedMetrics``.  The aggregated payload is persisted on
``benchmark_runs.metrics_json`` only (never writes back case rows).

**Semantic difference from ScoreFuser**:
- :class:`ScoreFuser` is *item-level*: fuses one item's scores across
  criteria; compares individual scores against ``pass_threshold``.
- :class:`AggregationCalculator` is *run-level*: aggregates all items'
  stats per criterion; compares ``CriterionStats.pass_rate`` against
  ``pass_threshold``.

The same ``AggregationConfig.method`` field drives both, but the semantics
differ by level — ``pass_threshold`` is a *score* threshold at item level
and a *pass-rate* threshold at run level.
"""

import importlib
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.reporting.framework.schema import (
    AggregatedMetrics,
    AggregationConfig,
    AggregationMethod,
    AttributionStats,
    Criterion,
    CriterionStats,
)

logger = get_logger("agent_eval.benchmark.aggregation")


# ════════════════════════════════════════════════════════════════════════════
# EvaluatorResultEntry — a single criterion's result read from PG
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EvaluatorResultEntry:
    """A single criterion's aggregate stats (from SQL or in-memory progress).

    Kept as a dataclass (not ORM) so AggregationCalculator stays testable
    without a live database.  Job-level payload is persisted only as part of
    ``benchmark_runs.metrics_json``; case rows stay in ``benchmark_run_items``.
    """

    criterion_id: str
    evaluator_ref: str
    sample_count: int
    success_count: int
    failed_count: int
    criterion_stats_json: dict[str, Any] | None = None
    attribution_stats_json: dict[str, Any] | None = None
    react_rounds_json: dict[str, Any] | None = None


# ════════════════════════════════════════════════════════════════════════════
# AggregationCalculator
# ════════════════════════════════════════════════════════════════════════════


class AggregationCalculator:
    """Run-level aggregation — combines per-criterion stats into final score.

    Methods (driven by ``AggregationConfig.method``):
        - ``weighted_avg``: ``sum(criterion.mean * weight) / sum(weights)``
        - ``all_pass``: all criteria ``pass_rate ≥ threshold`` → 1.0, else 0.0
        - ``majority_vote``: pass count > ``len(rubric) / 2`` → 1.0, else 0.0
        - ``custom``: importlib-resolved function from ``custom_fn_ref``
    """

    async def aggregate(
        self,
        agg_config: AggregationConfig,
        rubric: list[Criterion],
        evaluator_results: list[EvaluatorResultEntry],
    ) -> tuple[float, dict[str, CriterionStats], AggregatedMetrics]:
        """Aggregate per-criterion results into final score + metrics.

        Args:
            agg_config: Aggregation configuration (method, pass_threshold, ...).
            rubric: Ordered criteria list (provides weights and IDs).
            evaluator_results: Per-criterion entries (in-memory from runner).

        Returns:
            ``(final_score, per_criterion_stats, aggregated_metrics)``
        """
        # 1) Build per-criterion CriterionStats + attribution + react rounds
        per_criterion: dict[str, CriterionStats] = {}
        attribution_metrics: dict[str, AttributionStats] = {}
        react_rounds_summary: dict[int, int] = {}

        for entry in evaluator_results:
            stats_json = entry.criterion_stats_json or {}
            cs = CriterionStats(
                criterion_id=entry.criterion_id,
                evaluator_ref=entry.evaluator_ref,
                sample_count=entry.sample_count,
                success_count=entry.success_count,
                failed_count=entry.failed_count,
                mean=stats_json.get("mean", 0.0),
                median=stats_json.get("median", 0.0),
                p95=stats_json.get("p95", 0.0),
                min=stats_json.get("min", 0.0),
                max=stats_json.get("max", 0.0),
                std=stats_json.get("std", 0.0),
                pass_rate=stats_json.get("pass_rate", 0.0),
            )
            per_criterion[entry.criterion_id] = cs

            # Attribution stats (may be None if triage hasn't run yet)
            attr_json = entry.attribution_stats_json
            if attr_json:
                attribution_metrics[entry.criterion_id] = AttributionStats(
                    field_distributions=attr_json.get(
                        "field_distributions",
                        {},
                    ),
                    value_labels=attr_json.get("value_labels", {}),
                    total_attributed=attr_json.get("total_attributed", 0),
                    # legacy keys (historical metrics_json rows)
                    triage_level_distribution=attr_json.get(
                        "triage_level_distribution",
                        {},
                    ),
                    root_cause_distribution=attr_json.get(
                        "root_cause_distribution",
                        {},
                    ),
                )

            # ReAct rounds distribution
            react_json = entry.react_rounds_json
            if react_json:
                dist = react_json.get("distribution", {})
                for k, v in dist.items():
                    try:
                        # Handle "5+" style keys
                        rounds = int(k.rstrip("+"))
                        react_rounds_summary[rounds] = react_rounds_summary.get(rounds, 0) + v
                    except (ValueError, TypeError):
                        logger.debug(
                            "aggregation.skip_react_key key=%r value=%r",
                            k,
                            v,
                        )

        # 2) Calculate final score based on method
        final_score = self._compute_final_score(
            agg_config,
            rubric,
            per_criterion,
        )

        # 3) Build aggregated metrics
        total_samples = max(
            (e.sample_count for e in evaluator_results),
            default=0,
        )

        metrics = AggregatedMetrics(
            criterion_metrics=per_criterion,
            attribution_metrics=attribution_metrics,
            react_rounds_summary=react_rounds_summary,
            aggregation_method=agg_config.method,
            total_samples=total_samples,
            total_duration_ms=0,
        )

        return final_score, per_criterion, metrics

    @staticmethod
    def _compute_final_score(
        agg_config: AggregationConfig,
        rubric: list[Criterion],
        per_criterion: dict[str, CriterionStats],
    ) -> float:
        """Compute the run-level final score from per-criterion stats."""

        method = agg_config.method

        if method == AggregationMethod.WEIGHTED_AVG.value:
            total_weight = sum(c.weight for c in rubric)
            if total_weight <= 0:
                return 0.0
            weighted_sum = 0.0
            for c in rubric:
                stats = per_criterion.get(c.id)
                mean = stats.mean if stats is not None else 0.0
                weighted_sum += mean * c.weight
            return weighted_sum / total_weight

        if method == AggregationMethod.ALL_PASS.value:
            return (
                1.0
                if all(
                    (stats := per_criterion.get(c.id)) is not None
                    and stats.pass_rate >= agg_config.pass_threshold
                    for c in rubric
                )
                else 0.0
            )

        if method == AggregationMethod.MAJORITY_VOTE.value:
            passed = sum(
                1
                for c in rubric
                if (stats := per_criterion.get(c.id)) is not None
                and stats.pass_rate >= agg_config.pass_threshold
            )
            return 1.0 if passed > len(rubric) / 2 else 0.0

        if method == AggregationMethod.CUSTOM.value:
            if not agg_config.custom_fn_ref:
                raise ValueError(
                    "custom_fn_ref is required when method='custom'",
                )
            module_path, func_name = agg_config.custom_fn_ref.rsplit(":", 1)
            module = importlib.import_module(module_path)
            fn = getattr(module, func_name)
            return fn(rubric, per_criterion, agg_config)

        raise ValueError(f"Unknown aggregation method: {method!r}")


__all__ = [
    "EvaluatorResultEntry",
    "AggregationCalculator",
]
