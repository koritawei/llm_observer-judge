"""ScoreFuser — per-item score fusion across criteria.

Fuses multiple criterion scores for a single dataset item into a single
score (0.0~1.0).  Used by :class:`BenchmarkRunner` during the fused triage
phase (Phase 2) after all per-criterion evaluations complete.

**Failed evaluations count as 0.0** — an :class:`EvalFailure` entry keeps
its weight in the denominator so the same benchmark's ``final_score`` stays
comparable across runs (denominator is constant), and evaluation failures
produce a visible penalty rather than being silently ignored.

Custom item-level fusion (``AggregationConfig.fusion_fn_ref``) is a plugin:
RemoteConfig ``eval.plugins.fusion_functions.bindings`` keyed by the ref may redirect
it to ``remote`` (HTTP) or another local ``class_ref``; otherwise the ref is
imported directly.  Async callers use :meth:`ScoreFuser.afuse` so a remote
fusion never blocks the event loop.
"""

import importlib
from collections.abc import Callable
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.evaluators.base import EvaluatorResult
from benchmark.reporting.framework.schema import (
    AggregationConfig,
    AggregationMethod,
    Criterion,
)
from benchmark.plugins.host import PluginResolution, call_plugin_callable, resolve_plugin
from benchmark.plugins.remote_adapters import RemoteFusionFunction
from benchmark.plugins.remote_client import RemotePluginError

logger = get_logger("agent_eval.benchmark.fusion")

#: ``fusion_fn_ref`` signature: ``(scores, rubric, aggregation, score_of) -> float``
FusionFunction = Callable[..., float]


# ════════════════════════════════════════════════════════════════════════════
# EvalFailure — marker for a failed per-item evaluation
# ════════════════════════════════════════════════════════════════════════════


class EvalFailure:
    """Marker object for a failed evaluation (distinct from ``EvaluatorResult``).

    Carries the exception that caused the failure.  In fusion and aggregation
    the score is treated as ``0.0``, and the weight / count stays in the
    denominator so that failures produce a visible penalty.

    Instances are created by :meth:`BenchmarkRunner._run_single_evaluator`
    when a per-item evaluation raises or times out.
    """

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __repr__(self) -> str:
        return f"EvalFailure({type(self.error).__name__}: {self.error})"


def is_failure(result: object) -> bool:
    """Return ``True`` if *result* is an :class:`EvalFailure`."""

    return isinstance(result, EvalFailure)


# ════════════════════════════════════════════════════════════════════════════
# ScoreFuser — item-level fusion
# ════════════════════════════════════════════════════════════════════════════


class ScoreFuser:
    """Fuse per-item scores across criteria into a single score (0.0~1.0).

    The fusion strategy is determined by ``AggregationConfig.method`` —
    the **same** config field used by :class:`AggregationCalculator`, but at
    a different semantic level:

    - **ScoreFuser** (this class): *item-level* — fuses one item's scores
      across criteria; compares individual scores against ``pass_threshold``.
    - **AggregationCalculator**: *run-level* — aggregates all items' stats
      per criterion; compares ``CriterionStats.pass_rate`` against
      ``pass_threshold``.

    Methods:
        - ``weighted_avg``: ``sum(score * weight) / sum(weights)``
        - ``all_pass``: all criteria score ≥ threshold → 1.0, else 0.0
        - ``majority_vote``: pass count > ``len(rubric) / 2`` → 1.0, else 0.0
    """

    @staticmethod
    def fuse(
        scores: dict[str, EvaluatorResult | EvalFailure],
        rubric: list[Criterion],
        aggregation: AggregationConfig,
    ) -> float:
        """Return the fused score for a single item (sync; remote fusion blocks).

        Args:
            scores: ``{criterion_id: EvaluatorResult | EvalFailure}`` —
                one entry per criterion for the same dataset item.
            rubric: Ordered criteria list (provides weights and IDs).
            aggregation: Aggregation config (provides method + pass_threshold).

        Returns:
            Fused score in [0.0, 1.0].
        """
        score_of = _score_getter(scores)
        custom = _resolve_custom_fusion(aggregation)
        if custom is not None:
            try:
                return custom(scores, rubric, aggregation, score_of)
            except RemotePluginError:
                if not getattr(custom, "fallback_local", False):
                    raise
        return _fuse_by_method(score_of, rubric, aggregation)

    @staticmethod
    async def afuse(
        scores: dict[str, EvaluatorResult | EvalFailure],
        rubric: list[Criterion],
        aggregation: AggregationConfig,
    ) -> float:
        """Async variant of :meth:`fuse`; a remote fusion function is awaited."""
        score_of = _score_getter(scores)
        custom = _resolve_custom_fusion(aggregation)
        if custom is not None:
            try:
                return await call_plugin_callable(custom, scores, rubric, aggregation, score_of)
            except RemotePluginError:
                if not getattr(custom, "fallback_local", False):
                    raise
        return _fuse_by_method(score_of, rubric, aggregation)


def _score_getter(
    scores: dict[str, EvaluatorResult | EvalFailure],
) -> Callable[[str], float]:
    def score_of(cid: str) -> float:
        r = scores.get(cid)
        if r is None or isinstance(r, EvalFailure):
            return 0.0
        return r.score

    return score_of


def _import_fusion_fn(ref: str) -> FusionFunction | None:
    """Import ``module:func``; resolution failures log and return ``None``.

    Exceptions raised *inside* the custom fn itself are NOT swallowed by the
    caller — they propagate so genuine bugs stay visible.
    """
    try:
        module_path, func_name = ref.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    except (ImportError, AttributeError, ValueError):
        logger.warning(
            "fusion_fn_ref resolve failed, falling back to method: ref=%s",
            ref,
            exc_info=True,
        )
        return None


def _resolve_custom_fusion(aggregation: AggregationConfig) -> FusionFunction | None:
    """``fusion_fn_ref`` → RemoteConfig binding (remote / class_ref) → direct import."""
    ref = aggregation.fusion_fn_ref
    if not ref:
        return None
    plugin: Any = resolve_plugin(
        "fusion_functions",
        ref,
        resolution=PluginResolution(
            registry={},
            default=None,
            remote_factory=RemoteFusionFunction,
        ),
    )
    if plugin is not None:
        return plugin
    return _import_fusion_fn(ref)


def _fuse_by_method(
    score_of: Callable[[str], float],
    rubric: list[Criterion],
    aggregation: AggregationConfig,
) -> float:
    method = aggregation.method

    if method == AggregationMethod.WEIGHTED_AVG.value:
        total_weight = sum(c.weight for c in rubric)
        if total_weight <= 0:
            return 0.0
        weighted_sum = sum(score_of(c.id) * c.weight for c in rubric)
        return weighted_sum / total_weight

    if method == AggregationMethod.ALL_PASS.value:
        return 1.0 if all(score_of(c.id) >= aggregation.pass_threshold for c in rubric) else 0.0

    if method == AggregationMethod.MAJORITY_VOTE.value:
        passed = sum(1 for c in rubric if score_of(c.id) >= aggregation.pass_threshold)
        return 1.0 if passed > len(rubric) / 2 else 0.0

    if method == AggregationMethod.CUSTOM.value:
        raise ValueError(
            "method='custom' governs run-level aggregation only; "
            "item-level fusion requires fusion_fn_ref to be set"
        )

    raise ValueError(f"Unknown fusion method: {method!r}")


__all__ = [
    "EvalFailure",
    "FusionFunction",
    "is_failure",
    "ScoreFuser",
]
