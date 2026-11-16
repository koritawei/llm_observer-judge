"""L1 effectiveness statistics — pass rate / score distribution / D0-D6 failure rate / checker hits / coverage.

All pure code arithmetic, with no LLM. The score scale matches existing reports: 2 = pass,
1 = partially correct, 0 = fail (see the evaluation-side D0-D6 2-point total scale).
"""

from collections import Counter
from collections.abc import Iterable
from typing import Any

from benchmark.reporting.cluster.constants import DIMENSIONS, FAILURE_STATUSES, PARTIAL_SCORE, PASS_SCORE
from benchmark.reporting.cluster.schema import BatchCaseRecord


def build_effectiveness_stats(
    records: list[BatchCaseRecord],
    *,
    expected_total: int = 0,
) -> dict[str, Any]:
    """Effectiveness-block statistics.

    ``expected_total`` is the successful dispatch count (passed in by the trigger) and is
    used for ``eval_coverage``, divided into the actually recalled judged count. When it is
    0, fall back to the recalled count (avoid division by zero and stay self-consistent).
    """

    effective_scored = [r for r in records if r.effective_score is not None]
    total = len(records)
    low_score = [r for r in effective_scored if r.effective_score < PASS_SCORE]

    judge_summary = _score_summary([r.judge_score for r in records])
    attribution_summary = _score_summary([r.attribution_score for r in records])
    effective_summary = _score_summary([r.effective_score for r in records])
    dist = effective_summary["distribution"]
    return {
        "total": total,
        "scored": effective_summary["scored"],
        "expected_total": expected_total or total,
        "eval_coverage": _ratio(total, expected_total or total),
        "pass_rate": _ratio(dist["2"], effective_summary["scored"]),
        "partial_rate": _ratio(dist["1"], effective_summary["scored"]),
        "fail_rate": _ratio(dist["0"], effective_summary["scored"]),
        "avg_score": effective_summary["avg_score"],
        "score_distribution": dist,
        "score_views": {
            "judge_score": judge_summary,
            "attribution_score": attribution_summary,
            "effective_score": effective_summary,
        },
        "dimension_fail_rate": _dimension_fail_rate(records),
# Numerator: pure checker (deterministic rule) hits, excluding the hard_rule_hits subset
# from the LLM sub judge. Denominator: low-score case count (score < 2). The hit rate
# answers "how many low scores are deterministic failures explained by hard rules",
# not "how many of all cases touched a hard rule".
        "hard_rule_hit_rate": _ratio(
            sum(1 for r in low_score if _checker_hard_rule_hits(r)),
            len(low_score),
        ),
        "hard_rule_hits": _count_flat(r.hard_rule_hits for r in records),
        "checker_fail_hits": _checker_fail_hits(records),
        "risk_flags": _count_flat(r.risk_flags for r in records),
        "failure_modes": _failure_mode_counts(records),
        "triage_levels": _count_values(r.triage_level for r in records),
        "suites": _count_values(r.suite for r in records),
    }


def _score_summary(values: list[float | None]) -> dict[str, Any]:
    scored = [value for value in values if value is not None]
    return {
        "scored": len(scored),
        "avg_score": _avg(scored),
        "distribution": _score_distribution(scored),
    }


def _score_distribution(scores: list[float]) -> dict[str, int]:
    """Counts for the 0/1/2 tiers. Non-integer scores (such as 1.5, the A/B two-round average) floor into their interval."""

    dist = {"0": 0, "1": 0, "2": 0}
    for score in scores:
        if score >= PASS_SCORE:
            dist["2"] += 1
        elif score >= PARTIAL_SCORE:
            dist["1"] += 1
        else:
            dist["0"] += 1
    return dist


def _dimension_fail_rate(records: list[BatchCaseRecord]) -> dict[str, dict[str, Any]]:
    """Per-dimension failure rate for D0-D6.

    The denominator is the number of cases judged on that dimension, not all cases. A
    dimension may be not_applicable (does not apply to a conditional scene); counting those
    in the denominator would understate the failure rate.
    """

    judged: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    for record in records:
        for item in record.dimension_results:
            if not isinstance(item, dict):
                continue
            dimension = str(item.get("dimension") or "")
            if dimension not in DIMENSIONS:
                continue
            status = str(item.get("status") or "")
            if status in ("not_applicable", ""):
                continue
            judged[dimension] += 1
            if status in FAILURE_STATUSES:
                failed[dimension] += 1

    return {
        dimension: {
            "judged": judged.get(dimension, 0),
            "failed": failed.get(dimension, 0),
            "fail_rate": _ratio(failed.get(dimension, 0), judged.get(dimension, 0)),
        }
        for dimension in DIMENSIONS
        if judged.get(dimension, 0) or failed.get(dimension, 0)
    }


def _checker_hard_rule_hits(record: BatchCaseRecord) -> list[str]:
    """Pure checker (deterministic rule) hits, taken from each checker signal's own hard_rule_hits.

    This is a different source from ``record.hard_rule_hits`` (checker + LLM sub judge merged
    and deduped; see aggregate_judge_result._dedupe). The latter mixes in hits marked by the
    sub judge and is not a "hard rule" hit in meaning.
    """

    hits: list[str] = []
    for item in record.checker_results:
        if not isinstance(item, dict):
            continue
        hits.extend(str(hit) for hit in (item.get("hard_rule_hits") or []) if hit)
    return list(dict.fromkeys(hits))


def _checker_fail_hits(records: list[BatchCaseRecord]) -> dict[str, int]:
    """Count failure hits by checker name (to locate high-frequency rule hits)."""

    counter: Counter[str] = Counter()
    for record in records:
        for item in record.checker_results:
            if not isinstance(item, dict):
                continue
            if str(item.get("status") or "") not in FAILURE_STATUSES:
                continue
            name = str(item.get("source_name") or "")
            if name:
                counter[name] += 1
    return dict(counter.most_common())


def _failure_mode_counts(records: list[BatchCaseRecord]) -> dict[str, int]:
    """failure_mode counts (a multi-label array; one case may hit several items)."""

    counter: Counter[str] = Counter()
    for record in records:
        for entry in record.failure_modes:
            if not isinstance(entry, dict):
                continue
            mode = str(entry.get("mode") or "")
            if mode:
                counter[mode] += 1
    return dict(counter.most_common())


def _count_flat(iterables: Iterable[Iterable[str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for items in iterables:
        for item in items or []:
            if item:
                counter[str(item)] += 1
    return dict(counter.most_common())


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        if value:
            counter[str(value)] += 1
    return dict(counter.most_common())


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


__all__ = ["build_effectiveness_stats"]
