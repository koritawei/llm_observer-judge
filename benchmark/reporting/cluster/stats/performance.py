"""L1 performance statistics — three-bucket tokens/latency plus mean and percentiles for end-to-end and first-text.

Percentile scale (AC-10):
- Latency (per-call latency of the three buckets, end-to-end, first text): mean + P50/P95/P99
- tokens: mean + P95 (catch abnormally long context / runaway output)
- calls / cost: mean + total

The mean is flattened by the long tail (LLM call latency is right-skewed); percentiles are
what expose tail degradation. Extraction and bucketing are pinned by the T13 comparison
test to match ``generate-eval-report.py``. Percentiles are a pure incremental statistic.
"""

from collections import Counter
from typing import Any

from benchmark.reporting.cluster.constants import MODEL_BUCKETS
from benchmark.reporting.cluster.schema import BatchCaseRecord


def build_performance_stats(
    records: list[BatchCaseRecord],
    *,
    model_allowlist: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Performance-block statistics.

    Only cases whose performance collection succeeded (``perf`` is non-empty and
    ``perf.error`` is empty) are counted. Failures go into ``missing`` with a reason and
    do not enter the mean/percentile (so a 0 does not pull the numbers down).
    """

    collected = [r for r in records if r.perf is not None and not r.perf.error]
    failed = [r for r in records if r.perf is not None and r.perf.error]
    absent = [r for r in records if r.perf is None]

    bucket_names = tuple(
        dict.fromkeys(
            (
                *MODEL_BUCKETS,
                *(bucket for record in collected if record.perf for bucket in record.perf.buckets),
            )
        )
    )
    model_names = {
        model
        for record in collected
        if record.perf
        for model in record.perf.models
        if model_allowlist is None or model in model_allowlist
    }
    return {
        "collected": len(collected),
        "missing": len(failed) + len(absent),
        "missing_reasons": _missing_reasons(failed, len(absent)),
        "buckets": {bucket: _bucket_stats(collected, bucket) for bucket in bucket_names},
        "models": {model: _model_stats(collected, model) for model in sorted(model_names)},
        "e2e_ms": _latency_stats(
            [r.perf.e2e_ms for r in collected if r.perf and r.perf.e2e_ms is not None]
        ),
        "first_text_ms": _latency_stats(
            [r.perf.first_text_ms for r in collected if r.perf and r.perf.first_text_ms is not None]
        ),
        "total_cost": round(
            sum(sum(b.cost for b in r.perf.buckets.values()) for r in collected if r.perf),
            6,
        ),
        "total_tokens": sum(
            sum(b.total_tokens for b in r.perf.buckets.values()) for r in collected if r.perf
        ),
        "total_tokens_per_case": _token_stats(
            [
                float(sum(bucket.total_tokens for bucket in r.perf.buckets.values()))
                for r in collected
                if r.perf
            ]
        ),
        "route_modes": _route_mode_stats(records),
        "aone_defects": _notification_defect_stats(records),
    }


def _route_mode_stats(records: list[BatchCaseRecord]) -> dict[str, Any]:
    """Aggregate by the VUI route tag already persisted in Langfuse; do not infer a winner from a missing tag."""

    expected_modes = ("lite_sdm", "lite_dag", "standard")
    counts = {mode: 0 for mode in expected_modes}
    unclassified = 0
    for record in records:
        route_mode = record.perf.route_mode if record.perf else ""
        if route_mode in counts:
            counts[route_mode] += 1
        else:
            unclassified += 1
    classified = sum(counts.values())
    return {
        "classified": classified,
        "unclassified": unclassified,
        "modes": {
            mode: {
                "count": count,
                "share": round(count / classified, 4) if classified else 0.0,
            }
            for mode, count in counts.items()
        },
    }


def _notification_defect_stats(records: list[BatchCaseRecord]) -> dict[str, int]:
    """Deduplicate by dispatched ticket id; do not call Aone and do not expose ticket content."""

    workitem_ids = {record.notification_workitem_id for record in records if record.notification_workitem_id}
    related_cases = sum(1 for record in records if record.notification_workitem_id)
    return {"unique_workitems": len(workitem_ids), "related_cases": related_cases}


def _bucket_stats(records: list[BatchCaseRecord], bucket: str) -> dict[str, Any]:
    """Single-bucket statistics. The denominator is the number of cases that called this
    bucket. When a bucket was not triggered (for example verify runs only on some paths),
    cases that did not use it must not enter the denominator and pull the mean down.
    """

    entries = [r.perf.buckets[bucket] for r in records if r.perf and bucket in r.perf.buckets]
    if not entries:
        return {"cases": 0}

    calls = [e.calls for e in entries]
    models: Counter[str] = Counter()
    for entry in entries:
        models.update(entry.models)
    return {
        "cases": len(entries),
        "models": dict(models.most_common()),
        "calls": {
            "avg": _avg([float(e.calls) for e in entries]),
            "total": sum(calls),
        },
        "prompt_tokens": _token_stats([float(e.prompt_tokens) for e in entries], calls=calls),
        "completion_tokens": _token_stats(
            [float(e.completion_tokens) for e in entries], calls=calls
        ),
        "total_tokens": _token_stats([float(e.total_tokens) for e in entries], calls=calls),
        # Mean per-call latency: accumulated latency in the bucket / call count in the bucket (aligned with script-side avg_latency_ms)
        "latency_ms": _latency_stats([e.latency_ms / e.calls for e in entries if e.calls]),
        "cost": {
            "avg": _avg([e.cost for e in entries]),
            "total": round(sum(e.cost for e in entries), 6),
        },
    }


def _model_stats(records: list[BatchCaseRecord], model: str) -> dict[str, Any]:
    """Cross-case performance statistics for one actual Langfuse model."""

    entries = [
        record.perf.models[model]
        for record in records
        if record.perf and model in record.perf.models
    ]
    if not entries:
        return {"cases": 0}
    calls = [entry.calls for entry in entries]
    return {
        "cases": len(entries),
        "calls": {"avg": _avg([float(entry.calls) for entry in entries]), "total": sum(calls)},
        "total_tokens": _token_stats([float(entry.total_tokens) for entry in entries], calls=calls),
        "latency_ms": _latency_stats(
            [entry.latency_ms / entry.calls for entry in entries if entry.calls]
        ),
    }


def _latency_stats(values: list[float]) -> dict[str, Any]:
    """Latency: mean + P50/P95/P99 (sensitive to the long tail)."""

    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "avg": _avg(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": round(max(values), 2),
    }


def _token_stats(values: list[float], *, calls: list[int] | None = None) -> dict[str, Any]:
    """Tokens: mean + P95 (P99 is of limited use for tokens).

    ``avg`` is per case: the mean of accumulated tokens of one case in the bucket (which
    may come from several calls), used to compare overall consumption across cases.
    ``avg_per_call`` = Σtokens / Σcalls is the mean of a single call. calls.total is used
    for division only here, so the total of a multi-call case does not inflate the answer
    to "how expensive is one call" (aligned with the existing per-call conversion of latency_ms).
    """

    if not values:
        return {"count": 0}
    stats: dict[str, Any] = {
        "count": len(values),
        "avg": _avg(values),
        "p95": _percentile(values, 95),
        "max": round(max(values), 2),
    }
    if calls is not None:
        total_calls = sum(calls)
        stats["avg_per_call"] = round(sum(values) / total_calls, 2) if total_calls else 0.0
    return stats


def _percentile(values: list[float], percent: float) -> float:
    """Linear-interpolation percentile (same scale as numpy.percentile's default linear, without adding the dependency)."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _missing_reasons(failed: list[BatchCaseRecord], absent_count: int) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for record in failed:
        if record.perf is None:
            continue
        reasons[record.perf.error] = reasons.get(record.perf.error, 0) + 1
    if absent_count:
        reasons["not_collected"] = absent_count
    return reasons


__all__ = ["build_performance_stats"]
