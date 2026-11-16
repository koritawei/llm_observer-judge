"""Aggregation of report "performance and efficiency" metrics (latency / TTFT / Token / ReAct rounds).

Data scale (every part may be missing — a missing part is not counted, and the report
does not render the matching card / chart):

- End-to-end latency: case ``output.elapsed_seconds`` (seconds; a CSV string can be parsed)
- First-packet latency: case ``output.ttft_ms`` (milliseconds)
- Token: ``total_input_tokens`` / ``total_output_tokens`` / ``total_tokens`` in the ``odps``
  section of the judge raw payload. The callback raw payload hangs on
  ``judge_json.raw_json`` (external trigger) or ``judge_json.raw_json[cid]``
  (inline multi-criterion); both shapes are normalized here.
- ReAct rounds: aggregation-layer ``react_rounds_summary`` (``{round: count}``).

The product of ``summarize_performance_metrics`` is written to
``ReportSpec.performance_metrics`` and ``benchmark_runs.metrics_json.performance_metrics``.
``performance_trend_point`` projects it into a daily trend point. Trends and day-over-day
comparisons are read only from DB ``metrics_json`` and do not rescan item details.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.runs.run_helpers import percentile_cont

logger = get_logger("agent_eval.eval_framework.performance_metrics")

#: Trend-series keys (concatenated with the metric_keys of ``load_daily_trend``).
PERF_TREND_KEYS: tuple[str, ...] = (
    "latency_p50_ms",
    "ttft_p50_ms",
    "token_avg",
    "react_avg_rounds",
)

#: Latency columns in case ``output`` → factor that converts them to milliseconds.
_LATENCY_COLUMNS: tuple[tuple[str, float], ...] = (
    ("elapsed_seconds", 1000.0),
    ("elapsed_ms", 1.0),
)

#: Token columns in the judge raw ``odps`` section → aggregation keys.
_TOKEN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("total_tokens", "total"),
    ("total_input_tokens", "input"),
    ("total_output_tokens", "output"),
)


def summarize_performance_metrics(
    case_results: Sequence[Mapping[str, Any]] | None,
    *,
    react_rounds_summary: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Case details + ReAct distribution → performance and efficiency metrics.

    Returns:
        A subset of ``{latency, ttft, tokens, react}``. Each section appears only when it
        has valid samples. When none do, returns an empty dict (the caller then skips the whole column).
    """
    latency_ms: list[float] = []
    ttft_ms: list[float] = []
    token_totals: dict[str, float] = {"total": 0.0, "input": 0.0, "output": 0.0}
    token_sample_count = 0

    for case in case_results or ():
        if not isinstance(case, Mapping):
            continue
        output = _as_mapping(case.get("output"))
        if output:
            latency = _first_float(output, _LATENCY_COLUMNS)
            if latency is not None:
                latency_ms.append(latency)
            ttft = _as_float(output.get("ttft_ms"))
            if ttft is not None:
                ttft_ms.append(ttft)
        odps = _odps_section(case)
        if odps and _accumulate_tokens(odps, token_totals):
            token_sample_count += 1

    metrics: dict[str, Any] = {}
    latency_stats = _duration_stats(latency_ms)
    if latency_stats:
        metrics["latency"] = latency_stats
    ttft_stats = _duration_stats(ttft_ms)
    if ttft_stats:
        metrics["ttft"] = ttft_stats
    if token_sample_count:
        metrics["tokens"] = _token_stats(token_totals, token_sample_count)
    react_stats = _react_stats(react_rounds_summary)
    if react_stats:
        metrics["react"] = react_stats
    return metrics


def performance_trend_point(
    performance_metrics: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Performance metrics → a daily trend point (a missing metric writes no key, and that point stays empty in the series)."""
    if not isinstance(performance_metrics, Mapping):
        return {}
    point: dict[str, float] = {}
    for trend_key, section, field in (
        ("latency_p50_ms", "latency", "p50_ms"),
        ("ttft_p50_ms", "ttft", "p50_ms"),
        ("token_avg", "tokens", "avg_total"),
        ("react_avg_rounds", "react", "avg_rounds"),
    ):
        value = _as_float(_as_mapping(performance_metrics.get(section)).get(field))
        if value is not None:
            point[trend_key] = value
    return point


def _duration_stats(values: list[float]) -> dict[str, Any]:
    """Latency samples → ``{avg_ms, p50_ms, p95_ms, p99_ms, sample_count}``."""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "avg_ms": round(sum(ordered) / len(ordered), 2),
        "p50_ms": round(percentile_cont(ordered, 0.5), 2),
        "p95_ms": round(percentile_cont(ordered, 0.95), 2),
        "p99_ms": round(percentile_cont(ordered, 0.99), 2),
        "sample_count": len(ordered),
    }


def _token_stats(totals: Mapping[str, float], sample_count: int) -> dict[str, Any]:
    """Accumulated tokens → total + mean per request."""
    total = totals.get("total") or (totals.get("input", 0.0) + totals.get("output", 0.0))
    return {
        "total_tokens": round(total, 2),
        "total_input_tokens": round(totals.get("input", 0.0), 2),
        "total_output_tokens": round(totals.get("output", 0.0), 2),
        "avg_total": round(total / sample_count, 2),
        "avg_input": round(totals.get("input", 0.0) / sample_count, 2),
        "avg_output": round(totals.get("output", 0.0) / sample_count, 2),
        "sample_count": sample_count,
    }


def _react_stats(summary: Mapping[Any, Any] | None) -> dict[str, Any]:
    """ReAct round distribution → mean rounds + a distribution snapshot (keys normalized to string rounds)."""
    if not isinstance(summary, Mapping) or not summary:
        return {}
    distribution: dict[str, int] = {}
    weighted = 0.0
    count = 0
    for raw_rounds, raw_count in summary.items():
        rounds = _as_float(raw_rounds)
        cases = _as_float(raw_count)
        if rounds is None or cases is None or cases <= 0:
            continue
        distribution[str(int(rounds))] = int(cases)
        weighted += rounds * cases
        count += int(cases)
    if not count:
        return {}
    return {
        "avg_rounds": round(weighted / count, 2),
        "distribution": distribution,
        "sample_count": count,
    }


def _accumulate_tokens(odps: Mapping[str, Any], totals: dict[str, float]) -> bool:
    """Add one ``odps`` row's token columns into totals; return False when no column is valid."""
    hit = False
    for column, bucket in _TOKEN_COLUMNS:
        value = _as_float(odps.get(column))
        if value is None:
            continue
        totals[bucket] += value
        hit = True
    return hit


def _odps_section(case: Mapping[str, Any]) -> Mapping[str, Any]:
    """Locate the ``odps`` section in the case's judge raw payload (the fused row wins)."""
    for judge in _iter_judge_payloads(case):
        odps = _as_mapping(judge.get("odps"))
        if odps:
            return odps
        raw_json = _as_mapping(judge.get("raw_json"))
        if not raw_json:
            continue
        odps = _as_mapping(raw_json.get("odps"))
        if odps:
            return odps
        # inline multi-criterion: raw_json is bucketed by criterion_id
        for nested in raw_json.values():
            odps = _as_mapping(_as_mapping(nested).get("odps"))
            if odps:
                return odps
    return {}


def _iter_judge_payloads(case: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Iterate the judge raw payload of each dimension row of a case (the fused overall row comes first)."""
    scores = case.get("scores")
    if not isinstance(scores, Mapping):
        return
    fused = _as_mapping(_as_mapping(scores.get("fused")).get("judge"))
    if fused:
        yield fused
    for criterion_id, row in scores.items():
        if criterion_id == "fused":
            continue
        judge = _as_mapping(_as_mapping(row).get("judge"))
        if judge:
            yield judge


def _first_float(
    source: Mapping[str, Any],
    columns: Sequence[tuple[str, float]],
) -> float | None:
    """Take the first parseable number by column priority and multiply by the unit-conversion factor."""
    for column, scale in columns:
        value = _as_float(source.get(column))
        if value is not None:
            return value * scale
    return None


def _as_mapping(value: object) -> Mapping[str, Any]:
    """dict / JSON string → Mapping; anything else (including None) → an empty dict."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(loaded, Mapping):
            return loaded
    return {}


def _as_float(value: object) -> float | None:
    """Number / numeric string → float; bool, empty string, and invalid values → None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


__all__ = [
    "PERF_TREND_KEYS",
    "performance_trend_point",
    "summarize_performance_metrics",
]
