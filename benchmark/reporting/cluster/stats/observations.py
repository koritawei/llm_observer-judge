"""L1 observation-slice statistics — Metric over-limit / Diagnostic hits / business/scene/Profile version.

The source is the directly readable fields of the shared core in the unified envelope
``agent_eval_extra.v1`` (``BatchCaseRecord.metric_observations`` / ``diagnostic_findings`` /
envelope identity slice keys). Historical non-envelope batches are entirely empty, the stats
block is omitted as a whole, and report output matches the baseline.

Boundary with the effectiveness block: Metric over-limit and Diagnostic hits are independent
observation slices only (query, trend, alert, attribution evidence). They do not produce
dimension conclusions and do not affect ``primary_score`` (design §5.3).
"""

from collections import Counter
from typing import Any

from benchmark.reporting.cluster.schema import BatchCaseRecord

#: Over-limit status value of MetricObservation.status (matches the envelope contract Literal).
_METRIC_EXCEEDED_STATUS = "exceeded"


def build_observation_stats(records: list[BatchCaseRecord]) -> dict[str, Any]:
    """Observation-slice stats; returns an empty dict when there are no envelope observations (the renderer skips the block)."""

    stats: dict[str, Any] = {}
    metrics = _metric_breakdown(records)
    if metrics:
        stats["metric_observations"] = metrics
    diagnostics = _diagnostic_breakdown(records)
    if diagnostics:
        stats["diagnostic_findings"] = diagnostics
    profiles = _profile_breakdown(records)
    if profiles:
        stats["profiles"] = profiles
    return stats


def _metric_breakdown(records: list[BatchCaseRecord]) -> dict[str, dict[str, Any]]:
    """Slice by ``metric_key``: observation count, over-limit count, and over-limit rate (compare the same key across businesses)."""

    observed: Counter[str] = Counter()
    exceeded: Counter[str] = Counter()
    for record in records:
        for item in record.metric_observations:
            if not isinstance(item, dict):
                continue
            metric_key = str(item.get("metric_key") or "")
            if not metric_key:
                continue
            observed[metric_key] += 1
            if str(item.get("status") or "") == _METRIC_EXCEEDED_STATUS:
                exceeded[metric_key] += 1

    return {
        metric_key: {
            "observed": observed[metric_key],
            "exceeded": exceeded.get(metric_key, 0),
            "exceeded_rate": _ratio(exceeded.get(metric_key, 0), observed[metric_key]),
        }
        for metric_key in sorted(observed)
    }


def _diagnostic_breakdown(records: list[BatchCaseRecord]) -> dict[str, Any]:
    """Slice diagnostic hits by ``finding_key`` / ``severity`` (one case may hit multiple items)."""

    by_finding: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    for record in records:
        for item in record.diagnostic_findings:
            if not isinstance(item, dict):
                continue
            finding_key = str(item.get("finding_key") or "")
            if finding_key:
                by_finding[finding_key] += 1
            severity = str(item.get("severity") or "")
            if severity:
                by_severity[severity] += 1

    if not by_finding and not by_severity:
        return {}
    return {
        "by_finding_key": dict(by_finding.most_common()),
        "by_severity": dict(by_severity.most_common()),
    }


def _profile_breakdown(records: list[BatchCaseRecord]) -> dict[str, Any]:
    """Slice case distribution by business / scene / Profile version (envelope identity keys; empty for historical runs)."""

    business_ids = _count(record.business_id for record in records)
    scenarios = _count(record.scenario for record in records)
    profiles = _count(
        _profile_label(record.profile_id, record.profile_version) for record in records
    )
    if not (business_ids or scenarios or profiles):
        return {}
    return {
        "business_ids": business_ids,
        "scenarios": scenarios,
        "profiles": profiles,
    }


def _profile_label(profile_id: str, profile_version: str) -> str:
    if not profile_id and not profile_version:
        return ""
    return f"{profile_id}@{profile_version}"


def _count(values: object) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        if value:
            counter[str(value)] += 1
    return dict(counter.most_common())


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


__all__ = ["build_observation_stats"]
