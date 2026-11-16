"""L1 crosstab statistics — cross analysis of the effectiveness and attribution halves (the real increment of the cluster report).

Each half alone is straightforward; the value is in the cross:
1. checker hit → root-cause mapping gap: a checker hits often but attribution is unfinished
   or has no root_cause_code, which means that checker lacks a ruled root-cause mapping.
2. ``dimension × owner_group``: failures of one dimension all point at the same team → a
   clear entry for a focused governance effort.
3. tier vs overturn rate: many low scores under a tag/suite are overturned → the rubric or
   checker for that scene is misjudging.

Older versions included a ``failure_mode × root_cause_code`` co-occurrence matrix. On small
samples the Cartesian expansion produced many low-information pairs (including same-name
diagonal rows that are always true). It was removed in T5.
"""

from collections import Counter, defaultdict
from typing import Any

from benchmark.reporting.cluster.constants import DIMENSIONS, FAILURE_STATUSES, PASS_SCORE
from benchmark.reporting.cluster.schema import BatchCaseRecord
from benchmark.reporting.cluster.stats.attribution import root_cause_label

_UNCLASSIFIED = "unclassified"


def build_crosstab_stats(records: list[BatchCaseRecord]) -> dict[str, Any]:
    """Cross-analysis block."""

    return {
        "checker_root_cause_gap": _checker_root_cause_gap(records),
        "dimension_x_owner_group": _dimension_x_owner(records),
        "suite_overturn": _suite_overturn(records),
    }


def _checker_root_cause_gap(records: list[BatchCaseRecord]) -> list[dict[str, Any]]:
    """Gap where a checker hit but attribution did not produce a root_cause_code.

    A checker with a high ``unmapped`` count has no matching ruled root-cause mapping on
    the attribution side after it hits. That gap is a direct clue for feeding checkers/rules back.
    """

    hits: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    for record in records:
        if not (record.attribution.is_completed or record.attribution.is_pending):
            continue
        names = _failed_checker_names(record)
        # Overturned evaluations and objective no-fault cases never emit a formal root-cause code, so they are not mapping gaps.
        has_root_cause = root_cause_label(record) != _UNCLASSIFIED
        for name in names:
            hits[name] += 1
            if not has_root_cause:
                unmapped[name] += 1

    return [
        {
            "checker": name,
            "hits": count,
            "unmapped": unmapped.get(name, 0),
            "unmapped_rate": round(unmapped.get(name, 0) / count, 4) if count else 0.0,
        }
        for name, count in hits.most_common()
    ]


def _dimension_x_owner(records: list[BatchCaseRecord]) -> list[dict[str, Any]]:
    """D0-D6 failures × owner_group (entry for focused governance)."""

    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if not record.attribution.is_completed:
            continue
        owner = record.attribution.owner_group or _UNCLASSIFIED
        for dimension in _failed_dimensions(record):
            matrix[dimension][owner] += 1

    return [
        {
            "dimension": dimension,
            "owners": dict(owners.most_common()),
            "total": sum(owners.values()),
        }
        for dimension, owners in sorted(
            matrix.items(),
            key=lambda item: (
                DIMENSIONS.index(item[0]) if item[0] in DIMENSIONS else len(DIMENSIONS)
            ),
        )
    ]


def _suite_overturn(records: list[BatchCaseRecord]) -> list[dict[str, Any]]:
    """Count low scores and overturn rate by suite — a high overturn rate means the evaluation rules for that scene are suspect."""

    grouped: dict[str, list[BatchCaseRecord]] = defaultdict(list)
    for record in records:
        grouped[record.suite or _UNCLASSIFIED].append(record)

    rows: list[dict[str, Any]] = []
    for suite, items in grouped.items():
        completed = [r for r in items if r.attribution.is_completed]
        overturned = [r for r in completed if r.attribution.is_overturn]
        low_score = [
            r for r in items if r.effective_score is not None and r.effective_score < PASS_SCORE
        ]
        rows.append(
            {
                "suite": suite,
                "total": len(items),
                "low_score": len(low_score),
                "attributed": len(completed),
                "overturned": len(overturned),
                "overturn_rate": round(len(overturned) / len(completed), 4) if completed else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (-row["overturned"], row["suite"]))


def _failed_checker_names(record: BatchCaseRecord) -> list[str]:
    names: list[str] = []
    for item in record.checker_results:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "") not in FAILURE_STATUSES:
            continue
        name = str(item.get("source_name") or "")
        if name:
            names.append(name)
    return names


def _failed_dimensions(record: BatchCaseRecord) -> list[str]:
    dimensions: list[str] = []
    for item in record.dimension_results:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or "")
        if dimension not in DIMENSIONS:
            continue
        if str(item.get("status") or "") in FAILURE_STATUSES:
            dimensions.append(dimension)
    return dimensions


__all__ = ["build_crosstab_stats"]
