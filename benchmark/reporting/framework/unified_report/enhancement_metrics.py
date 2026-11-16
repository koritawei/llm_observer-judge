"""Deterministic enhancement metrics shared by the daily report and Cluster Report."""

from collections import defaultdict
from math import isfinite
from typing import Any

from pydantic import BaseModel, Field

UNATTRIBUTED_ROOT_CAUSE_LABEL = "未归因"
_AGGREGATED_OTHER_LABEL = "__aggregated_other__"


class EnhancementSample(BaseModel):
    """Per-case input contract shared by the waterfall chart and the root-cause dimension matrix."""

    root_cause: str | None = None
    score: float | None = None
    dimensions: dict[str, float] = Field(default_factory=dict)


def build_score_loss_waterfall(
    *,
    final_score: float | None,
    samples: list[EnhancementSample],
    full_score: float = 2.0,
) -> dict[str, Any] | None:
    """Allocate the overall-score gap by root cause and return data ready for HTML rendering."""

    buckets: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        score = numeric_score(sample.score)
        if score is None:
            continue
        label = root_cause_label(sample.root_cause)
        buckets[label].append(min(full_score, max(0.0, full_score - score)))
    if not buckets:
        return None

    raw_totals = {label: sum(losses) for label, losses in buckets.items()}
    total_raw_loss = sum(raw_totals.values())
    if total_raw_loss <= 0:
        return None

    end_score = numeric_score(final_score) or 0.0
    total_loss = min(full_score, max(0.0, full_score - end_score))
    steps = [
        {
            "label": label,
            "loss": round(total_loss * raw_loss / total_raw_loss, 3),
            "count": len(buckets[label]),
            "raw_loss": round(raw_loss, 3),
        }
        for label, raw_loss in raw_totals.items()
    ]
    return {
        "start": full_score,
        "end": round(end_score, 4),
        "delta": round(total_loss, 4),
        "steps": _limit_waterfall_steps(steps),
    }


def build_root_cause_dimension_matrix(
    *,
    samples: list[EnhancementSample],
    columns: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compute the root-cause × dimension mean score-loss matrix; full-score samples count as zero loss in the denominator."""

    valid_columns = [
        {
            "id": str(column["id"]),
            "name": str(column.get("name") or column["id"]),
            "full_score": float(column.get("full_score", 2.0)),
        }
        for column in columns
        if isinstance(column, dict) and column.get("id")
    ]
    if not valid_columns:
        return None

    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    row_totals: dict[str, float] = defaultdict(float)
    for sample in samples:
        dimensions = sample.dimensions
        label = root_cause_label(sample.root_cause)
        for column in valid_columns:
            score = numeric_score(dimensions.get(column["id"]))
            if score is None:
                continue
            loss = min(column["full_score"], max(0.0, column["full_score"] - score))
            sums[label][column["id"]] += loss
            counts[label][column["id"]] += 1
            row_totals[label] += loss
    if not sums:
        return None

    ranked_rows = sorted(row_totals, key=lambda label: (-row_totals[label], label))
    rows = ranked_rows[:8]
    if UNATTRIBUTED_ROOT_CAUSE_LABEL in ranked_rows and UNATTRIBUTED_ROOT_CAUSE_LABEL not in rows:
        rows = rows[:7] + [UNATTRIBUTED_ROOT_CAUSE_LABEL]
    cells = {
        row: {
            dimension: round(total / counts[row][dimension], 3)
            for dimension, total in sums[row].items()
            if counts[row][dimension]
        }
        for row in rows
    }
    return {
        "rows": rows,
        "cols": [{"id": column["id"], "name": column["name"]} for column in valid_columns],
        "cells": cells,
        "unit": "mean_loss",
    }


def numeric_score(value: object) -> float | None:
    """Ignore booleans, non-numeric values, and NaN/Inf scores."""

    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    result = float(value)
    return result if isfinite(result) else None


def root_cause_label(value: object) -> str:
    """Stably bucket an empty root cause into the unattributed bucket."""

    if isinstance(value, str) and value.strip():
        return value.strip()
    return UNATTRIBUTED_ROOT_CAUSE_LABEL


def _limit_waterfall_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Show at most seven groups; unattributed is always shown on its own."""

    ordered = sorted(steps, key=lambda step: float(step["loss"]), reverse=True)
    unattributed = next(
        (step for step in ordered if step["label"] == UNATTRIBUTED_ROOT_CAUSE_LABEL),
        None,
    )
    attributed = [step for step in ordered if step["label"] != UNATTRIBUTED_ROOT_CAUSE_LABEL]
    top = attributed[: 5 if unattributed else 6]
    if unattributed:
        top.append(unattributed)
    rest = attributed[len(top) - (1 if unattributed else 0) :]
    if rest:
        top.append(
            {
                "label": _AGGREGATED_OTHER_LABEL,
                "loss": round(sum(float(step["loss"]) for step in rest), 3),
                "count": sum(int(step["count"]) for step in rest),
                "raw_loss": round(sum(float(step["raw_loss"]) for step in rest), 3),
            }
        )
    return sorted(top, key=lambda step: float(step["loss"]), reverse=True)


__all__ = [
    "EnhancementSample",
    "UNATTRIBUTED_ROOT_CAUSE_LABEL",
    "build_root_cause_dimension_matrix",
    "build_score_loss_waterfall",
    "numeric_score",
    "root_cause_label",
]
