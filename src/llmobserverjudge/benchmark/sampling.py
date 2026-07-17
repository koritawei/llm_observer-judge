"""Stratified sampling for daily Benchmark (design §11.1)."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any


def target_sample_size(count: int, *, rate: float = 0.1, min_size: int = 5000) -> int:
    """actual = max(count×rate, min(min_size, count)); count < min_size → all."""
    if count <= 0:
        return 0
    if count < min_size:
        return count
    return max(int(count * rate), min(min_size, count))


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    rate: float = 0.1,
    min_size: int = 5000,
    strata_keys: tuple[str, ...] = ("eval_agent_id", "score_status"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic stratified sample. Returns (sample, sampling_meta)."""
    count = len(rows)
    n = target_sample_size(count, rate=rate, min_size=min_size)
    if count == 0:
        return [], _meta(0, 0, rate)

    if n >= count:
        sample = list(rows)
    else:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            key = tuple(r.get(k) for k in strata_keys)
            buckets[key].append(r)

        # proportional allocation with at least 1 per non-empty stratum when possible
        sample = []
        remaining = n
        strata = sorted(buckets.keys(), key=lambda k: str(k))
        # First pass: allocate floor
        alloc: dict[tuple[Any, ...], int] = {}
        for key in strata:
            bucket = buckets[key]
            share = max(1, int(round(len(bucket) / count * n))) if count else 0
            share = min(share, len(bucket))
            alloc[key] = share
        total_alloc = sum(alloc.values())
        # Adjust to exactly n
        while total_alloc > n:
            # shrink largest
            key = max(strata, key=lambda k: alloc[k])
            if alloc[key] > 1:
                alloc[key] -= 1
                total_alloc -= 1
            else:
                break
        while total_alloc < n:
            key = max(strata, key=lambda k: len(buckets[k]) - alloc[k])
            if alloc[key] < len(buckets[key]):
                alloc[key] += 1
                total_alloc += 1
            else:
                break

        for key in strata:
            bucket = buckets[key]
            take = alloc[key]
            # stable pick by md5(trace_id|agent)
            ranked = sorted(
                bucket,
                key=lambda r: hashlib.md5(
                    f"{r.get('trace_id')}:{r.get('eval_agent_id')}".encode()
                ).hexdigest(),
            )
            sample.extend(ranked[:take])
            remaining -= take

        sample = sample[:n]

    actual_rate = len(sample) / count if count else 0.0
    meta = _meta(count, len(sample), actual_rate)
    meta["stratified_by"] = list(strata_keys)
    meta["timezone"] = "UTC+8"
    return sample, meta


def _meta(total: int, sampled: int, actual_rate: float) -> dict[str, Any]:
    z = 1.96
    p = actual_rate
    ci_margin = z * math.sqrt(p * (1 - p) / total) if total > 0 else 0.0
    return {
        "total": total,
        "sampled": sampled,
        "actual_sample_rate": round(actual_rate, 6),
        "confidence_interval_95": [
            round(max(0.0, p - ci_margin), 6),
            round(min(1.0, p + ci_margin), 6),
        ],
    }
