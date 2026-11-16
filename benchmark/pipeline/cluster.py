"""Public cluster step.

A domain keeps related datasets together. Each dataset scores only its own
portion. This step groups those partial results by dimension and reports
whether the domain's dimensions are all present.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def cluster_dimension_results(
    results: list[dict[str, Any]] | None,
    *,
    expected_dimensions: list[str] | None = None,
) -> dict[str, Any]:
    """Group partial scores into one multi-dimensional cluster report."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results or []:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension") or item.get("criterion_id") or "overall")
        grouped[dimension].append(item)
    dimensions = []
    for name, rows in grouped.items():
        scores = [float(row.get("score") or 0) for row in rows]
        mean = sum(scores) / len(scores) if scores else 0.0
        dimensions.append(
            {
                "dimension": name,
                "count": len(rows),
                "mean_score": mean,
                "portions": sorted({str(row.get("portion") or "") for row in rows if row.get("portion")}),
            }
        )
    expected = list(expected_dimensions or [])
    present = {item["dimension"] for item in dimensions}
    missing = [name for name in expected if name not in present]
    return {
        "dimensions": dimensions,
        "expected_dimensions": expected,
        "missing_dimensions": missing,
        "complete": not missing and bool(expected or dimensions),
    }
