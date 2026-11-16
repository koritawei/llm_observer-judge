"""Domain adapter from Cluster Report to the unified daily-report enhancement metrics."""

from typing import Any

from benchmark.reporting.cluster.constants import DIMENSION_LABELS, DIMENSIONS
from benchmark.reporting.cluster.schema import BatchCaseRecord
from benchmark.reporting.framework.unified_report.enhancement_metrics import (
    EnhancementSample,
    build_root_cause_dimension_matrix,
    build_score_loss_waterfall,
    numeric_score,
)


def build_enhancement_stats(records: list[BatchCaseRecord]) -> dict[str, Any]:
    """Adapt the ClusterReport case view into the daily report's shared waterfall and matrix inputs."""

    samples = [_sample(record) for record in records]
    scored = [record.effective_score for record in records if record.effective_score is not None]
    final_score = sum(scored) / len(scored) if scored else None
    return {
        "score_loss_waterfall": build_score_loss_waterfall(
            final_score=final_score,
            samples=samples,
        ),
        "root_cause_dim_matrix": build_root_cause_dimension_matrix(
            samples=samples,
            columns=[
                {
                    "id": dimension,
                    "name": DIMENSION_LABELS.get(dimension, dimension),
                    "full_score": 2.0,
                }
                for dimension in DIMENSIONS
            ],
        ),
    }


def _sample(record: BatchCaseRecord) -> EnhancementSample:
    """Project a single case; low-score cases without attribution are bucketed as "unattributed" by the shared module."""

    dimensions: dict[str, float] = {}
    for item in record.dimension_results:
        if not isinstance(item, dict) or item.get("status") == "not_applicable":
            continue
        dimension = str(item.get("dimension") or "")
        score = numeric_score(item.get("score_hint"))
        if dimension in DIMENSIONS and score is not None:
            dimensions[dimension] = score
    return EnhancementSample(
        root_cause=record.attribution.root_cause_code or record.attribution.root_cause,
        score=record.effective_score,
        dimensions=dimensions,
    )


__all__ = ["build_enhancement_stats"]
