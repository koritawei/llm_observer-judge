"""BenchmarkAgent — aggregate sample into a report (design §11.5)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from llmobserverjudge.contracts.models import BenchmarkReport


class BenchmarkAgent:
    """Deterministic report builder (no LLM tools in STEP-10)."""

    async def generate(
        self,
        *,
        tenant_id: str,
        sample: list[dict[str, Any]],
        meta: dict[str, Any],
        batch_job_id: str | None = None,
        day: str | None = None,
        annotation_stats: dict[str, Any] | None = None,
        historical: BenchmarkReport | None = None,
    ) -> BenchmarkReport:
        scores = [
            float(r["normalized_score"])
            for r in sample
            if r.get("normalized_score") is not None
            and r.get("score_status") == "scored"
        ]
        by_agent: dict[str, list[float]] = defaultdict(list)
        bad_cases: list[dict[str, Any]] = []
        status_counts: Counter[str] = Counter()

        for r in sample:
            status_counts[str(r.get("score_status") or "unknown")] += 1
            aid = str(r.get("eval_agent_id") or "unknown")
            if r.get("score_status") == "scored" and r.get("normalized_score") is not None:
                by_agent[aid].append(float(r["normalized_score"]))
            if r.get("is_bad_case"):
                bad_cases.append(
                    {
                        "trace_id": r.get("trace_id"),
                        "eval_agent_id": aid,
                        "normalized_score": r.get("normalized_score"),
                        "label": r.get("label"),
                    }
                )

        mean = round(sum(scores) / len(scores), 4) if scores else 0.0
        dimensions = {
            aid: {
                "count": len(vals),
                "mean_score": round(sum(vals) / len(vals), 4) if vals else 0.0,
            }
            for aid, vals in sorted(by_agent.items())
        }

        comparison: dict[str, Any] = {}
        insights: list[str] = []
        if historical and historical.summary.get("mean_score") is not None:
            prev = float(historical.summary["mean_score"])
            delta = round(mean - prev, 4)
            comparison = {
                "previous_report_id": historical.report_id,
                "mean_score_delta": delta,
            }
            if delta < -0.05:
                insights.append("mean_score dropped >0.05 vs previous benchmark")
            elif delta > 0.05:
                insights.append("mean_score improved >0.05 vs previous benchmark")

        dead = int(meta.get("dead_count") or 0)
        if dead > 0:
            insights.append(f"incomplete batch: dead_count={dead}")
            meta = {
                **meta,
                "dead_count": dead,
                "incomplete_count": dead
                + int(meta.get("failed_count") or 0),
            }

        if mean < 0.6:
            insights.append("overall quality below 0.6 threshold")
        if bad_cases:
            insights.append(f"bad_case_count={len(bad_cases)}")

        return BenchmarkReport(
            report_id=str(uuid4()),
            tenant_id=tenant_id,
            batch_job_id=batch_job_id,
            date=day,
            summary={
                "mean_score": mean,
                "scored_count": len(scores),
                "sample_size": len(sample),
                "status_counts": dict(status_counts),
                "bad_case_count": len(bad_cases),
            },
            dimensions=dimensions,
            comparison=comparison,
            bad_case_analysis={
                "count": len(bad_cases),
                "items": bad_cases[:50],
            },
            human_annotation_summary=annotation_stats or {"count": 0},
            insights=insights,
            sampling_meta=meta,
            created_at=datetime.now(UTC),
        )
