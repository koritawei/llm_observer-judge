"""L1 attribution statistics — root-cause clusters / owner distribution / overturn rate / confidence / backlog.

Depends on the bucket fields filled in by T15 (``root_cause_code`` / ``conclusion_type`` /
``category`` / ``needs_human_review`` / ``evidence_details``). Old data missing a field is
counted as "unclassified" via an empty value; it is neither raised nor silently dropped.
"""

from collections import Counter
from collections.abc import Iterable
from typing import Any

from benchmark.reporting.cluster.schema import BatchCaseRecord

#: Fallback-rule prefix for owner_group (the marker when a RemoteConfig `vehicle-agent/*` P2 fallback hits).
_FALLBACK_OWNER_PREFIX = "vehicle-agent/"

#: Unclassified placeholder (when root_cause_code / category / owner_group is missing).
_UNCLASSIFIED = "unclassified"

#: Non-defect conclusions listed on their own when there is no formal root-cause code: overturned evaluation and objective no-fault are both **successful** attribution
#: output. Putting them in unclassified would misreport "the gate worked" as "an attribution-capability gap".
_STANDALONE_CONCLUSIONS = frozenset({"judge_false_positive", "non_attributable_outcome"})

#: Rejection-reason buckets for unclassified cases: match the stable copy the attribution side writes into root_cause.
#: The copy comes from this project's gate messages in stages.py and is our own contract; a miss falls back to the other bucket.
_REJECTION_BUCKETS: tuple[tuple[str, str], ...] = (
    ("未逐项覆盖", "incomplete_mode_coverage"),
    ("未决 material", "unresolved_material_mode"),
    ("EvidenceFact", "refuted_without_fact_binding"),
    ("taxonomy", "taxonomy_candidate_mismatch"),
    ("validation error", "decision_parse_failed"),
    ("缺少完整 trace 证据", "trace_evidence_unavailable"),
    ("fetch_trace_skeleton", "trace_evidence_unavailable"),
)

#: Confidence tier boundaries.
_HIGH_CONFIDENCE = 0.8
_MID_CONFIDENCE = 0.5


def build_attribution_stats(
    records: list[BatchCaseRecord],
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """Attribution-block statistics.

    The denominator of ``attribution_coverage`` is "should be attributed" = pending +
    completed, i.e. the case count the triage classifier actually let through (the
    aggregation side does not reimplement classifier rules, to avoid scale drift).
    """

    completed = [r for r in records if r.attribution.is_completed]
    pending = [r for r in records if r.attribution.is_pending]
    should_attribute = len(completed) + len(pending)

    return {
        "should_attribute": should_attribute,
        "attributed": len(completed),
        "pending": len(pending),
        "attribution_coverage": _ratio(len(completed), should_attribute),
        "root_cause_top": _top_counts(
            (root_cause_label(r) for r in completed),
            top_n,
        ),
        # All-root-cause Top: expand all_root_cause_codes (one case with several causes is counted several times),
        # and fall back to the primary root cause when old data lacks the new field. Same scale as root_cause_top.
        "all_root_cause_top": _top_counts(
            (
                code or _UNCLASSIFIED
                for r in completed
                for code in (r.attribution.effective_root_cause_codes or [root_cause_label(r)])
            ),
            top_n,
        ),
        "category_distribution": _counts(
            r.attribution.category or _UNCLASSIFIED for r in completed
        ),
        "owner_group_distribution": _counts(
            r.attribution.owner_group or _UNCLASSIFIED for r in completed
        ),
        "owner_fallback_rate": _ratio(
            sum(
                1 for r in completed if r.attribution.owner_group.startswith(_FALLBACK_OWNER_PREFIX)
            ),
            len(completed),
        ),
        "conclusion_distribution": _counts(
            r.attribution.conclusion_type or _UNCLASSIFIED for r in completed
        ),
        "overturn": _overturn_stats(completed),
        "confidence": _confidence_stats(completed),
        "evidence": _evidence_stats(completed),
        "human_review": _human_review_list(records),
        "pending_backlog": _pending_backlog(pending),
        "root_cause_cases": _root_cause_cases(completed),
        # `other:` candidates are counted separately and never enter the formal root_cause Top or the owner distribution.
        "other_candidate_stats": _other_candidate_stats(completed),
        # Rejection-reason distribution of unclassified cases (no formal root cause and not a standalone conclusion), for the clustering LLM
        # and for humans to locate the nature of the gap. Do not let them attribute it themselves to "insufficient rule coverage".
        "unclassified_breakdown": _unclassified_breakdown(completed),
    }


def root_cause_label(record: BatchCaseRecord) -> str:
    """Root-cause label under the stats scale: formal code first; otherwise a non-defect conclusion is listed alone; the rest count as unclassified."""

    code = record.attribution.root_cause_code
    if code:
        return code
    conclusion = record.attribution.conclusion_type
    return conclusion if conclusion in _STANDALONE_CONCLUSIONS else _UNCLASSIFIED


def _unclassified_breakdown(completed: list[BatchCaseRecord]) -> dict[str, Any]:
    """Bucket unclassified cases by rejection reason and attach case details for L2 to cite."""

    unclassified = [r for r in completed if root_cause_label(r) == _UNCLASSIFIED]
    buckets: Counter[str] = Counter()
    for record in unclassified:
        buckets[
            _rejection_bucket(record.attribution.root_cause, record.attribution.conclusion_type)
        ] += 1
    return {
        "count": len(unclassified),
        "buckets": dict(buckets.most_common()),
        "cases": [
            {
                "trace_id": record.trace_id,
                "case_id": record.case_key.case_id,
                "suite": record.suite,
                "bucket": _rejection_bucket(
                    record.attribution.root_cause, record.attribution.conclusion_type
                ),
            }
            for record in unclassified
        ],
    }


def _rejection_bucket(root_cause: str, conclusion_type: str) -> str:
    for marker, bucket in _REJECTION_BUCKETS:
        if marker in root_cause:
            return bucket
    return conclusion_type or "other"


def _overturn_stats(completed: list[BatchCaseRecord]) -> dict[str, Any]:
    """Statistics for overturned-evaluation conclusions (judge_false_positive / eval_rule_issue / dataset_issue).

    A cluster of ``eval_rule_issue`` means the evaluation rules themselves have a bug —
    that should be fixed before a business defect — so it is listed on its own, not only as a total.
    """

    overturned = [r for r in completed if r.attribution.is_overturn]
    corrected = [r for r in completed if r.attribution.corrected_triage_level]
    return {
        "overturn_count": len(overturned),
        "overturn_rate": _ratio(len(overturned), len(completed)),
        "by_type": _counts(r.attribution.conclusion_type for r in overturned),
        "corrected_triage_count": len(corrected),
        "corrected_triage_rate": _ratio(len(corrected), len(completed)),
        "corrected_levels": _counts(r.attribution.corrected_triage_level for r in corrected),
        "cases": [
            {
                "trace_id": r.trace_id,
                "case_id": r.case_key.case_id,
                "suite": r.suite,
                "conclusion_type": r.attribution.conclusion_type,
                "root_cause": r.attribution.root_cause,
                "corrected_triage_level": r.attribution.corrected_triage_level,
            }
            for r in overturned
        ],
    }


def _confidence_stats(completed: list[BatchCaseRecord]) -> dict[str, Any]:
    """Confidence distribution. Confidence is the calibrated value from the attribution side's final_validation_gate; it is not recomputed."""

    values = [r.attribution.confidence for r in completed if r.attribution.confidence is not None]
    high = sum(1 for v in values if v >= _HIGH_CONFIDENCE)
    mid = sum(1 for v in values if _MID_CONFIDENCE <= v < _HIGH_CONFIDENCE)
    low = sum(1 for v in values if v < _MID_CONFIDENCE)
    return {
        "counted": len(values),
        "avg": round(sum(values) / len(values), 4) if values else 0.0,
        "high": high,
        "mid": mid,
        "low": low,
        "low_rate": _ratio(low, len(values)),
    }


def _evidence_stats(completed: list[BatchCaseRecord]) -> dict[str, Any]:
    """Evidence sufficiency: the share of attributions that have field-level evidence (a conclusion without evidence is low confidence)."""

    with_evidence = sum(1 for r in completed if r.attribution.evidence_details)
    return {
        "with_evidence": with_evidence,
        "without_evidence": len(completed) - with_evidence,
        "evidence_rate": _ratio(with_evidence, len(completed)),
    }


def _human_review_list(records: list[BatchCaseRecord]) -> dict[str, Any]:
    """List that needs human review (can be fed to Label Studio directly)."""

    flagged = [r for r in records if r.attribution.needs_human_review]
    return {
        "count": len(flagged),
        "cases": [
            {
                "trace_id": r.trace_id,
                "case_id": r.case_key.case_id,
                "suite": r.suite,
                "score": r.effective_score,
                "root_cause": r.attribution.root_cause,
                "confidence": r.attribution.confidence,
            }
            for r in flagged
        ],
    }


def _pending_backlog(pending: list[BatchCaseRecord]) -> dict[str, Any]:
    """Attribution backlog: accepted but not finished (the gap when the watcher closes on a stall)."""

    return {
        "count": len(pending),
        "cases": [
            {
                "trace_id": r.trace_id,
                "case_id": r.case_key.case_id,
                "suite": r.suite,
                "score": r.effective_score,
                "triage_level": r.triage_level,
            }
            for r in pending
        ],
    }


def _root_cause_cases(completed: list[BatchCaseRecord]) -> dict[str, dict[str, Any]]:
    """Index cases by every attributed root cause so L2 clustering can fill traces back programmatically."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in completed:
        root_causes = record.attribution.effective_root_cause_codes or [root_cause_label(record)]
        for code in root_causes:
            key = code or root_cause_label(record)
            grouped.setdefault(key, []).append(
                {
                    "trace_id": record.trace_id,
                    "case_id": record.case_key.case_id,
                    "suite": record.suite,
                }
            )
    return {
        code: {"count": len(cases), "cases": cases}
        for code, cases in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def _other_candidate_stats(completed: list[BatchCaseRecord]) -> dict[str, Any]:
    """Aggregate `other:` candidates by a deterministic key within the scope; do not merge across Agents."""

    grouped: dict[str, list[BatchCaseRecord]] = {}
    for record in completed:
        candidate = record.attribution.candidate_root_cause
        if candidate is not None:
            grouped.setdefault(candidate.candidate_key(), []).append(record)

    candidates: list[dict[str, Any]] = []
    without_evidence = 0
    for key, records in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        candidate = records[0].attribution.candidate_root_cause
        assert candidate is not None
        evidence_complete = sum(1 for record in records if record.attribution.evidence_details)
        without_evidence += len(records) - evidence_complete
        candidates.append(
            {
                "candidate_key": key,
                "agent_scope": candidate.agent_scope,
                "raw_codes": sorted(
                    {
                        record.attribution.candidate_root_cause.raw_code
                        for record in records
                        if record.attribution.candidate_root_cause is not None
                    }
                ),
                "title": candidate.title,
                "surface": candidate.surface,
                "component": candidate.component,
                "symptom": candidate.symptom,
                "count": len(records),
                "evidence_complete_rate": _ratio(evidence_complete, len(records)),
                "cases": [
                    {
                        "trace_id": record.trace_id,
                        "case_id": record.case_key.case_id,
                        "suite": record.suite,
                    }
                    for record in records
                ],
            }
        )
    return {
        "count": sum(len(records) for records in grouped.values()),
        "without_evidence": without_evidence,
        "candidates": candidates,
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        if value:
            counter[str(value)] += 1
    return dict(counter.most_common())


def _top_counts(values: Iterable[str], top_n: int) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    total = 0
    for value in values:
        if value:
            counter[str(value)] += 1
            total += 1
    return [
        {"code": code, "count": count, "share": _ratio(count, total)}
        for code, count in counter.most_common(top_n)
    ]


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


__all__ = ["build_attribution_stats", "root_cause_label"]
