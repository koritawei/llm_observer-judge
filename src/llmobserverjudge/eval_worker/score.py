"""ScoreNode helpers (SDD-EV-23)."""

from __future__ import annotations

from typing import Any

from llmobserverjudge.contracts.models import (
    EvalScoreEntry,
    ScoreResult,
    ScoreStatus,
)


def aggregate_scores(
    *,
    tenant_id: str,
    trace_id: str,
    results: list[dict[str, Any]],
    prompt_version_id: str | None = None,
) -> ScoreResult:
    eval_scores: list[EvalScoreEntry] = []
    scored_items: list[EvalScoreEntry] = []

    for r in results:
        agent_id = r.get("agent_id", "unknown")
        raw_status = r.get("score_status", "scored")
        try:
            status = ScoreStatus(raw_status)
        except ValueError:
            status = ScoreStatus.MODEL_ERROR

        if status == ScoreStatus.SKIPPED:
            eval_scores.append(
                EvalScoreEntry(
                    agent_id=agent_id,
                    score_status=status,
                    skip_reason=r.get("skip_reason", ""),
                )
            )
            continue
        if status == ScoreStatus.INFRA_ERROR:
            eval_scores.append(
                EvalScoreEntry(
                    agent_id=agent_id,
                    score_status=status,
                    error=r.get("error", ""),
                    error_type=r.get("error_type", ""),
                )
            )
            continue
        if status == ScoreStatus.MODEL_ERROR:
            entry = EvalScoreEntry(
                agent_id=agent_id,
                score_status=ScoreStatus.MODEL_ERROR,
                normalized_score=0.0,
                is_bad_case=True,
                label=r.get("label") or "PARSE_ERROR",
                error=r.get("error"),
                error_type=r.get("error_type"),
            )
            eval_scores.append(entry)
            continue

        raw = float(r.get("score", 0))
        scale_max = float(r.get("scale_max", 10) or 10)
        normalized = float(r.get("normalized_score") or (raw / scale_max if scale_max else 0.0))
        threshold = float(r.get("threshold", 0.5))
        weight = float(r.get("weight", 1.0))
        entry = EvalScoreEntry(
            agent_id=agent_id,
            score_status=ScoreStatus.SCORED,
            raw_score=raw,
            normalized_score=round(normalized, 4),
            threshold=threshold,
            weight=weight,
            is_bad_case=normalized < threshold,
            label=r.get("label", ""),
            reasoning=r.get("reasoning", ""),
            conflict_detected=False,
        )
        eval_scores.append(entry)
        scored_items.append(entry)

    apply_conflict_flags(eval_scores)

    total_w = sum(e.weight or 0 for e in scored_items) or 0
    final = (
        round(
            sum((e.normalized_score or 0) * (e.weight or 0) for e in scored_items) / total_w,
            4,
        )
        if total_w
        else 0.0
    )
    bad = [e.agent_id for e in scored_items if e.is_bad_case]
    # model_error also counts as bad_case per SDD-EV-23
    bad.extend(
        e.agent_id for e in eval_scores if e.score_status == ScoreStatus.MODEL_ERROR
    )
    return ScoreResult(
        tenant_id=tenant_id,
        trace_id=trace_id,
        prompt_version_id=prompt_version_id,
        eval_scores=eval_scores,
        final_score=final,
        bad_case_detected=len(bad) > 0,
        bad_case_evals=list(dict.fromkeys(bad)),
    )


def apply_conflict_flags(eval_scores: list[EvalScoreEntry]) -> None:
    """Mark conflict_detected on known agent pairs (design §6.5)."""
    by_id = {e.agent_id: e for e in eval_scores if e.score_status == ScoreStatus.SCORED}
    safety = by_id.get("judge_safety")
    quality = by_id.get("judge_quality")
    if safety is None or quality is None:
        return
    conflict = False
    if safety.is_bad_case is False and quality.is_bad_case is True:
        conflict = True
    s_score = safety.normalized_score or 0.0
    q_score = quality.normalized_score or 0.0
    if abs(s_score - q_score) > 0.4:
        conflict = True
    if conflict:
        safety.conflict_detected = True
        quality.conflict_detected = True
