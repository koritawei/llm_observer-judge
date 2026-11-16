"""Fallback display plugin: example-shaped judge / attribution parsing.

Streaming (``aone_source=trace``) and any ``benchmark_name`` other than the registered
ieval / search_eval / search_eval_v2 names (including amap_agent* on the items side)
use this plugin. Judge tries persisted shapes in example-first order (top-level extra →
sub_scores → raw_json.extra → criterion_scores) and keeps the non-empty view with the
richest dimensions; attribution uses the shared markdown / table fill.
"""

from collections.abc import Mapping

from benchmark.runs.case_display.helpers import (
    batch_attribution_view,
    batch_output_text,
    empty_judge,
    judge_from_criterion_scores,
    judge_from_extra_bucket,
    judge_from_extra_top,
    judge_from_sub_scores,
    raw_judge_blob,
)
from benchmark.runs.case_display.views import (
    CaseDisplayRequest,
    CaseDisplayResult,
    CaseJudgeView,
)


class DefaultCaseDisplayPlugin:
    """Fallback when no business claims the case. ``matches`` is always true and priority=0."""

    name = "default"
    priority = 0

    def matches(self, request: CaseDisplayRequest) -> bool:
        del request
        return True

    def normalize(self, request: CaseDisplayRequest) -> CaseDisplayResult:
        case = request.case
        return CaseDisplayResult(
            output=batch_output_text(case),
            judge=_default_judge(case),
            attribution=batch_attribution_view(case),
        )


def _default_judge(case: Mapping[str, object]) -> CaseJudgeView | None:
    """example-first: top-level extra → sub_scores → extra bucket → criterion_scores."""
    blob = raw_judge_blob(case)
    if blob is None:
        return None
    candidates = (
        judge_from_extra_top(blob),
        judge_from_sub_scores(blob),
        judge_from_extra_bucket(blob),
        judge_from_criterion_scores(blob),
    )
    best: CaseJudgeView | None = None
    best_rank = (-1, -1, -1)
    for view in candidates:
        if empty_judge(view):
            continue
        rank = (
            len(view.dimensions),
            1 if view.reason else 0,
            1 if view.score is not None else 0,
        )
        if rank > best_rank:
            best_rank = rank
            best = view
    return best


__all__ = ["DefaultCaseDisplayPlugin"]
