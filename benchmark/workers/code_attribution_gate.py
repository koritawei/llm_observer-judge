"""Code-attribution enqueue gate. Grades cover only FAIL / POOR; analysis failures are skipped."""

from typing import Any

from benchmark.settings.setting import settings
from business.evaluators.example.grading import (
    GRADE_FAIL,
    GRADE_POOR,
    rubric_v2_grade,
)

_CODE_ATTR_GRADES = frozenset({GRADE_FAIL, GRADE_POOR})


def should_run_code_attribution(document: dict[str, Any], fused_score: float | None) -> bool:
    """Enqueue only when the switch is on and the grade is FAIL or POOR."""
    if not settings.code_attribution_enabled:
        return False
    if document.get("analysis_failed"):
        return False
    category = str(document.get("primary_issue_category") or "")
    if category == "analysis_failed":
        return False
    if fused_score is None:
        return False
    return rubric_v2_grade(fused_score) in _CODE_ATTR_GRADES
