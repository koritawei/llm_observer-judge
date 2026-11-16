"""Score bands for the example benchmark. 0–10 maps onto the shared grade labels."""

GRADE_FAIL = "FAIL"
GRADE_POOR = "POOR"
GRADE_ACCEPTABLE = "ACCEPTABLE"
GRADE_GOOD = "GOOD"
GRADE_EXCELLENT = "EXCELLENT"

_TEN_POINT_MAX = 10.0
_POOR = 4.0
_ACCEPTABLE = 6.0
_GOOD = 8.0
_RUBRIC_V2_MAX = 2.0


def ten_point_grade(score: float) -> str:
    if score <= 0.0:
        return GRADE_FAIL
    if score < _POOR:
        return GRADE_POOR
    if score < _ACCEPTABLE:
        return GRADE_ACCEPTABLE
    if score < _GOOD:
        return GRADE_GOOD
    return GRADE_EXCELLENT


def should_trigger_ieval_attribution(
    fused_score: float,
    aggregation: object | None = None,
) -> bool:
    del aggregation
    return rubric_v2_grade(float(fused_score)) not in {GRADE_GOOD, GRADE_EXCELLENT}


def rubric_v2_grade(score: float) -> str:
    if score <= 0.0:
        return GRADE_FAIL
    if score < 1.0:
        return GRADE_POOR
    if score < 1.4:
        return GRADE_ACCEPTABLE
    if score < 1.8:
        return GRADE_GOOD
    return GRADE_EXCELLENT


def grade_fused_score(
    fused_score: float,
    *,
    score_scale_max: float | None = None,
    pass_threshold: float | None = None,
) -> str:
    score = float(fused_score)
    if score_scale_max is not None:
        scale_max = float(score_scale_max)
        if scale_max <= 0.0:
            raise ValueError("score_scale_max must be positive")
        if scale_max == _RUBRIC_V2_MAX:
            return rubric_v2_grade(score)
        return ten_point_grade(score / scale_max * _TEN_POINT_MAX)
    if pass_threshold is not None and float(pass_threshold) > _RUBRIC_V2_MAX:
        return ten_point_grade(score)
    if score > _RUBRIC_V2_MAX:
        return ten_point_grade(score)
    return rubric_v2_grade(score)
