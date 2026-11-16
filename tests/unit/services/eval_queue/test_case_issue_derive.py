"""Unit tests for the CaseIssue derivation functions — covers AC-1 through AC-8 and AC-11."""

CaseIssue = None
derive_case_issue_from_judge = None
normalize_actual = None

# ── helpers ──────────────────────────────────────────────


def _criterion(score: float, reason: str = "") -> dict:
    """Build a single criterion_score dict."""
    return {"score": score, "reason": reason, "sub_scores": None}


def _judge(
    scores: dict[str, dict] | None = None,
    error: str | None = None,
    expected: str = "",
    actual: str = "",
) -> dict:
    """Build a judge dict (simulating payload.model_dump() plus the enqueue injection)."""
    j: dict = {"criterion_scores": scores or {}, "error": error}
    if expected:
        j["expected"] = expected
    if actual:
        j["actual"] = actual
    return j


# ── AC-1: a single low dimension is bad ─────────────────────────────────────


def test_derive_bad_single_dim() -> None:
    """D3.score=0.0 and every other dimension is ≥1.5 → verdict=bad, dimension=D3."""
    judge = _judge(
        scores={
            "D0": _criterion(2.0, "good"),
            "D1": _criterion(2.0, "good"),
            "D2": _criterion(2.0, "good"),
            "D3": _criterion(0.0, "POI 错误"),
            "D4": _criterion(2.0, "good"),
        },
    )
    result = derive_case_issue_from_judge(judge)
    assert result.verdict == "bad"
    assert result.dimension == "D3"
    assert "D3" in result.issue_hint
    assert "POI 错误" in result.issue_hint


# ── AC-2: every dimension is good ───────────────────────────────────


def test_derive_good() -> None:
    """Every dimension has score ≥1.5 and there is no error → verdict=good."""
    judge = _judge(
        scores={
            "D0": _criterion(2.0, "good"),
            "D1": _criterion(2.0, "good"),
            "D2": _criterion(2.0, "good"),
        },
    )
    result = derive_case_issue_from_judge(judge)
    assert result.verdict == "good"
    assert result.dimension == ""
    assert result.issue_hint == ""


# ── AC-3: an error makes it bad ───────────────────────────────


def test_derive_bad_with_error() -> None:
    """An error is present → verdict=bad."""
    judge = _judge(
        scores={
            "D0": _criterion(2.0, "good"),
            "D1": _criterion(2.0, "good"),
        },
        error="judge timeout",
    )
    result = derive_case_issue_from_judge(judge)
    assert result.verdict == "bad"


# ── AC-4: a low dimension with no reason ──────────────────────────────


def test_derive_no_reason() -> None:
    """A dimension with score=0 and no reason → issue_hint uses the {dim} score={score} format."""
    judge = _judge(
        scores={
            "D0": _criterion(2.0, "good"),
            "D1": _criterion(0.0),  # No reason
        },
    )
    result = derive_case_issue_from_judge(judge)
    assert result.verdict == "bad"
    assert result.dimension == "D1"
    assert "D1 score=" in result.issue_hint


# ── AC-5: tied lowest dimensions D2/D3 ────────────────────────────────


def test_derive_tie_dims() -> None:
    """D2/D3 tie for the lowest score → dimension=D2, and issue_hint contains D2 and D3."""
    judge = _judge(
        scores={
            "D0": _criterion(2.0, "good"),
            "D1": _criterion(2.0, "good"),
            "D2": _criterion(0.0, "wrong poi"),
            "D3": _criterion(0.0, "wrong route"),
            "D4": _criterion(2.0, "good"),
        },
    )
    result = derive_case_issue_from_judge(judge)
    assert result.verdict == "bad"
    assert result.dimension == "D2"
    assert "D2" in result.issue_hint
    assert "D3" in result.issue_hint
    assert "；" in result.issue_hint


# ── AC-6: expected/actual passthrough ────────────────────────────


def test_derive_passthrough_expected_actual() -> None:
    """issue contains expected/actual → CaseIssue passes them through unchanged."""
    judge = _judge(
        scores={
            "D0": _criterion(0.0, "bad"),
        },
        expected="推荐3个附近的加油站",
        actual="推荐了无关餐厅",
    )
    result = derive_case_issue_from_judge(judge)
    assert result.expected == "推荐3个附近的加油站"
    assert result.actual == "推荐了无关餐厅"


# ── AC-7: missing expected/actual ────────────────────────────


def test_derive_missing_expected_actual() -> None:
    """issue has no expected/actual → empty strings."""
    judge = _judge(
        scores={
            "D0": _criterion(0.0, "bad"),
        },
    )
    result = derive_case_issue_from_judge(judge)
    assert result.expected == ""
    assert result.actual == ""


# ── AC-8: the worker injects 5 fields onto the issue top level ──────────────────────────


def test_worker_injects_case_issue() -> None:
    """After derive, the issue dict contains the 5 top-level CaseIssue fields."""
    judge = _judge(
        scores={
            "D0": _criterion(0.0, "POI wrong"),
        },
        expected="expected answer",
        actual="actual answer",
    )
    case_issue = derive_case_issue_from_judge(judge)
    issue = {**judge}
    issue["verdict"] = case_issue.verdict
    issue["dimension"] = case_issue.dimension
    issue["issue_hint"] = case_issue.issue_hint
    issue["expected"] = case_issue.expected
    issue["actual"] = case_issue.actual
    assert issue["verdict"] == "bad"
    assert issue["dimension"] == "D0"
    assert issue["expected"] == "expected answer"
    assert issue["actual"] == "actual answer"
    assert "POI wrong" in issue["issue_hint"]


# ── AC-11: actual normalized (dict → str)──────────────────────


def test_normalize_actual_dict_to_str() -> None:
    """item_output={"text":"..."} → actual is a string."""
    item_output = {"text": "推荐了无关餐厅", "meta": {"lat": 30.0}}
    result = normalize_actual(item_output)
    assert isinstance(result, str)
    assert result == "推荐了无关餐厅"


def test_normalize_actual_answer_key() -> None:
    """item_output has no text but has answer → extract answer."""
    item_output = {"answer": "answer text"}
    result = normalize_actual(item_output)
    assert result == "answer text"


def test_normalize_actual_json_fallback() -> None:
    """item_output has no known field → fall back to JSON serialization."""
    item_output = {"custom_field": "value"}
    result = normalize_actual(item_output)
    assert isinstance(result, str)
    assert "custom_field" in result


def test_normalize_actual_none() -> None:
    """item_output=None → an empty string."""
    assert normalize_actual(None) == ""


def test_normalize_actual_str() -> None:
    """item_output is already a str → return it unchanged."""
    assert normalize_actual("raw text") == "raw text"


# ── edge cases ──────────────────────────────────────────────


def test_derive_empty_criterion_scores() -> None:
    """criterion_scores is an empty dict → verdict=good, dimension=''."""
    judge = _judge(scores={})
    result = derive_case_issue_from_judge(judge)
    assert result.verdict == "good"
    assert result.dimension == ""


def test_derive_excludes_fused_dimension() -> None:
    """The fused dimension does not participate in choosing dimension."""
    judge = _judge(
        scores={
            "D0": _criterion(2.0, "good"),
            "fused": _criterion(0.0, "fused low"),  # It should be excluded
        },
    )
    result = derive_case_issue_from_judge(judge)
    assert result.verdict == "good"
    assert result.dimension == ""


def test_derive_non_dict_input() -> None:
    """A non-dict input → return the default CaseIssue."""
    result = derive_case_issue_from_judge("not a dict")  # type: ignore[arg-type]
    assert isinstance(result, CaseIssue)
    assert result.verdict == "good"
