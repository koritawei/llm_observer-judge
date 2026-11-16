"""Unit tests for normalize_case_identity (D5: case_id column priority / uniqueness / require_case_id)."""

import pytest
from benchmark.runs.common.parse_interceptor import (
    CaseIdentityScope,
    normalize_case_identity,
)


def _scope(seq: int, **overrides: object) -> CaseIdentityScope:
    values: dict[str, object] = {"run_id": "run-1", "seq": seq}
    values.update(overrides)
    return CaseIdentityScope(**values)  # type: ignore[arg-type]


def test_case_id_column_wins_over_id_and_task_id() -> None:
    identity = normalize_case_identity(
        {"case_id": "c-1", "id": "i-1", "task_id": "t-1"},
        _scope(0),
    )
    assert identity.case_id == "c-1"
    assert identity.id == "c-1"
    assert identity.item_id == "run-1:0"


def test_id_column_wins_over_task_id_when_case_id_missing() -> None:
    identity = normalize_case_identity(
        {"id": "i-1", "task_id": "t-1"},
        _scope(2),
    )
    assert identity.case_id == "i-1"
    assert identity.item_id == "run-1:2"


def test_task_id_used_when_case_id_and_id_missing() -> None:
    identity = normalize_case_identity(
        {"task_id": "t-1", "query": "q"},
        _scope(1),
    )
    assert identity.case_id == "t-1"
    assert identity.item_id == "run-1:1"


def test_blank_values_are_skipped() -> None:
    identity = normalize_case_identity(
        {"case_id": "  ", "id": "", "task_id": "t-1"},
        _scope(0),
    )
    assert identity.case_id == "t-1"


def test_fallback_to_seq_form_when_all_missing() -> None:
    identity = normalize_case_identity({"query": "q"}, _scope(7))
    assert identity.case_id == "run-1:7"
    assert identity.id == "run-1:7"
    assert identity.item_id == "run-1:7"


def test_non_string_column_value_is_coerced() -> None:
    identity = normalize_case_identity({"id": 10086}, _scope(0))
    assert identity.case_id == "10086"


def test_require_case_id_raises_when_all_missing() -> None:
    with pytest.raises(ValueError, match="missing all case_id source columns"):
        normalize_case_identity(
            {"query": "q"},
            _scope(0, require_case_id=True),
        )


def test_missing_columns_fall_back_when_not_required() -> None:
    identity = normalize_case_identity(
        {"query": "q"},
        _scope(0, require_case_id=False),
    )
    assert identity.case_id == "run-1:0"


def test_duplicate_case_id_falls_back_to_seq_form() -> None:
    seen: set[str] = set()
    first = normalize_case_identity(
        {"case_id": "c-1"},
        _scope(0, seen_case_ids=seen),
    )
    dup = normalize_case_identity(
        {"case_id": "c-1"},
        _scope(1, seen_case_ids=seen),
    )
    assert first.case_id == "c-1"
    assert dup.case_id == "run-1:1"
    assert dup.item_id == "run-1:1"
    # A fallback no longer occupies the case_id slot; a later equal value still falls back
    third = normalize_case_identity(
        {"case_id": "c-1"},
        _scope(2, seen_case_ids=seen),
    )
    assert third.case_id == "run-1:2"


def test_legacy_keyword_scope_remains_compatible() -> None:
    """Legacy keyword calls of the dataset / benchmark / OSS loader stay usable during the migration."""
    seen: set[str] = set()
    identity = normalize_case_identity(
        {"case_id": "legacy-case"},
        run_id="legacy-run",
        seq=4,
        require_case_id=True,
        seen_case_ids=seen,
    )

    assert identity.id == "legacy-case"
    assert identity.item_id == "legacy-run:4"
    assert seen == {"legacy-case"}


def test_item_id_is_always_run_id_seq_form() -> None:
    """item_id does not use {run_id}:{case_id} (D5 decision: the callback lookup key is not polluted by a business value)."""
    identity = normalize_case_identity(
        {"case_id": "业务/特殊字符"},
        _scope(3),
    )
    assert identity.item_id == "run-1:3"
    assert identity.case_id == "业务/特殊字符"
