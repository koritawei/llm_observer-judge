"""strip_per_criterion_failed_count regression — a non-Pydantic / non-mapping input must not raise TypeError.

On the callback path the value type of criterion_metrics is not under our control (None/int/str can all appear).
Before the fix ``dict(stats)`` raised TypeError immediately on a non-iterable input and aborted the whole callback.
"""

from benchmark.runs.common.config_helpers import strip_per_criterion_failed_count
from pydantic import BaseModel


class _Stats(BaseModel):
    failed_count: int = 3
    total: int = 10


class _PlainObject:
    def __init__(self) -> None:
        self.failed_count = 1
        self.total = 7


def test_pydantic_model_strips_failed_count() -> None:
    assert strip_per_criterion_failed_count(_Stats()) == {"total": 10}


def test_dict_input_strips_failed_count_without_mutating_caller() -> None:
    src = {"failed_count": 2, "total": 5}
    assert strip_per_criterion_failed_count(src) == {"total": 5}
    assert src == {"failed_count": 2, "total": 5}


def test_object_with_dunder_dict() -> None:
    assert strip_per_criterion_failed_count(_PlainObject()) == {"total": 7}


def test_none_scalar_and_uniterable_inputs_return_empty_dict() -> None:
    assert strip_per_criterion_failed_count(None) == {}
    assert strip_per_criterion_failed_count(42) == {}
    assert strip_per_criterion_failed_count("nope") == {}
    assert strip_per_criterion_failed_count([]) == {}


# ── resolve_oss_prefix（D4）─────────────────────────────────────────────────

from benchmark.runs.common.config_helpers import resolve_oss_prefix  # noqa: E402

_DEFAULT = "oss://agent-eval-oss/default/prefix"


def test_empty_raw_returns_default() -> None:
    assert resolve_oss_prefix(None, default_prefix=_DEFAULT) == _DEFAULT
    assert resolve_oss_prefix("", default_prefix=_DEFAULT) == _DEFAULT
    assert resolve_oss_prefix("   ", default_prefix=_DEFAULT) == _DEFAULT


def test_complete_prefix_is_kept_without_public_prefix() -> None:
    """The prefix is already the full path: do not join a common prefix (that would build a doubled path and NoSuchKey)."""
    assert (
        resolve_oss_prefix("search-eval/benchmark_result/20261023", default_prefix=_DEFAULT)
        == "search-eval/benchmark_result/20261023"
    )
    assert (
        resolve_oss_prefix("search-eval/benchmark_result", default_prefix=_DEFAULT)
        == "search-eval/benchmark_result"
    )


def test_full_path_with_scheme_is_kept() -> None:
    full = "oss://other-bucket/a/b"
    assert resolve_oss_prefix(full, default_prefix=_DEFAULT) == full


def test_absolute_path_is_kept() -> None:
    assert resolve_oss_prefix("/abs/path", default_prefix=_DEFAULT) == "/abs/path"


def test_dot_dot_segment_rejected_in_all_forms() -> None:
    import pytest

    with pytest.raises(ValueError, match=r"\.\."):
        resolve_oss_prefix("../escape", default_prefix=_DEFAULT)
    with pytest.raises(ValueError, match=r"\.\."):
        resolve_oss_prefix("oss://bucket/a/../b", default_prefix=_DEFAULT)
    with pytest.raises(ValueError, match=r"\.\."):
        resolve_oss_prefix("a/../../b", default_prefix=_DEFAULT)


def test_empty_segment_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="empty"):
        resolve_oss_prefix("oss://bucket/a//b", default_prefix=_DEFAULT)
    with pytest.raises(ValueError, match="empty"):
        resolve_oss_prefix("a//b", default_prefix=_DEFAULT)
