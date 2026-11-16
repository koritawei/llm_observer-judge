"""benchmark.common.utils — run_id parsing / to_dict type safety."""

from dataclasses import dataclass
from typing import Any

import pytest
from benchmark.runs.common.utils import extract_ds, to_dict


def test_extract_ds_plain_date_suffix() -> None:
    assert extract_ds("ieval_external-20260812") == "20260812"


def test_extract_ds_with_run_sequence_suffix() -> None:
    """Several triggers on the same day: ...-YYYYMMDD-NN must take the date segment, not NN."""
    assert extract_ds("ieval_external-20260812-02") == "20260812"
    assert extract_ds("ieval_external-20260812-00") == "20260812"


def test_extract_ds_ext_prefix() -> None:
    assert extract_ds("ext-bench-20260812-01") == "20260812"


def test_extract_ds_missing_date_raises() -> None:
    """A run_id with no date segment must fail explicitly; do not silently fall back to "today" and query the wrong partition."""
    with pytest.raises(ValueError, match="不含 YYYYMMDD 日期段"):
        extract_ds("run_1")
    with pytest.raises(ValueError, match="不含 YYYYMMDD 日期段"):
        extract_ds("")
    # Eight digits that are not all numeric do not count as a date segment
    with pytest.raises(ValueError, match="不含 YYYYMMDD 日期段"):
        extract_ds("bench-abcdefgh")


@dataclass
class _Row:
    ds: str
    score: float


class _Slotted:
    __slots__ = ("ds",)

    def __init__(self, ds: str) -> None:
        self.ds = ds


def test_to_dict_dict_passthrough() -> None:
    """A dict is returned as-is (the same object, not a copy)."""
    d = {"ds": "20260812", "score": 1}
    assert to_dict(d) is d


def test_to_dict_instance_attributes() -> None:
    """A plain object / dataclass instance uses __dict__."""
    row = _Row(ds="20260812", score=1.5)
    assert to_dict(row) == {"ds": "20260812", "score": 1.5}


def test_to_dict_class_object_returns_empty() -> None:
    """When the class itself is passed, __dict__ is a mappingproxy (not a dict), and that type must not leak out as the return type."""
    out: dict[str, Any] = to_dict(_Row)
    assert isinstance(out, dict)  # Contract: the return value must be a real dict
    assert out == {}


def test_to_dict_slotted_and_scalars_return_empty() -> None:
    """A __slots__ instance (no __dict__) and a scalar both fall back to an empty dict, and do not raise."""
    assert to_dict(_Slotted("20260812")) == {}
    assert to_dict(None) == {}
    assert to_dict(42) == {}
    assert to_dict("20260812") == {}
