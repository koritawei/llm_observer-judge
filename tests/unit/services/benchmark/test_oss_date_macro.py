"""Date macros in an OSS path prefix: expansion, offsets, and rejection of an illegal macro.

Semantics: ``${yyyymmdd}`` = the base day (this run's data date), ``${yyyymmdd+n}`` =
n days later, ``${yyyymmdd-n}`` = n days earlier. After expansion the prefix is the data-file directory.
"""

import pytest
from benchmark.runs.common.oss_date_macro import (
    expand_date_macros,
    find_invalid_date_macro,
    has_date_macro,
    object_key_for_dated_prefix,
    partition_date_from_path,
    prepare_oss_prefix_for_load,
    trailing_yyyymmdd,
    validate_date_macros,
)

_BASE = "20261021"


def test_has_date_macro() -> None:
    assert has_date_macro("agent-eval/x/${yyyymmdd}")
    assert has_date_macro("agent-eval/x/${yyyymmdd-1}")
    assert not has_date_macro("agent-eval/x/20261021")
    assert not has_date_macro(None)
    assert not has_date_macro("")


def test_expand_current_date_keeps_prefix_shape() -> None:
    assert (
        expand_date_macros("oss://agent-eval-oss/amap/${yyyymmdd}", base_date=_BASE)
        == "oss://agent-eval-oss/amap/20261021"
    )


def test_expand_offset_days_across_month_and_year() -> None:
    assert expand_date_macros("p/${yyyymmdd-1}", base_date=_BASE) == "p/20261020"
    assert expand_date_macros("p/${yyyymmdd+3}", base_date=_BASE) == "p/20261024"
    # Across a month / across a year
    assert expand_date_macros("p/${yyyymmdd-21}", base_date=_BASE) == "p/20260930"
    assert expand_date_macros("p/${yyyymmdd+2}", base_date="20261231") == "p/20270102"
    # An offset of 0 is the same as today
    assert expand_date_macros("p/${yyyymmdd+0}", base_date=_BASE) == "p/20261021"


def test_expand_multiple_macros_in_one_prefix() -> None:
    assert (
        expand_date_macros("p/${yyyymmdd-1}/cmp/${yyyymmdd}", base_date=_BASE)
        == "p/20261020/cmp/20261021"
    )


def test_no_macro_returns_text_unchanged() -> None:
    assert expand_date_macros("p/20261021", base_date=_BASE) == "p/20261021"
    assert expand_date_macros("  p/x  ", base_date=_BASE) == "p/x"
    assert expand_date_macros(None, base_date=_BASE) == ""


@pytest.mark.parametrize(
    "raw",
    [
        "${yyyymmdd+}",
        "${yyyymmdd-}",
        "${yyyymmdd+abc}",
        "${yyyymmddx}",
        "${ymd}",
        "${yyyymmdd + 3}",
        "p/${yyyymmdd",
    ],
)
def test_invalid_macro_rejected(raw: str) -> None:
    assert find_invalid_date_macro(raw) is not None
    with pytest.raises(ValueError, match="非法日期宏"):
        validate_date_macros(raw)
    with pytest.raises(ValueError, match="非法日期宏"):
        expand_date_macros(raw, base_date=_BASE)


def test_invalid_base_date_rejected() -> None:
    with pytest.raises(ValueError, match="基准日"):
        expand_date_macros("p/${yyyymmdd}", base_date="2026-10-21")


def test_trailing_yyyymmdd_any_partition_date() -> None:
    assert trailing_yyyymmdd("search-eval/benchmark_result/20261102") == "20261102"
    assert trailing_yyyymmdd("oss://bkt/a/20261102/") == "20261102"
    assert trailing_yyyymmdd("search-eval/benchmark_result") is None
    assert trailing_yyyymmdd("a/20261199") is None  # An illegal calendar day
    assert trailing_yyyymmdd("a/2026110") is None


def test_partition_date_from_path_reads_dir_or_file() -> None:
    assert (
        partition_date_from_path("ieval-service/benchmark/benchmark_result/20261111")
        == "20261111"
    )
    assert (
        partition_date_from_path(
            "ieval-service/benchmark/benchmark_result/20261110/benchmark_final_results.csv",
        )
        == "20261110"
    )
    assert (
        partition_date_from_path(
            "oss://agent-eval-oss/ieval-service/benchmark/benchmark_result/20261110/xxx.csv",
        )
        == "20261110"
    )
    assert partition_date_from_path("ieval-service/benchmark/benchmark_result") is None
    assert partition_date_from_path(None) is None


def test_object_key_for_dated_prefix() -> None:
    assert (
        object_key_for_dated_prefix("search-eval/benchmark_result/20261102")
        == "search-eval/benchmark_result/20261102/benchmark_final_results.csv"
    )
    assert object_key_for_dated_prefix("search-eval/benchmark_result") is None


def test_prepare_oss_prefix_for_load_expands_and_pins() -> None:
    prefix, key = prepare_oss_prefix_for_load(
        "search-eval/benchmark_result/${yyyymmdd-1}",
        base_date="20261103",
    )
    assert prefix == "search-eval/benchmark_result/20261102"
    assert key == "search-eval/benchmark_result/20261102/benchmark_final_results.csv"

    plain, plain_key = prepare_oss_prefix_for_load(
        "search-eval/benchmark_result",
        base_date="20261103",
    )
    assert plain == "search-eval/benchmark_result"
    assert plain_key is None

    dated, dated_key = prepare_oss_prefix_for_load(
        "search-eval/benchmark_result/20261102",
        base_date="20261103",
    )
    assert dated == "search-eval/benchmark_result/20261102"
    assert dated_key == ("search-eval/benchmark_result/20261102/benchmark_final_results.csv")
