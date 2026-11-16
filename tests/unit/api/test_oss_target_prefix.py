"""OSS object-key normalization for imported datasets: the prefix is the full path, and no common prefix is joined on.

Regression point: after ``search-eval/benchmark_result/${yyyymmdd}/`` expands, it maps directly to the date
partition under the bucket root. Joining ``ieval-service/benchmark/benchmark_result`` on top fetches a
key that does not exist (log ``oss.load_daily no_data ... NoSuchKey``).
"""

from types import SimpleNamespace

import pytest
from benchmark.routes.benchmark import _macro_oss_object_key, _resolve_oss_target
from benchmark.runs.common.config_helpers import resolve_oss_prefix
from benchmark.runs.schema import ExternalBenchmarkRequest

_DEFAULT = "oss://agent-eval-oss/ieval-service/benchmark/benchmark_result"


def test_macro_object_key_keeps_business_prefix_only() -> None:
    effective = resolve_oss_prefix(
        "search-eval/benchmark_result/20261023",
        default_prefix=_DEFAULT,
    )
    assert effective == "search-eval/benchmark_result/20261023"
    assert (
        _macro_oss_object_key(effective)
        == "search-eval/benchmark_result/20261023/benchmark_final_results.csv"
    )
    assert "ieval-service" not in _macro_oss_object_key(effective)


def test_macro_object_key_strips_oss_scheme() -> None:
    effective = resolve_oss_prefix(
        "oss://agent-eval-oss/search-eval/benchmark_result/20261023",
        default_prefix=_DEFAULT,
    )
    assert (
        _macro_oss_object_key(effective)
        == "search-eval/benchmark_result/20261023/benchmark_final_results.csv"
    )


def test_resolve_oss_target_pins_preexpanded_immediate_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Immediate trigger: after the page pre-expands the macro and forwards a dated prefix, the accept side pins the object key.

    Regression: if it is not pinned, the loader appends data_date again →
    ``.../20261103/20261103/benchmark_final_results.csv``.
    """
    from benchmark.routes import benchmark as benchmark_api

    monkeypatch.setattr(
        benchmark_api,
        "get_default_download_prefix",
        lambda: _DEFAULT,
    )
    document = SimpleNamespace(
        id="ds-search",
        source="oss",
        oss_prefix="search-eval/benchmark_result/${yyyymmdd}",
        oss_partition_date=None,
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval_v2",
        date="20261103",
        # The page's buildManualRunRequest has already expanded the macro
        oss_prefix="search-eval/benchmark_result/20261103",
    )
    effective = _resolve_oss_target(payload, document, base_date="20261103")
    assert effective == "search-eval/benchmark_result/20261103"
    assert payload.oss_object_key == (
        "search-eval/benchmark_result/20261103/benchmark_final_results.csv"
    )


@pytest.mark.parametrize(
    ("doc_macro", "literal", "poison_base", "forbidden"),
    [
        ("${yyyymmdd-1}", "20261110", "20261110", "20261109"),
        ("${yyyymmdd+1}", "20261112", "20261112", "20261113"),
        ("${yyyymmdd}", "20261111", "20261110", "20261110"),
    ],
)
def test_resolve_oss_target_keeps_request_literal_when_document_has_macro(
    monkeypatch: pytest.MonkeyPatch,
    doc_macro: str,
    literal: str,
    poison_base: str,
    forbidden: str,
) -> None:
    """When the request is already a literal partition date, trust the request; the document's ``±n`` / today macros must not recompute it.

    ``poison_base`` simulates treating an already-written partition date as the macro base again. The old logic
    applied a second offset for ``-1``/``+1``, and for ``${yyyymmdd}`` could rewrite the literal date to the wrong base date.
    """
    from benchmark.routes import benchmark as benchmark_api

    monkeypatch.setattr(
        benchmark_api,
        "get_default_download_prefix",
        lambda: _DEFAULT,
    )
    document = SimpleNamespace(
        id="ds-offset",
        source="oss",
        oss_prefix=f"search-eval/benchmark_result/{doc_macro}",
        oss_partition_date=None,
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval_v2",
        date="20261111",
        oss_prefix=f"search-eval/benchmark_result/{literal}",
    )
    effective = _resolve_oss_target(payload, document, base_date=poison_base)
    assert effective == f"search-eval/benchmark_result/{literal}"
    assert payload.oss_object_key == (
        f"search-eval/benchmark_result/{literal}/benchmark_final_results.csv"
    )
    if forbidden != literal:
        assert forbidden not in (payload.oss_object_key or "")


def test_resolve_oss_target_expands_request_macro_without_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no dataset document, still expand macros on the request side (direct API / RemoteConfig calls)."""
    from benchmark.routes import benchmark as benchmark_api

    monkeypatch.setattr(
        benchmark_api,
        "get_default_download_prefix",
        lambda: _DEFAULT,
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval_v2",
        date="20261103",
        oss_prefix="search-eval/benchmark_result/${yyyymmdd}",
    )
    effective = _resolve_oss_target(payload, None, base_date="20261103")
    assert effective == "search-eval/benchmark_result/20261103"
    assert payload.oss_object_key == (
        "search-eval/benchmark_result/20261103/benchmark_final_results.csv"
    )


def test_resolve_oss_target_pins_offset_literal_without_matching_data_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A literal offset-date tail ≠ data_date is also pinned, to avoid a doubled date."""
    from benchmark.routes import benchmark as benchmark_api

    monkeypatch.setattr(
        benchmark_api,
        "get_default_download_prefix",
        lambda: _DEFAULT,
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval_v2",
        date="20261103",
        oss_prefix="search-eval/benchmark_result/20261102",
    )
    effective = _resolve_oss_target(payload, None, base_date="20261103")
    assert effective == "search-eval/benchmark_result/20261102"
    assert payload.oss_object_key == (
        "search-eval/benchmark_result/20261102/benchmark_final_results.csv"
    )


def test_resolve_external_run_identity_expands_offset_macro_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run-external: ``${yyyymmdd-1}`` expands once relative to the explicit date, with no second offset."""
    from benchmark.routes import benchmark as benchmark_api
    from benchmark.routes.benchmark import _resolve_external_run_identity

    monkeypatch.setattr(
        benchmark_api,
        "get_default_download_prefix",
        lambda: _DEFAULT,
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="ieval_external",
        date="20260905",
        oss_prefix="ieval-service/benchmark/benchmark_result/${yyyymmdd-1}",
    )
    identity = _resolve_external_run_identity(
        payload,
        None,
        assigned_run_id=None,
        data_date=None,
    )
    assert identity.date == "20260904"
    assert identity.run_id == "ieval_external-20260904"
    assert identity.effective_oss_prefix.endswith("/20260904")
    assert payload.oss_object_key == (
        "ieval-service/benchmark/benchmark_result/20260904/benchmark_final_results.csv"
    )
    assert "20260903" not in (payload.oss_object_key or "")


def test_resolve_external_run_identity_assigned_keeps_preexpanded_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """manual-runs expands macros before taking a sequence number: accept does not expand the offset date a second time."""
    from benchmark.routes import benchmark as benchmark_api
    from benchmark.routes.benchmark import _resolve_external_run_identity

    monkeypatch.setattr(
        benchmark_api,
        "get_default_download_prefix",
        lambda: _DEFAULT,
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="ieval_external",
        date="20260904",
        oss_prefix="ieval-service/benchmark/benchmark_result/20260904",
        oss_object_key=(
            "ieval-service/benchmark/benchmark_result/20260904/"
            "benchmark_final_results.csv"
        ),
    )
    identity = _resolve_external_run_identity(
        payload,
        None,
        assigned_run_id="ieval_external-20260904-m1",
        data_date="20260904",
    )
    assert identity.date == "20260904"
    assert identity.run_id == "ieval_external-20260904-m1"
    assert identity.effective_oss_prefix.endswith("/20260904")
    assert payload.oss_object_key == (
        "ieval-service/benchmark/benchmark_result/20260904/benchmark_final_results.csv"
    )


@pytest.mark.parametrize(
    ("macro", "expected", "forbidden"),
    [
        ("${yyyymmdd-1}", "20261110", "20261109"),
        ("${yyyymmdd+1}", "20261112", "20261113"),
        ("${yyyymmdd}", "20261111", "20261112"),
    ],
)
def test_resolve_external_run_identity_macros_expand_once_on_20261111(
    monkeypatch: pytest.MonkeyPatch,
    macro: str,
    expected: str,
    forbidden: str,
) -> None:
    """Passing 20261111 into ``±n`` / the today macro expands once relative to the explicit date only."""
    from benchmark.routes import benchmark as benchmark_api
    from benchmark.routes.benchmark import _resolve_external_run_identity

    monkeypatch.setattr(
        benchmark_api,
        "get_default_download_prefix",
        lambda: _DEFAULT,
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="ieval_external",
        date="20261111",
        oss_prefix=f"ieval-service/benchmark/benchmark_result/{macro}",
    )
    identity = _resolve_external_run_identity(
        payload,
        None,
        assigned_run_id=None,
        data_date=None,
    )
    assert identity.date == expected
    assert identity.run_id == f"ieval_external-{expected}"
    assert payload.oss_object_key == (
        f"ieval-service/benchmark/benchmark_result/{expected}/"
        "benchmark_final_results.csv"
    )
    assert forbidden not in (payload.oss_object_key or "")


def test_resolve_external_run_identity_assigned_with_doc_macro_keeps_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the sequence number is taken the path is already 20261110; if the document still contains a macro it must not expand again into 20261109."""
    from benchmark.routes import benchmark as benchmark_api
    from benchmark.routes.benchmark import _resolve_external_run_identity

    monkeypatch.setattr(
        benchmark_api,
        "get_default_download_prefix",
        lambda: _DEFAULT,
    )
    document = SimpleNamespace(
        id="ds-ieval",
        source="oss",
        oss_prefix="ieval-service/benchmark/benchmark_result/${yyyymmdd-1}",
        oss_partition_date=None,
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="ieval_external",
        date="20261110",
        oss_prefix="ieval-service/benchmark/benchmark_result/20261110",
        oss_object_key=(
            "ieval-service/benchmark/benchmark_result/20261110/"
            "benchmark_final_results.csv"
        ),
    )
    identity = _resolve_external_run_identity(
        payload,
        document,
        assigned_run_id="ieval_external-20261110-s1",
        data_date="20261110",
    )
    assert identity.date == "20261110"
    assert payload.oss_object_key == (
        "ieval-service/benchmark/benchmark_result/20261110/benchmark_final_results.csv"
    )
    assert "20261109" not in (payload.oss_prefix or "")
    assert "20261109" not in (payload.oss_object_key or "")
