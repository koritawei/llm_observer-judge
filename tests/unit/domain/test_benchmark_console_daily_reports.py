"""Match-matrix unit tests for the new semantics of ``_daily_report_matches_benchmark`` (the page dataset name).

The new-semantics branch aligns the task registration with the ``stem@registrationId`` composite key stored on ``benchmark_name``
(:func:`benchmark.domain.eval_config.parse_registration_eval_key`, filled in by this MR;
that symbol was missing before, so the new-semantics branch raised ImportError).
``agent_catalog.canonical_agent_name`` lands in a later benchmark-common MR.
The real ``_benchmark_name_aliases`` depends on it too, so an equivalent
stub that swaps -/_ isolates it (fixture autouse). Only this function's own match matrix is verified.
"""

import pytest
from benchmark.records.models.benchmark_run import BenchmarkRun
from benchmark.domain.benchmark_console import (
    DailyReportMatchContext,
    _daily_report_matches_benchmark,
)
from benchmark.domain.eval_config import (
    parse_eval_identity_key,
    parse_registration_eval_key,
    registration_eval_key,
    stored_benchmark_run_name,
)


def _stub_aliases(name: str) -> set[str]:
    """A lightweight alias equivalent to canonical_agent_name (the original value, plus -/_ swapped)."""

    return {name, name.replace("_", "-"), name.replace("-", "_")}


@pytest.fixture(autouse=True)
def _isolate_agent_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "benchmark.domain.benchmark_console._benchmark_name_aliases",
        _stub_aliases,
    )


def _run(dataset_name: str, benchmark_name: str) -> BenchmarkRun:
    return BenchmarkRun(
        run_id="bm_daily_report_test",
        benchmark_name=benchmark_name,
        dataset_name=dataset_name,
    )


def _ctx(**overrides: object) -> DailyReportMatchContext:
    base: dict[str, object] = {
        "agent_name": "example_benchmark",
        "dataset_name": "demo-ds",
        "registration_id": None,
        "require_version_isolation": False,
    }
    base.update(overrides)
    return DailyReportMatchContext(**base)  # type: ignore[arg-type]


def test_parse_registration_eval_key_shapes() -> None:
    """How a composite key splits into stem@reg, and how a bare key / empty key / broken key is normalized."""

    assert parse_registration_eval_key("agx_example_1@reg-123") == ("agx_example_1", "reg-123")
    assert parse_registration_eval_key("reg-123") == ("reg-123", None)
    assert parse_registration_eval_key("") == ("", None)
    assert parse_registration_eval_key(None) == ("", None)
    # A broken composite key (stem@ / @reg) is not split; the whole segment is treated as a bare key
    assert parse_registration_eval_key("agx_example_1@") == ("agx_example_1@", None)
    assert parse_registration_eval_key("@reg-123") == ("@reg-123", None)


def test_stream_only_identity_key_uses_explicit_prefix() -> None:
    """A stream with no registration round-trips without loss; an ordinary two-segment key still means only reg."""

    key = registration_eval_key("agx_example", None, "dataset-a")
    assert key == "agx_example@stream:dataset-a"
    assert parse_eval_identity_key(key) == ("agx_example", None, "dataset-a")
    assert parse_registration_eval_key(key) == ("agx_example", None)
    assert parse_eval_identity_key("agx_example@dataset-a") == (
        "agx_example",
        "dataset-a",
        None,
    )
    assert registration_eval_key("agx_example", "reg-1", "dataset-a") == (
        "agx_example@reg-1@dataset-a"
    )


def test_stored_benchmark_run_name_uses_injected_agent_stem() -> None:
    """The stored composite key uses the stem passed in by the services layer; domain does not maintain the canonical map."""

    assert (
        stored_benchmark_run_name(
            logical_name="example_benchmark",
            registration_id="reg-1",
            evaluator_stem="example",
        )
        == "example@reg-1"
    )
    assert (
        stored_benchmark_run_name(
            logical_name="search_eval",
            registration_id="reg-1",
            evaluator_stem="search_ai_card",
        )
        == "search_ai_card@reg-1"
    )
    assert stored_benchmark_run_name(logical_name="example_benchmark") == "example_benchmark"


def test_new_semantics_matches_by_versioned_column_key() -> None:
    """The column already stores ``stem@registrationId``: it belongs when that lines up with the task registration id."""

    run = _run("demo-ds", "agx_example_1@reg-123")
    assert (
        _daily_report_matches_benchmark(
            run,
            {},
            ctx=_ctx(registration_id="reg-123"),
        )
        is True
    )
    # The task-side registration id does not match the column's registration segment → it does not belong
    assert (
        _daily_report_matches_benchmark(
            run,
            {},
            ctx=_ctx(registration_id="reg-456"),
        )
        is False
    )


def test_new_semantics_matches_by_bare_registration_id() -> None:
    """The column stores a bare registrationId (no stem): it still aligns by registration id."""

    run = _run("demo-ds", "reg-123")
    assert (
        _daily_report_matches_benchmark(
            run,
            {},
            ctx=_ctx(registration_id="reg-123"),
        )
        is True
    )


def test_new_semantics_soft_matches_by_component() -> None:
    """No registration id: config.benchmark_component soft-matches the agent name."""

    run = _run("demo-ds", "agx_example_1")
    config = {"benchmark_component": "example-benchmark"}
    assert (
        _daily_report_matches_benchmark(
            run,
            config,
            ctx=_ctx(),
        )
        is True
    )
    # The component and the agent name have nothing in common → it does not belong
    assert (
        _daily_report_matches_benchmark(
            run,
            {"benchmark_component": "someone_else"},
            ctx=_ctx(),
        )
        is False
    )


def test_new_semantics_rejects_dataset_mismatch() -> None:
    """The column's dataset name does not match the dataset name registered on the task page → it does not belong."""

    run = _run("other-ds", "agx_example_1@reg-123")
    assert (
        _daily_report_matches_benchmark(
            run,
            {},
            ctx=_ctx(registration_id="reg-123"),
        )
        is False
    )


def test_new_semantics_requires_exact_stream_identity() -> None:
    """Different streams of the same dataset name and the same registration must not share a daily report."""
    run = _run("demo-ds", "agx_example_1@reg-123")
    ctx = _ctx(registration_id="reg-123", stream_instance_id="ds-rt-a")
    assert _daily_report_matches_benchmark(
        run,
        {"stream_instance_id": "ds-rt-a"},
        ctx=ctx,
    )
    assert not _daily_report_matches_benchmark(
        run,
        {"stream_instance_id": "ds-rt-b"},
        ctx=ctx,
    )
    assert not _daily_report_matches_benchmark(run, {}, ctx=ctx)


def test_stamped_run_excluded_when_registration_differs() -> None:
    """The run is already stamped with a registration that does not match the task registration id → drop it."""

    run = _run("demo-ds", "agx_example_1")
    config = {"agent_registration_id": "reg-999"}
    assert (
        _daily_report_matches_benchmark(
            run,
            config,
            ctx=_ctx(registration_id="reg-123"),
        )
        is False
    )


def test_amap_agent_benchmark_merges_legacy_daily_column() -> None:
    """The amap_agent_benchmark task page merges the old column name amap_agent_daily; other components do not merge."""

    legacy = _run("trace_pipeline:kafka", "amap_agent_daily")
    assert (
        _daily_report_matches_benchmark(
            legacy,
            {"trace_daily_report": True},
            ctx=_ctx(
                agent_name="amap_agent_benchmark",
                dataset_name="amap-agent-sampling-2026-10-30-1793370626586",
                registration_id="ba-82a6e3f2-724c-4e82-8c4d-111935b81bc7",
                require_version_isolation=True,
            ),
        )
        is True
    )
    assert (
        _daily_report_matches_benchmark(
            legacy,
            {"trace_daily_report": True},
            ctx=_ctx(
                agent_name="search_eval",
                dataset_name="amap-agent-sampling-2026-10-30-1793370626586",
            ),
        )
        is False
    )
