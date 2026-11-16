"""Unit tests for finalize_trigger — the CAS decision boundary / constant routing / conditional-branch SQL.

Locks three fix contracts:
- ``_cas_won`` treats a structurally abnormal execute_query result (None / empty / a single element) as not won,
  and does not crash with IndexError
- tick routing uses the EVAL_TYPE_* constants (not a comparison against a bare string literal)
- the benchmark CAS threshold SQL picks a complete statement with a conditional branch (no f-string concatenation)
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.runs.run_service import BenchmarkService
from benchmark.workers import finalize_trigger
from benchmark.workers.finalize_trigger import (
    EVAL_TYPE_BENCHMARK,
    EVAL_TYPE_EXTERNAL,
    _cas_won,
    tick_and_maybe_finalize,
    try_finalize_benchmark,
)


class TestCasWon:
    """_cas_won: a structurally abnormal execute_query result is treated as not won (no IndexError)."""

    @pytest.mark.parametrize(
        ("rows", "expected"),
        [
            (None, False),
            ((), False),
            ((1,), False),  # Only the affected count, with no rows element
            ((1, []), False),  # rows is an empty list
            ((0, []), False),
            ((1, [("r1",)]), True),
        ],
    )
    def test_boundaries(self, rows: Any, expected: bool) -> None:
        assert _cas_won(rows) is expected


def test_eval_type_constants() -> None:
    """The EVAL_TYPE_* constant values align with the routing literals (a runtime anchor against spelling drift)."""
    assert (EVAL_TYPE_BENCHMARK, EVAL_TYPE_EXTERNAL) == (
        "benchmark",
        "external",
    )


def _make_recorder(calls: list[str], name: str) -> Any:
    async def _record(*_args: Any, **_kwargs: Any) -> bool:
        calls.append(name)
        return True

    return _record


@pytest.mark.asyncio
async def test_tick_routes_by_eval_type_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tick routes to the matching finalize function by the EVAL_TYPE_* constant."""
    calls: list[str] = []
    for fn_name in (
        "try_finalize_benchmark",
        "try_finalize_external",
    ):
        monkeypatch.setattr(finalize_trigger, fn_name, _make_recorder(calls, fn_name))

    svc = MagicMock(spec=BenchmarkService)
    await tick_and_maybe_finalize("r2", EVAL_TYPE_BENCHMARK, benchmark_run_service=svc)
    await tick_and_maybe_finalize("r3", EVAL_TYPE_EXTERNAL, benchmark_run_service=svc)

    assert calls == [
        "try_finalize_benchmark",
        "try_finalize_external",
    ]


@pytest.mark.asyncio
async def test_tick_unknown_eval_type_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered eval_type triggers no finalize (a runtime defense)."""
    calls: list[str] = []
    for fn_name in (
        "try_finalize_benchmark",
        "try_finalize_external",
    ):
        monkeypatch.setattr(finalize_trigger, fn_name, _make_recorder(calls, fn_name))

    await tick_and_maybe_finalize(
        "r1",
        "other",  # type: ignore[arg-type]
        benchmark_run_service=MagicMock(spec=BenchmarkService),
    )
    assert calls == []


def _mock_tortoise_capturing(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, list[Any]]]:
    """Mock the Tortoise connection and capture execute_query's (sql, vals)."""
    captured: list[tuple[str, list[Any]]] = []

    async def _execute_query(
        sql: str,
        vals: list[Any] | None = None,
    ) -> tuple[int, list[Any]]:
        captured.append((sql, vals or []))
        return (1, [("r1",)])

    conn = MagicMock()
    conn.execute_query = _execute_query
    fake = MagicMock()
    fake.get_connection.return_value = conn
    monkeypatch.setattr(finalize_trigger, "Tortoise", fake)
    return captured


def _run_service_with_mode(triage_mode: str) -> MagicMock:
    svc = MagicMock(spec=BenchmarkService)
    run = MagicMock()
    run.config_json = {"triage_mode": triage_mode}
    svc.get_run = AsyncMock(return_value=run)
    svc.load_fused_score_average = AsyncMock(return_value=0.5)
    svc.complete_run = AsyncMock()
    return svc


@pytest.mark.asyncio
async def test_benchmark_cas_sql_fused_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fused: the complete SQL threshold compares only total_samples (not total_evaluators)."""
    captured = _mock_tortoise_capturing(monkeypatch)
    await try_finalize_benchmark("r1", _run_service_with_mode("fused"))
    cas_sql = captured[0][0]
    assert ">= total_samples" in cas_sql
    assert "total_evaluators" not in cas_sql
    # [stop only] the latch bypass can also win the CAS; a normal completion still takes "processed ≥ total"
    assert "stop_requested_at" in cas_sql
    # Emitting the report on stop must atomically confirm there are no running items
    assert "benchmark_run_items" in cas_sql
    assert "status = 'running'" in cas_sql


@pytest.mark.asyncio
async def test_benchmark_cas_sql_per_eval_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """per_eval: the complete SQL threshold compares total_samples * total_evaluators (branch selection)."""
    captured = _mock_tortoise_capturing(monkeypatch)
    await try_finalize_benchmark("r1", _run_service_with_mode("per_eval"))
    cas_sql = captured[0][0]
    assert "total_samples * total_evaluators" in cas_sql
    assert "stop_requested_at" in cas_sql
    assert "benchmark_run_items" in cas_sql
