"""A manually triggered run_id is isolated from a scheduled run_id."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.manual_run_id import (
    ManualRunClaimScope,
    ManualRunIdExhaustedError,
    ManualRunIdTakenError,
    claim_manual_run_id,
    list_manual_run_ids,
    manual_suffix,
    plan_manual_run_id,
    resolve_external_data_date,
    scheduled_run_id,
    scheduled_task_run_id,
    validate_idempotency_key,
)


def test_scheduled_run_id_stays_unsuffixed() -> None:
    assert scheduled_run_id("ieval_external", "20261010") == "ieval_external-20261010"


def test_scheduled_task_run_id_compat_helper_keeps_t_suffix() -> None:
    """The compatibility helper still emits ``-t-``; WebUI schedules now use scheduled_run_id."""
    a = scheduled_task_run_id("ieval_external", "20261010", "task-daily")
    b = scheduled_task_run_id("ieval_external", "20261010", "task-weekly")
    assert a == "ieval_external-20261010-t-task-daily"
    assert b == "ieval_external-20261010-t-task-weekly"
    assert a != b
    assert scheduled_run_id("ieval_external", "20261010") == "ieval_external-20261010"
    with pytest.raises(ValueError, match="task_id"):
        scheduled_task_run_id("ieval_external", "20261010", "  ")


def test_manual_sequence_skips_scheduled_and_remote_config_suffix() -> None:
    existing = [
        "ieval_external-20261010",
        "ieval_external-20261010-q1",
        "ieval_external-20261010-m1",
        "ieval_external-20261010-m-deadbeef",
        "other-20261010-m9",
    ]
    run_id = plan_manual_run_id(
        ManualRunClaimScope(benchmark_name="ieval_external", date="20261010"),
        existing,
    )
    assert run_id == "ieval_external-20261010-m2"
    assert run_id != scheduled_run_id("ieval_external", "20261010")


def test_first_manual_trigger_starts_at_m1() -> None:
    run_id = plan_manual_run_id(
        ManualRunClaimScope(benchmark_name="ieval_external", date="20261010"),
        ["ieval_external-20261010"],
    )
    assert run_id == "ieval_external-20261010-m1"
    assert manual_suffix(run_id, "ieval_external", "20261010") == "m1"


def test_scheduled_retrigger_uses_independent_s_sequence() -> None:
    run_id = plan_manual_run_id(
        ManualRunClaimScope(
            benchmark_name="ieval_external",
            date="20261010",
            namespace="s",
        ),
        [
            "ieval_external-20261010-m9",
            "ieval_external-20261010-s1",
        ],
    )
    assert run_id == "ieval_external-20261010-s2"
    assert manual_suffix(run_id, "ieval_external", "20261010") == "s2"


def test_idempotency_key_is_stable_and_distinct_from_sequence() -> None:
    first = plan_manual_run_id(
        ManualRunClaimScope(
            benchmark_name="ieval_external",
            date="20261010",
            idempotency_key="click-1",
        ),
        ["ieval_external-20261010-m1"],
    )
    second = plan_manual_run_id(
        ManualRunClaimScope(
            benchmark_name="ieval_external",
            date="20261010",
            idempotency_key="click-1",
        ),
        [],
    )
    other = plan_manual_run_id(
        ManualRunClaimScope(
            benchmark_name="ieval_external",
            date="20261010",
            idempotency_key="click-2",
        ),
        [],
    )
    assert first == second
    assert first == "ieval_external-20261010-m-0778bb08"
    assert first.startswith("ieval_external-20261010-m-")
    assert first != other
    assert manual_suffix(first, "ieval_external", "20261010").startswith("m-")

    scheduled = plan_manual_run_id(
        ManualRunClaimScope(
            benchmark_name="ieval_external",
            date="20261010",
            idempotency_key="click-1",
            namespace="s",
        ),
        [],
    )
    assert scheduled.startswith("ieval_external-20261010-s-")
    assert scheduled != first


def test_idempotency_key_rejects_blank_and_spaces() -> None:
    assert validate_idempotency_key(None) is None
    assert validate_idempotency_key("click-1") == "click-1"
    with pytest.raises(ValueError):
        validate_idempotency_key("")
    with pytest.raises(ValueError):
        validate_idempotency_key("has space")


def test_resolve_date_uses_shanghai_calendar_day() -> None:
    # 2026-10-09 16:30 UTC == 2026-10-10 00:30 CST
    now = datetime(2026, 10, 9, 16, 30, tzinfo=UTC)
    assert resolve_external_data_date(None, now=now) == "20261010"
    assert resolve_external_data_date("20260101", now=now) == "20260101"
    assert (
        resolve_external_data_date(
            None,
            now=now,
            storage_path=(
                "ieval-service/benchmark/benchmark_result/20261110/"
                "benchmark_final_results.csv"
            ),
        )
        == "20261110"
    )
    # A literal partition date wins over an explicit date
    assert (
        resolve_external_data_date(
            "20261111",
            now=now,
            storage_path="ieval-service/benchmark/benchmark_result/20261110/x.csv",
        )
        == "20261110"
    )
    # A path that contains a macro is not expanded here; fall back to the explicit date / today (the accept chain expands the macro once)
    assert (
        resolve_external_data_date(
            "20261111",
            now=now,
            storage_path="ieval-service/benchmark/benchmark_result/${yyyymmdd-1}",
        )
        == "20261111"
    )
    assert (
        resolve_external_data_date(
            None,
            now=now,
            storage_path="ieval-service/benchmark/benchmark_result",
        )
        == "20261010"
    )


@pytest.mark.asyncio
async def test_claim_retries_when_sequence_is_taken() -> None:
    seen: list[str] = []
    occupied = ["ieval_external-20261010-m1"]

    async def existing() -> list[str]:
        return list(occupied)

    async def claim(run_id: str) -> str:
        seen.append(run_id)
        if len(seen) == 1:
            occupied.append(run_id)
            raise ManualRunIdTakenError(run_id)
        return run_id

    claimed = await claim_manual_run_id(
        ManualRunClaimScope(benchmark_name="ieval_external", date="20261010"),
        existing_run_ids=existing,
        claim=claim,
        max_attempts=3,
    )
    assert seen == [
        "ieval_external-20261010-m2",
        "ieval_external-20261010-m3",
    ]
    assert claimed == "ieval_external-20261010-m3"


@pytest.mark.asyncio
async def test_claim_stops_after_max_attempts() -> None:
    async def existing() -> list[str]:
        return []

    async def claim(run_id: str) -> str:
        raise ManualRunIdTakenError(run_id)

    with pytest.raises(ManualRunIdExhaustedError):
        await claim_manual_run_id(
            ManualRunClaimScope(benchmark_name="ieval_external", date="20261010"),
            existing_run_ids=existing,
            claim=claim,
            max_attempts=2,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", ["m", "s"])
async def test_list_manual_run_ids_queries_namespace_prefix(namespace: str) -> None:
    """list_manual_run_ids looks up only manual runs already taken in the current namespace."""
    qs = MagicMock()
    qs.limit.return_value = qs
    qs.values_list = AsyncMock(
        return_value=[
            "ieval_external-20261010-m1",
            "ieval_external-20261010-m-deadbeef",
        ],
    )
    with patch(
        "benchmark.runs.manual_run_id.BenchmarkRun.filter",
        return_value=qs,
    ) as filtered:
        rows = await list_manual_run_ids("ieval_external", "20261010", namespace=namespace)
    filtered.assert_called_once_with(
        run_id__startswith=f"ieval_external-20261010-{namespace}",
    )
    qs.limit.assert_called_once()
    assert rows == [
        "ieval_external-20261010-m1",
        "ieval_external-20261010-m-deadbeef",
    ]
