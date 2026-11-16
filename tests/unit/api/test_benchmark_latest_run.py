"""Unit tests for GET /v1/benchmarks/runs/latest endpoint.

Verifies:
- Without run_id: returns the most recent run (list_runs limit=1)
- With run_id: returns the specific run (get_run)
- 404 when run_id not found
- 404 when no runs exist at all
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.routes import benchmark as api
from fastapi import HTTPException


def _make_run(
    run_id: str = "bm_test_001",
    dataset_name: str = "test-dataset",
    status: str = "completed",
    config_json: dict[str, Any] | None = None,
    report_url: str | None = None,
) -> SimpleNamespace:
    """Create a fake BenchmarkRun-like object."""
    report_meta: dict[str, Any] = {}
    if report_url is not None:
        report_meta["report_url"] = report_url
    return SimpleNamespace(
        run_id=run_id,
        dataset_name=dataset_name,
        status=status,
        config_json=config_json or {"sample_size": 100},
        report_meta_json=report_meta,
        created_at=datetime(2025, 1, 15, 12, 0, tzinfo=UTC),
    )


def _mock_service(
    *,
    get_run_return: Any = None,
    list_runs_return: list[Any] | None = None,
) -> MagicMock:
    """Create a mock BenchmarkService with configurable returns."""
    svc = MagicMock()
    svc.get_run = AsyncMock(return_value=get_run_return)
    svc.list_runs = AsyncMock(return_value=list_runs_return or [])
    return svc


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_latest_run_without_run_id_returns_latest() -> None:
    """Without run_id, calls list_runs(limit=1) and returns the first result."""
    run = _make_run(run_id="bm_latest_001", report_url="https://oss.example.com/r.html")
    svc = _mock_service(list_runs_return=[run])

    resp = await api.get_latest_run(benchmark_service=svc)

    assert resp["run_id"] == "bm_latest_001"
    assert resp["dataset_name"] == "test-dataset"
    assert resp["status"] == "completed"
    assert resp["config_json"] == {"sample_size": 100}
    assert resp["report_url"] == "https://oss.example.com/r.html"
    assert resp["created_at"] is not None

    # Verify list_runs was called with limit=1, get_run was NOT called
    svc.list_runs.assert_awaited_once()
    call_kwargs = svc.list_runs.call_args.kwargs
    assert call_kwargs["limit"] == 1
    svc.get_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_latest_run_with_run_id_returns_that_run() -> None:
    """With run_id, calls get_run(run_id) and returns that specific run."""
    run = _make_run(
        run_id="bm_target_002",
        dataset_name="specific-dataset",
        status="running",
        report_url="https://oss.example.com/target.html",
    )
    svc = _mock_service(get_run_return=run)

    resp = await api.get_latest_run(benchmark_service=svc, run_id="bm_target_002")

    assert resp["run_id"] == "bm_target_002"
    assert resp["dataset_name"] == "specific-dataset"
    assert resp["status"] == "running"
    assert resp["report_url"] == "https://oss.example.com/target.html"

    # Verify get_run was called with the specified run_id, list_runs was NOT called
    svc.get_run.assert_awaited_once_with("bm_target_002")
    svc.list_runs.assert_not_awaited()


@pytest.mark.asyncio
async def test_latest_run_with_run_id_not_found_raises_404() -> None:
    """With run_id that doesn't exist, raises 404."""
    svc = _mock_service(get_run_return=None)

    with pytest.raises(HTTPException) as exc_info:
        await api.get_latest_run(benchmark_service=svc, run_id="nonexistent_run")

    assert exc_info.value.status_code == 404
    assert "No benchmark runs found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_latest_run_no_runs_exist_raises_404() -> None:
    """Without run_id and no runs in DB, raises 404."""
    svc = _mock_service(list_runs_return=[])

    with pytest.raises(HTTPException) as exc_info:
        await api.get_latest_run(benchmark_service=svc)

    assert exc_info.value.status_code == 404
    assert "No benchmark runs found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_latest_run_report_url_none_when_no_report_meta() -> None:
    """report_url is None when report_meta_json is empty/not a dict."""
    run = _make_run(report_url=None)
    # report_meta_json is {} when report_url is None (per _make_run)
    svc = _mock_service(list_runs_return=[run])

    resp = await api.get_latest_run(benchmark_service=svc)

    assert resp["report_url"] is None


@pytest.mark.asyncio
async def test_latest_run_with_run_id_report_url_none() -> None:
    """With run_id, report_url is None when report_meta_json has no report_url key."""
    run = _make_run(report_url=None)
    svc = _mock_service(get_run_return=run)

    resp = await api.get_latest_run(benchmark_service=svc, run_id="bm_test_001")

    assert resp["report_url"] is None
    assert resp["run_id"] == "bm_test_001"
