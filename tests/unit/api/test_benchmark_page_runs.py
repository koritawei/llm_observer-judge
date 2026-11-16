"""Unit tests for GET /v1/benchmarks/runs query aliases (benchmarkName / taskId)."""

import warnings
from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.routes import benchmark as api
from benchmark.runs.run_lifecycle import ListRunsQuery
from fastapi import FastAPI, Query
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_list_page_runs_forwards_filters() -> None:
    """Endpoint forwards camelCase-resolved query fields into ListRunsQuery."""
    svc = MagicMock()
    svc.list_runs = AsyncMock(return_value=[])

    result = await api.list_page_runs(
        benchmark_service=svc,
        limit=10,
        offset=5,
        benchmark_name="example_benchmark",
        task_id="task-1",
    )

    assert result == []
    svc.list_runs.assert_awaited_once()
    query = svc.list_runs.call_args.args[0]
    assert isinstance(query, ListRunsQuery)
    assert query.benchmark_name == "example_benchmark"
    assert query.task_id == "task-1"
    assert query.limit == 10
    assert query.offset == 5


def test_page_runs_query_aliases_have_no_pydantic_warning() -> None:
    """``Query(alias=...)`` on params must not emit UnsupportedFieldAttributeWarning.

    Regression for nested ``BaseModel`` + ``Field(alias=)`` + ``Annotated[..., Query()]``,
    which Pydantic 2.13 warns on OpenAPI generation and effectively drops the alias.
    """
    app = FastAPI()

    @app.get("/runs")
    async def _runs(
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
        benchmark_name: Annotated[str | None, Query(alias="benchmarkName")] = None,
        task_id: Annotated[str | None, Query(alias="taskId")] = None,
    ) -> dict[str, Any]:
        return {
            "benchmark_name": benchmark_name,
            "task_id": task_id,
            "limit": limit,
            "offset": offset,
        }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema = app.openapi()

    unsupported = [
        w
        for w in caught
        if w.category.__name__ == "UnsupportedFieldAttributeWarning"
        or "benchmarkName" in str(w.message)
        or "taskId" in str(w.message)
    ]
    assert unsupported == []

    params = schema["paths"]["/runs"]["get"]["parameters"]
    assert {p["name"] for p in params} == {
        "limit",
        "offset",
        "benchmarkName",
        "taskId",
    }

    client = TestClient(app)
    resp = client.get(
        "/runs",
        params={"benchmarkName": "amap-agent", "taskId": "t-9", "limit": 3},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "benchmark_name": "amap-agent",
        "task_id": "t-9",
        "limit": 3,
        "offset": 0,
    }
