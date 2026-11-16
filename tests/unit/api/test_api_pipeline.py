"""Unit tests for POST /v1/benchmarks/{name}/run-pipeline endpoint.

Verifies async 202 response, 404 for unknown benchmark name, and 400 for
invalid date range.  Reuses the same ``_job_lock`` / ``_pending_benchmark_tasks``
infrastructure as the regular benchmark endpoint.
"""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.routes import benchmark as api
from benchmark.runs.schema import BenchmarkStatus
from benchmark.reporting.framework.base import BENCHMARK_METADATA_ATTR, BenchmarkComponentMetadata
from benchmark.reporting.framework.schema import (
    AggregationConfig,
    Criterion,
    DatasetRef,
)
from fastapi import HTTPException

# ════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _reset_lock() -> None:
    """The Lock mechanism was removed with the BullMQ migration; the fixture remains as a placeholder to keep tests isolated."""
    yield


@pytest.fixture(autouse=True)
def _benchmark_switch_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The benchmark switch defaults to True (do not process): API cases uniformly mock the guard so requests are allowed."""
    monkeypatch.setattr(api, "raise_if_benchmark_disabled", lambda: None)


def _meta(name: str = "bm_a") -> BenchmarkComponentMetadata:
    return BenchmarkComponentMetadata(
        name=name,
        service_name="svc",
        rubric=[
            Criterion(
                id="c1",
                name="C1",
                description="d",
                weight=1.0,
                evaluator_ref="example",
            ),
        ],
        aggregation=AggregationConfig(method="weighted_avg", pass_threshold=0.5),
        dataset=DatasetRef(dataset_name="ds", project_id="p1"),
    )


def _mock_agent(name: str = "bm_a") -> MagicMock:
    meta = _meta(name)
    agent = MagicMock()
    agent.get_metadata = MagicMock(return_value=meta)
    setattr(agent, BENCHMARK_METADATA_ATTR, meta)
    return agent


def _pipeline_payload(**overrides: object) -> api.PipelineBenchmarkRequest:
    data: dict[str, object] = {
        "date_start": datetime(2025, 1, 1, tzinfo=UTC),
        "date_end": datetime(2025, 1, 31, tzinfo=UTC),
        "sample_size": 100,
        "generate_report": False,
    }
    data.update(overrides)
    return api.PipelineBenchmarkRequest.model_validate(data)


async def _drain(timeout: float = 2.0) -> None:
    """Wait for pending background tasks to finish (run-pipeline is started via ``asyncio.create_task``)."""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.wait(pending, timeout=timeout)


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_post_run_pipeline_returns_202() -> None:
    gate = asyncio.Event()

    async def slow_run(**kwargs: object) -> object:
        await gate.wait()
        return SimpleNamespace(
            run_id=kwargs["run_id"],
            status=BenchmarkStatus.COMPLETED,
            final_score=1.0,
            metrics=None,
            report_url=None,
            error=None,
        )

    agent = _mock_agent()
    runtime = MagicMock()
    runtime.agent_manager.get = MagicMock(return_value=agent)

    service = MagicMock()
    service.create_run = AsyncMock(side_effect=lambda spec: "run-1")

    runner = MagicMock()
    runner.run_pipeline_benchmark = AsyncMock(side_effect=slow_run)

    resp = await api.run_pipeline_benchmark(
        name="bm_a",
        payload=_pipeline_payload(),
        benchmark_service=service,
        runner=runner,
        runtime=runtime,
    )

    assert resp.status_code == 202
    content = json.loads(resp.body)
    assert content["run_id"] == "run-1"
    assert content["status"] == "initializing"
    assert "poll_url" in content

    # Wait until runner acquired lock and started
    for _ in range(50):
        if runner.run_pipeline_benchmark.await_count >= 1:
            break
        await asyncio.sleep(0.01)
    assert runner.run_pipeline_benchmark.await_count == 1

    gate.set()
    await _drain()


@pytest.mark.asyncio
async def test_post_run_pipeline_invalid_benchmark_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither the static document nor the page-registered document matches → 404 (document lookup is stubbed when there is no DB)."""

    async def _no_document(name: str):
        return None

    monkeypatch.setattr(
        "benchmark.runs.common.dynamic_component.find_agent_document_by_name",
        _no_document,
    )

    runtime = MagicMock()
    runtime.agent_manager.get = MagicMock(side_effect=KeyError("not found"))

    service = MagicMock()
    runner = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await api.run_pipeline_benchmark(
            name="nonexistent",
            payload=_pipeline_payload(),
            benchmark_service=service,
            runner=runner,
            runtime=runtime,
        )

    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_post_run_pipeline_not_a_benchmark_component() -> None:
    """Agent exists but lacks BENCHMARK_METADATA_ATTR → 404."""
    agent = MagicMock()
    delattr(agent, BENCHMARK_METADATA_ATTR)  # ensure attr is absent

    runtime = MagicMock()
    runtime.agent_manager.get = MagicMock(return_value=agent)

    service = MagicMock()
    runner = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await api.run_pipeline_benchmark(
            name="regular_agent",
            payload=_pipeline_payload(),
            benchmark_service=service,
            runner=runner,
            runtime=runtime,
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_post_run_pipeline_invalid_date_range() -> None:
    runtime = MagicMock()
    service = MagicMock()
    runner = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await api.run_pipeline_benchmark(
            name="bm_a",
            payload=_pipeline_payload(
                date_start=datetime(2025, 1, 31, tzinfo=UTC),
                date_end=datetime(2025, 1, 1, tzinfo=UTC),
            ),
            benchmark_service=service,
            runner=runner,
            runtime=runtime,
        )

    assert exc_info.value.status_code == 400
    assert "date_end" in exc_info.value.detail


@pytest.mark.asyncio
async def test_post_run_pipeline_config_json_has_pipeline_source() -> None:
    """Verify config_json stores source=pipeline-kafka and date range."""
    agent = _mock_agent()
    runtime = MagicMock()
    runtime.agent_manager.get = MagicMock(return_value=agent)

    service = MagicMock()
    service.create_run = AsyncMock(side_effect=lambda spec: "run-cfg")

    runner = MagicMock()
    runner.run_pipeline_benchmark = AsyncMock()

    payload = _pipeline_payload(sample_size=50, generate_report=False)

    await api.run_pipeline_benchmark(
        name="bm_a",
        payload=payload,
        benchmark_service=service,
        runner=runner,
        runtime=runtime,
    )

    await _drain()

    create_spec = service.create_run.await_args.args[0]
    assert create_spec.dataset_name == "(pipeline-kafka)"
    config = create_spec.config_json
    assert config["source"] == "pipeline-kafka"
    assert config["date_start"] == payload.date_start.isoformat()
    assert config["date_end"] == payload.date_end.isoformat()
    assert config["sample_size"] == 50
