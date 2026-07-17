"""Benchmark / batch-eval HTTP API (design §11.2 / admin benchmarks)."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llmobserverjudge.benchmark.service import BenchmarkService
from llmobserverjudge.contracts.models import Priority


class BatchTriggerBody(BaseModel):
    tenant_id: str = Field(min_length=1)
    name: str = "batch"
    traces: list[dict[str, Any]] = Field(min_length=1)
    eval_configs: list[dict[str, Any]] = Field(min_length=1)
    priority: Priority = Priority.P2


class DailyBenchmarkBody(BaseModel):
    tenant_id: str = Field(min_length=1)
    date: date
    rate: float = 0.1
    min_size: int = 5000


def create_benchmark_app(service: BenchmarkService | None = None) -> FastAPI:
    svc = service or BenchmarkService()
    app = FastAPI(title="LLMJudge Benchmark", version="0.1.0")
    app.state.benchmark_service = svc

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/batch-eval/trigger")
    async def batch_trigger(body: BatchTriggerBody) -> dict[str, Any]:
        job = await svc.trigger_batch_eval(
            tenant_id=body.tenant_id,
            name=body.name,
            traces=body.traces,
            eval_configs=body.eval_configs,
            priority=body.priority,
        )
        report = await svc.store.get_report_by_batch(job.batch_job_id)
        return {
            "batch_job": job.model_dump(mode="json"),
            "report_id": report.report_id if report else None,
        }

    @app.post("/admin/v1/benchmarks")
    async def daily_benchmark(body: DailyBenchmarkBody) -> dict[str, Any]:
        report = await svc.trigger_daily_benchmark(
            body.tenant_id,
            body.date,
            rate=body.rate,
            min_size=body.min_size,
        )
        return report.model_dump(mode="json")

    @app.get("/api/v1/benchmarks/{report_id}")
    async def get_report(report_id: str) -> dict[str, Any]:
        report = await svc.store.get_report(report_id)
        if report is None:
            raise HTTPException(404, "not found")
        return report.model_dump(mode="json")

    @app.get("/api/v1/batch-eval/{batch_job_id}")
    async def get_batch(batch_job_id: str) -> dict[str, Any]:
        job = await svc.store.get_job(batch_job_id)
        if job is None:
            raise HTTPException(404, "not found")
        tasks = await svc.store.list_tasks(batch_job_id)
        return {
            "batch_job": job.model_dump(mode="json"),
            "tasks": [t.model_dump(mode="json") for t in tasks],
            "signal": await svc.signal.snapshot(batch_job_id),
        }

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        "llmobserverjudge.benchmark.api:create_benchmark_app",
        factory=True,
        host="0.0.0.0",
        port=8082,
    )
