"""API router assembly."""

from fastapi import APIRouter

from benchmark.routes import (
    benchmark,
    benchmark_case_reassign,
    benchmark_console,
    benchmark_run_cases,
    daily_reports,
    eval,
    health,
    report_providers,
    runtime_config,
    storage,
    streaming_tasks,
)

api_router = APIRouter()
api_router.include_router(eval.router)
api_router.include_router(health.router)
api_router.include_router(benchmark.router)
api_router.include_router(benchmark_run_cases.router)
api_router.include_router(benchmark_case_reassign.router)
api_router.include_router(daily_reports.router)
api_router.include_router(benchmark_console.router)
api_router.include_router(runtime_config.router)
api_router.include_router(report_providers.router)
api_router.include_router(storage.router)
api_router.include_router(streaming_tasks.router)
