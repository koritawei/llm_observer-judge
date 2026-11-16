"""Benchmark run case list / detail API.

``GET /v1/benchmarks/runs/{run_id}/cases`` — paginated case rows for report #cases-data.
``GET /v1/benchmarks/runs/{run_id}/cases/{case_id}`` — GitBook detail for one case (includes input).
``GET /v1/benchmarks/runs/{run_id}/report-summary`` — report HTML before #case-section.
"""

from typing import Annotated
from urllib.error import HTTPError, URLError

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from benchmark.routes.benchmark import BenchmarkServiceDep
from benchmark.runs.case_display.views import (
    CaseAttributionView,
    CaseJudgeView,
)
from benchmark.runs.report_summary_html import fetch_report_summary_html
from benchmark.runs.run_case_page import (
    CaseDetailQuery,
    CasePageQuery,
    build_case_detail,
    build_case_page,
    report_url_of,
)

router = APIRouter(prefix="/v1/benchmarks", tags=["benchmarks"])


class BenchmarkRunCaseItem(BaseModel):
    """One case (display fields already normalized on the server)."""

    id: str
    query: str = ""
    score: float | None = None
    status: str | None = None
    attribution_category: str | None = None
    output: str = ""
    judge: CaseJudgeView | None = None
    attribution: CaseAttributionView | None = None
    trace_url: str | None = None
    notification_url: str | None = None
    notification_ref: str | None = None
    notification_assigned_to: str | None = None


class BenchmarkRunCaseDetailItem(BenchmarkRunCaseItem):
    """GitBook detail: list fields plus the full input / expected."""

    model_config = ConfigDict(populate_by_name=True)

    input: str = ""
    expected_output: str = Field(default="", alias="expectedOutput")


class BenchmarkRunCasePageMeta(BaseModel):
    """Pagination metadata. ``pageSize`` is the public field name."""

    model_config = ConfigDict(populate_by_name=True)

    page: int
    page_size: int = Field(alias="pageSize")
    total: int
    owners: list[str] = Field(default_factory=list)
    notification_source: str | None = Field(default=None, alias="notificationSource")
    supports_attribution_category: bool = Field(
        default=False,
        alias="supportsAttributionCategory",
    )
    attribution_categories: list[str] = Field(
        default_factory=list,
        alias="attributionCategories",
    )
    report_url: str | None = Field(default=None, alias="reportUrl")


class BenchmarkRunCasesResponse(BaseModel):
    """Response body for ``GET /v1/benchmarks/runs/{run_id}/cases``."""

    data: list[BenchmarkRunCaseItem]
    meta: BenchmarkRunCasePageMeta


class BenchmarkRunCasesQuery(BaseModel):
    """Case-list query (FUNC-PARAMS-001: fold pagination and filter parameters)."""

    model_config = ConfigDict(populate_by_name=True)

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100, alias="pageSize")
    query: str = ""
    status: str | None = "FAIL,POOR"
    attribution_category: str = Field(default="", alias="attributionCategory")


class BenchmarkRunCaseDetailMeta(BaseModel):
    """Single-case detail meta (no pagination)."""

    model_config = ConfigDict(populate_by_name=True)

    owners: list[str] = Field(default_factory=list)
    notification_source: str | None = Field(default=None, alias="notificationSource")
    supports_attribution_category: bool = Field(
        default=False,
        alias="supportsAttributionCategory",
    )
    report_url: str | None = Field(default=None, alias="reportUrl")


class BenchmarkRunCaseDetailResponse(BaseModel):
    """Response body for ``GET /v1/benchmarks/runs/{run_id}/cases/{case_id}``."""

    data: BenchmarkRunCaseDetailItem
    meta: BenchmarkRunCaseDetailMeta


@router.get("/runs/{run_id}/cases", response_model=BenchmarkRunCasesResponse)
async def list_run_cases(
    run_id: str,
    benchmark_service: BenchmarkServiceDep,
    params: Annotated[BenchmarkRunCasesQuery, Query()],
) -> BenchmarkRunCasesResponse:
    """Return a page of case rows for one run.

    ``output`` / ``judge`` / ``attribution`` are already normalized for the realtime and
    batch shapes, so the WebUI can render them directly. ``notification_url`` comes from
    ``notification_dispatch_json.notification_url``.

    ``status`` defaults to ``FAIL,POOR`` (not passing); ``ALL`` does not filter by status.
    ``query`` is a case-insensitive substring match on the case query.
    ``attributionCategory`` is an exact match on the attribution category column;
    search_eval / v2 ignore it.
    """
    run = await benchmark_service.get_run_by_id_or_pk(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    payload = await build_case_page(
        benchmark_service,
        run,
        page_query=CasePageQuery(
            page=params.page,
            page_size=params.page_size,
            query=params.query,
            status=params.status,
            attribution_category=params.attribution_category,
        ),
    )
    return BenchmarkRunCasesResponse.model_validate(payload)


@router.get(
    "/runs/{run_id}/cases/{case_id}",
    response_model=BenchmarkRunCaseDetailResponse,
)
async def get_run_case(
    run_id: str,
    case_id: str,
    benchmark_service: BenchmarkServiceDep,
) -> BenchmarkRunCaseDetailResponse:
    """Return the GitBook detail for one case (full input / expected_output)."""
    run = await benchmark_service.get_run_by_id_or_pk(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    payload = await build_case_detail(
        benchmark_service,
        run,
        detail_query=CaseDetailQuery(case_id=case_id),
    )
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    return BenchmarkRunCaseDetailResponse.model_validate(payload)


@router.get("/runs/{run_id}/report-summary", response_class=HTMLResponse)
async def get_run_report_summary(
    run_id: str,
    benchmark_service: BenchmarkServiceDep,
) -> HTMLResponse:
    """Return the report HTML summary before the case detail (hides ``#case-section``)."""
    run = await benchmark_service.get_run_by_id_or_pk(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    report_url = report_url_of(run)
    if report_url is None:
        raise HTTPException(status_code=404, detail="No report available for this run")
    try:
        html = await fetch_report_summary_html(report_url)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (HTTPError, URLError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch report: {exc}",
        ) from exc
    return HTMLResponse(content=html, headers={"Cache-Control": "private, max-age=60"})
