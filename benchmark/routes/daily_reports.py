"""Daily report catalog API.

Protocol shell: no business parameters; delegates a full assembly to domain
``build_daily_report_catalog``. The response model is shared with domain
(``DailyReportCatalog``) so the fields are not defined twice.
"""

from fastapi import APIRouter

from benchmark.domain.daily_report_catalog import (
    DailyReportCatalog,
    build_daily_report_catalog,
)

router = APIRouter(prefix="/v1/benchmarks", tags=["benchmark-reports"])


class DailyReportUrlsResponse(DailyReportCatalog):
    """Response body for ``GET /v1/benchmarks/daily-report-urls``.

    - ``realtime``: T+1 daily reports; key = the registration document's ``agent_scope``.
    - ``batch``: today's batch reports; key is the component short name
      (``ieval_external`` → ``ieval``).
    - Every value is an object (the latest one for today). Keys without a valid Pages URL
      are omitted.

    **BREAKING** (versus the old URL string / array): the value is now
    ``{url, run_id, overall, rubrics}``; multiple rows for one scope keep only the latest.
    Old consumers should read ``.url`` and stop comparing ``catalog.realtime[scope] === "…"``.
    """


@router.get("/daily-report-urls", response_model=DailyReportUrlsResponse)
async def get_daily_report_urls() -> DailyReportUrlsResponse:
    """Return today's latest generated Pages reports and score snapshots (full DB read, no cache)."""
    catalog = await build_daily_report_catalog()
    return DailyReportUrlsResponse.model_validate(catalog.model_dump())
