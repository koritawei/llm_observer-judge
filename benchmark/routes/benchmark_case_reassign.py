"""Benchmark case Aone assignee reassignment API."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from benchmark.routes.benchmark import BenchmarkServiceDep
from benchmark.runs.notification_assignee import (
    NotificationAssigneeError,
    reassign_case_notification_assignee,
)

router = APIRouter(prefix="/v1/benchmarks", tags=["benchmarks"])


class ReassignCaseNotificationBody(BaseModel):
    """Reassign-assignee request body. ``assignee`` is a flower name or a numeric employee id."""

    assignee: str = Field(min_length=1, max_length=64)


class ReassignCaseNotificationResponse(BaseModel):
    """Bucket fields returned after a successful assignee change."""

    notification_ref: str
    notification_url: str
    notification_assigned_to: str


@router.post(
    "/runs/{run_id}/cases/{case_id}/notification-assignee",
    response_model=ReassignCaseNotificationResponse,
)
async def reassign_run_case_notification_assignee(
    run_id: str,
    case_id: str,
    body: ReassignCaseNotificationBody,
    benchmark_service: BenchmarkServiceDep,
) -> ReassignCaseNotificationResponse:
    """Change the Aone assignee on a case that has ``notification_url``, and write back ``notification_assigned_to``."""
    run = await benchmark_service.get_run_by_id_or_pk(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    try:
        updated = await reassign_case_notification_assignee(
            benchmark_service,
            run,
            case_id=case_id,
            assignee=body.assignee,
        )
    except NotificationAssigneeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return ReassignCaseNotificationResponse.model_validate(updated)
