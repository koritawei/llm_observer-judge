"""Label Studio webhook API."""

import hmac
from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/label-studio", tags=["label-studio"])


class LSWebhookBody(BaseModel):
    """LS webhook callback body schema."""

    annotation: dict[str, Any] = {}
    project: dict[str, Any] | None = None
    task: dict[str, Any] | None = None


@router.post("/webhook")
async def label_studio_webhook(
    body: LSWebhookBody,
    request: Request,
    authorization: str = Header(default=""),
) -> JSONResponse:
    webhook_service = getattr(request.app.state, "ls_webhook_service", None)
    if webhook_service is None:
        return JSONResponse({"status": "disabled"}, status_code=503)

    secret = request.app.state.settings.label_studio_webhook_secret
    if secret and not hmac.compare_digest(authorization, f"Bearer {secret}"):
        return JSONResponse({"status": "unauthorized"}, status_code=401)

    result = await webhook_service.handle_annotation(
        body.annotation,
        body.project,
        body.task,
    )
    http_status = result.pop("_http_status", 200)
    return JSONResponse(result, status_code=http_status)
