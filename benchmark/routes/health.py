"""Health check endpoints (Aone container probe)."""

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Liveness probe. Structure is stable (Aone health.sh parses status/detail).

    ``status`` is always ``ok`` (a live process passes the probe, so the instance is not
    pulled by mistake). When the BullMQ queue stack is degraded
    (``_ls_bullmq_init_failed_fallback`` records the reason), an ``eval_queue`` field
    exposes ``degraded`` and the reason for operators and callers — a degraded instance
    answers benchmark requests with 503. Only the explicit degraded flag is consulted, so
    a service that never assembled a queue stack (a lightweight app that only mounts this
    router) is not reported as degraded.
    """
    body = {"status": "ok", "detail": "success"}
    degraded_reason = getattr(request.app.state, "eval_queue_degraded_reason", None)
    if degraded_reason:
        body["eval_queue"] = f"degraded ({degraded_reason})"
    return body


@router.get("/status.taobao", response_class=PlainTextResponse)
async def status_taobao() -> str:
    """Aone/Tengine-compatible probe (plain text ``success``, aligned with vehicle-agent health.sh)."""

    return "success"
