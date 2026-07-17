"""Ingestion Gateway — FastAPI (SDD-IG-11) STEP-03 hardened."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from llmobserverjudge.contracts.models import IngestAccepted, IngestRequest
from llmobserverjudge.contracts.topics import Component, Topic
from llmobserverjudge.gateway.auth import ApiKeyAuthenticator
from llmobserverjudge.gateway.health import HealthTracker
from llmobserverjudge.gateway.rate_limit import RateLimiter
from llmobserverjudge.shared.mq import InMemoryMQ, MessageQueue
from llmobserverjudge.shared.project import DEFAULT_PROJECT_ID
from llmobserverjudge.shared.sql_project import InMemoryProjectStore, ProjectStore

StartupHook = Callable[[FastAPI], Awaitable[None]]
ShutdownHook = Callable[[FastAPI], Awaitable[None]]


class TelemetryPayload(BaseModel):
    sent_spans_total: int = Field(ge=0)
    dropped_spans_total: int = Field(ge=0)
    buffer_usage: int | None = None
    buffer_capacity: int | None = None


class TelemetryStore:
    def __init__(self) -> None:
        self.by_tenant: dict[str, TelemetryPayload] = {}

    def record(self, tenant_id: str, payload: TelemetryPayload) -> None:
        self.by_tenant[tenant_id] = payload


def create_app(
    mq: MessageQueue | None = None,
    authenticator: ApiKeyAuthenticator | None = None,
    rate_limiter: RateLimiter | None = None,
    health_tracker: HealthTracker | None = None,
    project_store: ProjectStore | None = None,
    *,
    on_startup: StartupHook | None = None,
    on_shutdown: ShutdownHook | None = None,
) -> FastAPI:
    state_mq = mq or InMemoryMQ()
    auth = authenticator or ApiKeyAuthenticator()
    limiter = rate_limiter or RateLimiter()
    health = health_tracker or HealthTracker()
    telemetry = TelemetryStore()
    projects = project_store or InMemoryProjectStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mq = state_mq
        app.state.auth = auth
        app.state.limiter = limiter
        app.state.health = health
        app.state.telemetry = telemetry
        app.state.projects = projects
        app.state.shutting_down = False
        if on_startup is not None:
            await on_startup(app)
        yield
        app.state.shutting_down = True
        if on_shutdown is not None:
            await on_shutdown(app)

    app = FastAPI(title="LLMJudge Gateway", version="0.1.0", lifespan=lifespan)
    app.state.mq = state_mq
    app.state.auth = auth
    app.state.limiter = limiter
    app.state.health = health
    app.state.telemetry = telemetry
    app.state.projects = projects
    app.state.shutting_down = False

    def _auth(api_key: str | None):
        try:
            return auth.authenticate(api_key)
        except PermissionError as exc:
            raise HTTPException(401, str(exc)) from exc

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, Any]:
        if request.app.state.shutting_down:
            raise HTTPException(503, "shutting down")
        tracker: HealthTracker = request.app.state.health
        try:
            await request.app.state.mq.health_check()
            tracker.reset("mq")
        except Exception:
            n = tracker.incr("mq")
            if tracker.is_unhealthy("mq"):
                raise HTTPException(
                    503, f"mq unavailable ({n} consecutive failures)"
                ) from None
            return {"status": "degraded", "mq_failures": n}
        return {"status": "ready"}

    @app.post("/api/v1/ingest", response_model=IngestAccepted)
    async def ingest(
        body: IngestRequest,
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_project_id: str | None = Header(default=None, alias="X-Project-Id"),
    ) -> IngestAccepted:
        if request.app.state.shutting_down:
            raise HTTPException(503, "shutting down")
        tenant = _auth(x_api_key)
        cost = float(len(body.spans))
        limiter = request.app.state.limiter
        allow = getattr(limiter, "allow_ingest_async", None)
        ok = (
            await allow(tenant.id, cost=cost)
            if allow is not None
            else limiter.allow_ingest(tenant.id, cost=cost)
        )
        if not ok:
            raise HTTPException(
                429, "rate limited", headers={"Retry-After": "10"}
            )

        # Resolve project: header > body span.project_id > default
        requested = (x_project_id or "").strip() or None
        if not requested and body.spans:
            requested = (body.spans[0].project_id or "").strip() or None
        try:
            project = await request.app.state.projects.resolve(
                tenant.id, project_id=requested or DEFAULT_PROJECT_ID
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

        enriched: list[dict[str, Any]] = []
        for span in body.spans:
            data = span.model_dump(mode="json", exclude_none=True)
            data["tenant_id"] = tenant.id
            data["project_id"] = project.project_id
            enriched.append(data)

        try:
            await request.app.state.mq.publish_batch(
                Component.GATEWAY, Topic.SPANS, enriched
            )
            request.app.state.health.reset("mq")
        except Exception as exc:  # noqa: BLE001
            request.app.state.health.incr("mq")
            raise HTTPException(503, "mq publish failed") from exc

        return IngestAccepted(status="accepted", count=len(enriched))

    @app.post("/api/v1/telemetry")
    async def sdk_telemetry(
        body: TelemetryPayload,
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, str]:
        tenant = _auth(x_api_key)
        limiter = request.app.state.limiter
        allow = getattr(limiter, "allow_telemetry_async", None)
        ok = (
            await allow(tenant.id)
            if allow is not None
            else limiter.allow_telemetry(tenant.id)
        )
        if not ok:
            raise HTTPException(
                429, "telemetry rate limited", headers={"Retry-After": "30"}
            )
        request.app.state.telemetry.record(tenant.id, body)
        return {"status": "ok"}

    return app


app = create_app()


def main() -> None:
    """Prefer Redis Streams gateway via ``llmj-worker gateway`` when splitting processes."""
    from llmobserverjudge.runtime.workers import main as worker_main

    worker_main(["gateway"])
