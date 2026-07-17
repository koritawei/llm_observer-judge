"""Project admin HTTP API — multi-tenant project registry."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from llmobserverjudge.gateway.auth import ApiKeyAuthenticator
from llmobserverjudge.shared.project import DEFAULT_PROJECT_ID
from llmobserverjudge.shared.sql_project import InMemoryProjectStore, ProjectStore


class CreateProjectBody(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    project_id: str | None = None


def create_project_app(
    store: ProjectStore | None = None,
    authenticator: ApiKeyAuthenticator | None = None,
) -> FastAPI:
    projects = store or InMemoryProjectStore()
    auth = authenticator or ApiKeyAuthenticator()
    app = FastAPI(title="LLMJudge Projects", version="0.1.0")
    app.state.project_store = projects
    app.state.auth = auth

    def _tenant(api_key: str | None) -> str:
        try:
            return auth.authenticate(api_key).id
        except PermissionError as exc:
            raise HTTPException(401, str(exc)) from exc

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/projects")
    async def list_projects(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        tid = tenant_id or _tenant(x_api_key)
        await projects.ensure_default(tid)
        rows = await projects.list_for_tenant(tid)
        return {"items": [r.to_dict() for r in rows], "count": len(rows), "tenant_id": tid}

    @app.post("/api/v1/projects")
    async def create_project(
        body: CreateProjectBody,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        tid = tenant_id or _tenant(x_api_key)
        await projects.ensure_default(tid)
        try:
            row = await projects.create(tid, body.name, project_id=body.project_id)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return row.to_dict()

    @app.get("/api/v1/projects/{project_id}")
    async def get_project(
        project_id: str,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        tid = tenant_id or _tenant(x_api_key)
        row = await projects.get(tid, project_id)
        if row is None and project_id == DEFAULT_PROJECT_ID:
            row = await projects.ensure_default(tid)
        if row is None:
            raise HTTPException(404, "project not found")
        return row.to_dict()

    return app
