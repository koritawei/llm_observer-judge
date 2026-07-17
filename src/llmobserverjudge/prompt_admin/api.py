"""Prompt Admin HTTP API — versions + A/B experiments (design §5.5 / §5.8)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class CreatePromptBody(BaseModel):
    name: str = ""
    content: str = Field(min_length=1)
    variables: list[str] = Field(default_factory=list)
    prompt_id: str | None = None
    category: str = "JUDGE"
    tenant_id: str | None = None
    publish: bool = False


class CreateVersionBody(BaseModel):
    content: str = Field(min_length=1)
    variables: list[str] = Field(default_factory=list)
    changelog: str | None = None


class CreateExperimentBody(BaseModel):
    control_version_id: str = Field(min_length=1)
    treatment_version_id: str = Field(min_length=1)
    traffic_pct: int = Field(default=20, ge=0, le=100)
    metric_primary: str = "mean_score"


class ConcludeBody(BaseModel):
    winner: str = Field(description="control | treatment")
    conclusion: str | None = None


def _version_dict(v: Any) -> dict[str, Any]:
    return {
        "version_id": v.version_id,
        "prompt_id": v.prompt_id,
        "version": v.version,
        "content": v.content,
        "status": v.status,
        "variables": list(v.variables or []),
        "published_at": v.published_at.isoformat() if v.published_at else None,
        "changelog": getattr(v, "changelog", None),
    }


def create_prompt_admin_app(prompt_service: Any) -> FastAPI:
    app = FastAPI(title="LLMJudge Prompt Admin", version="0.1.0")
    app.state.prompt_service = prompt_service

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/prompts")
    async def list_prompts(tenant_id: str | None = None) -> dict[str, Any]:
        items = await prompt_service.list_prompts(tenant_id=tenant_id)
        return {"items": items, "count": len(items)}

    @app.post("/api/v1/prompts")
    async def create_prompt(body: CreatePromptBody) -> dict[str, Any]:
        ver = await prompt_service.create_prompt(
            prompt_id=body.prompt_id,
            content=body.content,
            variables=body.variables,
            name=body.name,
            category=body.category,
            tenant_id=body.tenant_id,
        )
        if body.publish:
            ver = await prompt_service.publish(ver.prompt_id, ver.version_id)
        return _version_dict(ver)

    @app.get("/api/v1/prompts/{prompt_id}/versions")
    async def list_versions(prompt_id: str) -> dict[str, Any]:
        versions = await prompt_service.list_versions(prompt_id)
        return {"items": [_version_dict(v) for v in versions], "count": len(versions)}

    @app.put("/api/v1/prompts/{prompt_id}/versions")
    async def create_version(prompt_id: str, body: CreateVersionBody) -> dict[str, Any]:
        try:
            ver = await prompt_service.create_version(
                prompt_id,
                content=body.content,
                variables=body.variables,
                changelog=body.changelog,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)) from e
        return _version_dict(ver)

    @app.post("/api/v1/prompts/{prompt_id}/versions/{version_id}/publish")
    async def publish(prompt_id: str, version_id: str) -> dict[str, Any]:
        try:
            ver = await prompt_service.publish(prompt_id, version_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return _version_dict(ver)

    @app.post("/api/v1/prompts/{prompt_id}/versions/{version_id}/experimental")
    async def mark_experimental(prompt_id: str, version_id: str) -> dict[str, Any]:
        try:
            ver = await prompt_service.publish_experimental(prompt_id, version_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return _version_dict(ver)

    @app.get("/api/v1/experiments")
    async def list_all_experiments(
        prompt_id: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        rows = await prompt_service.list_experiments(prompt_id=prompt_id, status=status)
        return {"items": [e.to_dict() for e in rows], "count": len(rows)}

    @app.post("/api/v1/prompts/{prompt_id}/experiments")
    async def create_experiment(prompt_id: str, body: CreateExperimentBody) -> dict[str, Any]:
        try:
            exp = await prompt_service.create_experiment(
                prompt_id,
                control_version_id=body.control_version_id,
                treatment_version_id=body.treatment_version_id,
                traffic_pct=body.traffic_pct,
                metric_primary=body.metric_primary,
            )
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return exp.to_dict()

    @app.get("/api/v1/prompts/{prompt_id}/experiments/{exp_id}")
    async def get_experiment(prompt_id: str, exp_id: str) -> dict[str, Any]:
        exp = await prompt_service.get_experiment(exp_id)
        if exp is None or exp.prompt_id != prompt_id:
            raise HTTPException(404, "experiment not found")
        return exp.to_dict()

    @app.post("/api/v1/prompts/{prompt_id}/experiments/{exp_id}/conclude")
    async def conclude(prompt_id: str, exp_id: str, body: ConcludeBody) -> dict[str, Any]:
        exp = await prompt_service.get_experiment(exp_id)
        if exp is None or exp.prompt_id != prompt_id:
            raise HTTPException(404, "experiment not found")
        try:
            out = await prompt_service.conclude_experiment(
                exp_id, winner=body.winner, conclusion=body.conclusion
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return out.to_dict()

    return app
