"""Minimal annotation platform HTTP API (claim / complete / list)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from llmobserverjudge.annotation_worker.store import (
    AnnotationStore,
    BACKLOG_ALERT_THRESHOLD,
    InMemoryAnnotationStore,
    backlog_alert,
)
from llmobserverjudge.contracts.models import AnnotationStatus


class ClaimBody(BaseModel):
    annotator_id: str = Field(min_length=1)


class CompleteBody(BaseModel):
    annotator_id: str = Field(min_length=1)
    human_score: float | None = None
    human_label: str | None = None
    human_root_cause: str | None = None
    evidence_sufficient: bool | None = None
    fix_feasible: bool | None = None


def create_annotation_app(store: AnnotationStore | None = None) -> FastAPI:
    store = store or InMemoryAnnotationStore()
    app = FastAPI(title="LLMJudge Annotation Platform", version="0.1.0")
    app.state.annotation_store = store

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/annotations")
    async def list_annotations(
        tenant_id: str | None = None,
        status: AnnotationStatus | None = None,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = await store.list(tenant_id=tenant_id, status=status, limit=limit)
        return {"items": [r.model_dump(mode="json") for r in rows], "count": len(rows)}

    @app.get("/api/v1/annotations/{annotation_id}")
    async def get_annotation(annotation_id: str) -> dict[str, Any]:
        row = await store.get(annotation_id)
        if row is None:
            raise HTTPException(404, "not found")
        return row.model_dump(mode="json")

    @app.post("/api/v1/annotations/{annotation_id}/claim")
    async def claim(annotation_id: str, body: ClaimBody) -> dict[str, Any]:
        try:
            row = await store.claim(annotation_id, body.annotator_id)
        except KeyError:
            raise HTTPException(404, "not found") from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return row.model_dump(mode="json")

    @app.post("/api/v1/annotations/{annotation_id}/complete")
    async def complete(annotation_id: str, body: CompleteBody) -> dict[str, Any]:
        try:
            row = await store.complete(
                annotation_id,
                annotator_id=body.annotator_id,
                human_score=body.human_score,
                human_label=body.human_label,
                human_root_cause=body.human_root_cause,
                evidence_sufficient=body.evidence_sufficient,
                fix_feasible=body.fix_feasible,
            )
        except KeyError:
            raise HTTPException(404, "not found") from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return row.model_dump(mode="json")

    @app.post("/api/v1/annotations/expire")
    async def expire() -> dict[str, int]:
        n = await store.expire_pending()
        return {"expired": n}

    @app.get("/api/v1/tenants/{tenant_id}/annotation-backlog")
    async def backlog(tenant_id: str) -> dict[str, Any]:
        pending = await store.pending_count(tenant_id)
        return {
            "tenant_id": tenant_id,
            "pending": pending,
            "alert": backlog_alert(pending, BACKLOG_ALERT_THRESHOLD),
            "threshold": BACKLOG_ALERT_THRESHOLD,
        }

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        "llmobserverjudge.annotation_worker.api:create_annotation_app",
        factory=True,
        host="0.0.0.0",
        port=8081,
    )
