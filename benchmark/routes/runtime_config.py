"""Page writes for realtime eval config and ODPS export jobs.

Business fields come from the request. Environment-level fields are stamped by
``runtime_config_defaults`` before the row is saved.
"""

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

from benchmark.records.models.runtime_config_record import (
    KIND_EVALUATOR,
    KIND_ODPS_JOB,
    RuntimeConfigRecord,
    compose_evaluator_key_tail,
    evaluator_record_key,
    odps_record_key,
    project_scope_key,
    split_evaluator_record_key,
)
from benchmark.domain.eval_config import ingest_conflict_message
from benchmark.domain.runtime_config_defaults import stamp_eval_entry, stamp_odps_job
from benchmark.domain.runtime_config_store import (
    current_eval_config_file,
    refresh_eval_config_snapshot,
    refresh_odps_export_snapshot,
    retire_runtime_config,
    upsert_runtime_config,
)
from benchmark.domain.schemas import EvalConfigEntry
from benchmark.domain.trace_report_config import (
    TraceReportSeed,
    benchmark_name_for_evaluator,
    ensure_default_trace_report,
)

router = APIRouter(prefix="/v1/runtime-config", tags=["runtime-config"])


class EvalEntryWrite(BaseModel):
    """One eval config submitted by the page. Environment-level fields are not on this model."""

    model_config = {"extra": "forbid"}

    project_id: str = Field(min_length=1, max_length=128)
    evaluator_name: str = Field(min_length=1, max_length=128)
    sampling: float = Field(ge=0.0, le=1.0)
    ingest: str = "kafka"
    agent_name: str = ""
    agent_registration_id: str = ""
    stream_instance_id: str = ""
    name: list[str] | None = None
    tags: list[str] | None = None
    start_time: str | None = None
    end_time: str | None = None
    prompt: str | None = None


class OdpsJobWrite(BaseModel):
    """One ODPS job submitted by the page. Timezone, file name, and SQL are not on this model."""

    model_config = {"extra": "forbid"}

    id: str = "default"
    enabled: bool = True
    schedule_time: str
    limit: int | None = Field(default=None, ge=1)
    oss_prefix: str
    project_id: str = ""


class DeleteEvaluatorQuery(BaseModel):
    """Query parameters for deleting an eval config (FUNC-PARAMS-001 fold)."""

    model_config = {"extra": "forbid"}

    project_id: str
    evaluator_name: str
    agent_name: str = ""
    agent_registration_id: str = ""
    stream_instance_id: str = ""


def _envelope(row: RuntimeConfigRecord) -> dict[str, Any]:
    """Read-only envelope. Shown on the page, not written back."""
    return {
        "schema_version": row.schema_version,
        "revision": row.revision,
        "scope_key": row.scope_key,
    }


def _public_eval(
    project_id: str,
    entry: EvalConfigEntry,
    row: RuntimeConfigRecord,
) -> dict[str, Any]:
    model: dict[str, Any] | None = None
    if entry.model is not None and not entry.model.is_empty:
        model = entry.model.model_dump(by_alias=True)
    return {
        "project_id": project_id,
        "evaluator_name": entry.evaluator_name,
        "sampling": entry.sampling,
        "ingest": entry.ingest,
        "agent_name": entry.agent_name or "",
        "name": entry.name or [],
        "tags": entry.tags or [],
        "start_time": entry.start_time,
        "end_time": entry.end_time,
        "prompt": entry.prompt,
        "environment": entry.environment,
        "variable_mapping": [item.model_dump(by_alias=True) for item in entry.variable_mapping],
        "model": model,
        **_envelope(row),
    }


def _public_job(payload: dict[str, Any], row: RuntimeConfigRecord) -> dict[str, Any]:
    return {
        "id": payload.get("id", "default"),
        "enabled": payload.get("enabled", True),
        "schedule_time": payload.get("schedule_time", ""),
        "schedule_timezone": payload.get("schedule_timezone", ""),
        "limit": payload.get("limit"),
        "oss_prefix": payload.get("oss_prefix", ""),
        "object_name": payload.get("object_name", ""),
        "project_id": payload.get("project_id", ""),
        "sql_template": payload.get("sql_template", ""),
        **_envelope(row),
    }


def _stamp_or_400(
    builder: Callable[[dict[str, Any]], dict[str, Any]],
    raw: dict[str, Any],
) -> dict[str, Any]:
    try:
        return builder(raw)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/evaluators")
async def list_evaluators() -> list[dict[str, Any]]:
    """List eval entries that are still active, including fields written from env vars."""
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVALUATOR,
        deleted_at__isnull=True,
    ).order_by("record_key")
    items: list[dict[str, Any]] = []
    for row in rows:
        project_id, _, _, _ = split_evaluator_record_key(row.record_key)
        payload = row.payload if isinstance(row.payload, dict) else {}
        entry = EvalConfigEntry.model_validate(payload)
        items.append(_public_eval(project_id, entry, row))
    return items


@router.put("/evaluators")
async def upsert_evaluator(body: EvalEntryWrite) -> dict[str, Any]:
    """Save one eval config. Environment, variable mapping, and model are written from env vars.

    After a successful save, a T+1 report item is registered by default (fixed 09:00, everything
    else default). When the evaluator stem resolves to a benchmark component, the item is
    named "evaluator-agent/project". When it does not resolve (no rubric source), the
    registration is skipped with a warning and the entry save is not blocked.
    """
    stamped = _stamp_or_400(stamp_eval_entry, body.model_dump())
    agent_name = body.agent_name.strip()
    registration_id = body.agent_registration_id.strip()
    stream_instance_id = body.stream_instance_id.strip()
    if stream_instance_id:
        stamped["streamInstanceId"] = stream_instance_id
    conflict = ingest_conflict_message(
        current_eval_config_file().configs.get(body.project_id, []),
        evaluator_name=body.evaluator_name,
        ingest=body.ingest,
    )
    if conflict is not None:
        raise HTTPException(status_code=400, detail=conflict)
    record_key = evaluator_record_key(
        body.project_id,
        body.evaluator_name,
        agent_name,
        compose_evaluator_key_tail(registration_id, stream_instance_id),
    )
    saved = await upsert_runtime_config(
        kind=KIND_EVALUATOR,
        record_key=record_key,
        payload=stamped,
        scope_key=project_scope_key(body.project_id),
    )
    await refresh_eval_config_snapshot()
    benchmark_name = benchmark_name_for_evaluator(body.evaluator_name)
    if benchmark_name is not None:
        await ensure_default_trace_report(
            TraceReportSeed(
                name=f"{body.evaluator_name}-{agent_name or body.project_id}",
                benchmark_name=benchmark_name,
                source=body.ingest,
                agent_name=agent_name or None,
                project_id=body.project_id,
                agent_registration_id=registration_id or None,
                stream_instance_id=stream_instance_id or None,
            )
        )
    entry = EvalConfigEntry.model_validate(stamped)
    return _public_eval(body.project_id, entry, saved)


@router.delete("/evaluators")
async def delete_evaluator(
    query: Annotated[DeleteEvaluatorQuery, Query()],
) -> dict[str, Any]:
    """Delete one eval config."""
    record_key = evaluator_record_key(
        query.project_id,
        query.evaluator_name,
        query.agent_name,
        compose_evaluator_key_tail(
            query.agent_registration_id,
            query.stream_instance_id,
        ),
    )
    deleted = await retire_runtime_config(KIND_EVALUATOR, record_key)
    if not deleted:
        raise HTTPException(status_code=404, detail="evaluator config not found")
    await refresh_eval_config_snapshot()
    return {"deleted": True}


@router.get("/odps-jobs")
async def list_odps_jobs() -> list[dict[str, Any]]:
    """List ODPS jobs already stored, including fields written from env vars."""
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_ODPS_JOB,
        deleted_at__isnull=True,
    ).order_by("record_key")
    return [_public_job(row.payload if isinstance(row.payload, dict) else {}, row) for row in rows]


@router.put("/odps-jobs")
async def upsert_odps_job(body: OdpsJobWrite) -> dict[str, Any]:
    """Save one ODPS job. Timezone, file name, and SQL are written from env vars."""
    stamped = _stamp_or_400(stamp_odps_job, body.model_dump())
    job_id = str(stamped["id"])
    project_id = str(stamped.get("project_id") or "")
    saved = await upsert_runtime_config(
        kind=KIND_ODPS_JOB,
        record_key=odps_record_key(job_id),
        payload=stamped,
        scope_key=project_scope_key(project_id),
    )
    await refresh_odps_export_snapshot()
    return _public_job(stamped, saved)


@router.delete("/odps-jobs/{job_id}")
async def delete_odps_job(job_id: str) -> dict[str, Any]:
    """Delete one ODPS job."""
    deleted = await retire_runtime_config(KIND_ODPS_JOB, odps_record_key(job_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="odps job not found")
    await refresh_odps_export_snapshot()
    return {"deleted": True}


__all__ = ["router"]
