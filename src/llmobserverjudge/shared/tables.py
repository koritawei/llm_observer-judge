"""SQLAlchemy ORM tables for STEP-11 persistence + multi-project."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from llmobserverjudge.shared.db import Base
from llmobserverjudge.shared.project import DEFAULT_PROJECT_ID


class ProjectRow(Base):
    """Project registry: names unique within a tenant; same project_id OK across tenants."""

    __tablename__ = "projects"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_project_tenant_name"),
    )

    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PendingEvalJobRow(Base):
    __tablename__ = "pending_eval_jobs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "project_id", "trace_id", name="uq_pending_tenant_project_trace"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    project_id: Mapped[str] = mapped_column(
        String(64), default=DEFAULT_PROJECT_ID, index=True
    )
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    trace_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    prompt_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    published: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class HumanAnnotationRow(Base):
    __tablename__ = "human_annotations"

    annotation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    project_id: Mapped[str] = mapped_column(
        String(64), default=DEFAULT_PROJECT_ID, index=True
    )
    eval_agent_id: Mapped[str] = mapped_column(String(128))
    trigger_reason: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    auto_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    auto_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    conflict_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(256), unique=True, nullable=True
    )
    annotator_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    human_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    human_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    score_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    label_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    human_root_cause: Mapped[str | None] = mapped_column(String(256), nullable=True)
    evidence_sufficient: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    fix_feasible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    annotation_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class AttributionResultRow(Base):
    __tablename__ = "bad_case_attributions"

    attribution_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(256), unique=True, nullable=True
    )
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    project_id: Mapped[str] = mapped_column(
        String(64), default=DEFAULT_PROJECT_ID, index=True
    )
    eval_agent_id: Mapped[str] = mapped_column(String(128))
    attribution_agent: Mapped[str] = mapped_column(String(128))
    root_cause: Mapped[str] = mapped_column(String(256))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=list)
    tools_called: Mapped[list[Any]] = mapped_column(JSON, default=list)
    rounds: Mapped[int] = mapped_column(Integer, default=0)
    early_stop: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(32), default="COMPLETED")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)


class PromptVersionRow(Base):
    __tablename__ = "prompt_versions"

    version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    prompt_id: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="DRAFT", index=True)
    variables: Mapped[list[Any]] = mapped_column(JSON, default=list)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class BenchmarkReportRow(Base):
    __tablename__ = "benchmark_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    project_id: Mapped[str] = mapped_column(
        String(64), default=DEFAULT_PROJECT_ID, index=True
    )
    batch_job_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class SpanRow(Base):
    __tablename__ = "spans"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "project_id", "span_id", name="uq_span_tenant_project_span"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    project_id: Mapped[str] = mapped_column(
        String(64), default=DEFAULT_PROJECT_ID, index=True
    )
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    span_id: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EvalResultRow(Base):
    __tablename__ = "eval_results"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "project_id", "trace_id", name="uq_eval_tenant_project_trace"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    project_id: Mapped[str] = mapped_column(
        String(64), default=DEFAULT_PROJECT_ID, index=True
    )
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(256), unique=True, nullable=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PromptMetaRow(Base):
    __tablename__ = "prompt_meta"

    prompt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    category: Mapped[str] = mapped_column(String(64), default="JUDGE")
    tenant_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    project_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class PromptExperimentRow(Base):
    __tablename__ = "prompt_experiments"

    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    prompt_id: Mapped[str] = mapped_column(String(128), index=True)
    control_version_id: Mapped[str] = mapped_column(String(64))
    treatment_version_id: Mapped[str] = mapped_column(String(64))
    traffic_pct: Mapped[int] = mapped_column(Integer, default=20)
    status: Mapped[str] = mapped_column(String(32), default="RUNNING", index=True)
    metric_primary: Mapped[str] = mapped_column(String(64), default="mean_score")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    conclusion: Mapped[str | None] = mapped_column(Text, nullable=True)
    winner_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AttributionCheckpointRow(Base):
    __tablename__ = "attribution_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "project_id",
            "trace_id",
            name="uq_checkpoint_tenant_project_trace",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(128), default="", index=True
    )
    project_id: Mapped[str] = mapped_column(
        String(64), default=DEFAULT_PROJECT_ID, index=True
    )
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class IdempotencyKeyRow(Base):
    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
