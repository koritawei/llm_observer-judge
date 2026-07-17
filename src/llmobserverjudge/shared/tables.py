"""SQLAlchemy ORM tables for STEP-11 persistence."""

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


class PendingEvalJobRow(Base):
    __tablename__ = "pending_eval_jobs"
    __table_args__ = (UniqueConstraint("tenant_id", "trace_id", name="uq_pending_tenant_trace"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    trace_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    prompt_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    published: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class HumanAnnotationRow(Base):
    __tablename__ = "human_annotations"

    annotation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    eval_agent_id: Mapped[str] = mapped_column(String(128))
    trigger_reason: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    auto_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    auto_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    conflict_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(256), unique=True, nullable=True)
    annotator_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    human_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    human_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    score_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    label_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    human_root_cause: Mapped[str | None] = mapped_column(String(256), nullable=True)
    evidence_sufficient: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    fix_feasible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    annotation_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class AttributionResultRow(Base):
    __tablename__ = "bad_case_attributions"

    attribution_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(256), unique=True, nullable=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
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
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BenchmarkReportRow(Base):
    __tablename__ = "benchmark_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    batch_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
