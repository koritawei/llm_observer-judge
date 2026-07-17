"""Pydantic models aligned with Frozen STEP-01 JSON Schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpanKind(StrEnum):
    LLM = "LLM"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    RETRIEVAL = "RETRIEVAL"
    OTHER = "OTHER"


class SpanStatus(StrEnum):
    OK = "OK"
    ERROR = "ERROR"


class Priority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class ScoreStatus(StrEnum):
    SCORED = "scored"
    SKIPPED = "skipped"
    INFRA_ERROR = "infra_error"
    MODEL_ERROR = "model_error"


class Span(BaseModel):
    """Implements: docs/sdd/04-contracts/schemas/span.schema.json"""

    model_config = ConfigDict(extra="allow")

    tenant_id: str = Field(min_length=1)
    project_id: str = Field(default="default", min_length=1)
    trace_id: str
    span_id: str
    name: str
    timestamp: datetime
    parent_span_id: str | None = None
    kind: SpanKind | None = None
    duration_ms: int | None = None
    status: SpanStatus | None = None
    model: str | None = None
    prompt: str | None = None
    response: str | None = None
    tool_calls: str | None = None
    token_input: int = 0
    token_output: int = 0
    temperature: float | None = None
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    sampled: bool | None = None
    traceparent: str | None = None

    @field_validator("trace_id")
    @classmethod
    def _trace_id_hex(cls, v: str) -> str:
        if len(v) != 32 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("trace_id must be 32 lowercase hex chars")
        return v

    @field_validator("span_id", "parent_span_id")
    @classmethod
    def _span_id_hex(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 16 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("span_id must be 16 lowercase hex chars")
        return v


class IngestRequest(BaseModel):
    spans: list[Span] = Field(min_length=1, max_length=500)


class IngestAccepted(BaseModel):
    status: str = "accepted"
    count: int


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    agent_id: str
    model: str
    prompt_id: str | None = None
    rubric: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    condition: str | None = None
    threshold: float = 0.5
    weight: float = 1.0
    scale_max: float = 10.0


class EvalJob(BaseModel):
    """Implements: eval-job.schema.json — publisher: Dispatch only."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    tenant_id: str
    project_id: str = "default"
    trace_summary: dict[str, Any]
    eval_configs: list[EvalConfig] = Field(min_length=1)
    priority: Priority
    idempotency_key: str
    created_at: datetime
    prompt_version_id: str | None = None
    eval_config_hash: str | None = None
    outbox_id: str | None = None


class EvalScoreEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    agent_id: str
    score_status: ScoreStatus
    raw_score: float | None = None
    normalized_score: float | None = None
    is_bad_case: bool | None = None
    label: str | None = None
    threshold: float | None = None
    weight: float | None = None
    reasoning: str | None = None
    conflict_detected: bool | None = None
    skip_reason: str | None = None
    error: str | None = None
    error_type: str | None = None


class ScoreResult(BaseModel):
    """Implements: score-result.schema.json"""

    model_config = ConfigDict(extra="allow")

    tenant_id: str
    project_id: str = "default"
    trace_id: str
    eval_scores: list[EvalScoreEntry]
    final_score: float
    bad_case_detected: bool
    prompt_version_id: str | None = None
    bad_case_evals: list[str] = Field(default_factory=list)


class GraphState(BaseModel):
    """Implements: graph-state.schema.json"""

    model_config = ConfigDict(extra="allow")

    trace_id: str
    tenant_id: str
    project_id: str = "default"
    trace_summary: dict[str, Any]
    prompt_version_id: str | None = None
    eval_configs: list[dict[str, Any]] = Field(default_factory=list)
    eval_results: list[dict[str, Any]] = Field(default_factory=list)
    eval_scores: list[dict[str, Any]] = Field(default_factory=list)
    eval_execution_layers: list[list[str]] = Field(default_factory=list)
    final_score: float = 0.0
    bad_case_detected: bool = False
    bad_case_evals: list[str] = Field(default_factory=list)
    attribution_results: list[dict[str, Any]] = Field(default_factory=list)
    annotation_task_ids: list[str] = Field(default_factory=list)
    annotation_task_count: int = 0
    node_errors: dict[str, str] = Field(default_factory=dict)
    optional_skipped: list[str] = Field(default_factory=list)
    execution_graph: list[dict[str, Any]] = Field(default_factory=list)


class PendingEvalJob(BaseModel):
    """Outbox intent row — not an MQ message."""

    id: str
    trace_id: str
    tenant_id: str
    project_id: str = "default"
    trace_summary: dict[str, Any]
    prompt_version_id: str | None = None
    published: bool = False
    attempt_count: int = 0


class AnnotationStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"


class AnnotationTask(BaseModel):
    """MQ payload — asyncapi/annotation-jobs.yaml."""

    model_config = ConfigDict(extra="allow")

    annotation_id: str
    trace_id: str
    tenant_id: str
    project_id: str = "default"
    eval_agent_id: str
    trigger_reason: str
    auto_score: float | None = None
    auto_label: str | None = None
    conflict_detected: bool = False
    idempotency_key: str | None = None


class HumanAnnotation(BaseModel):
    """Durable annotation record (design §9 / HumanAnnotation)."""

    model_config = ConfigDict(extra="allow")

    annotation_id: str
    trace_id: str
    tenant_id: str
    project_id: str = "default"
    eval_agent_id: str
    trigger_reason: str
    status: AnnotationStatus = AnnotationStatus.PENDING
    auto_score: float | None = None
    auto_label: str | None = None
    conflict_detected: bool = False
    idempotency_key: str | None = None
    annotator_id: str | None = None
    human_score: float | None = None
    human_label: str | None = None
    score_delta: float | None = None
    label_correct: bool | None = None
    human_root_cause: str | None = None
    evidence_sufficient: bool | None = None
    fix_feasible: bool | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    annotation_time_ms: int | None = None


class AttributionJob(BaseModel):
    """Implements: asyncapi/attribution-jobs.yaml — publisher: Dispatch only."""

    model_config = ConfigDict(extra="allow")

    trace_id: str
    tenant_id: str
    project_id: str = "default"
    eval_agent_id: str
    priority: Priority
    idempotency_key: str
    attribution_agent: str = "attributor_default"
    score_snapshot: dict[str, Any] = Field(default_factory=dict)
    trace_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class AttributionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    attribution_id: str
    trace_id: str
    tenant_id: str
    project_id: str = "default"
    eval_agent_id: str
    attribution_agent: str
    root_cause: str
    confidence: float
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    rounds: int = 0
    early_stop: bool = False
    status: str = "COMPLETED"
    error: str | None = None
    duration_ms: int = 0


class BatchTaskStatus(StrEnum):
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEAD = "DEAD"


class BatchJobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BatchJob(BaseModel):
    model_config = ConfigDict(extra="allow")

    batch_job_id: str
    tenant_id: str
    project_id: str = "default"
    name: str = ""
    source_config: dict[str, Any] = Field(default_factory=dict)
    eval_configs: list[dict[str, Any]] = Field(default_factory=list)
    sampling_config: dict[str, Any] = Field(default_factory=dict)
    priority: Priority = Priority.P2
    status: BatchJobStatus = BatchJobStatus.QUEUED
    total_tasks: int = 0
    timeout_seconds: int = 3600
    created_at: datetime | None = None
    completed_at: datetime | None = None


class BatchEvalTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    batch_job_id: str
    trace_id: str
    tenant_id: str = ""
    project_id: str = "default"
    eval_config_id: str = ""
    status: BatchTaskStatus = BatchTaskStatus.PENDING
    attempt: int = 0
    max_retries: int = 3
    worker_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    trace_summary: dict[str, Any] = Field(default_factory=dict)
    dispatched_at: datetime | None = None
    completed_at: datetime | None = None


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    report_id: str
    tenant_id: str
    project_id: str = "default"
    batch_job_id: str | None = None
    date: str | None = None  # YYYY-MM-DD for T+1
    summary: dict[str, Any] = Field(default_factory=dict)
    dimensions: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)
    bad_case_analysis: dict[str, Any] = Field(default_factory=dict)
    human_annotation_summary: dict[str, Any] = Field(default_factory=dict)
    insights: list[str] = Field(default_factory=list)
    sampling_meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
