"""Pydantic port of Langfuse ``@langfuse/shared`` Zod schemas.

These models are **manual ports** (not imports) because the streaming
pipeline is a Python module while the source schemas are TypeScript / Zod.

Eval config schemas (``VariableMapping`` / ``RemoteConfigModelConfig`` /
``EvalConfigEntry`` / ``EvalConfigFile``) live in :mod:`benchmark.domain.schemas`.

Source files:
    - ``packages/shared/src/features/evals/observationForEval.ts`` — observationForEvalSchema
    - ``packages/shared/src/server/repositories/definitions.ts`` — observationRecordInsertSchema

.. note::
    Migrated from ``benchmark.judge.schemas`` (streaming migration P1, move only).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, JsonValue

# ---------------------------------------------------------------------------
# Observation data (used by evaluate)
# ---------------------------------------------------------------------------


class ObservationForEval(BaseModel):
    """S3 observation data — ported from Langfuse ``observationForEvalSchema``.

    Downloaded from S3 path ``{prefix}evals/{projectId}/observations/{spanId}.json``.

    Only the fields used by the streaming pipeline are typed strictly; the rest are
    kept as ``Any`` for forward-compatibility (Langfuse may add new fields).
    """

    model_config = {"extra": "allow"}

    # -- Identifiers ------------------------------------------------------
    span_id: str = Field(alias="span_id", description="Observation ID")
    trace_id: str = Field(alias="trace_id", description="所属 trace ID")
    project_id: str = Field(alias="project_id", description="项目 ID")
    parent_span_id: str | None = Field(default=None, alias="parent_span_id")

    # -- Core attributes --------------------------------------------------
# Native Langfuse enum: SPAN / EVENT / GENERATION / AGENT / TOOL.
# TRACE is a judge-service internal extension — used when trace_to_observation() turns a trace record into a
# synthetic observation, meaning "trace-level evaluation" (not a Langfuse observation).
    type: Literal["SPAN", "EVENT", "GENERATION", "AGENT", "TOOL", "TRACE"] = Field(
        description="Observation 类型（TRACE 为 judge-service 内部扩展，用于 trace 级别评估）",
    )
    name: str = Field(description="Observation 名称")
    environment: str = Field(description="环境")
    version: str | None = None
    level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] = Field(
        default="DEFAULT",
        description="日志级别",
    )
    status_message: str | None = None

    # -- Trace-level ------------------------------------------------------
    trace_name: str | None = None
    trace_timestamp: str | int | None = None
    user_id: str | None = None
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    release: str | None = None

    # -- Model ------------------------------------------------------------
    provided_model_name: str | None = None
    model_parameters: dict[str, Any] | None = None

    # -- Prompt -----------------------------------------------------------
    prompt_id: str | None = None
    prompt_name: str | None = None
    prompt_version: str | int | None = None

    # -- Usage & Cost -----------------------------------------------------
    provided_usage_details: dict[str, Any] = Field(default_factory=dict)
    provided_cost_details: dict[str, Any] = Field(default_factory=dict)
    usage_details: dict[str, Any] = Field(default_factory=dict)
    cost_details: dict[str, Any] = Field(default_factory=dict)

    # -- Tool calls -------------------------------------------------------
    tool_definitions: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[Any] = Field(default_factory=list)
    tool_call_names: list[str] = Field(default_factory=list)

    # -- Experiment -------------------------------------------------------
    experiment_id: str | None = None
    experiment_name: str | None = None
    experiment_description: str | None = None
    experiment_dataset_id: str | None = None
    experiment_item_id: str | None = None
    experiment_item_expected_output: str | None = None
    experiment_item_root_span_id: str | None = None

    # -- Data (used by evaluate) -----------------------------------------
    input: JsonValue | None = None
    output: JsonValue | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Record stream schemas (Langfuse RecordStreamMessage)
# ---------------------------------------------------------------------------


class TraceRecord(BaseModel):
    """Trace record — aligned with Langfuse ``TraceRecordInsertType``.

    Source: ``packages/shared/src/server/repositories/definitions.ts``
    (traceRecordInsertSchema).  Pushed to ``langfuse.records``
    topic by ``RecordStreamProducer`` after ClickHouse write.

    Field mapping to :class:`ObservationForEval`::

        id          → span_id + trace_id  (trace ID used for both)
        name        → name + trace_name
        (rest direct)
    """

    model_config = {"extra": "allow"}

    id: str
    project_id: str
    name: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    environment: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)
    input: JsonValue | None = None
    output: JsonValue | None = None
    tags: list[str] = Field(default_factory=list)
    version: str | None = None
    release: str | None = None
    timestamp: str | int | None = None
    is_deleted: int = 0
    public: bool | None = None


class RecordStreamMessage(BaseModel):
    """Message envelope — aligned with Langfuse ``RecordStreamMessage``.

    Structure::

        {"record_type": "trace" | "observation" | "score",
         "record": { ... }}

    The ``record`` field is kept as a raw dict; callers parse it into the
    appropriate schema (e.g. :class:`TraceRecord`) based on
    ``record_type``.
    """

    model_config = {"extra": "allow"}

    record_type: Literal["trace", "observation", "score"]
    record: dict[str, Any]
