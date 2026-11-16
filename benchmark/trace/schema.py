"""trace_resolver 数据模型：skeleton / detail 契约。

实例凭证模型 :class:`LangfuseInstance` 定义在 ``benchmark.settings.trace_endpoints``
（config 层，与加载器同地）；本模块 re-export 供包内使用，保持
``from benchmark.trace.schema import LangfuseInstance`` 兼容。

一个 project_id 对应一套实例凭证。入口用 ``metadata_key``（默认
``trace_id``）把业务 id filter 成 Langfuse 主键；下游读写一律用主键。
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from benchmark.settings.trace_endpoints import TraceEndpoint


def _to_preview(value: object, max_chars: int) -> str | None:
    """Serialize any value to a short string preview, truncating if needed."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "\u2026"
    return text


class TraceDocument(BaseModel):
    """A single Langfuse trace, parsed from the public API response.

    The list endpoint (``TraceWithDetails``) and the detail endpoint
    (``TraceWithFullDetails``) differ only in how much they inline. We declare
    the common trace-level fields for typed access and keep everything else
    (``observations`` / ``scores`` / future fields) verbatim via
    ``extra="allow"`` — Langfuse can add fields without breaking parsing.

    Langfuse emits camelCase, so ``alias_generator=to_camel`` maps it onto these
    snake_case attrs; ``serialize_by_alias=True`` makes ``model_dump()`` emit the
    original camelCase keys again, so a parsed trace round-trips back to the raw
    payload (declared fields and extras alike). Freeform fields are typed
    ``Any`` so an unexpected upstream shape never drops an otherwise valid trace.
    """

    model_config = ConfigDict(
        extra="allow",
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    id: str | None = None
    name: str | None = None
    timestamp: str | None = None
    environment: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    tags: list[str] | None = None
    input: Any = None
    output: Any = None
    metadata: Any = None
    latency: float | None = None
    total_cost: float | None = None


class TraceListResponse(BaseModel):
    """The ``GET /api/public/traces`` envelope (``{"data": [...], ...}``)."""

    model_config = ConfigDict(extra="allow")

    data: list[TraceDocument] = Field(default_factory=list)


class TraceSpanSkeleton(BaseModel):
    """Lightweight representation of one Langfuse observation (span / generation / event).

    ``input_preview`` and ``output_preview`` are client-side truncations of the
    full input/output — never the raw values. Use ``ObservationDetail`` for
    full content.  ``status`` maps from the API's ``level`` field.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    id: str
    name: str | None = None
    type: str | None = None
    status: str | None = None
    parent_observation_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    input_preview: str | None = None
    output_preview: str | None = None


class TraceSkeletonResult(BaseModel):
    """Result of a skeleton fetch: primary_id, project context, and span list.

    ``truncated`` is ``True`` when the observation list was capped to
    ``max_spans`` and more observations exist upstream.
    """

    primary_id: str
    business_trace_id: str
    project_id: str
    spans: list[TraceSpanSkeleton]
    truncated: bool = False
    instance_name: str | None = None


class ResolvedTrace(BaseModel):
    """Full trace + originating instance context.

    Returned by :meth:`TraceResolver.resolve` — combines the fetched
    ``LangfuseTrace`` with the ``LangfuseInstance`` that served it, so
    downstream code (e.g. trace_meta digest) has both the trace data and
    the instance metadata (name, base_url, metadata_key) in one object.
    """

    trace: TraceDocument
    instance: TraceEndpoint


class ObservationDetail(BaseModel):
    """Full detail for a single Langfuse observation.

    ``extra="allow"`` preserves any upstream fields not declared here.
    """

    model_config = ConfigDict(
        extra="allow",
        alias_generator=to_camel,
        populate_by_name=True,
    )

    id: str | None = None
    trace_id: str | None = None
    name: str | None = None
    type: str | None = None
    input: Any = None
    output: Any = None
    metadata: Any = None
    status_message: str | None = None


class GenerationSnapshot(BaseModel):
    """通用 generation observation 的无损读取结果（不含任何业务语义）。

    预算裁剪按 message 边界对齐，禁止无标记截断：

    - ``read_message_end`` 为已读区间的开区间右端（``read_message_start +
      已读条数``）；小于 ``input_message_count`` 表示仅读到
      ``messages[read_message_start:read_message_end]``，需通过 message range
      补读剩余部分；无 message 结构时为 None。
    - ``evidence_incomplete=True`` 时 ``incomplete_reasons`` 记录原因码
      （如 ``missing_generation`` / ``cross_trace_rejected`` / ``over_budget``），
      由调用方按自身状态机解释，本模型不内置业务枚举。
    """

    observation_id: str
    name: str | None = None
    type: str | None = None
    status: str | None = None
    parent_observation_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    #: 已读 input：消息仅保留 read_message_* 区间，字典保留兄弟字段。
    #: 完整输入保存在 TraceRequestBound.detail_cache，供后续补读。
    input: Any = None
    #: 原始 output（超长时截断并置 ``output_truncated``）。
    output: Any = None
    #: input 中识别出的 message 总数；无 message 结构时为 None。
    input_message_count: int | None = None
    read_message_start: int = 0
    read_message_end: int | None = None
    output_truncated: bool = False
    evidence_incomplete: bool = False
    incomplete_reasons: list[str] = Field(default_factory=list)
    #: 读取失败的诊断文本，与 missing_generation 原因码区分。
    read_error: str | None = None
