"""``trace`` ToolProvider：按需读 observation。

Runtime 单例，禁止在实例上挂请求态；请求态用 :class:`TraceRequestBound`
（``bound=`` / ContextVar via :func:`bind_trace_request`）。
YAML ``tools.providers: [trace]`` 注册，业务 agent 无需白名单。

默认走 Langfuse Public API v1。sidecar MetaQ 入口在
``TraceRequestBound.sidecar_observations`` / ``bind_sidecar_observations``
上挂 observation 树时，四个读工具只裁剪这份数据，不打 Langfuse。
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from benchmark.libs.agenthost.app.providers import ProviderRegistry
from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.agenthost.tools.base import ToolDefinition
from benchmark.libs.agenthost.tools.component_metadata import (
    ToolProviderContext,
    _build_tool_component_provider,
    tool_component,
)
from benchmark.libs.agenthost.tools.content import ToolResult
from benchmark.libs.agenthost.tools.context import ToolExecutionContext
from benchmark.libs.agenthost.tools.providers.base_provider import BaseToolProvider
from pydantic import BaseModel, Field, ValidationError

from benchmark.trace.http_backend import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RESOLVE_LOOKBACK_DAYS,
    DEFAULT_TIMEOUT_SECONDS,
    TraceHttpError,
)
from benchmark.trace.resolver import FIND_MAX_OBSERVATIONS, TraceResolver
from benchmark.trace.schema import (
    GenerationSnapshot,
    ObservationDetail,
    TraceSpanSkeleton,
)
from benchmark.ingest.sidecar.observations import (
    _SIDECAR_OBSERVATIONS_KEY,
    dict_observations,
    filter_observations,
    find_observation,
    observations_to_full_trace,
    observations_to_trace_evidence,
    to_observation_detail,
    to_span_skeleton,
)

# agentgraph.tools 不像 agents 那样 eager 注册 adapter；agent-eval 又不依赖
# libs/tools（那边 import 时才会 register）。在本模块加载时补上，否则
# @tool_component 只有元数据、coerce 失败 → tools.providers: [trace] LookupError。
# 同时重导出 _build_tool_component_provider，供 ProviderRegistry convention 路径 1
# （查 provider 所在 module 属性）兜底。
ProviderRegistry.register_metadata_adapter("tool", _build_tool_component_provider)

logger = get_logger("agent_eval.trace_tools")

_ARGS_SUMMARY_STRING_LIMIT = 200
#: skeleton 摘要字段瘦身后的保留长度。
_SKELETON_PREVIEW_CHARS = 64

#: 归因快照独立预算默认值：按归因模型 context window 确定，不复用
#: ``max_observation_chars``（8000）的通用正文 cap。
DEFAULT_MAX_ATTRIBUTION_SNAPSHOT_CHARS = 60_000

#: Request-local bound; set via :func:`bind_trace_request` so
#: ``ToolManager.call_tool(name, args)`` works without a custom ``bound=`` kwarg.
_trace_bound_var: ContextVar[Any] = ContextVar("agent_eval_trace_bound", default=None)

#: Sidecar MetaQ unpacked observation tree for this evaluate() request.
_sidecar_observations_var: ContextVar[list[Any] | None] = ContextVar(
    "agent_eval_sidecar_observations",
    default=None,
)


@dataclass(frozen=True, kw_only=True)
class GenerationMessageRange:
    """单次 generation 消息范围读取参数，结束位置不包含在内。"""

    message_start: int
    message_end: int
    field: str = "input"
    max_chars: int | None = None


class FetchSkeletonArgs(BaseModel):
    """Arguments for ``fetch_trace_skeleton``."""

    max_spans: int = Field(default=300, ge=1, le=500)
    preview_chars: int = Field(default=180, ge=1, le=2_000)


class FetchObservationDetailArgs(BaseModel):
    """Arguments for ``fetch_observation_detail``."""

    observation_id: str = Field(min_length=1)
    max_chars: int | None = Field(default=None, ge=1, le=100_000)


class FetchFullTraceArgs(BaseModel):
    fields: str | None = None


class FindObservationsArgs(BaseModel):
    """Arguments for ``find_observations``.

    At least one of ``name`` / ``type`` / ``status`` is required — unfiltered
    full-trace listing belongs to ``fetch_trace_skeleton``, not find.
    """

    name: str | None = None
    type: str | None = None
    status: str | None = None

    def model_post_init(self, __context: object) -> None:
        if not any(
            (
                (self.name or "").strip(),
                (self.type or "").strip(),
                (self.status or "").strip(),
            )
        ):
            raise ValueError(
                "find_observations requires at least one of name/type/status; "
                "use fetch_trace_skeleton for an unfiltered span list"
            )


@contextmanager
def bind_trace_request(bound: "TraceRequestBound") -> Iterator["TraceRequestBound"]:
    """Bind ``TraceRequestBound`` for the current async/task context."""

    token = _trace_bound_var.set(bound)
    try:
        yield bound
    finally:
        _trace_bound_var.reset(token)


def current_trace_bound() -> "TraceRequestBound | None":
    """Return the ContextVar-bound request context, if any."""

    return _trace_bound_var.get()


@contextmanager
def bind_sidecar_observations(
    observations: list[Any] | None,
) -> Iterator[list[Any] | None]:
    """Bind sidecar observation tree for the current evaluate() request."""

    token = _sidecar_observations_var.set(observations)
    try:
        yield observations
    finally:
        _sidecar_observations_var.reset(token)


def current_sidecar_observations() -> list[Any] | None:
    """Return the request-scoped sidecar observation tree, if any."""

    value = _sidecar_observations_var.get()
    return value if isinstance(value, list) else None


def sidecar_trace_evidence(
    observations: object,
    trace_id: str | None,
) -> dict[str, Any] | None:
    """Build preprocess evidence from a local sidecar tree, else ContextVar."""
    local = observations if isinstance(observations, list) else current_sidecar_observations()
    if not isinstance(local, list):
        return None
    return observations_to_trace_evidence(local, trace_id)


def _sidecar_observations(bound: "TraceRequestBound") -> list[Any] | None:
    if isinstance(bound.sidecar_observations, list):
        return bound.sidecar_observations
    return current_sidecar_observations()


def _sidecar_observations_from_metadata(metadata: dict[str, Any]) -> list[Any] | None:
    top = metadata.get(_SIDECAR_OBSERVATIONS_KEY)
    if isinstance(top, list):
        return top
    er = metadata.get("evaluator_request")
    er_dict = er if isinstance(er, dict) else {}
    nested_meta = er_dict.get("metadata") if isinstance(er_dict.get("metadata"), dict) else {}
    nested = nested_meta.get(_SIDECAR_OBSERVATIONS_KEY)
    if isinstance(nested, list):
        return nested
    return current_sidecar_observations()


def _attribution_error(
    detail: ObservationDetail,
    obs_id: str,
    expected_trace_id: str,
) -> str | None:
    if detail.trace_id is None:
        logger.warning(
            "trace.detail_no_trace_id obs_id=%s expected=%s",
            obs_id,
            expected_trace_id,
        )
        return (
            f"cannot validate observation attribution: "
            f"observation {obs_id!r} has no trace_id "
            f"(expected {expected_trace_id!r})"
        )
    if detail.trace_id != expected_trace_id:
        logger.warning(
            "trace.detail_cross_trace obs_id=%s detail_trace=%s expected=%s",
            obs_id,
            detail.trace_id,
            expected_trace_id,
        )
        return (
            f"observation {obs_id!r} belongs to trace {detail.trace_id!r}, "
            f"expected {expected_trace_id!r}"
        )
    return None


def _is_bound_like(value: Any) -> bool:  # noqa: ANN401
    """TraceRequestBound 鸭子型检查。

    importlib.reload 等场景会产生多份类身份，严格 isinstance 会误拒；
    鸭子型检查在类身份不一致时仍可识别合法 bound。
    """

    return isinstance(value, TraceRequestBound) or (
        hasattr(value, "detail_cache")
        and hasattr(value, "known_refs")
        and hasattr(value, "primary_id")
    )


def _resolve_bound(kwargs: dict[str, Any]) -> "TraceRequestBound | None":
    """Resolve bound from explicit kwarg, ToolExecutionContext.extra, or ContextVar."""

    bound = kwargs.get("bound")
    if _is_bound_like(bound):
        return bound
    context = kwargs.get("context")
    if isinstance(context, ToolExecutionContext):
        extra_bound = context.extra.get("bound") or context.extra.get("trace_bound")
        if _is_bound_like(extra_bound):
            return extra_bound
    return _trace_bound_var.get()


def _dumps(obj: Any) -> str:  # noqa: ANN401
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _summarize_args(args: Any) -> dict[str, Any]:  # noqa: ANN401
    """Trim args for trace input without recording full large string values."""

    if args is None:
        return {}
    if isinstance(args, dict):
        return {
            key: (
                "<str>"
                if isinstance(value, str) and len(value) > _ARGS_SUMMARY_STRING_LIMIT
                else value
            )
            for key, value in args.items()
        }
    return {"_type": type(args).__name__}


def _trace_span_output(result: Any, max_chars: int) -> dict[str, Any]:  # noqa: ANN401
    """Build observable tool output with bounded concrete result data."""

    body = getattr(result, "body", None)
    if body is None:
        body = getattr(result, "content", "") or getattr(result, "error", "") or ""
    if not isinstance(body, str):
        body = str(body)

    body_chars = len(body)
    truncated = max_chars > 0 and body_chars > max_chars
    visible_body = f"{body[:max_chars]}…[truncated]" if truncated else body
    data: Any = visible_body
    if not truncated:
        try:
            data = json.loads(visible_body)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "success": getattr(result, "success", None),
        "data": data,
        "body_chars": body_chars,
        "body_truncated": truncated,
    }


def _span_ids(items: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in items:
        raw = item.get("id")
        if isinstance(raw, str) and raw.strip():
            out.append(raw.strip())
    return out


def _slim_span(item: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy previews so an oversized single span can still emit its id."""

    slimmed = dict(item)
    for key in ("input_preview", "output_preview"):
        val = slimmed.get(key)
        if isinstance(val, str) and len(val) > _SKELETON_PREVIEW_CHARS:
            slimmed[key] = val[:_SKELETON_PREVIEW_CHARS] + "…"
    return slimmed


def _fit_span_list(
    items: list[dict[str, Any]],
    max_chars: int,
    *,
    envelope: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Drop trailing spans until JSON fits ``max_chars``. Always valid JSON."""

    def _empty() -> tuple[str, list[str]]:
        if envelope is None:
            return "[]", []
        return _dumps({**envelope, "spans": [], "body_truncated": True}), []

    if not items:
        return _empty() if envelope is not None else ("[]", [])

    for slim_round in (False, True):
        candidates = [_slim_span(x) for x in items] if slim_round else list(items)
        # Prefer keeping at least one span; empty is only the last resort
        # after slim also fails (avoids accepting n=0 before trying slim).
        n = len(candidates)
        while n > 0:
            kept = candidates[:n]
            if envelope is None:
                payload: Any = kept
            else:
                payload = {
                    **envelope,
                    "spans": kept,
                    "body_truncated": n < len(items) or slim_round,
                }
            body = _dumps(payload)
            if max_chars <= 0 or len(body) <= max_chars:
                return body, _span_ids(kept)
            n -= 1

    return _empty()


def _preview_value(value: Any, *, limit: int = 240) -> Any:  # noqa: ANN401
    if value is None:
        return None
    text = value if isinstance(value, str) else _dumps(value)
    if len(text) <= limit:
        return value if isinstance(value, str) else text
    return text[:limit] + "…[truncated]"


def _apply_arg_max_chars(data: dict[str, Any], max_chars: int | None) -> dict[str, Any]:
    """Apply per-call ``max_chars`` to input/output before body-cap fitting."""

    if max_chars is None:
        return data
    out = dict(data)
    if out.get("input") is not None:
        out["input"] = _preview_value(out.get("input"), limit=max_chars)
    if out.get("output") is not None:
        out["output"] = _preview_value(out.get("output"), limit=max_chars)
    return out


def _fit_detail(data: dict[str, Any], max_chars: int) -> str:
    """Shrink detail JSON until it fits ``max_chars`` (always valid JSON)."""

    body = _dumps(data)
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    shrunk = {
        **data,
        "input": _preview_value(data.get("input"), limit=8_000),
        "output": _preview_value(data.get("output"), limit=8_000),
        "body_truncated": True,
    }
    body = _dumps(shrunk)
    if len(body) <= max_chars:
        return body
    shrunk["input"] = None
    shrunk["output"] = None
    body = _dumps(shrunk)
    if len(body) <= max_chars:
        return body

    # Drop metadata / extras / status_message; then compact + strip until fit.
    base_minimal: dict[str, Any] = {
        "id": data.get("id"),
        "trace_id": data.get("trace_id"),
        "name": data.get("name"),
        "type": data.get("type"),
        "body_truncated": True,
    }
    for name_limit in (64, 32, 16, 8, None):
        candidate = dict(base_minimal)
        if name_limit is None:
            candidate.pop("name", None)
            candidate.pop("type", None)
        else:
            name = candidate.get("name")
            if isinstance(name, str) and len(name) > name_limit:
                candidate["name"] = name[:name_limit] + "…"
        body = _dumps(candidate)
        if len(body) <= max_chars:
            return body
        compact = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= max_chars:
            return compact
    return json.dumps(
        {"id": data.get("id"), "body_truncated": True},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _render_detail(
    detail: ObservationDetail,
    *,
    arg_max_chars: int | None,
    body_cap: int,
) -> str:
    """Render a cached/full detail with per-call and provider body caps."""

    data = _apply_arg_max_chars(detail.model_dump(by_alias=False), arg_max_chars)
    return _fit_detail(data, body_cap)


def _serialize_value(value: Any) -> str:  # noqa: ANN401
    """Deterministic JSON serialization for budget accounting (no truncation)."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _extract_messages(value: Any) -> tuple[list[Any] | None, str]:  # noqa: ANN401
    """Best-effort generic message extraction from a generation input.

    Returns ``(messages, field_path)``: a top-level list is treated as the
    message sequence (path ``input``); a dict with a ``messages`` list uses
    path ``input.messages``. Anything else yields ``(None, "")`` — no
    message-boundary trimming applies.
    """

    if isinstance(value, list):
        return value, "input"
    if isinstance(value, dict):
        messages = value.get("messages")
        if isinstance(messages, list):
            return messages, "input.messages"
    return None, ""


def _snapshot_from_detail(detail: ObservationDetail) -> GenerationSnapshot:
    """Build a snapshot skeleton from a validated detail (no budget applied).

    ``ObservationDetail`` extras keep the upstream key casing (Langfuse emits
    camelCase), so lookups accept both camelCase and snake_case.
    """

    extras = getattr(detail, "__pydantic_extra__", None) or {}

    def _extra(*keys: str) -> Any:  # noqa: ANN401
        for key in keys:
            if extras.get(key) is not None:
                return extras[key]
        return None

    return GenerationSnapshot(
        observation_id=str(detail.id or ""),
        name=detail.name,
        type=detail.type,
        status=str(_extra("level") or "") or None,
        parent_observation_id=(
            str(v) if (v := _extra("parentObservationId", "parent_observation_id")) else None
        ),
        start_time=str(v) if (v := _extra("startTime", "start_time")) else None,
        end_time=str(v) if (v := _extra("endTime", "end_time")) else None,
        input=detail.input,
        output=detail.output,
    )


def _apply_snapshot_budget(
    snapshot: GenerationSnapshot,
    detail: ObservationDetail,
    budget: int,
) -> None:
    """Apply the attribution budget in-place: output first, then input messages.

    Message-boundary aligned: whole messages are kept until the budget is
    exhausted; the first message that overflows is dropped and ``over_budget``
    is recorded with the read range. No generative summarization, ever.
    """

    remaining = budget

    output_text = _serialize_value(detail.output) if detail.output is not None else ""
    if output_text and len(output_text) > remaining:
        snapshot.output = output_text[:remaining] + "…[truncated]"
        snapshot.output_truncated = True
        snapshot.evidence_incomplete = True
        snapshot.incomplete_reasons.append("over_budget")
        remaining = 0
    else:
        remaining -= len(output_text)

    messages, _path = _extract_messages(detail.input)
    if messages is None:
        snapshot.read_message_end = None
        if detail.input is None:
            return
        if remaining <= 0 or len(_serialize_value(detail.input)) > remaining:
            snapshot.evidence_incomplete = True
            if "over_budget" not in snapshot.incomplete_reasons:
                snapshot.incomplete_reasons.append("over_budget")
        return

    snapshot.input_message_count = len(messages)
    _fit_snapshot_messages(snapshot, messages, remaining, offset=0)


def _fit_snapshot_messages(
    snapshot: GenerationSnapshot,
    messages: list[Any],
    budget: int,
    *,
    offset: int,
) -> None:
    """Keep whole messages while they fit ``budget``; mark overflow explicitly."""

    kept: list[Any] = []
    envelope = (
        {**snapshot.input, "messages": []}
        if isinstance(snapshot.input, dict)
        else None
    )
    used = len(_serialize_value(envelope)) if envelope is not None else 0
    for message in messages:
        text = _serialize_value(message)
        if kept and used + len(text) > budget:
            break
        if not kept and used + len(text) > budget:
            # 单条 message 超预算仍保留整条（禁止无标记截断），但标记超预算。
            kept.append(message)
            used += len(text)
            snapshot.evidence_incomplete = True
            if "over_budget" not in snapshot.incomplete_reasons:
                snapshot.incomplete_reasons.append("over_budget")
            break
        kept.append(message)
        used += len(text)

    if used > budget or len(kept) < len(messages):
        snapshot.evidence_incomplete = True
        if "over_budget" not in snapshot.incomplete_reasons:
            snapshot.incomplete_reasons.append("over_budget")
        snapshot.read_message_end = offset + len(kept)
    else:
        snapshot.read_message_end = offset + len(messages)
    if (
        snapshot.input_message_count is not None
        and snapshot.read_message_end < snapshot.input_message_count
    ):
        snapshot.evidence_incomplete = True
        if not {"over_budget", "message_range_unread"}.intersection(snapshot.incomplete_reasons):
            snapshot.incomplete_reasons.append("message_range_unread")
    #: 已读 message 序列经 ``input`` 字段承载（超预算时仅保留已读部分，
    #: 完整原文仍在 ``bound.detail_cache``，可通过 message range 补读）。
    snapshot.input = (
        {**snapshot.input, "messages": kept}
        if envelope is not None
        else kept if kept else None
    )


@dataclass
class TraceRequestBound:
    """Per-request mutable context for one trace / project.

    Owned by the **caller** (e.g. AttributionAgent)，not by the singleton
    provider. Pass the same instance to every ``call_tool(..., bound=...)``
    in a request so ``primary_id`` / ``detail_cache`` / ``known_refs`` stay
    isolated under concurrency.

    ``primary_id`` is the Langfuse primary id (pre-set by ``create_bound``
    after eval-entry rewrite; also refreshed on skeleton fetch).
    ``detail_cache`` deduplicates observation-detail calls within the request.
    ``known_refs`` accumulates observation ids actually emitted in tool
    bodies (evidence gate source of truth — do not re-parse truncated text).

    Per-round tool-call caps live on the agent
    (``DEFAULT_MAX_TOOL_CALLS_PER_ROUND`` / ``max_react_rounds``); body size is
    capped by the provider's ``max_observation_chars``.
    ``detail_cache`` stores **full** ObservationDetail; per-call
    ``max_chars`` is applied only when rendering a response.
    """

    trace_id: str
    project_id: str
    primary_id: str | None = field(default=None)
    detail_cache: dict[str, ObservationDetail] = field(default_factory=dict)
    known_refs: set[str] = field(default_factory=set)
    sidecar_observations: list[Any] | None = None


class TraceToolExt(BaseModel):
    """``tools.extensions.trace`` 可选扩展。"""

    max_observation_chars: int = Field(
        default=8_000,
        ge=1,
        description=(
            "Upper bound on the body length of any tool result that flows "
            "back into the LLM prompt (detail / skeleton / find). Prevents "
            "a single oversized observation from ballooning the ReAct "
            "prompt and stalling / OOM-ing the model."
        ),
    )
    enable_fetch_full_trace: bool = Field(
        default=False,
        description=(
            "慎用：为 True 时注册 fetch_full_trace（GET /traces/{id} 全字段）。"
            "框架默认 False；仅业务 YAML opt-in。大 trace 易 500/OOM/撑爆 context。"
        ),
    )
    max_attribution_snapshot_chars: int = Field(
        default=DEFAULT_MAX_ATTRIBUTION_SNAPSHOT_CHARS,
        ge=1,
        description=(
            "归因分层取证的 generation 快照独立预算（按归因模型 context window "
            "确定）。不复用 max_observation_chars；读取按 message 边界对齐裁剪，"
            "超预算显式标记而不做生成式摘要。"
        ),
    )
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        description=(
            "httpx timeout for all Langfuse Public API calls "
            "(entry resolve + skeleton / detail / find / full_trace)."
        ),
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        description=("Retries for ConnectError and HTTP 429/5xx only. Timeouts are never retried."),
    )
    resolve_lookback_days: int = Field(
        default=DEFAULT_RESOLVE_LOOKBACK_DAYS,
        ge=1,
        description=(
            "Bounded timestamp lookback (days) for entry "
            "business-id → primary-id resolve. Required by self-hosted "
            "Langfuse to avoid unbounded GET /traces scans (HTTP 422)."
        ),
    )
    enable_vui_observation_context_fallback: bool = Field(
        default=False,
        description=(
            "Temporarily backfill missing VUI Judge context from one Langfuse "
            "observations page. Keep disabled after trace-level input rollout."
        ),
    )


_TOOL_FETCH_SKELETON: ToolDefinition = {
    "name": "fetch_trace_skeleton",
    "description": (
        "Fetch a lightweight skeleton of all Langfuse observations for the "
        "currently bound trace. Returns span id, name, type, status and "
        "short input/output previews. Call this first to discover observation ids."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "max_spans": {
                "type": "integer",
                "description": "Max number of spans to return (default 300).",
            },
            "preview_chars": {
                "type": "integer",
                "description": "Max chars for input/output previews (default 180).",
            },
        },
    },
    "metadata": {"source": "langfuse"},
}

_TOOL_FETCH_OBS_DETAIL: ToolDefinition = {
    "name": "fetch_observation_detail",
    "description": (
        "Fetch full detail (input, output, metadata) for a single Langfuse "
        "observation by id. Results are cached for the duration of the request."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "observation_id": {
                "type": "string",
                "description": "Langfuse observation id (from skeleton spans).",
            },
            "max_chars": {
                "type": "integer",
                "description": "If given, truncate input/output to this many chars.",
            },
        },
        "required": ["observation_id"],
    },
    "metadata": {"source": "langfuse"},
}

_TOOL_FETCH_FULL_TRACE: ToolDefinition = {
    "name": "fetch_full_trace",
    "description": (
        "CAUTION / 慎用: Fetch the FULL Langfuse trace via "
        "GET /api/public/traces/{id} (all fields including nested "
        "observations). Large traces may 500, OOM, or blow LLM context. "
        "No page/limit on this endpoint — prefer fetch_trace_skeleton + "
        "fetch_observation_detail. Only available when "
        "tools.extensions.trace.enable_fetch_full_trace=true. "
        "Optional fields: core,io,scores,observations,metrics "
        "(omit for official full payload)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "string",
                "description": ("Optional field groups. Omit for full payload."),
            },
        },
    },
    "metadata": {"source": "langfuse", "caution": "full_trace"},
}

_TOOL_FIND_OBS: ToolDefinition = {
    "name": "find_observations",
    "description": (
        "Find observations for the bound trace with server-side filters. "
        "Requires at least one of name / type / status. For an unfiltered "
        "span list use fetch_trace_skeleton instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Filter by observation name (server-side).",
            },
            "type": {
                "type": "string",
                "description": "Filter by type, e.g. GENERATION or SPAN (server-side).",
            },
            "status": {
                "type": "string",
                "description": "Filter by status/level, e.g. ERROR (server-side level).",
            },
        },
    },
    "metadata": {"source": "langfuse"},
}


@tool_component(name="trace", ext_schema=TraceToolExt)
class TraceToolProvider(BaseToolProvider):
    """agentgraph ToolProvider backed by :class:`TraceResolver`.

    Stateless w.r.t. requests: only holds shared ``TraceResolver`` + body-size
    defaults. Callers create a :class:`TraceRequestBound` (via
    :meth:`bound_from_metadata` / :meth:`create_bound`) and pass it on every
    ``call_tool(..., bound=...)``.

    Provider name: ``trace``, source_type: ``trace`` (backend may still be Langfuse).
    """

    def __init__(
        self,
        resolver: TraceResolver,
        *,
        max_observation_chars: int = 8_000,
        enable_fetch_full_trace: bool = False,
        max_attribution_snapshot_chars: int = DEFAULT_MAX_ATTRIBUTION_SNAPSHOT_CHARS,
        enable_vui_observation_context_fallback: bool = False,
    ) -> None:
        super().__init__("trace", "trace")
        self._resolver = resolver
        self._max_observation_chars = max_observation_chars
        self._enable_fetch_full_trace = enable_fetch_full_trace
        self._max_attribution_snapshot_chars = max_attribution_snapshot_chars
        self._enable_vui_observation_context_fallback = enable_vui_observation_context_fallback

    @classmethod
    def from_config(
        cls,
        cfg: TraceToolExt | None,
        ctx: ToolProviderContext,  # noqa: ARG003
    ) -> "TraceToolProvider | None":
        """YAML ``tools.providers: [trace]`` 装配入口。

        无 ``AGENT_EVAL_LANGFUSE_INSTANCES`` 时返回 ``None``（框架跳过注册）。
        """

        from benchmark.settings import Settings, load_trace_instances  # noqa: PLC0415

        # Prefer Settings injected on AppRuntime (create_app(settings=...));
        # fall back to process env only when runtime has no settings attr.
        runtime = getattr(ctx, "runtime", None)
        settings = getattr(runtime, "settings", None) if runtime is not None else None
        if settings is None:
            logger.warning(
                "trace.from_config_settings_fallback "
                "reason=runtime_settings_missing using=Settings()"
            )
            settings = Settings()
        instances = load_trace_instances(settings.trace_instances)
        if not instances:
            logger.warning("trace: AGENT_EVAL_LANGFUSE_INSTANCES empty; skip provider")
            return None
        max_obs_chars = cfg.max_observation_chars if cfg is not None else 8_000
        enable = cfg.enable_fetch_full_trace if cfg is not None else False
        snapshot_chars = (
            cfg.max_attribution_snapshot_chars
            if cfg is not None
            else DEFAULT_MAX_ATTRIBUTION_SNAPSHOT_CHARS
        )
        timeout_seconds = cfg.timeout_seconds if cfg is not None else DEFAULT_TIMEOUT_SECONDS
        max_retries = cfg.max_retries if cfg is not None else DEFAULT_MAX_RETRIES
        lookback_days = (
            cfg.resolve_lookback_days if cfg is not None else DEFAULT_RESOLVE_LOOKBACK_DAYS
        )
        enable_vui_context_fallback = (
            cfg.enable_vui_observation_context_fallback if cfg is not None else False
        )
        return cls(
            TraceResolver(
                instances,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                resolve_lookback_days=lookback_days,
            ),
            max_observation_chars=max_obs_chars,
            enable_fetch_full_trace=enable,
            max_attribution_snapshot_chars=snapshot_chars,
            enable_vui_observation_context_fallback=enable_vui_context_fallback,
        )

    @property
    def enable_vui_observation_context_fallback(self) -> bool:
        """Whether the VUI evaluator may use its one-page legacy fallback."""

        return self._enable_vui_observation_context_fallback

    @property
    def resolver(self) -> TraceResolver:
        """Expose TraceResolver for lifespan shutdown / diagnostics."""

        return self._resolver

    async def aclose(self) -> None:
        await self._resolver.aclose()

    async def close(self) -> None:
        """AppRuntime.shutdown 调 ``provider.close()``；与 ``aclose`` 同义。"""

        await self.aclose()

    # ── per-request bound factories (no instance state) ─────────────────────

    def create_bound(self, *, trace_id: str, project_id: str) -> TraceRequestBound:
        """Create a request-local bound for this Langfuse primary id / project.

        ``trace_id`` must already be the Langfuse primary id (rewritten at eval
        entry). ``primary_id`` is pre-filled so tools never metadata-filter again.
        """

        tid = str(trace_id)
        return TraceRequestBound(
            trace_id=tid,
            project_id=str(project_id),
            primary_id=tid,
            sidecar_observations=current_sidecar_observations(),
        )

    def bound_from_metadata(
        self,
        metadata: dict[str, Any],
        *,
        trace_id: str | None = None,
    ) -> TraceRequestBound:
        """Build a :class:`TraceRequestBound` from request metadata (no side effects).

        ``trace_id`` overrides ``metadata["trace_id"]`` when provided.
        Raises :exc:`ValueError` if ``trace_id`` or ``project_id`` cannot be resolved.
        """
        er = metadata.get("evaluator_request")
        er_dict = er if isinstance(er, dict) else {}
        extra = er_dict.get("extra") if isinstance(er_dict.get("extra"), dict) else {}
        tid = (
            trace_id
            or metadata.get("trace_id")
            or er_dict.get("trace_id")
            or er_dict.get("traceid")
        )
        pid = metadata.get("project_id") or er_dict.get("project_id") or extra.get("project_id")
        if not tid:
            raise ValueError("trace_id is required: pass it explicitly or include it in metadata")
        if not pid:
            raise ValueError("project_id is required in metadata")
        bound = self.create_bound(trace_id=str(tid), project_id=str(pid))
        local = _sidecar_observations_from_metadata(metadata)
        if local is not None:
            bound.sidecar_observations = local
        return bound

    # ── BaseToolProvider overrides ─────────────────────────────────────────

    async def _fetch_tools(self) -> list[ToolDefinition]:
        tools = [
            _TOOL_FETCH_SKELETON,
            _TOOL_FETCH_OBS_DETAIL,
            _TOOL_FIND_OBS,
        ]
        if self._enable_fetch_full_trace:
            tools.append(_TOOL_FETCH_FULL_TRACE)
        return tools

    async def health_check(self) -> bool:
        return True

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        **kwargs: Any,  # noqa: ANN401
    ) -> ToolResult:
        """Dispatch tool and nest a child Langfuse span under the active report trace.

        Spans reuse :func:`benchmark.utils.trace_report.get_report_client`
        (``LangfuseMiddleware`` singleton, pinned to ``LANGFUSE_PUBLIC_KEY``) and
        only start when a parent ``get_current_trace_id()`` exists — never a second
        root. ``AGENT_EVAL_LANGFUSE_INSTANCES`` is read-only for fetching evaluated
        traces and is never used for write.
        """

        bound = _resolve_bound(kwargs)
        try:
            span_info = self._start_trace_tool_span(tool_name, arguments, bound)
        except Exception:
            logger.warning(
                "trace_tool.span_start_failed tool=%s",
                tool_name,
                exc_info=True,
            )
            span_info = None
        span_context, span = span_info if span_info is not None else (None, None)
        try:
            result = await self._call_tool_inner(
                tool_name,
                arguments,
                bound=bound,
            )
            self._finish_trace_tool_span(span_context, span, result=result)
            return result
        except Exception as exc:
            self._finish_trace_tool_span(span_context, span, exc=exc)
            raise

    async def _call_tool_inner(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        bound: TraceRequestBound | None,
    ) -> ToolResult:
        """Dispatch tool; bound resolved by :meth:`call_tool`."""

        if not _is_bound_like(bound):
            logger.warning(
                "trace.call_tool_missing_bound tool=%s hint=bind_trace_request_or_bound_kwarg",
                tool_name,
            )
            return ToolResult.from_error(
                "bound is required: use bind_trace_request(...) or "
                "pass call_tool(..., bound=TraceRequestBound(...)) / "
                "ToolExecutionContext(extra={'bound': ...})"
            )

        if tool_name == "fetch_trace_skeleton":
            return await self._fetch_skeleton(bound, arguments)
        if tool_name == "fetch_observation_detail":
            return await self._fetch_observation_detail(bound, arguments)
        if tool_name == "find_observations":
            return await self._find_observations(bound, arguments)
        if tool_name == "fetch_full_trace":
            return await self._fetch_full_trace(bound, arguments)
        return ToolResult.from_error(f"unknown tool: {tool_name!r}")

    def _start_trace_tool_span(
        self,
        name: str,
        args: Any,  # noqa: ANN401
        bound: TraceRequestBound | None,
    ) -> tuple[Any, Any] | None:
        """Nest a child span under the active agent-eval report trace, or no-op.

        Must **not** construct a fresh ``Langfuse(...)`` client: that uses a
        separate OTel tracer provider and ``start_as_current_observation`` then
        exports a second root (one HTTP eval → two Langfuse traces).

        Reuse :func:`get_report_client` (same singleton as ``LangfuseMiddleware``,
        pinned to ``LANGFUSE_PUBLIC_KEY``) and only ``start_observation`` when a
        parent ``get_current_trace_id()`` already exists.
        ``AGENT_EVAL_LANGFUSE_INSTANCES`` is never used for write.
        """

        bound_payload: dict[str, str] | None = None
        if bound is not None:
            bound_payload = {
                "trace_id": bound.trace_id,
                "project_id": bound.project_id,
            }
        try:
            from benchmark.utils.trace_report import (  # noqa: PLC0415
                get_report_client,
                with_current_node_otel_parent,
            )

            client = get_report_client()
            get_trace_id = getattr(client, "get_current_trace_id", None)
            if not callable(get_trace_id) or not get_trace_id():
                # No Middleware/workflow root yet → skip; never create an orphan root.
                return None
            # LangGraph node CHAIN is often not OTel-current; attach it so this
            # leaf nests under preprocess/judge_* instead of the report root.
            with with_current_node_otel_parent():
                span = client.start_observation(
                    name=f"trace.{name}",
                    as_type="span",
                    input={
                        "tool": name,
                        "args_summary": _summarize_args(args),
                        "bound": bound_payload,
                    },
                )
            # Second slot unused; kept for call_tool's (context, span) unpack.
            return None, span
        except Exception:
            # Span setup is best-effort and must never break tool execution.
            logger.warning(
                "trace_tool.span_start_failed tool=%s",
                name,
                exc_info=True,
            )
            return None

    def _finish_trace_tool_span(
        self,
        span_context: Any,  # noqa: ANN401
        span: Any,  # noqa: ANN401
        *,
        result: Any = None,  # noqa: ANN401
        exc: BaseException | None = None,
    ) -> None:
        """Update and end a child observation; safe when setup returned ``None``."""

        if span is None:
            return
        try:
            if exc is not None:
                span.update(level="ERROR", status_message=str(exc))
            elif result is not None:
                span.update(output=_trace_span_output(result, self._max_observation_chars))
        except Exception:
            logger.warning("trace_tool.span_update_failed", exc_info=True)
        finally:
            try:
                end = getattr(span, "end", None)
                if callable(end):
                    end()
                elif span_context is not None:
                    span_context.__exit__(None, None, None)
            except Exception:
                logger.warning("trace_tool.span_close_failed", exc_info=True)

    # ── tool handlers ──────────────────────────────────────────────────────

    async def _fetch_skeleton(
        self,
        bound: TraceRequestBound,
        arguments: dict[str, Any],
    ) -> ToolResult:
        try:
            args = FetchSkeletonArgs.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolResult.from_error(f"invalid fetch_trace_skeleton args: {exc}")

        sidecar = _sidecar_observations(bound)
        if sidecar is not None:
            return self._skeleton_from_sidecar(bound, sidecar, args)

        try:
            result = await self._resolver.fetch_skeleton(
                bound.trace_id,
                bound.project_id,
                max_spans=args.max_spans,
                preview_chars=args.preview_chars,
            )
        except TraceHttpError as exc:
            return ToolResult.from_error(str(exc))

        if result is None:
            return ToolResult.from_error(
                f"unknown project_id for trace tools: "
                f"trace_id={bound.trace_id!r} project_id={bound.project_id!r}"
            )

        bound.primary_id = result.primary_id  # equals bound.trace_id after entry rewrite
        payload = result.model_dump(by_alias=False)
        spans = payload.pop("spans", []) or []
        if not isinstance(spans, list):
            spans = []
        span_dicts = [s if isinstance(s, dict) else {} for s in spans]
        body, ids = _fit_span_list(
            span_dicts,
            self._max_observation_chars,
            envelope=payload,
        )
        bound.known_refs.update(ids)
        return ToolResult.from_text(body)

    async def _fetch_observation_detail(
        self,
        bound: TraceRequestBound,
        arguments: dict[str, Any],
    ) -> ToolResult:
        try:
            args = FetchObservationDetailArgs.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolResult.from_error(f"invalid fetch_observation_detail args: {exc}")
        obs_id = args.observation_id.strip()
        if not obs_id:
            return ToolResult.from_error("observation_id is required")

        detail, error, _status = await self._load_bound_detail(bound, obs_id)
        if detail is None:
            return ToolResult.from_error(error or f"observation not found: {obs_id!r}")

        body = _render_detail(
            detail,
            arg_max_chars=args.max_chars,
            body_cap=self._max_observation_chars,
        )
        bound.known_refs.add(obs_id)
        return ToolResult.from_text(body)

    async def _load_bound_detail(  # noqa: PLR0911 - explicit attribution safety exits
        self,
        bound: TraceRequestBound,
        obs_id: str,
    ) -> tuple[ObservationDetail | None, str | None, str | None]:
        """加载并校验 observation detail（请求内缓存 + 跨 trace 归属校验）。

        返回 ``(detail, error_message, status_code)``；成功时 error/status 为 None，
        失败时 status ∈ ``missing_generation`` / ``generation_read_failed`` /
        ``cross_trace_rejected``（原因码由
        调用方按自身状态机解释）。detail 校验通过后写入 ``bound.detail_cache``。
        """

        if obs_id in bound.detail_cache:
            return bound.detail_cache[obs_id], None, None

        try:
            sidecar = _sidecar_observations(bound)
            if sidecar is not None:
                item = find_observation(sidecar, obs_id)
                detail = (
                    to_observation_detail(item, bound.primary_id or bound.trace_id)
                    if item is not None
                    else None
                )
            else:
                detail = await self._resolver.fetch_observation(
                    bound.project_id,
                    obs_id,
                    max_chars=None,
                )
        except TraceHttpError as exc:
            return None, str(exc), "generation_read_failed"

        if detail is None:
            return None, f"observation not found: {obs_id!r}", "missing_generation"

        effective_primary = bound.primary_id or bound.trace_id
        bound.primary_id = effective_primary
        # 跨 trace 校验（含 trace_id 缺失 fail-closed）由 _attribution_error 承担。
        attr_error = _attribution_error(detail, obs_id, effective_primary)
        if attr_error is not None:
            return None, attr_error, "cross_trace_rejected"

        bound.detail_cache[obs_id] = detail
        return detail, None, None

    # ── generation snapshot Python API（归因分层取证专用，不注册为 LLM tool）──

    async def find_generation_snapshots(
        self,
        bound: TraceRequestBound,
        *,
        name: str | None = None,
        status: str | None = None,
    ) -> list[TraceSpanSkeleton]:
        """通用 generation 查找：按 (start_time, id) 稳定排序返回骨架列表。

        不理解任何业务语义（Summary/ReAct 等由调用方通过 ``name`` 指定）。
        HTTP 失败向上抛 :class:`LangfuseHttpError`，由编排层决定降级。
        """

        primary_id = bound.primary_id or bound.trace_id
        bound.primary_id = primary_id
        spans = await self._resolver.find_observations(
            bound.project_id,
            primary_id=primary_id,
            name=name,
            type="GENERATION",
            status=status,
        )
        return sorted(spans, key=lambda span: (span.start_time or "", span.id))

    async def find_bound_observations(
        self,
        bound: TraceRequestBound,
    ) -> tuple[list[TraceSpanSkeleton], bool]:
        """返回已绑定 trace 的 observation 骨架及完整性标记。

        这是归因私有的 Python API，不注册为 LLM tool。达到 resolver 的分页上限时
        无法证明列表完整，调用方必须保守回退，不能据此断言某个 observation 不存在。
        """

        primary_id = bound.primary_id or bound.trace_id
        bound.primary_id = primary_id
        spans = await self._resolver.find_observations(
            bound.project_id,
            primary_id=primary_id,
        )
        return spans, len(spans) < FIND_MAX_OBSERVATIONS

    async def fetch_bound_observation_details(
        self,
        bound: TraceRequestBound,
        observation_ids: list[str],
    ) -> list[ObservationDetail] | None:
        """批量读取已绑定 observation 的完整 detail，任一缺失即整体失败。

        每条读取均复用 ``_load_bound_detail`` 的请求缓存与跨 trace 校验；调用方
        仅可在可控的小 observation 集合上使用，避免把整包 raw trace 带入归因。
        """

        ids = list(dict.fromkeys(obs_id.strip() for obs_id in observation_ids if obs_id.strip()))
        if not ids:
            return []
        details: list[ObservationDetail] = []
        for obs_id in ids:
            detail, _error, _status = await self._load_bound_detail(bound, obs_id)
            if detail is None:
                return None
            details.append(detail)
            bound.known_refs.add(obs_id)
        return details

    async def fetch_generation_snapshot(
        self,
        bound: TraceRequestBound,
        observation_id: str,
        *,
        max_chars: int | None = None,
    ) -> GenerationSnapshot:
        """无损读取单个 generation 的 input/output（独立预算 + message 边界对齐）。

        不套用 ``max_observation_chars`` 的二次裁剪；超预算时保留完整
        message 边界并标记 ``over_budget``，不做生成式摘要。跨 trace /
        缺失 observation 时返回带 ``evidence_incomplete`` 的快照而非抛错。
        """

        obs_id = observation_id.strip()
        budget = max_chars or self._max_attribution_snapshot_chars
        detail, error, status_code = await self._load_bound_detail(bound, obs_id)
        if detail is None:
            return GenerationSnapshot(
                observation_id=obs_id,
                evidence_incomplete=True,
                incomplete_reasons=[status_code or "missing_generation"],
                read_error=error,
            )

        snapshot = _snapshot_from_detail(detail)
        _apply_snapshot_budget(snapshot, detail, budget)
        bound.known_refs.add(obs_id)
        return snapshot

    async def fetch_generation_message_range(
        self,
        bound: TraceRequestBound,
        observation_id: str,
        request: GenerationMessageRange,
    ) -> GenerationSnapshot:
        """定向补读 input messages[start:end]（无摘要，继续接受 trace 归属校验）。

        仅支持 ``field="input"``；结果同样按 message 边界对齐预算。
        """

        obs_id = observation_id.strip()
        budget = request.max_chars or self._max_attribution_snapshot_chars
        if request.field != "input":
            raise ValueError(
                f"message range read only supports field='input', got {request.field!r}"
            )
        if request.message_start < 0 or request.message_end < request.message_start:
            raise ValueError(
                f"invalid message range: [{request.message_start}, {request.message_end})"
            )

        detail, error, status_code = await self._load_bound_detail(bound, obs_id)
        if detail is None:
            return GenerationSnapshot(
                observation_id=obs_id,
                evidence_incomplete=True,
                incomplete_reasons=[status_code or "missing_generation"],
                read_error=error,
            )

        messages, _path = _extract_messages(detail.input)
        if messages is None:
            return GenerationSnapshot(
                observation_id=obs_id,
                name=detail.name,
                type=detail.type,
                evidence_incomplete=True,
                incomplete_reasons=["missing_input"],
            )

        selected = messages[request.message_start:request.message_end]
        snapshot = _snapshot_from_detail(detail)
        snapshot.input_message_count = len(messages)
        snapshot.read_message_start = request.message_start
        _fit_snapshot_messages(snapshot, selected, budget, offset=request.message_start)
        bound.known_refs.add(obs_id)
        return snapshot

    def _skeleton_from_sidecar(
        self,
        bound: TraceRequestBound,
        sidecar: list[Any],
        args: FetchSkeletonArgs,
    ) -> ToolResult:
        items = dict_observations(sidecar)
        truncated = len(items) > args.max_spans
        items = items[: args.max_spans]
        bound.primary_id = bound.primary_id or bound.trace_id
        payload = {
            "primary_id": bound.primary_id,
            "business_trace_id": bound.trace_id,
            "project_id": bound.project_id,
            "truncated": truncated,
            "instance_name": None,
        }
        span_dicts = [
            to_span_skeleton(item, args.preview_chars).model_dump(by_alias=False) for item in items
        ]
        body, ids = _fit_span_list(
            span_dicts,
            self._max_observation_chars,
            envelope=payload,
        )
        bound.known_refs.update(ids)
        return ToolResult.from_text(body)

    async def _fetch_full_trace(
        self,
        bound: TraceRequestBound,
        arguments: dict[str, Any],
    ) -> ToolResult:
        sidecar = _sidecar_observations(bound)
        if sidecar is not None:
            primary_id = bound.primary_id or bound.trace_id
            bound.primary_id = primary_id
            return ToolResult.from_text(_dumps(observations_to_full_trace(sidecar, primary_id)))

        if not self._enable_fetch_full_trace:
            return ToolResult.from_error(
                "fetch_full_trace is disabled; set "
                "tools.extensions.trace.enable_fetch_full_trace=true"
            )

        try:
            args = FetchFullTraceArgs.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolResult.from_error(f"invalid fetch_full_trace args: {exc}")

        primary_id = bound.primary_id or bound.trace_id
        bound.primary_id = primary_id
        fields = (args.fields or "").strip() or None

        try:
            trace = await self._resolver.fetch_full_trace(
                bound.project_id,
                primary_id,
                fields=fields,
            )
        except TraceHttpError as exc:
            return ToolResult.from_error(str(exc))

        if trace is None:
            return ToolResult.from_error(
                f"full trace not found or unknown project_id: "
                f"trace_id={primary_id!r} project_id={bound.project_id!r}"
            )

        if isinstance(trace, dict):
            payload = trace
        else:
            payload = trace.model_dump(by_alias=True)
        tid = payload.get("id")
        if isinstance(tid, str) and tid.strip():
            bound.primary_id = tid.strip()

        return ToolResult.from_text(_dumps(payload))

    async def _find_observations(
        self,
        bound: TraceRequestBound,
        arguments: dict[str, Any],
    ) -> ToolResult:
        try:
            args = FindObservationsArgs.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolResult.from_error(f"invalid find_observations args: {exc}")

        primary_id = bound.primary_id or bound.trace_id
        bound.primary_id = primary_id

        sidecar = _sidecar_observations(bound)
        if sidecar is not None:
            matched = filter_observations(
                sidecar,
                name=args.name,
                type=args.type,
                status=args.status,
            )
            result_data = [
                to_span_skeleton(item, 180).model_dump(by_alias=False) for item in matched
            ]
            body, ids = _fit_span_list(result_data, self._max_observation_chars)
            bound.known_refs.update(ids)
            return ToolResult.from_text(body)

        try:
            spans = await self._resolver.find_observations(
                bound.project_id,
                primary_id=primary_id,
                name=args.name,
                type=args.type,
                status=args.status,
            )
        except TraceHttpError as exc:
            return ToolResult.from_error(str(exc))

        result_data = [s.model_dump(by_alias=False) for s in spans]
        body, ids = _fit_span_list(result_data, self._max_observation_chars)
        bound.known_refs.update(ids)
        return ToolResult.from_text(body)
