"""Aone work-item content rendering.

This module only renders an attribution result into a work-item title/body/payload; `aone.py` only calls Aone.
The body is always two parts: the top half is for a person to judge quickly, and the bottom half is for later AI automation to read.
"""

import json
from dataclasses import dataclass

from benchmark.attribution.badcase.schemas import (
    NotificationAIContextPayload,
    NotificationRuntimeOptions,
    NotificationWorkitemPayload,
    AttributionResult,
)

DEFAULT_NOTIFICATION_TITLE_PREFIX = "[AgentEval Badcase]"
NOTIFICATION_AI_CONTEXT_SCHEMA_VERSION = "badcase-attribution.aone.v1"

_MAX_NOTIFICATION_TITLE_LENGTH = 120

_PROJECT_URL_FOOTER = "\n\n[关联看板]({url})"


@dataclass(frozen=True)
class NotificationWorkitemOverrides:
    """Work-item trace-route overrides (assignee / priority / project_url).

    Resolved by the caller and passed in (see ``create_aone_task``); all empty by default.
    Kept separate from ``options`` (work-item style and dispatch fallback) so the two are not mixed in
    the same argument list.
    """

    assignee: str = ""
    priority: str = ""
    project_url: str = ""
    #: When not None, override options.participants (the final participants after multi-person route expansion).
    participants: list[str] | None = None


def build_workitem_payload(
    result: AttributionResult,
    *,
    project: str,
    category: str,
    options: NotificationRuntimeOptions | None = None,
    overrides: NotificationWorkitemOverrides | None = None,
) -> NotificationWorkitemPayload:
    """Turn an attribution result into a payload the Aone CLI can consume.

    ``options`` supplies work-item style and dispatch fallback (``workitem_type`` / ``title_prefix``
    / ``participants`` / ``tags``; production passes :class:`AoneConfig`, which is duck-type
    compatible); ``overrides`` supplies trace-route overrides, resolved by the caller and passed in
    (see ``create_aone_task``).
    """
    opts = options or NotificationRuntimeOptions()
    ov = overrides or NotificationWorkitemOverrides()

    body = render_workitem_body(result)
    if ov.project_url:
        body = body + _PROJECT_URL_FOOTER.format(url=ov.project_url)

    return NotificationWorkitemPayload(
        project=project,
        category=category,
        type=opts.workitem_type,
        title=build_workitem_title(result, title_prefix=opts.title_prefix),
        body=body,
        assignee=ov.assignee,
        participants=(
            list(ov.participants) if ov.participants is not None else list(opts.participants)
        ),
        tags=list(opts.tags),
        priority=ov.priority,
        project_url=ov.project_url,
    )


def build_workitem_title(
    result: AttributionResult,
    *,
    title_prefix: str = DEFAULT_NOTIFICATION_TITLE_PREFIX,
) -> str:
    """Build a stable short title so the Aone title does not grow too long."""

    trace_label = result.trace_id or result.case_id or "unknown-trace"
    title = f"{title_prefix} {result.category}: {trace_label}"
    return title[:_MAX_NOTIFICATION_TITLE_LENGTH]


def render_workitem_body(result: AttributionResult) -> str:
    """Build the Aone body: for a person first, then for AI to read."""

    return "\n\n".join(
        [
            render_human_summary(result),
            render_ai_context(result),
        ]
    )


def render_human_summary(result: AttributionResult) -> str:
    """Render the top half of the work item for engineers and PMs to scan quickly."""

    return "\n".join(
        [
            "## 归因摘要（给人看）",
            "",
            f"- Trace ID: {result.trace_id or '-'}",
            f"- Query: {result.query or '-'}",
            f"- Case ID: {result.case_id or '-'}",
            f"- Trace URL: {result.trace_url or '-'}",
            f"- 归因类别: {result.category}",
            f"- 置信度: {result.confidence:.2f}",
            f"- 需要人工复核: {result.needs_human_review}",
            "",
            "### 根因",
            result.root_cause or "-",
            "",
            "### 判断依据",
            result.reasoning or "-",
            "",
            "### 证据引用",
            _render_refs(result.evidence_refs),
            "",
            "### 建议动作",
            result.action_hint or "-",
        ]
    )


def render_ai_context(result: AttributionResult) -> str:
    """Render the bottom half of the work item for later AI automation."""

    payload = _ai_context_payload(result)
    return "\n".join(
        [
            "## AI Context（给后续自动化读取）",
            "",
            "```json",
            json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, indent=2),
            "```",
        ]
    )


def _ai_context_payload(result: AttributionResult) -> NotificationAIContextPayload:
    """Build the smallest context that later automation can parse stably."""

    return NotificationAIContextPayload(
        schema_version=NOTIFICATION_AI_CONTEXT_SCHEMA_VERSION,
        trace_id=result.trace_id,
        query=result.query,
        case_id=result.case_id,
        trace_url=result.trace_url,
        category=result.category,
        confidence=result.confidence,
        root_cause=result.root_cause,
        reasoning=result.reasoning,
        evidence_refs=result.evidence_refs,
        action_hint=result.action_hint,
        needs_human_review=result.needs_human_review,
    )


def _render_refs(refs: list[str]) -> str:
    """Render evidence refs as a markdown list."""

    return "\n".join(f"- {ref}" for ref in refs) if refs else "-"


__all__ = [
    "NOTIFICATION_AI_CONTEXT_SCHEMA_VERSION",
    "DEFAULT_NOTIFICATION_TITLE_PREFIX",
    "build_workitem_payload",
    "build_workitem_title",
    "render_ai_context",
    "render_human_summary",
    "render_workitem_body",
]
