"""Idempotency key helpers (G-05 / idempotency-keys.md).

Keys are scoped by tenant + project so cross-tenant / cross-project
collision is impossible even when span_id / trace_id collide.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llmobserverjudge.shared.project import DEFAULT_PROJECT_ID


def span_key(
    tenant_id: str, span_id: str, *, project_id: str = DEFAULT_PROJECT_ID
) -> str:
    return f"{tenant_id}:{project_id}:{span_id}"


def trace_closed_key(
    tenant_id: str, trace_id: str, *, project_id: str = DEFAULT_PROJECT_ID
) -> str:
    return f"{tenant_id}:{project_id}:{trace_id}:trace_closed"


def eval_config_hash(eval_configs: list[dict[str, Any]]) -> str:
    normalized = sorted(eval_configs, key=lambda c: c.get("agent_id", ""))
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def eval_job_key(
    trace_id: str,
    configs: list[dict[str, Any]],
    *,
    tenant_id: str = "",
    project_id: str = DEFAULT_PROJECT_ID,
) -> str:
    scope = f"{tenant_id}:{project_id}:" if tenant_id else f"{project_id}:"
    return f"{scope}{trace_id}:{eval_config_hash(configs)}"


def attribution_key(
    trace_id: str,
    eval_agent_id: str,
    *,
    tenant_id: str = "",
    project_id: str = DEFAULT_PROJECT_ID,
) -> str:
    scope = f"{tenant_id}:{project_id}:" if tenant_id else f"{project_id}:"
    return f"{scope}{trace_id}:{eval_agent_id}:attr"


def annotation_key(
    trace_id: str,
    eval_agent_id: str,
    *,
    tenant_id: str = "",
    project_id: str = DEFAULT_PROJECT_ID,
) -> str:
    scope = f"{tenant_id}:{project_id}:" if tenant_id else f"{project_id}:"
    return f"{scope}{trace_id}:{eval_agent_id}:annot"
