"""Set libs ``init_request_context(workflow_name=...)`` per HTTP request.

Why: LangfuseMiddleware reads `workflow_name` from the request-scope ContextVar
and forwards it to the Langfuse trace root `name`
(libs/agent_graph/src/agentgraph/init/context.py:151-158).
Without this middleware, `bind_trace_metadata(trace_name=...)` writes to a sibling
ContextVar that surfaces only as `metadata.trace_name`, NOT as the root trace `name`.

Naming: ``agent-eval.{agent_name}``. ``agent_name`` in the body is not yet available in middleware,
so this reads query ``agent_name`` first; the default is ``unknown``. After ``EvaluatorService.evaluate`` parses the
envelope it calls ``update_request_context`` to correct the name to the real business name.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from benchmark.libs.agenthost.init import clear_request_context, init_request_context
from benchmark.libs.agenthost.observability.logger import get_logger
from fastapi import Request, Response

from benchmark.settings.setting import settings
from benchmark.utils.trace_naming import build_eval_trace_name

logger = get_logger("agent_eval.middleware.langfuse_trace_context")


def _resolve_agent_name(request: Request) -> str | None:
    """Pick the business agent name from request query params (optional)."""
    agent = request.query_params.get("agent_name")
    if agent and agent.strip():
        return agent.strip()
    return None


async def trace_context_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Bind the libs request context for the duration of the request."""
    agent_name = _resolve_agent_name(request)
    workflow_name = build_eval_trace_name(agent_name)
    env = settings.env

    token = init_request_context(
        workflow_name=workflow_name,
        service="agent-eval",
        environment=env,
    )
    try:
        response = await call_next(request)
    finally:
        clear_request_context()
    return response
