"""OpenAI-compatible eval entry.

The router is a thin protocol shell: FastAPI validates the OpenAI Chat Completions
request against :class:`ChatCompletionRequest`, the :class:`EvaluatorService`
(injected via :data:`EvaluatorServiceDep`) selects the evaluator by ``model``, runs
the evaluator graph, and hands the result to triage; the route returns the
OpenAI-compatible body. No proxying / upstream forwarding happens here — the verdict
is produced entirely by the selected evaluator agent.

Validation failures and :class:`EvaluatorError` are turned into the OpenAI error
envelope by the exception handlers in :mod:`benchmark.utils.error_handler` (registered
via ``install_error_handlers``), so this route holds no error branching of its own.
"""

from benchmark.libs.agenthost.observability.logger import get_logger
from fastapi import APIRouter

from benchmark.routes.schema import (
    ChatCompletion,
    ChatCompletionRequest,
    has_eval_envelope,
    to_chat_completion,
    to_evaluator_request,
)
from benchmark.services.evaluation import EvalRunOptions, EvaluatorServiceDep
from benchmark.utils.trace_report import capture_report_trace_id

router = APIRouter(prefix="/v1", tags=["eval"])
logger = get_logger("agent_eval.eval")


@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    service: EvaluatorServiceDep,
) -> ChatCompletion:
    """Evaluate one OpenAI-compatible Chat Completions request.

    Two paths share this route:

    - **Evaluator** (default): when the message content carries the evaluator JSON
      envelope (``{input, output, metadata, …}``), the selected evaluator runs the
      LangGraph judge and the route returns a ``chat.completion`` carrying the
      ``{score, reason}`` verdict.
    - **Proxy fallback**: any other shape (Langfuse save-time connectivity probe,
      function-calling test, ad-hoc chat) bypasses the judge and forwards the raw
      messages to the evaluator's LLM, returning whatever the model said (text or
      ``tool_calls``) so structured-output clients get the shape they negotiated.

    Failures raise :class:`EvaluatorError`, mapped to the OpenAI error envelope by
    the handlers in :mod:`benchmark.utils.error_handler`.
    """

    if has_eval_envelope(payload):
        logger.info("eval.request evaluator_name=%s", payload.model)
        eval_request = to_evaluator_request(payload)
        with capture_report_trace_id() as report:
            result = await service.evaluate(
                payload.model,
                eval_request,
                options=EvalRunOptions(resolve_trace=not payload.skip_trace_resolve),
            )
        return to_chat_completion(
            payload.model,
            result,
            trace_id=report.trace_id,
        )
    logger.info("eval.fallback_proxy evaluator_name=%s", payload.model)
    return await service.proxy_chat_completion(payload)
