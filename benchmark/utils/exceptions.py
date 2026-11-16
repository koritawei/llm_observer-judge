"""Unified error hierarchy for the eval endpoint (OpenAI-compatible envelope).

One base, two families:

- :class:`BaseError` — the common shape: an int ``code`` (HTTP status), a string
  ``error_type`` (the machine-readable identifier), and a human ``message``.
- :class:`EvaluatorError` — the evaluator dev contract; **uses BaseError directly**
  (evaluators raise it inside the graph with a ``*_CODE`` error_type + a status).
- :class:`APIError` — the API error; extends BaseError with the OpenAI ``param`` and
  per-subclass class-attr defaults. :func:`from_evaluator_error` projects an
  EvaluatorError onto an APIError.

Wire shape stays OpenAI-shaped — ``{"error": {message, type, param, code}}`` — but
``code`` is now the **int HTTP status** and ``type`` carries the fine identifier
(``invalid_json`` / ``model_not_found`` / …) so clients still distinguish errors.

**重试语义**（MQ 消费链路据此分流，见 :mod:`benchmark.ingest.mq.executor`）：``code`` 的
首位决定可恢复性 —— **4xx = 永久性错误**（数据 / 配置缺陷，重试永远不会成功 → 跳过
并提交 offset）；**5xx = 瞬时错误**（外部依赖故障 → 退避重试）。新增 ``error_type`` 时按
“再试一次有可能成功吗”选状态码，**而不是按错误的严重程度**；``evaluation_failed``
（500）仅作为“未知失败”兜底，已知的永久失败均应用专门的 4xx 类型。
"""

import json
from http import HTTPStatus
from typing import Any

# ── Machine-readable error_type identifiers (envelope ``type``) ──
# Evaluator-side (raised in the graph; survive across the agent boundary):
MODEL_NOT_FOUND_CODE = "model_not_found"
PROMPT_NOT_FOUND_CODE = "evaluator_prompt_not_found"
LLM_FAILED_CODE = "llm_failed"
#: LLM 鉴权/参数类永久错误（401/403/400 无效 api key 等）。
#: 映射 400：重试不会改变凭证，worker 侧按永久失败直接 mark_failed，
#: 避免无效 key 被当瞬态 502 重试（无限 requeue 导致 run 卡住）。
LLM_AUTH_FAILED_CODE = "llm_auth_failed"
#: 非鉴权的永久 LLM 4xx（参数/资源/上下文等），映射 400，worker 不重试。
LLM_PERMANENT_FAILED_CODE = "llm_permanent_failed"
LLM_UNPARSEABLE_CODE = "llm_unparseable"
#: 完整 JSON 但 verdict schema 不符。temperature=0 时重试不会改变结果，故 422。
LLM_SCHEMA_INVALID_CODE = "llm_schema_invalid"
EVALUATION_FAILED_CODE = "evaluation_failed"
INTERNAL_ERROR_TYPE = "internal_error"
#: Required metadata absent on the inbound request — 400, langfuse judge 视为永久错误不重试。
METADATA_MISSING_CODE = "metadata_missing"
#: Business trace_id 在 Langfuse 中无法 resolve 为主键 — 400, 永久错误不重试。
TRACE_NOT_FOUND_CODE = "trace_not_found"
#: Langfuse Public API 传输失败 / 非 200（含 422 无界查询）— 502, 可重试。
TRACE_UNAVAILABLE_CODE = "trace_unavailable"
#: Trace 的 input 或 output 尚未完整；Kafka 后续更新可重新进入评测。
INCOMPLETE_TRACE_IO_CODE = "incomplete_trace_io"

#: HTTP status 语义分界：≥ 该值视为服务端错误（error 级日志），否则为客户端错误（warning 级）。
SERVER_ERROR_STATUS = 500

#: HTTP status for each evaluator error_type (the int ``code`` it surfaces as).
_EVALUATOR_STATUS: dict[str, int] = {
    MODEL_NOT_FOUND_CODE: 400,
    PROMPT_NOT_FOUND_CODE: 400,
    METADATA_MISSING_CODE: 400,
    TRACE_NOT_FOUND_CODE: 400,
    LLM_AUTH_FAILED_CODE: 400,
    LLM_PERMANENT_FAILED_CODE: 400,
    "invalid_request_body": 400,
    INCOMPLETE_TRACE_IO_CODE: 400,
    TRACE_UNAVAILABLE_CODE: 502,
    LLM_FAILED_CODE: 502,
    LLM_UNPARSEABLE_CODE: 502,
    LLM_SCHEMA_INVALID_CODE: 422,
    EVALUATION_FAILED_CODE: SERVER_ERROR_STATUS,
}
#: error_type values that surface the OpenAI ``param: "model"`` hint.
_MODEL_PARAM_TYPES = frozenset(
    {
        MODEL_NOT_FOUND_CODE,
        PROMPT_NOT_FOUND_CODE,
        METADATA_MISSING_CODE,
        TRACE_NOT_FOUND_CODE,
    }
)


def evaluator_status(error_type: str) -> int:
    """HTTP status an evaluator error_type maps to (unknown → 500)."""

    return _EVALUATOR_STATUS.get(error_type, SERVER_ERROR_STATUS)


def status_error_type(status: int) -> str:
    """A fine ``error_type`` for a bare HTTP status (e.g. 404 → ``not_found``)."""

    try:
        return HTTPStatus(status).name.lower()
    except ValueError:
        return INTERNAL_ERROR_TYPE


class BaseError(Exception):
    """Common error base — int ``code`` (HTTP status) + str ``error_type`` + ``message``."""

    def __init__(self, code: int, error_type: str, message: str = "") -> None:
        super().__init__(message or error_type)
        self.code = code
        self.error_type = error_type
        self.message = message or error_type


class EvaluatorError(BaseError):
    """Business error raised inside the evaluator graph — the evaluator dev contract.

    Uses :class:`BaseError` directly: ``code`` is the HTTP status to surface
    (:func:`evaluator_status`), ``error_type`` a ``*_CODE`` identifier, ``message`` human.

    :meth:`to_payload` / :meth:`from_payload` are its JSON wire form — a FAILED
    ``AgentResponse`` carries the error as content JSON (the boundary passes data, not
    raised exceptions), and the service round-trips it back into an EvaluatorError.

    ``details`` 可选：失败时仍需落库的旁路字段（如 ``prompt_customization``
    溯源），经 payload 跨 agent 边界带回 worker；构造方按需在构造后赋值
    （FUNC-PARAMS-001 收敛，与 :meth:`APIError.of` 的构造后赋值同一惯例）。
    """

    #: 失败时仍需落库的旁路字段；``None`` 时 ``to_payload`` 不落 details 键。
    details: dict[str, Any] | None = None

    def __init__(self, code: int, error_type: str, message: str = "") -> None:
        super().__init__(code, error_type, message)

    def to_payload(self) -> str:
        """Serialize fields to the JSON carried in a FAILED response."""

        payload: dict[str, Any] = {
            "code": self.code,
            "error_type": self.error_type,
            "message": self.message,
        }
        if isinstance(self.details, dict) and self.details:
            payload["details"] = self.details
        return json.dumps(payload)

    @classmethod
    def from_payload(cls, text: str) -> "EvaluatorError":
        """Reconstruct from the boundary JSON (tolerant of missing / garbled fields)."""

        try:
            data = json.loads(text or "{}")
        except (json.JSONDecodeError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        raw_details = data.get("details")
        err = cls(
            code=int(data.get("code", 500)),
            error_type=str(data.get("error_type") or EVALUATION_FAILED_CODE),
            message=str(data.get("message") or ""),
        )
        if isinstance(raw_details, dict) and raw_details:
            err.details = raw_details
        return err


class APIError(BaseError):
    """API error → OpenAI envelope; extends :class:`BaseError` with ``param``.

    ``code`` (inherited) is the HTTP status; ``error_type`` is the envelope ``type``.
    Subclasses set the ``code`` / ``error_type`` (/ ``param``) class attributes.
    :meth:`to_envelope` renders the ``{"error": {message, type, param, code}}`` body;
    :meth:`of` builds one with explicit ``code`` / ``error_type`` (framework / raw cases).
    """

    code: int = 500
    error_type: str = INTERNAL_ERROR_TYPE
    param: str | None = None

    def __init__(self, message: str = "", *, param: str | None = None) -> None:
        super().__init__(self.code, self.error_type, message)
        if param is not None:
            self.param = param

    @classmethod
    def of(cls, *, code: int, error_type: str, message: str = "") -> "APIError":
        """Build an APIError with explicit ``code`` / ``error_type`` (raw / framework cases).

        FUNC-PARAMS-001 收敛：``param`` 不再入参，调用方按需在返回值上赋值。
        """

        err = cls(message)
        err.code = code
        err.error_type = error_type
        return err

    def to_envelope(self) -> dict[str, Any]:
        """Render the OpenAI error body — ``{"error": {message, type, param, code}}``."""

        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


# ── request validation (raised in the API layer / mapped from pydantic) ──


class InvalidJSONError(APIError):
    code, error_type = 400, "invalid_json"


class InvalidRequestBodyError(APIError):
    code, error_type = 400, "invalid_request_body"


class MissingParameterError(APIError):
    code, error_type = 400, "missing_required_parameter"


class StreamingNotSupportedError(APIError):
    code, error_type, param = 400, "streaming_not_supported", "stream"


def from_evaluator_error(exc: EvaluatorError) -> APIError:
    """Project an :class:`EvaluatorError` onto an :class:`APIError` for the envelope.

    Carries the evaluator's ``code`` / ``error_type`` / ``message`` straight through;
    the model/prompt-not-found types also surface ``param: "model"``.
    """

    err = APIError.of(
        code=exc.code,
        error_type=exc.error_type,
        message=exc.message,
    )
    if exc.error_type in _MODEL_PARAM_TYPES:
        err.param = "model"
    return err
