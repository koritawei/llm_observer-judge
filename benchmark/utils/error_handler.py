"""Exception handlers — turn exceptions into the OpenAI error envelope.

Mirrors the reference ``core/error_handler.py`` (routers only raise), but lives under
``app/utils`` and is driven by the :class:`APIError` hierarchy (each exception carries
its own ``code`` (HTTP status) / ``error_type`` / ``param``) rather than an external
mapping table. These 4 handlers cover the *known* exception types:

1. :class:`APIError` — API exceptions, read straight off the instance.
2. :class:`EvaluatorError` — business failures, projected via
   :func:`~benchmark.utils.exceptions.from_evaluator_error`.
3. :class:`RequestValidationError` — pydantic failures, mapped to an APIError.
4. :class:`StarletteHTTPException` — framework HTTP errors (404 / 405 / …).

The catch-all for **unhandled** exceptions is :class:`ErrorHandlingMiddleware` (a
``BaseHTTPMiddleware``, not a handler), so anything that leaks past these still gets
the stable 500 envelope.

The wire shape stays OpenAI-compatible (``{"error": {message, type, param, code}}``;
``request_id`` is surfaced via the ``X-Request-Id`` header set by
:class:`RequestContextMiddleware`, not the body).
"""

import uuid
from collections.abc import Sequence
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from benchmark.utils.exceptions import (
    APIError,
    EvaluatorError,
    InvalidJSONError,
    InvalidRequestBodyError,
    MissingParameterError,
    StreamingNotSupportedError,
    from_evaluator_error,
    status_error_type,
)

logger = get_logger("agent_eval.eval")

#: pydantic error types that mean "a required field is absent or empty".
_MISSING_FIELD_TYPES = frozenset({"missing", "string_too_short"})


def get_request_id(request: Request) -> str:
    """The request_id stamped by :class:`RequestContextMiddleware`, or a fresh one."""

    return getattr(request.state, "request_id", "") or uuid.uuid4().hex


def _response(exc: APIError) -> JSONResponse:
    """Render an :class:`APIError` as its OpenAI-envelope JSONResponse (status = code)."""

    return JSONResponse(status_code=exc.code, content=exc.to_envelope())


def _validation_to_exception(errors: Sequence[dict[str, Any]]) -> APIError:
    """Map a pydantic ``RequestValidationError`` to the matching APIError.

    Reads the first error's ``type`` / ``loc`` to recover invalid JSON, a
    missing/empty required field (with its name as ``param``), the rejected
    ``stream`` flag, or a body that is not a JSON object.
    """

    if not errors:
        return InvalidRequestBodyError("Request body is invalid.")
    first = errors[0]
    etype = str(first.get("type", ""))
    loc = first.get("loc", ())
# Take the last segment of loc: under a nested model loc looks like ("body", "form", "agentName"),
# and the first segment is the model name rather than a field name, so reporting it would point at a form field that does not exist.
    field = next((part for part in reversed(loc[1:]) if isinstance(part, str)), None)

    if etype == "json_invalid":
        return InvalidJSONError("Request body must be valid JSON.")
    if field == "stream":
        return StreamingNotSupportedError(
            "Streaming responses are not supported by the eval endpoint."
        )
    if etype in _MISSING_FIELD_TYPES and field:
        return MissingParameterError(f"Missing required field: {field}.", param=field)
    return InvalidRequestBodyError("Request body must be a JSON object.")


def install_error_handlers(app: FastAPI) -> None:
    """Register the eval API's exception handlers on ``app``.

    Call from ``create_app`` (prod) and the unit-test app builder.
    """

    @app.exception_handler(APIError)
    async def _on_api_error(request: Request, exc: APIError) -> JSONResponse:
        _log_reject(get_request_id(request), exc.code, exc.error_type)
        return _response(exc)

    @app.exception_handler(EvaluatorError)
    async def _on_evaluator_error(request: Request, exc: EvaluatorError) -> JSONResponse:
        _log_reject(get_request_id(request), exc.code, exc.error_type)
        return _response(from_evaluator_error(exc))

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        api_exc = _validation_to_exception(exc.errors())
        logger.info(
            "eval.invalid_request request_id=%s type=%s param=%s",
            get_request_id(request),
            api_exc.error_type,
            api_exc.param,
        )
        return _response(api_exc)

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        request_id = get_request_id(request)
        if exc.status_code >= 500:
            logger.error(
                "eval.http_error request_id=%s status=%s", request_id, exc.status_code
            )
        elif exc.status_code == 404:
            logger.info(
                "eval.http_error request_id=%s status=%s", request_id, exc.status_code
            )
        else:
            logger.warning(
                "eval.http_error request_id=%s status=%s", request_id, exc.status_code
            )
        return _response(
            APIError.of(
                code=exc.status_code,
                error_type=status_error_type(exc.status_code),
                message=str(exc.detail),
            )
        )


def _log_reject(request_id: str, status_code: int, error_type: str) -> None:
    """4xx client/business rejects stay warning; 5xx surface as error for alerting."""

    if status_code >= 500:
        logger.error(
            "eval.reject request_id=%s type=%s status=%s",
            request_id,
            error_type,
            status_code,
        )
    else:
        logger.warning(
            "eval.reject request_id=%s type=%s status=%s",
            request_id,
            error_type,
            status_code,
        )
