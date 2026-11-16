"""HTTP middlewares for the eval API.

- :class:`RequestContextMiddleware` — request_id, timing, access log. Every request
  gets a ``request_id`` (propagated from the inbound ``X-Request-Id`` header or freshly
  minted), stashed on ``request.state.request_id`` so the error handlers
  (:mod:`benchmark.utils.error_handler`) can stamp it into their logs, and echoed back on
  the response with the processing time.
- :class:`ErrorHandlingMiddleware` — normalizes **error responses** into the OpenAI
  envelope: any 4xx/5xx whose body isn't already ``{"error": {...}}`` (framework
  defaults, sub-app responses) is repackaged by reading the error info off the
  response, and any unhandled exception is caught as the final 500 net.

Order matters: ``ErrorHandlingMiddleware`` sits *inside* ``RequestContextMiddleware``,
so its (re)built error response flows back out and still picks up the
``X-Request-Id`` header + access log line.
"""

import json
import time
import traceback
import uuid
from http import HTTPStatus

from benchmark.libs.agenthost.observability.logger import get_logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from benchmark.utils.exceptions import APIError, status_error_type

logger = get_logger("agent_eval.eval")

#: Inbound/outbound correlation-id header (echoed; propagated if the client sends one).
REQUEST_ID_HEADER = "X-Request-Id"
#: Outbound processing-time header (milliseconds).
PROCESS_TIME_HEADER = "X-Process-Time"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Inject ``request_id``, time the request, and log one access line."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.2f}ms"
        # Probe-friendly access log: query string + content-type + user-agent let
        # us tell a real /v1/chat/completions call apart from a Langfuse / SDK
        # setup probe. Health/probe paths stay at debug to cut production noise.
        path = request.url.path
        log_fn = logger.debug if _is_probe_path(path) else logger.info
        log_fn(
            "eval.access request_id=%s method=%s path=%s query=%s status=%s "
            "content_type=%s user_agent=%s elapsed_ms=%.2f",
            request_id,
            request.method,
            path,
            request.url.query or "-",
            response.status_code,
            response.headers.get("content-type", "-"),
            request.headers.get("user-agent", "-"),
            elapsed_ms,
        )
        return response


#: Paths that are probe/health traffic — logged at debug, not info.
_PROBE_PATHS = frozenset({"/", "/health", "/healthz", "/ready", "/readyz", "/v1/models"})


def _is_probe_path(path: str) -> bool:
    return path in _PROBE_PATHS or path.endswith("/health")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Normalize error responses into the OpenAI envelope; catch unhandled exceptions.

    Two paths:

    1. **Error response** — after ``call_next`` returns, any 4xx/5xx whose body isn't
       already ``{"error": {...}}`` is repackaged: the message is read off the response
       (framework ``{"detail": ...}``, validation lists, plain text) and re-emitted as
       the stable envelope. Already-enveloped errors (from the registered handlers) are
       re-emitted verbatim, so there's no double-wrapping. Success (<400) is returned
       untouched — the body is never drained.
    2. **Unhandled exception** — a genuinely unhandled exception (one with no registered
       handler) is the final 500 net; returning a response instead of re-raising keeps
       the leaked detail out of the client.

    Sits *inside* :class:`RequestContextMiddleware`, so the (re)built response flows
    back out and still gets the ``X-Request-Id`` header + access log.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - final catch-all → stable envelope, never leak
            request_id = getattr(request.state, "request_id", "") or uuid.uuid4().hex
            logger.error(
                "eval.unhandled request_id=%s\n%s", request_id, traceback.format_exc()
            )
            err = APIError("Internal server error.")  # defaults: 500 / internal_error
            return JSONResponse(status_code=err.code, content=err.to_envelope())
        if response.status_code < HTTPStatus.BAD_REQUEST:
            return response  # success — never drain a (possibly streaming) body
        return await _normalize_error_response(response)


async def _normalize_error_response(response: Response) -> Response:
    """Re-emit an error response as the OpenAI envelope, repackaging if not already."""

    body = await _collect_body(response)
    headers = _carry_headers(response)
    if _is_openai_envelope(body):
        # Already enveloped (our handlers) — re-emit verbatim (iterator was consumed).
        return Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type="application/json",
            background=response.background,
        )
    status = response.status_code
    err = APIError.of(
        code=status,
        error_type=status_error_type(status),
        message=_error_message(body, status),
    )
    return JSONResponse(
        status_code=status,
        content=err.to_envelope(),
        headers=headers,
        background=response.background,
    )


async def _collect_body(response: Response) -> bytes:
    """Drain a (possibly streaming) response body into bytes."""

    raw = bytearray()
    async for chunk in response.body_iterator:  # type: ignore[attr-defined]
        raw += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
    return bytes(raw)


def _carry_headers(response: Response) -> dict[str, str]:
    """Copy headers except content-length/-type (recomputed when re-emitting)."""

    skip = {"content-length", "content-type"}
    return {k: v for k, v in response.headers.items() if k.lower() not in skip}


def _is_openai_envelope(body: bytes) -> bool:
    """True if the body already is ``{"error": {"message": ...}}``."""

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("error"), dict)
        and "message" in data["error"]
    )


def _error_message(body: bytes, status: int) -> str:
    """Pull a human message out of a non-enveloped error body."""

    text = body.decode("utf-8", "replace").strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text or _status_phrase(status)
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            msgs = [
                d["msg"]
                for d in detail
                if isinstance(d, dict) and isinstance(d.get("msg"), str)
            ]
            if msgs:
                return "; ".join(msgs)
        if isinstance(data.get("message"), str):
            return data["message"]
    return _status_phrase(status)


def _status_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"
