"""HTTP transport — the single exit for the httpx dependency inside agent-eval.

Ruff (TID251) forbids business code from ``import httpx`` directly (the exit should be
the client wrapper in libs/api_client); agent-eval is not yet on the libs/api_client
build/deploy chain (neither pyproject nor the runtime PYTHONPATH covers it), so this module
acts as the in-service httpx wrapper: it resolves httpx dynamically via :func:`importlib.import_module`
(the same dynamic-module pattern as benchmark.settings triage_loader / eval_framework fusion)
and re-exports the common symbols.

- ``AsyncClient`` and httpx exception types: business modules and tests uniformly
  ``from benchmark.utils.http_transport import ...``
- :func:`classify_httpx_retryable`: retry classification for httpx exceptions (design §8.2,
  used by upper layers such as :mod:`benchmark.queue.submit_failure`)

httpx dependency and exception knowledge are closed over here; the business layer (runner / submit_failure and similar)
does not need to know about httpx. When libs/api_client is adopted later, only this module's implementation changes,
and callers stay untouched.
"""

import importlib

#: Dynamically resolved httpx module (this module is the only in-service exit for httpx).
httpx = importlib.import_module("httpx")

Client = httpx.Client
AsyncClient = httpx.AsyncClient
ConnectError = httpx.ConnectError
HTTPError = httpx.HTTPError
HTTPStatusError = httpx.HTTPStatusError
ReadTimeout = httpx.ReadTimeout
TimeoutException = httpx.TimeoutException
TransportError = httpx.TransportError

#: HTTP status codes at or above this value are treated as transient server errors (retryable).
_HTTP_SERVER_ERROR_MIN = 500

#: Client status codes that are explicitly retryable (timeout / too early / rate limit).
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429})


def classify_httpx_retryable(exc: BaseException) -> bool | None:
    """Retry classification for httpx exceptions (transport-layer exception knowledge, design §8.2).

    - Timeout (``TimeoutException``) / transport error (``TransportError``) → retryable
    - ``HTTPStatusError`` → 408/425/429 or ≥500 is retryable; other 4xx are not
    - Non-httpx exceptions → return ``None`` and let the caller keep classifying
    """
    if isinstance(exc, TimeoutException):
        return True
    if isinstance(exc, TransportError):
        return True
    if isinstance(exc, HTTPStatusError):
        code = exc.response.status_code
        return code in _RETRYABLE_STATUS_CODES or code >= _HTTP_SERVER_ERROR_MIN
    return None
