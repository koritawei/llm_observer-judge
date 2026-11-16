"""Unit tests for the OpenAI-compatible /v1/chat/completions eval entry."""

import json
from unittest.mock import MagicMock

import pytest
from benchmark.routes import eval as eval_api
from benchmark.routes.schema import (
    ChatCompletion,
    ChatCompletionRequest,
    ToolCall,
    ToolCallFunction,
    build_proxy_chat_completion,
)
from benchmark.evaluators.base import EvaluatorRequest, EvaluatorResult
from benchmark.http.middleware.error_middleware import ErrorHandlingMiddleware, RequestContextMiddleware
from benchmark.services.evaluation import EvalRunOptions, get_evaluator_service
from benchmark.utils.error_handler import install_error_handlers
from benchmark.utils.exceptions import EvaluatorError
from benchmark.utils.trace_report import eval_report_root
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, PlainTextResponse


class _FakeService:
    def __init__(
        self,
        result: EvaluatorResult | None = None,
        error: Exception | None = None,
        proxy_response: ChatCompletion | None = None,
    ) -> None:
        self.result = result or EvaluatorResult(score=7.0, reason="ok")
        self.error = error
        self.proxy_response = proxy_response
        self.seen: list[str] = []
        self.proxy_seen: list[dict] = []
        self.requests: list[EvaluatorRequest] = []
        self.options: list[EvalRunOptions | None] = []

    async def evaluate(
        self,
        evaluator_name: str,
        request: EvaluatorRequest,
        *,
        options: EvalRunOptions | None = None,
    ) -> EvaluatorResult:
        self.seen.append(evaluator_name)
        self.requests.append(request)
        self.options.append(options)
        if self.error is not None:
            raise self.error
        return self.result

    async def proxy_chat_completion(self, payload: ChatCompletionRequest) -> ChatCompletion:
        self.proxy_seen.append(payload.model_dump())
        if self.proxy_response is not None:
            return self.proxy_response
        return build_proxy_chat_completion(model=payload.model, content="ok")


def _client(service: _FakeService, *, raise_server_exceptions: bool = True) -> TestClient:
    app = FastAPI()
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)
    app.include_router(eval_api.router)
    # Inject the fake via the DI seam (no request.app.state).
    app.dependency_overrides[get_evaluator_service] = lambda: service
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _ok_payload() -> dict:
    # Evaluator JSON envelope — has_eval_envelope() True → evaluator path.
    content = json.dumps({"input": "你是谁", "output": "我是助手"})
    return {"model": "example", "messages": [{"role": "user", "content": content}]}


@pytest.mark.parametrize("model", ["vehicle_agent", "vui_agent"])
@pytest.mark.parametrize("skip_resolve", [True, False])
@pytest.mark.parametrize("report_state", ["recording", "noop", "unavailable"])
def test_trace_only_request_reaches_evaluator(
    model: str, skip_resolve: bool, report_state, monkeypatch
) -> None:
    root = MagicMock()
    root.trace_id = "abcdef0123456789abcdef0123456789"
    root._otel_span.is_recording.return_value = report_state == "recording"
    client = MagicMock()
    client.start_as_current_observation.return_value.__enter__.return_value = root
    if report_state == "unavailable":
        client.start_as_current_observation.side_effect = RuntimeError("SDK unavailable")
    monkeypatch.setattr("benchmark.utils.trace_report.get_report_client", lambda: client)

    class ReportingService(_FakeService):
        async def evaluate(self, *args, **kwargs):
            with eval_report_root(name="test-report"):
                return await super().evaluate(*args, **kwargs)

    svc = ReportingService(result=EvaluatorResult(score=7.0, reason="ok"))
    response = _client(svc).post(
        "/v1/chat/completions",
        json={
            "model": model,
            "skip_trace_resolve": skip_resolve,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "metadata": {"trace_id": "trace-1"},
                            "project_id": "project-key",
                            "agent_name": model,
                        }
                    ),
                }
            ],
        },
    )
    assert response.status_code == 200
    expected_id = root.trace_id if report_state == "recording" else None
    assert response.json()["trace_id"] == expected_id
    assert svc.seen == [model] and svc.proxy_seen == []
    assert svc.requests[0].input is None and svc.requests[0].output is None
    assert svc.requests[0].trace_id == "trace-1"
    assert svc.options[0].resolve_trace is (not skip_resolve)
    assert json.loads(response.json()["choices"][0]["message"]["content"])["score"] == 7.0


def test_success_returns_service_body() -> None:
    svc = _FakeService()
    resp = _client(svc).post("/v1/chat/completions", json=_ok_payload())
    assert resp.status_code == 200
    assert resp.json()["object"] == "chat.completion"
    assert svc.seen[0] == "example"


def test_invalid_json_body_returns_400() -> None:
    resp = _client(_FakeService()).post(
        "/v1/chat/completions",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_json"


def test_missing_model_returns_400() -> None:
    resp = _client(_FakeService()).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["param"] == "model"
    assert body["error"]["type"] == "missing_required_parameter"


def test_missing_messages_returns_400() -> None:
    resp = _client(_FakeService()).post("/v1/chat/completions", json={"model": "example"})
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "messages"


def test_stream_true_is_rejected() -> None:
    payload = _ok_payload() | {"stream": True}
    resp = _client(_FakeService()).post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "streaming_not_supported"


def test_non_object_body_returns_400() -> None:
    resp = _client(_FakeService()).post("/v1/chat/completions", json=[1, 2, 3])
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_body"


def test_evaluator_error_maps_to_openai_envelope() -> None:
    err = EvaluatorError(
        code=400, error_type="model_not_found", message="Unknown evaluator model 'ghost'."
    )
    resp = _client(_FakeService(error=err)).post("/v1/chat/completions", json=_ok_payload())
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["type"] == "model_not_found"
    assert body["error"]["code"] == 400
    assert body["error"]["param"] == "model"
    assert body["error"]["message"] == "Unknown evaluator model 'ghost'."


def test_evaluator_error_llm_maps_to_502() -> None:
    err = EvaluatorError(code=502, error_type="llm_failed", message="llm failed")
    resp = _client(_FakeService(error=err)).post("/v1/chat/completions", json=_ok_payload())
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "llm_failed"


def test_unknown_error_code_maps_to_500() -> None:
    err = EvaluatorError(code=500, error_type="evaluation_failed", message="boom")
    resp = _client(_FakeService(error=err)).post("/v1/chat/completions", json=_ok_payload())
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "evaluation_failed"


def test_unexpected_exception_maps_to_500_envelope() -> None:
    # A non-EvaluatorError bubbling out of the service hits the catch-all handler:
    # still the OpenAI envelope (500 server_error), not a leaked default body.
    svc = _FakeService(error=RuntimeError("kaboom"))
    resp = _client(svc, raise_server_exceptions=False).post(
        "/v1/chat/completions", json=_ok_payload()
    )
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["type"] == "internal_error"
    assert body["error"]["code"] == 500
    # the leaked exception message is NOT surfaced to the client
    assert "kaboom" not in body["error"]["message"]


def test_response_carries_request_id_header() -> None:
    resp = _client(_FakeService()).post("/v1/chat/completions", json=_ok_payload())
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-Id")


def test_error_response_carries_request_id_header() -> None:
    # Handled error responses flow back out through the middleware → still stamped.
    resp = _client(_FakeService()).post("/v1/chat/completions", json={"model": "example"})
    assert resp.status_code == 400
    assert resp.headers.get("X-Request-Id")


def _bare_app() -> FastAPI:
    """An app with only the two middlewares — no registered handlers."""
    app = FastAPI()
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestContextMiddleware)
    return app


def test_middleware_repackages_json_detail_error_into_envelope() -> None:
    # A raw {"detail": ...} error (not from our handlers) is normalized into the OpenAI
    # envelope by ErrorHandlingMiddleware reading the error info off the response.
    app = _bare_app()

    @app.get("/raw")
    async def _raw() -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    resp = TestClient(app).get("/raw")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["message"] == "Not Found"
    assert body["error"]["type"] == "not_found"
    assert body["error"]["code"] == 404
    assert resp.headers.get("X-Request-Id")  # still stamped by the outer middleware


def test_middleware_repackages_plain_text_error_into_envelope() -> None:
    app = _bare_app()

    @app.get("/boom")
    async def _boom() -> PlainTextResponse:
        return PlainTextResponse(status_code=503, content="upstream down")

    resp = TestClient(app).get("/boom")
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["type"] == "service_unavailable"
    assert body["error"]["code"] == 503
    assert body["error"]["message"] == "upstream down"


def test_already_enveloped_error_is_not_double_wrapped() -> None:
    # An error from the registered handlers is re-emitted verbatim — no {"error":{"error"}}.
    resp = _client(_FakeService()).post("/v1/chat/completions", json={"model": "example"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["param"] == "messages"
    assert "error" not in body["error"]


# ── proxy fallback — non-eval requests (Langfuse connectivity probe, function-calling test) ──


def test_non_envelope_plain_text_falls_through_to_proxy() -> None:
    """Plain-text content (Langfuse save probe shape) → bypass evaluator, call proxy."""

    svc = _FakeService()
    resp = _client(svc).post(
        "/v1/chat/completions",
        json={
            "model": "example",
            "messages": [
                {
                    "role": "user",
                    "content": "Extract a score (1-5) and reasoning from this text: This is a test. It worked perfectly.",
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert svc.seen == []  # evaluator NOT called
    assert svc.proxy_seen and svc.proxy_seen[0]["model"] == "example"
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_non_envelope_xml_content_falls_through_to_proxy() -> None:
    """XML-style <input> content (working-branch shape) → bypass evaluator, call proxy."""

    svc = _FakeService()
    resp = _client(svc).post(
        "/v1/chat/completions",
        json={
            "model": "example",
            "messages": [{"role": "user", "content": "<input>foo</input><output>bar</output>"}],
        },
    )
    assert resp.status_code == 200
    assert svc.seen == []  # XML does not parse as JSON envelope → fallback
    assert svc.proxy_seen


def test_empty_envelope_falls_through_to_proxy() -> None:
    """A literal ``{}`` parses but carries no input/output → fallback (not the empty-eval verdict)."""

    svc = _FakeService()
    resp = _client(svc).post(
        "/v1/chat/completions",
        json={"model": "example", "messages": [{"role": "user", "content": "{}"}]},
    )
    assert resp.status_code == 200
    assert svc.seen == []
    assert svc.proxy_seen


def test_function_calling_response_shape_is_preserved() -> None:
    """Proxy response with tool_calls → tool_calls envelope, content=null, finish_reason=tool_calls."""

    svc = _FakeService(
        proxy_response=build_proxy_chat_completion(
            model="example",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_abc",
                    function=ToolCallFunction(
                        name="score_response",
                        arguments=json.dumps({"reasoning": "ok", "score": 5}),
                    ),
                )
            ],
        )
    )
    resp = _client(svc).post(
        "/v1/chat/completions",
        json={
            "model": "example",
            "messages": [{"role": "user", "content": "Extract a score..."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "score_response",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string"},
                                "score": {"type": "number"},
                            },
                        },
                    },
                }
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    tool_calls = choice["message"]["tool_calls"]
    assert tool_calls and tool_calls[0]["function"]["name"] == "score_response"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {"reasoning": "ok", "score": 5}


def test_envelope_with_input_routes_to_evaluator() -> None:
    """A real eval envelope (input + output) → evaluator path, NOT proxy."""

    svc = _FakeService()
    content = json.dumps(
        {
            "input": {"query": "你是谁"},
            "output": {"text": "我是助手"},
            "metadata": {"trace_id": "trace-1"},
            "agent_name": "vehicle-agent",
            "project_id": "p1",
        }
    )
    resp = _client(svc).post(
        "/v1/chat/completions",
        json={"model": "example", "messages": [{"role": "user", "content": content}]},
    )
    assert resp.status_code == 200
    assert svc.seen == ["example"]
    assert svc.proxy_seen == []
    body = resp.json()
    content_str = body["choices"][0]["message"]["content"]
    parsed = json.loads(content_str)
    assert parsed == {"score": 7.0, "reasoning": "ok"}
