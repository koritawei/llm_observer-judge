"""Unit tests for health probe routes."""

from benchmark.routes import health as health_api
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(health_api.router)
    return TestClient(app)


def test_health_returns_json_probe() -> None:
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "detail": "success"}


def test_status_taobao_returns_plain_success() -> None:
    client = _client()
    resp = client.get("/status.taobao")
    assert resp.status_code == 200
    assert resp.text == "success"


def test_health_reports_degraded_eval_queue() -> None:
    """When the BullMQ stack is degraded (app.state.eval_queue_manager missing), surface the degradation and its reason.

    status stays ok (the liveness probe must not fail, so the instance is not taken out by mistake). The eval_queue field carries
    degraded plus the degradation reason, for ops observation and alerting.
    """
    client = _client()
    # During TestClient, simulate the degraded state on app.state (fallback set to None plus a reason)
    with client:
        client.app.state.eval_queue_manager = None  # type: ignore[assignment]
        client.app.state.eval_queue_degraded_reason = (
            "RuntimeError: Redis connection not configured"
        )
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["eval_queue"].startswith("degraded")
    assert "Redis connection not configured" in body["eval_queue"]


def test_health_no_eval_queue_field_when_healthy() -> None:
    """When the queue stack is healthy, do not attach an eval_queue field (the shape stays compatible; the field appears only when degraded)."""
    client = _client()
    with client:
        client.app.state.eval_queue_manager = object()  # The queue stack is ready, with no degradation marker
        resp = client.get("/health")
    assert resp.status_code == 200
    assert "eval_queue" not in resp.json()
