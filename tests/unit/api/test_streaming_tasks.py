"""Unit tests for the benchmark streaming task-management API (§6.5)."""

from unittest.mock import MagicMock

import pytest
from benchmark.routes import streaming_tasks as streaming_api
from benchmark.domain.streaming_config import StreamingConfig, StreamingTaskConfig
from benchmark.streaming.registry import RunningTask, StreamingTaskRegistry
from benchmark.streaming.tasks import StreamingTaskSpec
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _FakeRunner:
    """Lightweight runner (registry plus pause/resume records)."""

    def __init__(self) -> None:
        self.registry = StreamingTaskRegistry()
        self._reconciler = None

    async def pause_task(self, name: str, **kwargs: object) -> bool:
        task = self.registry.get(name)
        if task is None:
            return False
        task.runtime_state = "paused"
        task.consumers = {}
        return True

    async def resume_task(self, name: str) -> bool:
        task = self.registry.get(name)
        if task is None:
            return False
        task.runtime_state = "running"
        task.consumers = {"langfuse.records": [MagicMock(_running=True)]}
        return True


def _client(
    runner: object | None,
    declared: StreamingConfig | None,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    monkeypatch.setattr(streaming_api, "fetch_streaming_config", lambda: declared)
    app = FastAPI()
    app.include_router(streaming_api.router)
    if runner is not None:
        app.state.streaming_runner = runner
    return TestClient(app)


def _declared_config() -> StreamingConfig:
    return StreamingConfig(
        enabled=True,
        tasks=[
            StreamingTaskConfig.model_validate(
                {
                    "name": "amap_agent_daily",
                    "topics": ["langfuse.records"],
                    "group_id": None,
                    "state": "running",
                    "rubric_class": "amap_agent_kafka_rubric",
                    "daily_report": {"enabled": True},
                },
            ),
        ],
    )


def _running_task(name: str = "amap_agent_daily") -> RunningTask:
    spec = StreamingTaskSpec(
        name=name,
        topics=["langfuse.records"],
        group_id="agent-eval-judge",
    )
    task = RunningTask(spec=spec, evaluator=MagicMock(), runtime_state="running")
    task.consumers = {"langfuse.records": [MagicMock(_running=True)]}
    return task


def test_tasks_returns_503_when_runner_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(None, _declared_config(), monkeypatch)
    resp = client.get("/v1/benchmarks/streaming/tasks")
    assert resp.status_code == 503
    assert "BENCHMARK_STREAMING_ENABLED" in resp.json()["detail"]


def test_tasks_returns_declared_and_runtime_views(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _FakeRunner()
    runner.registry.upsert(_running_task())
    client = _client(runner, _declared_config(), monkeypatch)

    resp = client.get("/v1/benchmarks/streaming/tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["declared"]["available"] is True
    assert body["declared"]["enabled"] is True
    declared_task = body["declared"]["tasks"]["amap_agent_daily"]
    assert declared_task["state"] == "running"
    assert declared_task["rubric_class"] == "amap_agent_kafka_rubric"
    assert declared_task["daily_report_enabled"] is True
    (runtime_task,) = body["tasks"]
    assert runtime_task["name"] == "amap_agent_daily"
    assert runtime_task["runtime_state"] == "running"
    assert runtime_task["is_running"] is True
    assert runtime_task["subscribed_topics"] == ["langfuse.records"]


def test_tasks_marks_declared_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _FakeRunner()
    client = _client(runner, None, monkeypatch)
    resp = client.get("/v1/benchmarks/streaming/tasks")
    assert resp.status_code == 200
    assert resp.json()["declared"]["available"] is False


def test_pause_unknown_task_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _FakeRunner()
    client = _client(runner, _declared_config(), monkeypatch)
    resp = client.post("/v1/benchmarks/streaming/tasks/ghost/pause")
    assert resp.status_code == 404


def test_pause_then_resume_task(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _FakeRunner()
    runner.registry.upsert(_running_task())
    client = _client(runner, _declared_config(), monkeypatch)

    resp = client.post("/v1/benchmarks/streaming/tasks/amap_agent_daily/pause")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["runtime_state"] == "paused"
    assert "RemoteConfig" in body["hint"]

    resp = client.post("/v1/benchmarks/streaming/tasks/amap_agent_daily/resume")
    assert resp.status_code == 200
    assert resp.json()["runtime_state"] == "running"


def test_pause_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _FakeRunner()
    runner.registry.upsert(_running_task())
    client = _client(runner, _declared_config(), monkeypatch)

    assert client.post("/v1/benchmarks/streaming/tasks/amap_agent_daily/pause").status_code == 200
    resp = client.post("/v1/benchmarks/streaming/tasks/amap_agent_daily/pause")
    assert resp.status_code == 200
    assert resp.json()["runtime_state"] == "paused"


def test_pause_stopped_task_returns_409(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stopped task cannot be paused (§6.2 state machine: recovery requires an explicit resume)."""
    runner = _FakeRunner()
    task = _running_task()
    task.runtime_state = "stopped"
    task.consumers = {}
    runner.registry.upsert(task)
    client = _client(runner, _declared_config(), monkeypatch)

    resp = client.post("/v1/benchmarks/streaming/tasks/amap_agent_daily/pause")
    assert resp.status_code == 409
    assert "stopped" in resp.json()["detail"]
    assert runner.registry.get("amap_agent_daily").runtime_state == "stopped"


def test_audit_log_emitted_on_pause(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    runner = _FakeRunner()
    runner.registry.upsert(_running_task())
    client = _client(runner, _declared_config(), monkeypatch)

    with caplog.at_level(logging.INFO, logger="agent_eval.api.streaming_tasks"):
        client.post(
            "/v1/benchmarks/streaming/tasks/amap_agent_daily/pause",
            headers={"x-request-id": "req-123"},
        )
    audit = [r.message for r in caplog.records if "streaming task audit" in r.message]
    assert len(audit) == 1
    assert "action=pause" in audit[0]
    assert "task=amap_agent_daily" in audit[0]
    assert "req-123" in audit[0]
