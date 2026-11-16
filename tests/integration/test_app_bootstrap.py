"""Real create_app bootstrap — verify agent_manager assembly (HTTP end-to-end is in test_eval_http_e2e)."""
# ruff: noqa: ANN001, ANN201, PLR2004

import pytest
from benchmark.settings import Settings
from benchmark.records.models.trace_record import TraceRecord
from benchmark.service.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_observers():
    TraceRecord.clear_observer()
    yield
    TraceRecord.clear_observer()


def _app(tmp_path):
    return create_app(settings=Settings(database_url=f"sqlite:///{tmp_path}/eval.db"))


def test_bootstrap_registers_example_evaluator(tmp_path) -> None:
    app = _app(tmp_path)
    with TestClient(app):
        assert "judge_example" in app.state.runtime.agent_manager.list()


def test_bootstrap_registers_example_attribution(tmp_path) -> None:
    app = _app(tmp_path)
    with TestClient(app):
        manager = app.state.runtime.agent_manager
        assert "attribution_example" in manager.list()


def test_attribution_agent_resolvable_after_real_bootstrap(tmp_path) -> None:
    """After a real bootstrap, attribution_example must be retrievable from agent_manager."""

    app = _app(tmp_path)
    with TestClient(app):
        manager = app.state.runtime.agent_manager
        assert "attribution_example" in manager.list()
        from business.agents.example.attribution.agent import AttributionAgent

        instance = manager.get("attribution_example")
        assert isinstance(instance, AttributionAgent)


# ── OpenAPI/Swagger docs endpoint gate (ALIBABA-2026-46066077) ──────────────────
# Not part of lifespan: doc endpoints are served by FastAPI directly and do not depend on DB/runtime.


def test_api_docs_open_on_daily(tmp_path) -> None:
    """daily/local opens OpenAPI docs by default (regression baseline)."""
    client = TestClient(_app(tmp_path))
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_api_docs_closed_on_pre_and_prod(tmp_path) -> None:
    """Security regression (ALIBABA-2026-46066077): pre/prod must not expose swagger/OpenAPI docs."""
    for env in ("pre", "prod"):
        app = create_app(
            settings=Settings(database_url=f"sqlite:///{tmp_path}/eval-{env}.db", env=env)
        )
        client = TestClient(app)
        for path in ("/openapi.json", "/docs", "/redoc"):
            assert client.get(path).status_code == 404, f"{env} {path} must be 404"


def test_api_docs_explicit_override(tmp_path) -> None:
    """docs_enabled explicitly overrides the environment gate: an escape hatch to temporarily enable docs on pre for investigation."""
    app = create_app(
        settings=Settings(
            database_url=f"sqlite:///{tmp_path}/eval.db",
            env="pre",
            docs_enabled=True,
        )
    )
    client = TestClient(app)
    assert client.get("/openapi.json").status_code == 200
