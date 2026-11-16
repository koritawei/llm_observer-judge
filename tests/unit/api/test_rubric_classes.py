"""Unit tests for GET /v1/benchmarks/rubric-classes (D2 read-only allowlist endpoint)."""

from benchmark.routes import benchmark as benchmark_api
from benchmark.runs.common.rubric_classes import RUBRIC_CLASS_REFS
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(benchmark_api.router)
    return TestClient(app)


def test_rubric_classes_returns_registry_aliases() -> None:
    resp = _client().get("/v1/benchmarks/rubric-classes")
    assert resp.status_code == 200
    entries = resp.json()
    aliases = {entry["alias"] for entry in entries}
    assert aliases == set(RUBRIC_CLASS_REFS)


def test_rubric_classes_entry_shape() -> None:
    entries = _client().get("/v1/benchmarks/rubric-classes").json()
    by_alias = {entry["alias"]: entry for entry in entries}

    kafka = by_alias["amap_agent_kafka_rubric"]
    assert kafka["class_ref"] == RUBRIC_CLASS_REFS["amap_agent_kafka_rubric"]
    assert kafka["aggregation_method"] is not None
    assert kafka["pass_threshold"] is not None
    assert isinstance(kafka["criteria"], list)
    assert kafka["criteria"], "amap_agent_kafka_rubric 应声明至少一个维度"
    criterion = kafka["criteria"][0]
    assert {"id", "name", "weight", "evaluator_ref", "description"} <= set(criterion)
    # §5.1 invariant: the bridge key that aligns evaluator_ref with the eval config scoreName
    assert kafka["criteria"][0]["evaluator_ref"] == "amap_agent"

    component = by_alias["component_rubric"]
    assert component["class_ref"] == RUBRIC_CLASS_REFS["component_rubric"]
    # component_rubric has no Rubric class attached (criteria are carried by component metadata)
    assert component["criteria"] == []
    assert component["aggregation_method"] is None
