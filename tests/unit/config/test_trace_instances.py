"""Unit tests for Langfuse instance config loading."""

from __future__ import annotations

import base64
import json

from benchmark.settings.trace_endpoints import (
    load_trace_instances,
    read_trace_instance_name,
    select_trace_instances_by_name,
)
from benchmark.trace import TraceEndpoint


def _encode(items: list[dict[str, object]]) -> str:
    return base64.b64encode(json.dumps(items).encode()).decode()


def test_load_trace_instances_empty_string() -> None:
    assert load_trace_instances("") == {}


def test_load_trace_instances_maps_by_project_id() -> None:
    encoded = _encode(
        [
            {
                "projectId": "proj-77",
                "name": "amap",
                "baseUrl": "https://lf.example.com",
                "publicKey": "pk-test",
                "secretKey": "sk-test",
            }
        ]
    )
    result = load_trace_instances(encoded)
    assert set(result) == {"proj-77"}
    inst = result["proj-77"]
    assert isinstance(inst, TraceEndpoint)
    assert inst.base_url == "https://lf.example.com"
    assert inst.name == "amap"


def test_load_trace_instances_falls_back_to_name() -> None:
    encoded = _encode(
        [
            {
                "name": "vehicle",
                "baseUrl": "https://lf.example.com",
                "publicKey": "pk-test",
                "secretKey": "sk-test",
            }
        ]
    )
    result = load_trace_instances(encoded)
    assert set(result) == {"vehicle"}


def test_load_trace_instances_invalid_base64_returns_empty() -> None:
    assert load_trace_instances("not-base64!!!") == {}


def test_load_trace_instances_skips_invalid_items() -> None:
    encoded = _encode(
        [
            {"name": "missing-url", "publicKey": "pk", "secretKey": "sk"},
            {
                "projectId": "proj-ok",
                "baseUrl": "https://lf.example.com",
                "publicKey": "pk-test",
                "secretKey": "sk-test",
            },
        ]
    )
    result = load_trace_instances(encoded)
    assert set(result) == {"proj-ok"}


def test_select_trace_instances_by_name_keeps_project_id() -> None:
    encoded = _encode(
        [
            {
                "projectId": "proj-v4",
                "name": "trace-v4",
                "baseUrl": "https://v4.example.com",
                "publicKey": "pk",
                "secretKey": "sk",
            },
            {
                "projectId": "proj-old",
                "name": "trace-old",
                "baseUrl": "https://old.example.com",
                "publicKey": "pk",
                "secretKey": "sk",
            },
        ]
    )
    result = select_trace_instances_by_name(encoded, "trace-v4")
    assert set(result) == {"proj-v4"}
    assert result["proj-v4"].base_url == "https://v4.example.com"


def test_read_trace_instance_name(tmp_path) -> None:
    path = tmp_path / "app.prod.yaml"
    path.write_text("custom:\n  trace:\n    instance_name: trace-v4\n", encoding="utf-8")
    assert read_trace_instance_name(path) == "trace-v4"
    assert read_trace_instance_name(tmp_path / "missing.yaml") is None
