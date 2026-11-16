"""Unit tests for project-scoped source-trace Langfuse score clients."""

import base64
import json

import pytest

from benchmark.services import source_score_client
from benchmark.services.source_score_client import ProjectScoreClientProvider


def _instances_config() -> str:
    payload = [
        {
            "projectId": "source-project",
            "baseUrl": "https://source-langfuse.example.com",
            "publicKey": "pk-source",
            "secretKey": "sk-source",
        }
    ]
    return base64.b64encode(json.dumps(payload).encode()).decode()


def test_provider_builds_and_caches_only_the_source_project_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    monkeypatch.setattr(source_score_client, "Langfuse", _Client)
    provider = ProjectScoreClientProvider(
        trace_instances=_instances_config(), timeout_seconds=12.0
    )

    first = provider.get_client("source-project")
    second = provider.get_client("source-project")

    assert first is second
    assert created == [
        {
            "public_key": "pk-source",
            "secret_key": "sk-source",
            "base_url": "https://source-langfuse.example.com",
            "timeout": 12.0,
        }
    ]


def test_provider_returns_none_for_an_unconfigured_source_project() -> None:
    provider = ProjectScoreClientProvider(
        trace_instances=_instances_config(), timeout_seconds=12.0
    )

    assert provider.get_client("report-project") is None
