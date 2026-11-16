"""L2 / resolution / overview / task visibility (extension of the main seam)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from benchmark.domain import customization_config as customization_config_module
from benchmark.domain.benchmark_console import (
    AgentDocument,
    ConsoleError,
    EvalTaskDocument,
    list_agents,
    list_all_agent_registrations,
    resolve_agent_registration,
    save_agent,
    save_task,
)
from benchmark.domain.customization_config import CustomizationConfig
from benchmark.customization.bundle_store import BundleStore, set_bundle_store
from benchmark.customization.manifest import bundle_sha256


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeFetcher:
    def __init__(self) -> None:
        self.contents: dict[str, bytes] = {}

    async def read_files(self, *, git_commit: str, paths: list[str]) -> dict[str, bytes]:
        return {path: self.contents[path] for path in paths if path in self.contents}


@pytest.fixture()
def _customization_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fetcher = _FakeFetcher()
    config = CustomizationConfig(enabled=True, cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    set_bundle_store(BundleStore(config, fetcher))
    yield fetcher
    set_bundle_store(None)


def _judge_ref(fetcher: _FakeFetcher, text: str = "prompt", version: int = 1) -> dict[str, Any]:
    data = text.encode()
    git_path = f"benchmark-customizations/example_benchmark/biz/v1/judge/v{version}"
    prompt_ref = {
        "slots": ["prompt"],
        "files": [{"path": "prompt/prompts/prompt.md", "sha256": _sha(data), "bytes": len(data)}],
    }
    digest = bundle_sha256("judge", prompt_ref, None)
    fetcher.contents[f"{git_path}/prompt/prompts/prompt.md"] = data
    fetcher.contents[f"{git_path}/manifest.json"] = json.dumps({"bundleSha256": digest}).encode()
    return {
        "stage": "judge",
        "contentVersion": version,
        "gitBranch": "master",
        "gitPath": git_path,
        "gitCommit": f"commit-judge-{version}",
        "bundleSha256": digest,
        "prompt": prompt_ref,
        "createdAt": "2026-11-02 10:00:00",
        "createdBy": "tester",
    }


def _custom_agent(
    fetcher: _FakeFetcher,
    *,
    scope: str,
    text: str = "prompt",
    version: int = 1,
    doc_id: str = "tmp-1",
) -> AgentDocument:
    return AgentDocument.model_validate(
        {
            "id": doc_id,
            "name": "example_benchmark",
            "owner": "eval-team",
            "registeredAt": "2026-11-02 10:00:00",
            "business": "biz",
            "variantKey": "v1",
            "agentScope": scope,
            "customization": {"judge": _judge_ref(fetcher, text, version)},
        }
    )


@pytest.mark.asyncio
async def test_customized_hidden_in_other_scope(_customization_env: _FakeFetcher) -> None:
    await save_agent(_custom_agent(_customization_env, scope="amap-agent"))
    await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-plain",
                "name": "example_benchmark",
                "owner": "eval-team",
                "registeredAt": "2026-11-02 10:00:00",
                "agentScope": "amap-agent",
            }
        )
    )
    amap = await list_agents("amap-agent")
    assert any(item.get("customization") for item in amap)
    # vehicle is not registered → the list is empty; the amap customization is not visible
    vehicle = await list_agents("vehicle-agent")
    assert vehicle == []


@pytest.mark.asyncio
async def test_plain_registration_not_reused_across_scopes() -> None:
    first = await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-amap-plain",
                "name": "example_benchmark",
                "owner": "eval-team",
                "registeredAt": "2026-11-02 10:00:00",
                "agentScope": "amap-agent",
            }
        )
    )
    second = await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-vehicle-plain",
                "name": "example_benchmark",
                "owner": "eval-team",
                "registeredAt": "2026-11-02 11:00:00",
                "agentScope": "vehicle-agent",
            }
        )
    )
    assert first["id"] != second["id"]
    assert first["agentScope"] == "amap-agent"
    assert second["agentScope"] == "vehicle-agent"


@pytest.mark.asyncio
async def test_resolve_scope_mismatch(_customization_env: _FakeFetcher) -> None:
    saved = await save_agent(_custom_agent(_customization_env, scope="amap-agent"))
    with pytest.raises(ConsoleError) as exc:
        await resolve_agent_registration(
            "example_benchmark",
            saved["id"],
            agent_scope="vehicle-agent",
        )
    assert exc.value.status_code == 409
    assert "customization_scope_mismatch" in exc.value.detail


@pytest.mark.asyncio
async def test_overview_lists_all_scopes(_customization_env: _FakeFetcher) -> None:
    await save_agent(_custom_agent(_customization_env, scope="amap-agent", text="a"))
    await save_agent(
        _custom_agent(
            _customization_env,
            scope="vehicle-agent",
            text="b",
            version=2,
            doc_id="tmp-2",
        )
    )
    overview = await list_all_agent_registrations()
    scopes = {item.get("agentScope") for item in overview if item.get("customization")}
    assert "amap-agent" in scopes
    assert "vehicle-agent" in scopes


@pytest.mark.asyncio
async def test_save_task_rejects_invisible_customization(
    _customization_env: _FakeFetcher,
) -> None:
    saved = await save_agent(_custom_agent(_customization_env, scope="amap-agent"))
    task = EvalTaskDocument.model_validate(
        {
            "id": "task-1",
            "agentScope": "vehicle-agent",
            "name": "t",
            "agentId": saved["id"],
            "datasetId": "ds-1",
            "formula": "weighted-average",
            "trigger": "immediate",
            "status": "active",
            "processedCount": 0,
            "totalItems": 0,
            "startedAt": "2026-11-02 10:00:00",
        }
    )
    with pytest.raises(ConsoleError) as exc:
        await save_task(task)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_save_task_rejects_plain_from_other_scope() -> None:
    saved = await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-plain-amap",
                "name": "example_benchmark",
                "owner": "eval-team",
                "registeredAt": "2026-11-02 10:00:00",
                "agentScope": "amap-agent",
            }
        )
    )
    task = EvalTaskDocument.model_validate(
        {
            "id": "task-plain",
            "agentScope": "vehicle-agent",
            "name": "t",
            "agentId": saved["id"],
            "datasetId": "ds-1",
            "formula": "weighted-average",
            "trigger": "immediate",
            "status": "active",
            "processedCount": 0,
            "totalItems": 0,
            "startedAt": "2026-11-02 10:00:00",
        }
    )
    with pytest.raises(ConsoleError) as exc:
        await save_task(task)
    assert exc.value.status_code == 403
