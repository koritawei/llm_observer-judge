"""Customization version-list API (/benchmark-agents/customizations/versions)."""

import json

import pytest
from benchmark.routes.benchmark_console import list_customization_versions
from benchmark.domain import customization_config as customization_config_module
from benchmark.domain.customization_config import CustomizationConfig


class _FakeStore:
    expected_suffix: str = "/example/biz/default/judge"

    async def list_stage_versions(self, base_path: str) -> list[int]:
        assert base_path.endswith(self.expected_suffix), base_path
        return [3, 2, 1]

    async def latest_commit_for_path(self, path: str) -> str:
        return f"sha-{path.rsplit('/', 1)[-1]}"

    async def read_files(self, *, git_commit: str, paths: list[str]) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for path in paths:
            version = path.rsplit("/v", 1)[-1].split("/", 1)[0]
            if path.endswith("manifest.json") and version != "1":
                manifest = {
                    "contentVersion": int(version),
                    "bundleSha256": f"bundle-{version}",
                    "prompt": {
                        "slots": ["prompt"],
                        "files": [
                            {
                                "path": "prompt/prompts/prompt.md",
                                "sha256": "a" * 64,
                                "bytes": 12,
                            }
                        ],
                    },
                    "skill": {
                        "entry": "SKILL.md",
                        "files": [{"path": "skill/SKILL.md", "sha256": "b" * 64, "bytes": 8}],
                    },
                    "createdAt": "2026-10-21T00:00:00Z",
                    "createdBy": "tester",
                }
                result[path] = json.dumps(manifest).encode("utf-8")
            # v1: no manifest → the list skips that version
        return result


@pytest.mark.asyncio
async def test_versions_returns_full_refs_desc(monkeypatch: pytest.MonkeyPatch) -> None:
    config = CustomizationConfig(enabled=True)
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    monkeypatch.setattr("benchmark.routes.benchmark_console.fetch_customization_config", lambda: config)
    monkeypatch.setattr(
        "benchmark.routes.benchmark_console.GitContentStore",
        lambda _config: _FakeStore(),
    )
    body = await list_customization_versions(
        agent_name="example",
        business="biz",
        variant_key="default",
        stage="judge",
    )
    # v1 missing a manifest is skipped; v3/v2 return a complete ref
    assert [item["contentVersion"] for item in body] == [3, 2]
    first = body[0]
    assert first["gitPath"].endswith("/judge/v3")
    assert first["gitCommit"] == "sha-v3"
    assert first["gitBranch"] == "master"
    assert first["bundleSha256"] == "bundle-3"
    assert first["prompt"]["slots"] == ["prompt"]
    assert first["skill"]["entry"] == "SKILL.md"
    assert first["createdBy"] == "tester"


@pytest.mark.asyncio
async def test_versions_invalid_stage_rejected() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await list_customization_versions(
            agent_name="example",
            business="",
            variant_key="",
            stage="other",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_versions_slugifies_non_slug_agent_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read side and write side share stage_base_path: a non-slug registration name can still hit the Git directory."""
    config = CustomizationConfig(enabled=True)
    store = _FakeStore()
    store.expected_suffix = "/amap_agent/biz/v1/judge"
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    monkeypatch.setattr("benchmark.routes.benchmark_console.fetch_customization_config", lambda: config)
    monkeypatch.setattr(
        "benchmark.routes.benchmark_console.GitContentStore",
        lambda _config: store,
    )
    body = await list_customization_versions(
        agent_name="Amap Agent",
        business="biz",
        variant_key="v1",
        stage="judge",
    )
    assert [item["contentVersion"] for item in body] == [3, 2]
    assert body[0]["gitPath"].endswith("/amap_agent/biz/v1/judge/v3")


@pytest.mark.asyncio
async def test_versions_rejects_path_traversal_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    config = CustomizationConfig(enabled=True)
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    monkeypatch.setattr("benchmark.routes.benchmark_console.fetch_customization_config", lambda: config)
    with pytest.raises(HTTPException) as exc:
        await list_customization_versions(
            agent_name="example",
            business="../x",
            variant_key="v1",
            stage="judge",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "invalid_path_segment"
