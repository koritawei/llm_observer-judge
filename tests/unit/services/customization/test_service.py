"""Upload orchestration: idempotent reuse, and the order of counter write-back after the Git commit."""

from types import SimpleNamespace

import pytest
from benchmark.domain.customization_config import CustomizationConfig
from benchmark.customization import service as customization_service
from benchmark.customization.service import StageUploadRequest
from benchmark.reporting.framework.base import PromptSlot, StageCustomizationPolicy


def _request() -> StageUploadRequest:
    return StageUploadRequest(
        agent_name="example",
        business="biz",
        variant_key="v1",
        stage="judge",
        prompt_texts={"prompt": "模板"},
        created_by="tester",
    )


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        bundle_sha256="sha256",
        files={},
        prompt_ref=None,
        skill_ref=None,
    )


def _patch_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    stage_policy = StageCustomizationPolicy(
        allow_prompt=True,
        allow_skill=True,
        prompt_slots=(PromptSlot(key="prompt"),),
    )
    monkeypatch.setattr(
        customization_service,
        "customization_policy_for_name",
        lambda name: SimpleNamespace(judge=stage_policy, attribution=stage_policy),
    )
    monkeypatch.setattr(customization_service, "prepare_bundle", lambda request: _bundle())


def _config() -> CustomizationConfig:
    return CustomizationConfig(enabled=True, base_path="base", branch="master")


@pytest.mark.asyncio
async def test_upload_is_idempotent_for_same_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_policy(monkeypatch)
    last_ref = {"git_commit": "old", "bundle_sha256": "sha256", "content_version": 3}

    async def _peek(key):
        return {"last_ref": last_ref}

    async def _unexpected_version(key):
        raise AssertionError("idempotent upload must not allocate a new version")

    monkeypatch.setattr(customization_service, "peek_customization_sequence", _peek)
    monkeypatch.setattr(customization_service, "next_customization_version", _unexpected_version)

    assert (
        await customization_service.upload_stage_customization(
            _request(),
            config=_config(),
        )
        == last_ref
    )


@pytest.mark.asyncio
async def test_git_commit_precedes_counter_ref_write(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_policy(monkeypatch)
    events: list[str] = []
    recorded: list[dict] = []

    async def _peek(key):
        return {}

    async def _next_version(key):
        return 4

    async def _record(key, ref):
        events.append("record")
        recorded.append(ref)

    class _GitStore:
        async def commit_bundle(self, *, git_path: str, files: dict, message: str) -> str:
            events.append("commit")
            return "deadbeef"

    monkeypatch.setattr(customization_service, "peek_customization_sequence", _peek)
    monkeypatch.setattr(customization_service, "next_customization_version", _next_version)
    monkeypatch.setattr(customization_service, "record_customization_ref", _record)
    monkeypatch.setattr(customization_service, "git_path_for", lambda base, key, version: "path/v4")
    monkeypatch.setattr(customization_service, "build_manifest", lambda spec: {"schemaVersion": 1})

    result = await customization_service.upload_stage_customization(
        _request(),
        config=_config(),
        git_store=_GitStore(),
    )

    assert events == ["commit", "record"]
    assert result["git_commit"] == "deadbeef"
    assert result["content_version"] == 4
    assert result["git_path"] == "path/v4"
    assert recorded == [result]
