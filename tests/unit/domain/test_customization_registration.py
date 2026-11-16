"""Business customization registration: version assignment / idempotency / immutability / switches / selection algorithm / counters."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from benchmark.records.models.runtime_config_record import CustomizationKey
from benchmark.domain import customization_config as customization_config_module
from benchmark.domain.benchmark_console import (
    AgentDocument,
    ConsoleError,
    list_agent_documents,
    registration_snapshot,
    resolve_agent_registration,
    save_agent,
    strip_agent_execution_fields,
)
from benchmark.domain.customization_config import CustomizationConfig
from benchmark.domain.runtime_config_store import (
    next_customization_version,
    peek_customization_sequence,
)
from benchmark.customization.bundle_store import BundleStore, set_bundle_store
from benchmark.customization.manifest import bundle_sha256
from benchmark.customization.service import (
    StageUploadRequest,
    upload_stage_customization,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeFetcher:
    def __init__(self) -> None:
        self.contents: dict[str, bytes] = {}

    async def read_files(self, *, git_commit: str, paths: list[str]) -> dict[str, bytes]:
        return {path: self.contents[path] for path in paths if path in self.contents}


def _stage_ref(
    fetcher: _FakeFetcher,
    stage: str,
    *,
    prompt_text: str | None = "业务模板 ${query}",
    version: int = 1,
) -> dict[str, Any]:
    prompt_ref = None
    contents: dict[str, bytes] = {}
    git_path = f"benchmark-customizations/example_benchmark/biz/v1/{stage}/v{version}"
    if prompt_text is not None:
        data = prompt_text.encode()
        prompt_ref = {
            "slots": ["prompt"],
            "files": [
                {"path": "prompt/prompts/prompt.md", "sha256": _sha(data), "bytes": len(data)}
            ],
        }
        contents[f"{git_path}/prompt/prompts/prompt.md"] = data
    digest = bundle_sha256(stage, prompt_ref, None)
    contents[f"{git_path}/manifest.json"] = json.dumps({"bundleSha256": digest}).encode()
    fetcher.contents.update(contents)
    return {
        "stage": stage,
        "contentVersion": version,
        "gitBranch": "master",
        "gitPath": git_path,
        "gitCommit": f"commit-{stage}-{version}",
        "bundleSha256": digest,
        "prompt": prompt_ref,
        "createdAt": "2026-10-21 10:00:00",
        "createdBy": "tester",
    }


def _agent(
    *,
    doc_id: str = "tmp-1",
    name: str = "example_benchmark",
    customization: dict[str, Any] | None = None,
    **overrides: Any,
) -> AgentDocument:
    body: dict[str, Any] = {
        "id": doc_id,
        "name": name,
        "owner": "eval-team",
        "registeredAt": "2026-10-21 10:00:00",
        "agentScope": "amap-agent",
    }
    if customization is not None:
        body["customization"] = customization
        body["business"] = "biz"
        body["variantKey"] = "v1"
        body.setdefault("agentScope", "amap-agent")
    body.update(overrides)
    return AgentDocument.model_validate(body)


@pytest.fixture()
def _customization_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fetcher = _FakeFetcher()
    config = CustomizationConfig(enabled=True, cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    set_bundle_store(BundleStore(config, fetcher))
    yield fetcher
    set_bundle_store(None)


async def test_save_assigns_version_and_deterministic_id(_customization_env: _FakeFetcher) -> None:
    ref = _stage_ref(_customization_env, "judge")
    result = await save_agent(_agent(customization={"judge": ref}))
    assert result["id"] == "example_benchmark--amap_agent--biz--v1--v1"
    assert result["version"] == 1
    assert result["agentScope"] == "amap-agent"


async def test_save_second_content_gets_next_version(_customization_env: _FakeFetcher) -> None:
    first = _stage_ref(_customization_env, "judge", prompt_text="第一版")
    await save_agent(_agent(customization={"judge": first}))
    second = _stage_ref(_customization_env, "judge", prompt_text="第二版", version=2)
    result = await save_agent(_agent(doc_id="tmp-2", customization={"judge": second}))
    assert result["id"] == "example_benchmark--amap_agent--biz--v1--v2"
    assert result["version"] == 2


async def test_save_same_content_is_idempotent(_customization_env: _FakeFetcher) -> None:
    ref = _stage_ref(_customization_env, "judge")
    await save_agent(_agent(customization={"judge": ref}))
    again = await save_agent(_agent(doc_id="tmp-9", customization={"judge": ref}))
    assert again["id"] == "example_benchmark--amap_agent--biz--v1--v1"
    assert again["version"] == 1


async def test_existing_id_content_change_conflicts(_customization_env: _FakeFetcher) -> None:
    ref = _stage_ref(_customization_env, "judge")
    saved = await save_agent(_agent(customization={"judge": ref}))
    other = _stage_ref(_customization_env, "judge", prompt_text="改动内容", version=7)
    with pytest.raises(ConsoleError) as exc:
        await save_agent(_agent(doc_id=saved["id"], customization={"judge": other}))
    assert exc.value.status_code == 409
    assert "customization_immutable" in exc.value.detail


async def test_disabled_blocks_customized_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetcher = _FakeFetcher()
    config = CustomizationConfig(enabled=False, cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    set_bundle_store(BundleStore(config, fetcher))
    try:
        ref = _stage_ref(fetcher, "judge")
        with pytest.raises(ConsoleError) as exc:
            await save_agent(_agent(customization={"judge": ref}))
        assert exc.value.status_code == 503
    finally:
        set_bundle_store(None)


async def test_policy_deny_for_unknown_agent(_customization_env: _FakeFetcher) -> None:
    ref = _stage_ref(_customization_env, "judge")
    # search_eval is on the amap allowlist, but prompt/skill customization is not open
    with pytest.raises(ConsoleError) as exc:
        await save_agent(_agent(name="search_eval", customization={"judge": ref}))
    assert exc.value.status_code == 403
    assert "customization_not_allowed" in exc.value.detail


async def test_attribution_toggle_allowed_by_default(
    _customization_env: _FakeFetcher,
) -> None:
    # Attribution is optional: every agent on the allowlist can turn it off (a policy exemption is no longer required)
    result = await save_agent(_agent(name="search_eval", attributionEnabled=False))
    assert result["attributionEnabled"] is False
    # The switch can be saved on its own, with no content customization
    result = await save_agent(_agent(name="example_benchmark", attributionEnabled=False))
    assert result["attributionEnabled"] is False
    # An agent not on the allowlist cannot be registered under that evaluation target
    with pytest.raises(ConsoleError) as exc:
        await save_agent(_agent(name="custom_business_agent", attributionEnabled=False))
    assert exc.value.status_code == 403
    assert "agent_not_allowed_in_scope" in exc.value.detail


async def test_snapshot_frozen_against_later_version_and_toggle(
    _customization_env: _FakeFetcher,
) -> None:
    """Design §12.5 C1/C3: the snapshot of a run in progress is not affected by a later revision or a flipped switch."""
    first = _stage_ref(_customization_env, "judge", prompt_text="第一版口径")
    saved = await save_agent(_agent(customization={"judge": first}))
    doc_v1 = next(item for item in await list_agent_documents() if item.id == saved["id"])
    snapshot_v1 = registration_snapshot(doc_v1)
    # A later revision (v2) plus flipping the attribution switch must not write back the frozen v1 snapshot
    second = _stage_ref(_customization_env, "judge", prompt_text="第二版口径", version=2)
    await save_agent(_agent(doc_id="tmp-2", customization={"judge": second}))
    await save_agent(_agent(doc_id=saved["id"], attributionEnabled=False))
    assert snapshot_v1["customization"]["judge"]["git_commit"] == "commit-judge-1"
    assert snapshot_v1["attribution_enabled"] is True
    assert snapshot_v1["version"] == 1


async def test_toggle_only_update_keeps_customization(_customization_env: _FakeFetcher) -> None:
    ref = _stage_ref(_customization_env, "judge")
    saved = await save_agent(_agent(customization={"judge": ref}))
    updated = await save_agent(_agent(doc_id=saved["id"], attributionEnabled=False))
    assert updated["attributionEnabled"] is False
    assert updated["customization"]["judge"]["gitCommit"] == "commit-judge-1"
    assert updated["version"] == 1


async def test_resolve_explicit_missing_and_offline(_customization_env: _FakeFetcher) -> None:
    with pytest.raises(ConsoleError) as exc:
        await resolve_agent_registration("example_benchmark", "nope--v1")
    assert exc.value.status_code == 404
    ref = _stage_ref(_customization_env, "judge")
    saved = await save_agent(_agent(customization={"judge": ref}, status="offline"))
    with pytest.raises(ConsoleError) as exc:
        await resolve_agent_registration("example_benchmark", saved["id"])
    assert exc.value.status_code == 409


async def test_resolve_explicit_name_mismatch(_customization_env: _FakeFetcher) -> None:
    ref = _stage_ref(_customization_env, "judge")
    saved = await save_agent(_agent(customization={"judge": ref}))
    with pytest.raises(ConsoleError) as exc:
        await resolve_agent_registration("other_agent", saved["id"])
    assert exc.value.status_code == 400


async def test_resolve_default_and_ambiguous(_customization_env: _FakeFetcher) -> None:
    ref1 = _stage_ref(_customization_env, "judge", prompt_text="甲")
    await save_agent(_agent(customization={"judge": ref1}))
    ref2 = _stage_ref(_customization_env, "judge", prompt_text="乙", version=2)
    await save_agent(_agent(doc_id="tmp-2", customization={"judge": ref2}))
    # No scope: the customization is not a candidate → None (not ambiguous)
    assert await resolve_agent_registration("example_benchmark") is None
    with pytest.raises(ConsoleError) as exc:
        await resolve_agent_registration("example_benchmark", agent_scope="amap-agent")
    assert exc.value.status_code == 400
    assert "ambiguous_agent_registration" in exc.value.detail
    ref3 = _stage_ref(_customization_env, "judge", prompt_text="丙", version=3)
    default_saved = await save_agent(
        _agent(doc_id="tmp-3", customization={"judge": ref3}, isDefault=True)
    )
    resolved = await resolve_agent_registration("example_benchmark", agent_scope="amap-agent")
    assert resolved is not None
    assert resolved.id == default_saved["id"]


async def test_resolve_plain_registration_reuses_existing(
    _customization_env: _FakeFetcher,
) -> None:
    """A same-name registration with no customization reuses the existing document and does not create a second plain copy."""
    first = await save_agent(_agent(doc_id="legacy-1"))
    second = await save_agent(_agent(doc_id="legacy-2"))
    assert second["id"] == first["id"] == "legacy-1"
    resolved = await resolve_agent_registration("example_benchmark")
    assert resolved is not None
    assert resolved.id == "legacy-1"


async def test_resolve_none_when_no_registration() -> None:
    assert await resolve_agent_registration("nothing_here") is None


async def test_sequence_versions_are_per_stage() -> None:
    judge_key = CustomizationKey("a", "b", "v", "judge")
    attribution_key = CustomizationKey("a", "b", "v", "attribution")
    assert await next_customization_version(judge_key) == 1
    assert await next_customization_version(judge_key) == 2
    assert await next_customization_version(attribution_key) == 1
    payload = await peek_customization_sequence(judge_key)
    assert payload["last_id"] == 2


class _FakeGitStore:
    def __init__(self) -> None:
        self.commits: list[dict[str, Any]] = []

    async def commit_bundle(self, *, git_path: str, files: dict[str, bytes], message: str) -> str:
        self.commits.append({"git_path": git_path, "files": files, "message": message})
        return f"commit-{len(self.commits)}"


async def test_upload_allocates_version_and_reuses_same_content() -> None:
    git = _FakeGitStore()
    config = CustomizationConfig(enabled=True)
    request = StageUploadRequest(
        agent_name="example_benchmark",
        business="biz",
        variant_key="v1",
        stage="judge",
        prompt_texts={"prompt": "模板内容"},
        created_by="tester",
    )
    ref = await upload_stage_customization(request, config=config, git_store=git)  # type: ignore[arg-type]
    assert ref["content_version"] == 1
    assert ref["git_commit"] == "commit-1"
    assert ref["git_path"].endswith("/judge/v1")
    again = await upload_stage_customization(request, config=config, git_store=git)  # type: ignore[arg-type]
    assert again["content_version"] == 1
    assert len(git.commits) == 1


async def test_upload_disabled_is_503() -> None:
    config = CustomizationConfig(enabled=False)
    with pytest.raises(Exception) as exc:
        await upload_stage_customization(
            StageUploadRequest(
                agent_name="example_benchmark",
                business="biz",
                variant_key="v1",
                stage="judge",
                prompt_texts={"prompt": "x"},
            ),
            config=config,
            git_store=_FakeGitStore(),  # type: ignore[arg-type]
        )
    assert getattr(exc.value, "status_code", None) == 503


def test_strip_keeps_customization(_customization_env: _FakeFetcher) -> None:
    ref = _stage_ref(_customization_env, "judge")
    document = _agent(customization={"judge": ref})
    stripped = strip_agent_execution_fields(document)
    assert stripped.customization is not None
    assert stripped.customization.judge is not None
