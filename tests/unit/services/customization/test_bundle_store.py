"""Bundle materialization: fetch by commit + sha256 recheck + an explicit failure when a sync bundle is not materialized."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from benchmark.domain.customization_config import CustomizationConfig
from benchmark.customization.bundle_store import BundleStore
from benchmark.customization.errors import CustomizationError
from benchmark.customization.manifest import bundle_sha256


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ref_and_contents(
    tmp_path: Path, *, prompt_text: str = "业务模板", skill_text: str = "前置规则"
) -> tuple[dict[str, Any], dict[str, bytes]]:
    prompt_data = prompt_text.encode()
    skill_data = skill_text.encode()
    prompt_ref = {
        "slots": ["prompt"],
        "files": [
            {
                "path": "prompt/prompts/prompt.md",
                "sha256": _sha(prompt_data),
                "bytes": len(prompt_data),
            }
        ],
    }
    skill_ref = {
        "entry": "SKILL.md",
        "files": [{"path": "skill/SKILL.md", "sha256": _sha(skill_data), "bytes": len(skill_data)}],
    }
    digest = bundle_sha256("judge", prompt_ref, skill_ref)
    manifest = {
        "schemaVersion": 1,
        "stage": "judge",
        "bundleSha256": digest,
    }
    git_path = "benchmark-customizations/agent/biz/v1/judge/v1"
    ref = {
        "stage": "judge",
        "content_version": 1,
        "git_branch": "master",
        "git_path": git_path,
        "git_commit": "abc123",
        "bundle_sha256": digest,
        "prompt": prompt_ref,
        "skill": skill_ref,
    }
    contents = {
        f"{git_path}/manifest.json": json.dumps(manifest).encode(),
        f"{git_path}/prompt/prompts/prompt.md": prompt_data,
        f"{git_path}/skill/SKILL.md": skill_data,
    }
    return ref, contents


class _FakeFetcher:
    def __init__(self, contents: dict[str, bytes]) -> None:
        self._contents = contents
        self.calls = 0

    async def read_files(self, *, git_commit: str, paths: list[str]) -> dict[str, bytes]:
        self.calls += 1
        return {path: self._contents[path] for path in paths if path in self._contents}


def _store(tmp_path: Path, fetcher: _FakeFetcher) -> BundleStore:
    config = CustomizationConfig(enabled=True, cache_dir=str(tmp_path / "cache"))
    return BundleStore(config, fetcher)


async def test_ensure_materializes_and_caches(tmp_path: Path) -> None:
    ref, contents = _ref_and_contents(tmp_path)
    fetcher = _FakeFetcher(contents)
    store = _store(tmp_path, fetcher)
    bundle = await store.ensure(ref)
    assert bundle.prompt == {"prompt": "业务模板"}
    assert bundle.skill_text == "前置规则"
    again = await store.ensure(ref)
    assert again is bundle
    assert fetcher.calls == 1


async def test_ensure_checksum_mismatch_is_409(tmp_path: Path) -> None:
    ref, contents = _ref_and_contents(tmp_path)
    key = f"{ref['git_path']}/prompt/prompts/prompt.md"
    contents[key] = "被篡改".encode()
    store = _store(tmp_path, _FakeFetcher(contents))
    with pytest.raises(CustomizationError) as exc:
        await store.ensure(ref)
    assert exc.value.status_code == 409
    assert exc.value.code == "checksum_mismatch"


async def test_ensure_missing_file_is_503(tmp_path: Path) -> None:
    ref, contents = _ref_and_contents(tmp_path)
    del contents[f"{ref['git_path']}/skill/SKILL.md"]
    store = _store(tmp_path, _FakeFetcher(contents))
    with pytest.raises(CustomizationError) as exc:
        await store.ensure(ref)
    assert exc.value.status_code == 503


def test_sync_load_unmaterialized_raises(tmp_path: Path) -> None:
    ref, _ = _ref_and_contents(tmp_path)
    store = _store(tmp_path, _FakeFetcher({}))
    with pytest.raises(CustomizationError) as exc:
        store.load_materialized_sync(ref)
    assert exc.value.status_code == 503


async def test_disk_cache_reused_across_store_instances(tmp_path: Path) -> None:
    ref, contents = _ref_and_contents(tmp_path)
    first = _store(tmp_path, _FakeFetcher(contents))
    await first.ensure(ref)
    failing = _FakeFetcher({})
    second = _store(tmp_path, failing)
    bundle = second.load_materialized_sync(ref)
    assert bundle.prompt["prompt"] == "业务模板"
    assert failing.calls == 0
