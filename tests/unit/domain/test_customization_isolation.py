"""Isolation of several instances of the same code (design §12): internal-name assignment/inheritance/quota/snapshot/realtime fields.

The server assigns the internal name by instance group (name+business+variant). Versions in the same group
share it, different groups must differ, and a client-supplied value is ignored. The external response does not return the internal name.

A same-name registration with no customization is plain-reused (see
``test_resolve_plain_registration_reuses_existing``). Multi-version and quota cases
must take the customization path (``example_benchmark``).
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from benchmark.domain import customization_config as customization_config_module
from benchmark.domain.agent_scope_ownership import OwnershipPolicy, set_ownership_policy
from benchmark.domain.benchmark_console import (
    AgentDocument,
    ConsoleError,
    list_agent_documents,
    registration_snapshot,
    save_agent,
)
from benchmark.domain.customization_config import CustomizationConfig
from benchmark.domain.runtime_config_defaults import stamp_eval_entry
from benchmark.customization.bundle_store import BundleStore, set_bundle_store
from benchmark.customization.manifest import bundle_sha256

_INTERNAL_RE = re.compile(r"^agx_[a-z0-9_]+_\d{10}$")


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
    prompt_text: str = "业务模板 ${query}",
    version: int = 1,
    business: str = "biz",
    variant: str = "v1",
) -> dict[str, Any]:
    data = prompt_text.encode()
    prompt_ref = {
        "slots": ["prompt"],
        "files": [{"path": "prompt/prompts/prompt.md", "sha256": _sha(data), "bytes": len(data)}],
    }
    git_path = f"benchmark-customizations/example_benchmark/{business}/{variant}/{stage}/v{version}"
    digest = bundle_sha256(stage, prompt_ref, None)
    fetcher.contents[f"{git_path}/prompt/prompts/prompt.md"] = data
    fetcher.contents[f"{git_path}/manifest.json"] = json.dumps({"bundleSha256": digest}).encode()
    return {
        "stage": stage,
        "contentVersion": version,
        "gitBranch": "master",
        "gitPath": git_path,
        "gitCommit": f"commit-{stage}-{version}-{business}",
        "bundleSha256": digest,
        "prompt": prompt_ref,
        "createdAt": "2026-10-21 10:00:00",
        "createdBy": "tester",
    }


def _agent(
    doc_id: str,
    name: str,
    *,
    customization: dict[str, Any] | None = None,
    **overrides: object,
) -> AgentDocument:
    body: dict[str, object] = {
        "id": doc_id,
        "name": name,
        "owner": "eval-team",
        "registeredAt": "2026-10-22 10:00:00",
        "agentScope": "amap-agent",
    }
    if customization is not None:
        body["customization"] = customization
        body.setdefault("business", "biz")
        body.setdefault("variantKey", "v1")
        body.setdefault("agentScope", "amap-agent")
    body.update(overrides)
    return AgentDocument.model_validate(body)


async def _stored(doc_id: str) -> AgentDocument:
    for item in await list_agent_documents():
        if item.id == doc_id:
            return item
    raise AssertionError(f"document {doc_id} not stored")


@pytest.fixture()
def _customization_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeFetcher:
    fetcher = _FakeFetcher()
    config = CustomizationConfig(enabled=True, cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    set_bundle_store(BundleStore(config, fetcher))
    yield fetcher
    set_bundle_store(None)


async def test_internal_name_assigned_by_server_and_hidden() -> None:
    set_ownership_policy(
        OwnershipPolicy(
            scopes={"amap-agent": frozenset({"iso_hidden_benchmark"})},
            global_agents=frozenset(),
            name_aliases={},
        )
    )
    try:
        result = await save_agent(
            _agent(
                "iso-hidden-1",
                "iso_hidden_benchmark",
                business="iso-biz",
                variantKey="v1",
                internalAgentName="agx_iso_hidden_9999999999",
            )
        )
    finally:
        set_ownership_policy(None)
    # The external response does not return the internal name; a value sent by the client is ignored (the server assigns it again)
    assert "internalAgentName" not in result
    stored = await _stored("iso-hidden-1")
    assert stored.internal_agent_name is not None
    assert _INTERNAL_RE.match(stored.internal_agent_name)
    assert stored.internal_agent_name != "agx_iso_hidden_9999999999"
    assert stored.internal_agent_name.startswith("agx_iso_hidden_")


async def test_internal_name_shared_within_group_distinct_across_groups(
    _customization_env: _FakeFetcher,
) -> None:
    first_ref = _stage_ref(_customization_env, "judge", prompt_text="组1-v1", version=1)
    second_ref = _stage_ref(_customization_env, "judge", prompt_text="组1-v2", version=2)
    other_ref = _stage_ref(
        _customization_env,
        "judge",
        prompt_text="组2",
        version=1,
        business="g2",
    )
    first = await save_agent(
        _agent(
            "iso-group-1",
            "example_benchmark",
            customization={"judge": first_ref},
            business="g1",
            variantKey="v1",
        )
    )
    second = await save_agent(
        _agent(
            "iso-group-2",
            "example_benchmark",
            customization={"judge": second_ref},
            business="g1",
            variantKey="v1",
        )
    )
    other = await save_agent(
        _agent(
            "iso-group-3",
            "example_benchmark",
            customization={"judge": other_ref},
            business="g2",
            variantKey="v1",
        )
    )
    first_doc = await _stored(first["id"])
    second_doc = await _stored(second["id"])
    other_doc = await _stored(other["id"])
    assert first_doc.internal_agent_name == second_doc.internal_agent_name
    assert other_doc.internal_agent_name != first_doc.internal_agent_name
    snapshot = registration_snapshot(first_doc)
    assert snapshot["internal_agent_name"] == first_doc.internal_agent_name


async def test_instance_quota_rejects_new_group_beyond_component_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fetcher = _FakeFetcher()
    config = CustomizationConfig(
        enabled=True,
        cache_dir=str(tmp_path / "cache"),
        max_instances_per_component=1,
    )
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    set_bundle_store(BundleStore(config, fetcher))
    try:
        ref1 = _stage_ref(fetcher, "judge", prompt_text="q1", business="q1")
        await save_agent(
            _agent(
                "iso-quota-1",
                "example_benchmark",
                customization={"judge": ref1},
                business="q1",
                variantKey="v1",
            )
        )
        ref2 = _stage_ref(fetcher, "judge", prompt_text="q2", business="q2")
        with pytest.raises(ConsoleError) as exc:
            await save_agent(
                _agent(
                    "iso-quota-2",
                    "example_benchmark",
                    customization={"judge": ref2},
                    business="q2",
                    variantKey="v2",
                )
            )
        assert exc.value.status_code == 403
        assert "instance_quota_exceeded" in exc.value.detail
        # A new content version in the same group is not a new instance, and is not hit by the component quota by mistake
        ref3 = _stage_ref(fetcher, "judge", prompt_text="q1-v2", version=2, business="q1")
        await save_agent(
            _agent(
                "iso-quota-3",
                "example_benchmark",
                customization={"judge": ref3},
                business="q1",
                variantKey="v1",
            )
        )
    finally:
        set_bundle_store(None)


async def test_version_quota_rejects_beyond_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fetcher = _FakeFetcher()
    config = CustomizationConfig(
        enabled=True,
        cache_dir=str(tmp_path / "cache"),
        max_versions_per_instance=1,
    )
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    set_bundle_store(BundleStore(config, fetcher))
    try:
        ref1 = _stage_ref(fetcher, "judge", prompt_text="ver1", version=1)
        first = await save_agent(
            _agent(
                "iso-ver-1",
                "example_benchmark",
                customization={"judge": ref1},
                business="vb",
                variantKey="v1",
            )
        )
        ref2 = _stage_ref(fetcher, "judge", prompt_text="ver2", version=2)
        with pytest.raises(ConsoleError) as exc:
            await save_agent(
                _agent(
                    "iso-ver-2",
                    "example_benchmark",
                    customization={"judge": ref2},
                    business="vb",
                    variantKey="v1",
                )
            )
        assert exc.value.status_code == 403
        assert "instance_quota_exceeded" in exc.value.detail
        # Updating an existing same id is not blocked by the version quota (existing is not None → skip)
        await save_agent(
            _agent(
                first["id"],
                "example_benchmark",
                customization={"judge": ref1},
                business="vb",
                variantKey="v1",
            )
        )
    finally:
        set_bundle_store(None)


def test_stamp_eval_entry_carries_internal_agent_name() -> None:
    stamped = stamp_eval_entry(
        {
            "evaluator_name": "example",
            "sampling": 0.5,
            "internal_agent_name": "agx_example_0000000001",
        }
    )
    assert stamped["internalAgentName"] == "agx_example_0000000001"
    # When it is not registered, do not pass it through (existing behavior stays)
    legacy = stamp_eval_entry({"evaluator_name": "example", "sampling": 0.5})
    assert "internalAgentName" not in legacy
