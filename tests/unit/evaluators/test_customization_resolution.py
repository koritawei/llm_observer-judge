"""Runtime customization resolution: skill prepended + slot override/fallback + zero change when there is no ref."""

from typing import Any

import pytest
from benchmark.evaluators.base import EvaluatorRequest, PromptBundleRef, SkillBundleRef
from benchmark.evaluators.customization_resolution import (
    bundle_refs_from_snapshot,
    prompt_provenance,
    resolve_stage_prompt_from_text,
    resolve_stage_prompt_text,
)
from benchmark.customization.bundle_store import (
    MaterializedStageBundle,
    set_bundle_store,
)
from benchmark.customization.errors import CustomizationError


class _FakeStore:
    """Minimal BundleStore stand-in: ensure returns the preset bundle directly, and a sync read returns the same value."""

    def __init__(self, bundle: MaterializedStageBundle | None) -> None:
        self._bundle = bundle

    async def ensure(self, ref: dict[str, Any]) -> MaterializedStageBundle:
        if self._bundle is None:
            raise CustomizationError(503, "customization_fetch_failed", "fake miss")
        return self._bundle

    def load_materialized_sync(self, ref: dict[str, Any]) -> MaterializedStageBundle:
        if self._bundle is None:
            raise CustomizationError(503, "customization_fetch_failed", "not materialized")
        return self._bundle


@pytest.fixture(autouse=True)
def _reset_store():
    yield
    set_bundle_store(None)  # type: ignore[arg-type]


def _request(**kwargs: Any) -> EvaluatorRequest:
    return EvaluatorRequest(input="q", output="a", **kwargs)


def _refs() -> tuple[PromptBundleRef, SkillBundleRef]:
    base = {"stage": "judge", "git_path": "p/v1", "git_commit": "abc", "bundle_sha256": "x"}
    return (
        PromptBundleRef(**base, slots=["prompt"], files=[]),
        SkillBundleRef(**base, entry="SKILL.md", files=[]),
    )


async def test_no_ref_returns_default_unchanged() -> None:
    set_bundle_store(_FakeStore(None))  # type: ignore[arg-type]
    assert await resolve_stage_prompt_from_text(_request(), "prompt", "默认模板") == "默认模板"
    assert resolve_stage_prompt_text(_request(), "prompt", "默认模板") == "默认模板"


async def test_skill_prefix_and_slot_override() -> None:
    prompt_ref, skill_ref = _refs()
    set_bundle_store(  # type: ignore[arg-type]
        _FakeStore(MaterializedStageBundle(prompt={"prompt": "业务模板"}, skill_text="前置规则"))
    )
    request = _request(prompt_bundle_ref=prompt_ref, skill_bundle_ref=skill_ref)
    assert await resolve_stage_prompt_from_text(request, "prompt", "默认") == "前置规则\n\n业务模板"


async def test_prompt_resolution_logs_preview_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ops requires INFO to print a truncated prompt preview; the metadata must still be complete."""
    import logging

    prompt_ref, skill_ref = _refs()
    set_bundle_store(  # type: ignore[arg-type]
        _FakeStore(MaterializedStageBundle(prompt={"prompt": "业务模板"}, skill_text="前置规则"))
    )
    request = _request(prompt_bundle_ref=prompt_ref, skill_bundle_ref=skill_ref)

    with caplog.at_level(logging.INFO, logger="benchmark.evaluators.customization_resolution"):
        await resolve_stage_prompt_from_text(request, "prompt", "默认")

    assert "customization.prompt_resolved" in caplog.text
    assert "chars=" in caplog.text
    assert "preview=" in caplog.text
    assert "业务模板" in caplog.text


async def test_uncovered_slot_falls_back_but_skill_still_prefixes() -> None:
    prompt_ref, skill_ref = _refs()
    set_bundle_store(  # type: ignore[arg-type]
        _FakeStore(MaterializedStageBundle(prompt={"other": "x"}, skill_text="前置规则"))
    )
    request = _request(prompt_bundle_ref=prompt_ref, skill_bundle_ref=skill_ref)
    assert await resolve_stage_prompt_from_text(request, "prompt", "默认") == "前置规则\n\n默认"


async def test_skill_only_ref_prefixes_default() -> None:
    _, skill_ref = _refs()
    set_bundle_store(_FakeStore(MaterializedStageBundle(skill_text="前置规则")))  # type: ignore[arg-type]
    request = _request(skill_bundle_ref=skill_ref)
    assert resolve_stage_prompt_text(request, "prompt", "默认") == "前置规则\n\n默认"


def test_sync_unmaterialized_raises_not_fallback() -> None:
    prompt_ref, _ = _refs()
    set_bundle_store(_FakeStore(None))  # type: ignore[arg-type]
    with pytest.raises(CustomizationError) as exc:
        resolve_stage_prompt_text(_request(prompt_bundle_ref=prompt_ref), "prompt", "默认")
    assert exc.value.status_code == 503


def test_bundle_refs_from_snapshot_shapes() -> None:
    snapshot = {
        "customization": {
            "judge": {
                "stage": "judge",
                "git_path": "p/v1",
                "git_commit": "abc",
                "bundle_sha256": "x",
                "content_version": 3,
                "prompt": {"slots": ["prompt"], "files": [{"path": "a"}]},
                "skill": {"entry": "SKILL.md", "files": []},
            }
        }
    }
    prompt_ref, skill_ref = bundle_refs_from_snapshot(snapshot, "judge")
    assert prompt_ref is not None and prompt_ref.slots == ["prompt"]
    assert prompt_ref.content_version == 3
    assert skill_ref is not None and skill_ref.entry == "SKILL.md"
    assert skill_ref.content_version == 3
    assert bundle_refs_from_snapshot(snapshot, "attribution") == (None, None)
    assert bundle_refs_from_snapshot(None, "judge") == (None, None)
    assert bundle_refs_from_snapshot({}, "judge") == (None, None)


def test_bundle_refs_derive_content_version_from_git_path() -> None:
    snapshot = {
        "customization": {
            "judge": {
                "stage": "judge",
                "git_path": "benchmark-customizations/example/biz/default/judge/v12",
                "git_commit": "deadbeef",
                "bundle_sha256": "abc",
                "prompt": {"slots": ["prompt"], "files": []},
            }
        }
    }
    prompt_ref, _ = bundle_refs_from_snapshot(snapshot, "judge")
    assert prompt_ref is not None
    assert prompt_ref.content_version == 12
    provenance = prompt_provenance(prompt_ref, None)
    assert provenance is not None
    assert provenance["content_version"] == 12
    assert provenance["has_prompt"] is True
    assert provenance["has_skill"] is False
