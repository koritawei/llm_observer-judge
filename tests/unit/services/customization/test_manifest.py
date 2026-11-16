"""Bundle assembly and validation: slot allowlist / variable scan / path and size gates."""

import pytest
from benchmark.records.models.runtime_config_record import CustomizationKey
from benchmark.customization.errors import CustomizationError
from benchmark.customization.manifest import (
    BundleInput,
    BundleLimits,
    bundle_sha256,
    git_path_for,
    prepare_bundle,
    prompt_slot_from_filename,
    stage_base_path,
)
from benchmark.reporting.framework.base import PromptSlot, StageCustomizationPolicy

_SINGLE = StageCustomizationPolicy(
    allow_prompt=True,
    allow_skill=True,
    prompt_slots=(PromptSlot(key="prompt", variables=("query", "output")),),
)
_DENY = StageCustomizationPolicy()


def _prepare(policy: StageCustomizationPolicy = _SINGLE, **kwargs: object):
    return prepare_bundle(
        BundleInput(
            stage="judge",
            stage_policy=policy,
            prompt_texts=kwargs.get("prompt_texts") or {},  # type: ignore[arg-type]
            skill_files=kwargs.get("skill_files") or {},  # type: ignore[arg-type]
            limits=BundleLimits(
                max_prompt_bytes=1024,
                max_skill_bytes=1024,
                max_files=8,
            ),
        )
    )


def test_prompt_only_bundle_ok() -> None:
    bundle = _prepare(prompt_texts={"prompt": "判断 ${query} 与 ${output}"})
    assert bundle.prompt_ref is not None
    assert bundle.prompt_ref["slots"] == ["prompt"]
    assert bundle.skill_ref is None
    assert bundle.files["prompt/prompts/prompt.md"].startswith("判断".encode())
    assert bundle.bundle_sha256 == bundle_sha256("judge", bundle.prompt_ref, None)


def test_unknown_template_variable_rejected() -> None:
    with pytest.raises(CustomizationError) as exc:
        _prepare(prompt_texts={"prompt": "使用 ${secret} 变量"})
    assert exc.value.status_code == 400
    assert exc.value.code == "unknown_template_variable"


def test_slot_not_in_whitelist_rejected() -> None:
    with pytest.raises(CustomizationError) as exc:
        _prepare(prompt_texts={"other": "文本"})
    assert exc.value.code == "slot_not_allowed"


def test_prompt_denied_by_policy_is_403() -> None:
    with pytest.raises(CustomizationError) as exc:
        _prepare(policy=_DENY, prompt_texts={"prompt": "文本"})
    assert exc.value.status_code == 403
    assert exc.value.code == "customization_not_allowed"


def test_empty_stage_rejected() -> None:
    with pytest.raises(CustomizationError) as exc:
        _prepare()
    assert exc.value.code == "empty_stage_bundle"


def test_skill_requires_entry() -> None:
    with pytest.raises(CustomizationError) as exc:
        _prepare(skill_files={"resources/a.md": b"x"})
    assert exc.value.code == "skill_entry_missing"


def test_skill_bundle_ok_with_resources() -> None:
    bundle = _prepare(
        skill_files={"SKILL.md": "# 规则".encode(), "resources/a.md": "补充".encode()}
    )
    assert bundle.skill_ref is not None
    assert bundle.skill_ref["entry"] == "SKILL.md"
    assert set(bundle.files) == {"skill/SKILL.md", "skill/resources/a.md"}


def test_skill_path_traversal_rejected() -> None:
    with pytest.raises(CustomizationError) as exc:
        _prepare(skill_files={"SKILL.md": b"x", "../evil.md": b"x"})
    assert exc.value.code == "invalid_path"


def test_skill_file_type_rejected() -> None:
    with pytest.raises(CustomizationError) as exc:
        _prepare(skill_files={"SKILL.md": b"x", "run.exe": b"x"})
    assert exc.value.code == "invalid_file_type"


def test_skill_non_utf8_rejected() -> None:
    with pytest.raises(CustomizationError) as exc:
        _prepare(skill_files={"SKILL.md": b"\xff\xfe"})
    assert exc.value.code == "invalid_encoding"


def test_prompt_slot_too_large_rejected() -> None:
    policy = StageCustomizationPolicy(
        allow_prompt=True,
        prompt_slots=(PromptSlot(key="prompt", max_bytes=4),),
    )
    with pytest.raises(CustomizationError) as exc:
        _prepare(policy=policy, prompt_texts={"prompt": "超过四个字节"})
    assert exc.value.code == "prompt_too_large"


def test_prompt_and_skill_combined() -> None:
    bundle = _prepare(
        prompt_texts={"prompt": "模板"},
        skill_files={"SKILL.md": "前置".encode()},
    )
    assert bundle.prompt_ref is not None
    assert bundle.skill_ref is not None


def test_git_path_for_slugifies_agent_and_validates_segments() -> None:
    key = CustomizationKey("Amap Agent", "biz", "v1", "judge")
    assert stage_base_path("benchmark-customizations", key) == (
        "benchmark-customizations/amap_agent/biz/v1/judge"
    )
    path = git_path_for("benchmark-customizations", key, 3)
    assert path == "benchmark-customizations/amap_agent/biz/v1/judge/v3"
    with pytest.raises(CustomizationError) as exc:
        git_path_for(
            "benchmark-customizations",
            CustomizationKey("a", "../x", "v1", "judge"),
            1,
        )
    assert exc.value.code == "invalid_path_segment"


def test_prompt_slot_from_filename() -> None:
    assert prompt_slot_from_filename("expression.md") == "expression"
    assert prompt_slot_from_filename("dir/core_completion.prompt") == "core_completion"
    with pytest.raises(CustomizationError):
        prompt_slot_from_filename("notes.txt")
