"""Customization policy: the framework registers it in one place / everything off by default / view derivation (existing agent declarations are not changed)."""

import pytest
from benchmark.runs.common.agent_capabilities import (
    CapabilityBindingView,
    customization_policy_for_name,
    customization_policy_view,
)
from benchmark.customization.errors import CustomizationError
from benchmark.customization.manifest import (
    BundleInput,
    BundleLimits,
    prepare_bundle,
)
from benchmark.reporting.framework.base import PromptSlot, StageCustomizationPolicy


def test_unknown_agent_defaults_to_deny() -> None:
    policy = customization_policy_for_name("some_custom_business_agent")
    assert not policy.judge.allow_prompt
    assert not policy.judge.allow_skill
    assert not policy.attribution.allow_prompt
    assert not policy.attribution.allow_skill
    # prompt/skill are off by default, but the attribution switch is on by default (attribution is an optional flow)
    assert policy.allow_attribution_toggle


def test_only_example_allows_prompt_skill() -> None:
    """prompt/skill customization is open only for example_benchmark (judge + attribution)."""
    policy = customization_policy_for_name("example_benchmark")
    assert [slot.key for slot in policy.judge.prompt_slots] == ["prompt"]
    assert policy.judge.allow_skill
    assert [slot.key for slot in policy.attribution.prompt_slots] == ["prompt"]
    assert policy.attribution.allow_skill
    assert policy.allow_attribution_toggle


_EXAMPLE_PROMPT_VARS = {
    "*",
    "answer",
    "axes",
    "checkpoints",
    "context_info",
    "env_city",
    "env_location",
    "env_time",
    "expected_output",
    "history",
    "input",
    "output",
    "query",
    "response_text",
    "score_max",
    "score_min",
    "trace_evidence",
    "used_tool",
}


def test_example_judge_and_attribution_slots_declare_common_variables() -> None:
    """The judge / attribution slots share the common variable set, so an upload of ${expected_output} and similar is not rejected."""
    policy = customization_policy_for_name("example_benchmark")
    assert set(policy.judge.prompt_slots[0].variables) == _EXAMPLE_PROMPT_VARS
    assert set(policy.attribution.prompt_slots[0].variables) == _EXAMPLE_PROMPT_VARS


def test_example_judge_template_with_runtime_variables_passes_upload_check() -> None:
    """Regression: common fields + extension macros (*) all pass example judge upload validation."""
    policy = customization_policy_for_name("example_benchmark")
    template = (
        "问题：${query}\n回答：${answer}\n期望：${expected_output}\n"
        "input=${input} output=${output}\n工具：${used_tool}\n证据：${trace_evidence}\n"
        "分项：${axes}\n分值区间：${score_min}-${score_max}\n"
        "城市：${env_city} 时间：${env_time} 位置：${env_location}\n"
        "history=${history}\ncontext_info=${context_info}\ncheckpoints=${checkpoints}"
    )
    limits = BundleLimits(max_prompt_bytes=65536, max_skill_bytes=262144, max_files=32)
    bundle = prepare_bundle(
        BundleInput(
            stage="judge",
            stage_policy=policy.judge,
            prompt_texts={"prompt": template},
            skill_files={},
            limits=limits,
        )
    )
    assert bundle.prompt_ref is not None

    # ``*`` allows input extension macros (such as scene_id); the render side still replaces only fields that exist in the bag.
    extended = prepare_bundle(
        BundleInput(
            stage="judge",
            stage_policy=policy.judge,
            prompt_texts={"prompt": "场景 ${scene_id} 自定义 ${biz_tag}"},
            skill_files={},
            limits=limits,
        )
    )
    assert extended.prompt_ref is not None


def test_example_attribution_template_with_common_variables_passes_upload_check() -> None:
    """A customized attribution prompt likewise allows common variables and extension macros."""
    policy = customization_policy_for_name("example_benchmark")
    limits = BundleLimits(max_prompt_bytes=65536, max_skill_bytes=262144, max_files=32)
    bundle = prepare_bundle(
        BundleInput(
            stage="attribution",
            stage_policy=policy.attribution,
            prompt_texts={
                "prompt": "归因 ${query} / ${answer} / ${expected_output} / ${input} / ${output}"
            },
            skill_files={},
            limits=limits,
        )
    )
    assert bundle.prompt_ref is not None

    extended = prepare_bundle(
        BundleInput(
            stage="attribution",
            stage_policy=policy.attribution,
            prompt_texts={"prompt": "归因扩展 ${scene_id}"},
            skill_files={},
            limits=limits,
        )
    )
    assert extended.prompt_ref is not None


def test_wildcard_absent_still_rejects_unknown_template_variable() -> None:
    """A slot without ``*`` still rejects an undeclared variable (the framework default is unchanged)."""
    stage = StageCustomizationPolicy(
        allow_prompt=True,
        prompt_slots=(PromptSlot(key="prompt", variables=("query", "answer")),),
    )
    limits = BundleLimits(max_prompt_bytes=65536, max_skill_bytes=262144, max_files=32)
    with pytest.raises(CustomizationError) as exc:
        prepare_bundle(
            BundleInput(
                stage="judge",
                stage_policy=stage,
                prompt_texts={"prompt": "使用 ${secret}"},
                skill_files={},
                limits=limits,
            )
        )
    assert exc.value.code == "unknown_template_variable"


def test_amap_and_search_deny_prompt_skill() -> None:
    amap = customization_policy_for_name("amap_agent_benchmark")
    assert not amap.judge.allow_prompt
    assert not amap.judge.allow_skill
    assert not amap.attribution.allow_prompt
    assert not amap.attribution.allow_skill
    # Keep only the ability to turn attribution off
    assert amap.allow_attribution_toggle

    search = customization_policy_for_name("search_eval")
    assert not search.judge.allow_prompt
    assert not search.judge.allow_skill
    assert not search.attribution.allow_prompt
    # The attribution switch is on by default (search has no attribution capability, so the view hides it with supported=False)
    assert search.allow_attribution_toggle


def test_ieval_toggle_only() -> None:
    policy = customization_policy_for_name("ieval_external")
    assert not policy.judge.allow_prompt
    assert not policy.judge.allow_skill
    assert policy.allow_attribution_toggle


def test_view_derives_attribution_supported_from_capabilities() -> None:
    policy = customization_policy_for_name("example_benchmark")
    without = customization_policy_view(policy, capabilities=())
    assert without.attribution_toggle.allowed
    assert not without.attribution_toggle.supported
    binding = CapabilityBindingView(
        pipeline="batch", kind="attribution", execution="local_sync", provider="local"
    )
    with_attr = customization_policy_view(policy, capabilities=[binding])
    assert with_attr.attribution_toggle.supported
    assert with_attr.judge.prompt.slots == ["prompt"]
    none_binding = CapabilityBindingView(
        pipeline="batch", kind="attribution", execution="none", provider="none"
    )
    none_attr = customization_policy_view(policy, capabilities=[none_binding])
    assert not none_attr.attribution_toggle.supported
