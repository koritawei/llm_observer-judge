"""Helper that isolates judge / attribution prompt stages."""

from benchmark.evaluators.base import EvaluatorRequest, PromptBundleRef, SkillBundleRef
from benchmark.evaluators.prompt_stages import (
    attribution_bundle_refs,
    dump_ref,
    judge_bundle_refs,
    sanitize_evaluator_request_for_attribution,
)
from benchmark.attribution.service.service import _build_attribution_request

_ATTR_CONTENT_VERSION = 2


def _judge_ref(**kwargs: object) -> PromptBundleRef:
    return PromptBundleRef(
        stage="judge",
        git_path="p/judge/v1",
        git_commit="abc",
        bundle_sha256="sha1",
        content_version=1,
        **kwargs,  # type: ignore[arg-type]
    )


def _attr_ref(**kwargs: object) -> PromptBundleRef:
    return PromptBundleRef(
        stage="attribution",
        git_path="p/attr/v2",
        git_commit="def",
        bundle_sha256="sha2",
        content_version=_ATTR_CONTENT_VERSION,
        **kwargs,  # type: ignore[arg-type]
    )


def test_judge_bundle_refs_accept_legacy_missing_stage() -> None:
    """A historical ref with no stage is still treated as usable by the judge."""
    legacy = {"git_path": "p/legacy", "git_commit": "abc", "bundle_sha256": "sha"}
    prompt_ref, skill_ref = judge_bundle_refs(
        {"prompt_bundle_ref": legacy, "skill_bundle_ref": {"stage": "judge"}},
    )
    assert prompt_ref == legacy
    assert skill_ref == {"stage": "judge"}


def test_dump_ref_supports_models_and_mappings() -> None:
    model = _judge_ref()
    mapping = {"stage": "attribution", "git_path": "p/attr"}
    dumped = dump_ref(model)
    assert dumped is not None and dumped["stage"] == "judge"
    assert dump_ref(mapping) == mapping
    assert dump_ref(None) is None


def test_judge_bundle_refs_ignore_attribution_stage() -> None:
    req = EvaluatorRequest(
        prompt_bundle_ref=_attr_ref(),
        skill_bundle_ref=SkillBundleRef(
            stage="attribution",
            git_path="p/attr/v2",
            git_commit="def",
            bundle_sha256="sha2",
        ),
    )
    assert judge_bundle_refs(req) == (None, None)


def test_attribution_bundle_refs_prefer_dedicated_fields() -> None:
    req = EvaluatorRequest(
        prompt_bundle_ref=_judge_ref(),
        attribution_prompt_bundle_ref=_attr_ref(),
        attribution_skill_bundle_ref=SkillBundleRef(
            stage="attribution",
            git_path="p/attr/v2",
            git_commit="def",
            bundle_sha256="sha2",
        ),
    )
    prompt_ref, skill_ref = attribution_bundle_refs(req)
    assert prompt_ref is not None and prompt_ref.stage == "attribution"
    assert skill_ref is not None and skill_ref.stage == "attribution"


def test_attribution_bundle_refs_accept_shared_field_when_stage_attribution() -> None:
    """The ieval path writes the attribution ref into the shared prompt_bundle_ref."""
    req = EvaluatorRequest(prompt_bundle_ref=_attr_ref())
    prompt_ref, _ = attribution_bundle_refs(req)
    assert prompt_ref is not None and prompt_ref.content_version == _ATTR_CONTENT_VERSION


def test_sanitize_strips_judge_prompt_fields() -> None:
    cleaned = sanitize_evaluator_request_for_attribution(
        {
            "prompt_text": "JUDGE_REMOTE_CONFIG",
            "judge_prompt_text": "JUDGE_CUSTOM",
            "attribution_prompt_text": "ATTR_CUSTOM",
            "prompt_bundle_ref": {"stage": "judge", "git_path": "x"},
            "attribution_prompt_bundle_ref": {
                "stage": "attribution",
                "git_path": "y",
                "git_commit": "c",
                "bundle_sha256": "s",
            },
        }
    )
    assert "prompt_text" not in cleaned
    assert "judge_prompt_text" not in cleaned
    assert "prompt_bundle_ref" not in cleaned
    assert cleaned["attribution_prompt_text"] == "ATTR_CUSTOM"
    assert cleaned["attribution_prompt_bundle_ref"]["stage"] == "attribution"


def test_build_attribution_request_does_not_pass_judge_prompt() -> None:
    req = EvaluatorRequest(
        trace_id="t1",
        project_id="p1",
        agent_name="amap-agent",
        prompt_text="JUDGE_REMOTE_CONFIG",
        judge_prompt_text="JUDGE_CUSTOM",
        attribution_prompt_text="ATTR_CUSTOM",
        prompt_bundle_ref=_judge_ref(),
        attribution_prompt_bundle_ref=_attr_ref(),
    )
    agent_req = _build_attribution_request(req, {"score": 1.0, "reason": "low"})
    er = agent_req.metadata["evaluator_request"]
    assert isinstance(er, dict)
    assert "prompt_text" not in er
    assert "judge_prompt_text" not in er
    assert "prompt_bundle_ref" not in er
    assert er.get("attribution_prompt_text") == "ATTR_CUSTOM"
    # The top level holds only attribution-stage refs, for the attribution agent to consume
    top = agent_req.metadata.get("prompt_bundle_ref")
    assert isinstance(top, dict)
    assert top.get("stage") == "attribution"
    assert top.get("content_version") == _ATTR_CONTENT_VERSION
