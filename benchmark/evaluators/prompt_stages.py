"""Judge / attribution prompt stage isolation.

Convention:
- judge only consumes ``judge_prompt_text`` / ``prompt_text`` (RemoteConfig score override) /
  ``prompt_bundle_ref`` (stage=judge)
- attribution only consumes ``attribution_prompt_text`` /
  ``attribution_prompt_bundle_ref`` (or a bundle whose stage=attribution)
- when a stage has no override, that stage falls back to the platform default template; do not use the other stage's
  prompt as the default.
"""

from collections.abc import Mapping
from typing import Any, Literal, cast

from benchmark.evaluators.base import EvaluatorRequest, PromptBundleRef, SkillBundleRef

PromptStage = Literal["judge", "attribution"]
BundleRef = PromptBundleRef | SkillBundleRef | Mapping[str, Any]
EvaluatorSource = EvaluatorRequest | Mapping[str, Any]

#: Judge-only keys stripped from the evaluator_request dump before writing attribution metadata.
_JUDGE_ONLY_REQUEST_KEYS: frozenset[str] = frozenset(
    {
        "prompt_text",
        "judge_prompt_text",
        "prompt_bundle_ref",
        "skill_bundle_ref",
    }
)


def ref_stage(ref: BundleRef | None) -> str | None:
    """Read the stage of a bundle ref (object or dict); return None when it cannot be recognized."""
    if ref is None:
        return None
    raw = getattr(ref, "stage", None) if not isinstance(ref, Mapping) else ref.get("stage")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def is_stage(ref: BundleRef | None, stage: PromptStage) -> bool:
    """Whether the ref clearly belongs to the given stage."""
    return ref_stage(ref) == stage


def dump_ref(ref: BundleRef | None) -> dict[str, Any] | None:
    """bundle ref → JSON shape that can go into metadata."""
    if ref is None:
        return None
    if isinstance(ref, Mapping):
        return dict(ref)
    return ref.model_dump(mode="json")


def attribution_bundle_refs(
    source: EvaluatorSource,
) -> tuple[BundleRef | None, BundleRef | None]:
    """Take the two refs that belong to the **attribution stage only** from an EvaluatorRequest / mapping.

    Prefer ``attribution_prompt_bundle_ref`` / ``attribution_skill_bundle_ref``;
    if those dedicated fields are absent, also accept ``prompt_bundle_ref`` / ``skill_bundle_ref`` when
    ``stage==attribution`` (the ieval local attribution path writes the attribution ref straight into
    the generic fields). Generic fields for the judge stage are always ignored.
    """
    prompt_ref = _get(source, "attribution_prompt_bundle_ref")
    skill_ref = _get(source, "attribution_skill_bundle_ref")
    if prompt_ref is None:
        shared = _get(source, "prompt_bundle_ref")
        if is_stage(shared, "attribution"):
            prompt_ref = shared
    if skill_ref is None:
        shared = _get(source, "skill_bundle_ref")
        if is_stage(shared, "attribution"):
            skill_ref = shared
    if prompt_ref is not None and not is_stage(prompt_ref, "attribution"):
        prompt_ref = None
    if skill_ref is not None and not is_stage(skill_ref, "attribution"):
        skill_ref = None
    return prompt_ref, skill_ref


def judge_bundle_refs(
    source: EvaluatorSource,
) -> tuple[BundleRef | None, BundleRef | None]:
    """Take the two refs that belong to the **judge stage only** from an EvaluatorRequest / mapping."""
    prompt_ref = _get(source, "prompt_bundle_ref")
    skill_ref = _get(source, "skill_bundle_ref")
    if prompt_ref is not None and ref_stage(prompt_ref) not in (None, "judge"):
        # An explicit attribution stage is not for the judge; a missing stage is treated as a legacy judge ref.
        prompt_ref = None
    if skill_ref is not None and ref_stage(skill_ref) not in (None, "judge"):
        skill_ref = None
    return prompt_ref, skill_ref


def sanitize_evaluator_request_for_attribution(
    dump: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Drop judge-only prompt fields so the attribution side does not misread the scoring template / judge bundle."""
    if not isinstance(dump, Mapping):
        return {}
    cleaned = {key: value for key, value in dump.items() if key not in _JUDGE_ONLY_REQUEST_KEYS}
    # Keep the dedicated attribution bundle fields; also drop a judge stage mistakenly written into attribution_*.
    for key in ("attribution_prompt_bundle_ref", "attribution_skill_bundle_ref"):
        ref = cleaned.get(key)
        if ref is not None and not is_stage(ref, "attribution"):
            cleaned.pop(key, None)
    return cleaned


def _get(source: EvaluatorSource, key: str) -> BundleRef | None:
    value = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
    return cast(BundleRef | None, value)


__all__ = [
    "PromptStage",
    "attribution_bundle_refs",
    "dump_ref",
    "is_stage",
    "judge_bundle_refs",
    "ref_stage",
    "sanitize_evaluator_request_for_attribution",
]
