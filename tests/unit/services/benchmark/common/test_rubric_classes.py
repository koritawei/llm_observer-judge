"""Unit tests for the Rubric class-name registry: alias resolution, allowlist, component fallback, and the request contract."""

import pytest
from benchmark.runs.common import rubric_classes as rc_mod
from benchmark.runs.common.profile_resolver import (
    ProfileResolutionError,
    ProfileResolver,
)
from benchmark.runs.common.rubric_classes import (
    RubricClassResolutionError,
    resolve_rubric_class,
    rubric_alias_for_ref,
)
from benchmark.runs.schema import ExternalBenchmarkRequest
from benchmark.reporting.framework.base import BenchmarkComponentMetadata
from benchmark.reporting.framework.schema import AggregationConfig, Criterion
from benchmark.workers.external_runner import profile_from_external_request

_AMAP_REF = "business.runs.ieval.amap_chat.rubric:AmapChatRubricV2"
_SEARCH_REF = "business.runs.search_eval.rubric:SearchEvalRubric"


def _request(name: str, rubric_class: str) -> ExternalBenchmarkRequest:
    return ExternalBenchmarkRequest(
        benchmark_name=name,
        judge_class="local_judge",
        attribution_class="no_attribution",
        rubric_class=rubric_class,
    )


@pytest.mark.parametrize(
    ("alias", "class_ref"),
    [
        ("amap_chat_rubric_v2", _AMAP_REF),
        ("search_eval_rubric", _SEARCH_REF),
        ("component_rubric", None),
    ],
)
def test_builtin_aliases_resolve(alias: str, class_ref: str | None) -> None:
    spec = resolve_rubric_class(alias)
    assert spec.alias == alias
    assert spec.class_ref == class_ref


def test_unknown_alias_lists_available_choices() -> None:
    with pytest.raises(RubricClassResolutionError, match="amap_chat_rubric_v2"):
        resolve_rubric_class("unknown_rubric")


def test_registered_class_must_subclass_base_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        rc_mod.RUBRIC_CLASS_REFS,
        "broken_rubric",
        "benchmark.runs.common.rubric_classes:RubricClassSpec",
    )
    with pytest.raises(RubricClassResolutionError, match="BaseRubric"):
        resolve_rubric_class("broken_rubric")


def test_class_outside_whitelist_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(rc_mod.RUBRIC_CLASS_REFS, "evil_rubric", "os:system")
    with pytest.raises(RubricClassResolutionError):
        resolve_rubric_class("evil_rubric")


def test_alias_reverse_lookup() -> None:
    assert rubric_alias_for_ref(_AMAP_REF) == "amap_chat_rubric_v2"
    assert rubric_alias_for_ref(None) == "component_rubric"
    assert rubric_alias_for_ref("pkg.mod:Unregistered") is None


def test_request_rubric_class_recommended_with_legacy_rubric_ref_window() -> None:
    """Migration window (§9): rubric_class is recommended but may be omitted (fall back to the default mapping);
    deprecated rubric_ref may still be passed through; when both are present the alias wins."""
    base = {
        "benchmark_name": "ieval_external",
        "judge_class": "ieval_external_judge",
        "attribution_class": "ieval_external_attribution",
    }
    # An omitted rubric_class is no longer a 422: fall back to the default mapping (rubric_ref=None)
    profile = profile_from_external_request(ExternalBenchmarkRequest(**base))
    assert profile.rubric_ref is None
    assert profile.component_rubric is False
    # Both present: the alias wins and rubric_ref is ignored
    profile2 = profile_from_external_request(
        ExternalBenchmarkRequest(
            **base, rubric_class="amap_chat_rubric_v2", rubric_ref="pkg.mod:Other"
        ),
    )
    assert profile2.rubric_ref == _AMAP_REF


def test_profile_carries_resolved_rubric_ref() -> None:
    profile = profile_from_external_request(_request("search_eval", "search_eval_rubric"))
    assert profile.rubric_ref == _SEARCH_REF
    assert profile.component_rubric is False


def test_unknown_rubric_class_maps_to_profile_error() -> None:
    with pytest.raises(ProfileResolutionError) as exc:
        profile_from_external_request(_request("search_eval", "unknown_rubric"))
    assert exc.value.reason == "invalid_rubric_class"


def test_component_rubric_skips_benchmark_default_mapping() -> None:
    """ieval_external maps to AmapChatRubricV2 by default; component_rubric explicitly switches to meta.rubric."""
    resolved = ProfileResolver().resolve(
        profile_from_external_request(_request("ieval_external", "component_rubric")),
        _meta("ieval_external"),
    )
    assert resolved.rubric_ref is None
    assert [c.id for c in resolved.rubric] == ["Q1"]
    # criteria come from meta, but the parse interceptor is still mounted via the default Rubric mapping fallback
    assert resolved.parse_interceptor_ref == (
        "business.runs.ieval.amap_chat.parse:AmapChatParseInterceptor"
    )


def test_different_businesses_use_their_own_rubric_class() -> None:
    """Under the same judge/attribution combination, each business picks its own Rubric class by rubric_class."""
    ieval = ProfileResolver().resolve(
        profile_from_external_request(_request("biz_a", "amap_chat_rubric_v2")),
        _meta(),
    )
    search = ProfileResolver().resolve(
        profile_from_external_request(_request("biz_b", "search_eval_rubric")),
        _meta(),
    )
    assert ieval.rubric_ref == _AMAP_REF
    assert search.rubric_ref == _SEARCH_REF
    assert [c.id for c in ieval.rubric] != [c.id for c in search.rubric]


def _meta(name: str = "biz") -> BenchmarkComponentMetadata:
    return BenchmarkComponentMetadata(
        name=name,
        service_name="demo_svc",
        rubric=[Criterion(id="Q1", name="q", description="", weight=1.0, evaluator_ref="demo")],
        aggregation=AggregationConfig(method="weighted_avg", pass_threshold=0.5),
    )
