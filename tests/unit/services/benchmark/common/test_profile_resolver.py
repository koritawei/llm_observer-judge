"""Unit tests for ProfileResolver (design doc §5.2).

Covers: rubric_ref class loading (allowlist / illegal 400) / execution derivation /
target checks / attribution default fallback / config_json snapshot (without api_key).
"""

import pytest
from benchmark.runs.common.external_client_factory import (
    ExternalClientFactory,
    ExternalClientFactoryConfig,
)
from benchmark.runs.common.profile_resolver import (
    ProfileResolutionError,
    ProfileResolver,
    load_class_ref,
)
class AmapChatParseInterceptor:
    pass
class AmapChatRubricV2:
    pass
from benchmark.runs.schema import (
    AttributionProfileConfig,
    BenchmarkRuntimeProfile,
    JudgeProfileConfig,
)
from benchmark.reporting.framework.base import BenchmarkComponentMetadata
from benchmark.reporting.framework.schema import AggregationConfig, Criterion

_BASE = "https://aone.example.com"

_META = BenchmarkComponentMetadata(
    name="demo_bench",
    service_name="demo",
    rubric=[Criterion(id="Q1", name="q", description="", weight=1.0, evaluator_ref="demo")],
    aggregation=AggregationConfig(method="weighted_avg", pass_threshold=0.5),
)


def _resolver(with_factory: bool = True) -> ProfileResolver:
    factory = (
        ExternalClientFactory(
            ExternalClientFactoryConfig(
                judge_base_url=_BASE,
                api_key="ak-test",
                default_judge_pipeline_id="100",
                default_attribution_pipeline_id="200",
            ),
        )
        if with_factory
        else None
    )
    return ProfileResolver(client_factory=factory)


def _profile(**overrides: object) -> BenchmarkRuntimeProfile:
    base: dict[str, object] = {"benchmark_name": "demo_bench"}
    base.update(overrides)
    return BenchmarkRuntimeProfile(**base)  # type: ignore[arg-type]


# ── load_class_ref allowlist ──────────────────────────────────────────────


def test_load_class_ref_whitelist_reject() -> None:
    with pytest.raises(ProfileResolutionError) as exc:
        load_class_ref("os.path:join", what="rubric_ref")
    assert exc.value.reason == "invalid_rubric_ref"


def test_load_class_ref_bad_format() -> None:
    with pytest.raises(ProfileResolutionError):
        load_class_ref("no-colon", what="rubric_ref")


def test_load_class_ref_import_failure() -> None:
    with pytest.raises(ProfileResolutionError):
        load_class_ref("benchmark.runs.no_such_module:Nope", what="rubric_ref")


# ── rubric_ref class loading ─────────────────────────────────────────────────


def test_resolve_rubric_ref_class_loading() -> None:
    """rubric_ref → class load, and criteria come from the class criteria()."""
    resolved = _resolver().resolve(
        _profile(
            rubric_ref="business.runs.ieval.amap_chat.rubric:AmapChatRubricV2",
        ),
        _META,
    )
    assert [c.id for c in resolved.rubric] == ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
    assert resolved.rubric_ref is not None
    # Interceptors mounted on the Rubric class take effect automatically (§7.5)
    assert isinstance(resolved.parse_interceptor, AmapChatParseInterceptor)
    assert resolved.parse_interceptor_ref == AmapChatRubricV2.parse_interceptor_ref
    # aggregation takes the Rubric class default (pass_threshold=1.0, not meta's 0.5)
    assert resolved.aggregation.pass_threshold == AmapChatRubricV2.pass_threshold
    assert resolved.aggregation.fusion_fn_ref == AmapChatRubricV2.fusion_fn_ref


def test_resolve_default_rubric_fallback_to_meta() -> None:
    """No rubric_ref and no default mapping → meta.rubric (default fallback, §6.2)."""
    resolved = _resolver().resolve(_profile(), _META)
    assert [c.id for c in resolved.rubric] == ["Q1"]
    assert resolved.rubric_ref is None
    assert resolved.parse_interceptor is None
    assert resolved.aggregation.pass_threshold == 0.5


def test_resolve_default_rubric_ref_registry() -> None:
    """§9 migration anchor: ieval_external mounts AmapChatRubricV2 by default (v1 behavior unchanged)."""
    resolved = _resolver().resolve(_profile(benchmark_name="ieval_external"), _META)
    assert [c.id for c in resolved.rubric] == ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
    assert isinstance(resolved.parse_interceptor, AmapChatParseInterceptor)


def test_resolve_explicit_aggregation_overrides_rubric_class() -> None:
    agg = AggregationConfig(method="all_pass", pass_threshold=0.9)
    resolved = _resolver().resolve(
        _profile(
            rubric_ref="business.runs.ieval.amap_chat.rubric:AmapChatRubricV2",
            aggregation=agg,
        ),
        _META,
    )
    assert resolved.aggregation.method == "all_pass"


def test_resolve_explicit_interceptor_overrides_rubric_class() -> None:
    """An explicit profile parse_interceptor_ref overrides what the Rubric class mounts (§7.5 priority)."""
    resolved = _resolver().resolve(
        _profile(
            rubric_ref="business.runs.ieval.amap_chat.rubric:AmapChatRubricV2",
            parse_interceptor_ref=(
                "business.runs.ieval.amap_chat.parse:AmapChatParseInterceptor"
            ),
        ),
        _META,
    )
    assert isinstance(resolved.parse_interceptor, AmapChatParseInterceptor)


def test_resolve_bad_interceptor_ref_rejected() -> None:
    with pytest.raises(ProfileResolutionError) as exc:
        _resolver().resolve(
            _profile(parse_interceptor_ref="os:system"),
            _META,
        )
    assert exc.value.reason == "invalid_parse_interceptor_ref"


# ── judge execution / target ──────────────────────────────────────────


def test_judge_default_async_when_factory_configured() -> None:
    """Default (no judge config): Aone is configured → async + the default URL (production F3 semantics)."""
    resolved = _resolver().resolve(_profile(), _META)
    assert resolved.judge.execution == "async"
    assert resolved.judge.target == f"{_BASE}/api/pipelines/100/trigger"
    assert resolved.judge.provider == "external"
    assert resolved.judge.dims == "Q1"


def test_judge_default_sync_without_factory() -> None:
    """Aone is not configured → default to sync local evaluation (production LocalEvalService fallback)."""
    resolved = _resolver(with_factory=False).resolve(_profile(), _META)
    assert resolved.judge.execution == "sync"
    assert resolved.judge.provider == "local"
    assert resolved.judge.target is None


def test_judge_execution_inferred_from_target_url() -> None:
    """Default derivation: a target of http(s):// → async (§6.3 constraint 3)."""
    resolved = _resolver().resolve(
        _profile(judge=JudgeProfileConfig(target=f"{_BASE}/api/pipelines/777/trigger")),
        _META,
    )
    assert resolved.judge.execution == "async"
    assert resolved.judge.target == f"{_BASE}/api/pipelines/777/trigger"


def test_judge_target_host_outside_whitelist_rejected() -> None:
    """A target host outside the process base_url host → invalid_judge_target (§6.3 SSRF)."""
    with pytest.raises(ProfileResolutionError) as exc:
        _resolver().resolve(
            _profile(
                judge=JudgeProfileConfig(
                    target="https://evil.example.com/api/pipelines/1/trigger",
                ),
            ),
            _META,
        )
    assert exc.value.reason == "invalid_judge_target"


def test_judge_sync_target_class_loaded() -> None:
    """sync target = the local evaluator class name (allowlist load, §6.3 constraint 2)."""
    resolved = _resolver().resolve(
        _profile(
            judge=JudgeProfileConfig(
                target=("business.runs.ieval.amap_chat.parse:AmapChatParseInterceptor"),
            ),
        ),
        _META,
    )
    assert resolved.judge.execution == "sync"
    assert resolved.judge.local_eval_class is AmapChatParseInterceptor


def test_judge_sync_target_outside_whitelist_rejected() -> None:
    with pytest.raises(ProfileResolutionError) as exc:
        _resolver().resolve(
            _profile(judge=JudgeProfileConfig(target="os:system")),
            _META,
        )
    assert exc.value.reason == "invalid_judge_target"


# ── attribution execution / target ────────────────────────────────────


def test_attribution_default_sync_example() -> None:
    """Default attribution = sync + example (production pull default, F14)."""
    resolved = _resolver().resolve(_profile(), _META)
    assert resolved.attribution.execution == "sync"
    assert resolved.attribution.provider == "example"


def test_attribution_none_explicit() -> None:
    resolved = _resolver().resolve(
        _profile(attribution=AttributionProfileConfig(execution="none", provider="none")),
        _META,
    )
    assert resolved.attribution.execution == "none"
    assert resolved.attribution.target is None


def test_attribution_async_default_falls_back_to_judge_url() -> None:
    """When an async attribution target is omitted, fall back to the judge.target URL (§6.4, aligned with production F3)."""
    resolved = _resolver().resolve(
        _profile(
            judge=JudgeProfileConfig(target=f"{_BASE}/api/pipelines/777/trigger"),
            attribution=AttributionProfileConfig(provider="external"),
        ),
        _META,
    )
    assert resolved.attribution.execution == "async"
    # A process-default attribution URL exists (pipeline 200) → use the default, not judge's 777
    assert resolved.attribution.target == f"{_BASE}/api/pipelines/200/trigger"


def test_attribution_async_explicit_target() -> None:
    resolved = _resolver().resolve(
        _profile(
            attribution=AttributionProfileConfig(
                target=f"{_BASE}/api/pipelines/888/trigger",
            ),
        ),
        _META,
    )
    assert resolved.attribution.execution == "async"
    assert resolved.attribution.provider == "external"
    assert resolved.attribution.target == f"{_BASE}/api/pipelines/888/trigger"


def test_attribution_async_without_factory_rejected() -> None:
    with pytest.raises(ProfileResolutionError) as exc:
        _resolver(with_factory=False).resolve(
            _profile(attribution=AttributionProfileConfig(provider="external")),
            _META,
        )
    assert exc.value.reason == "invalid_attribution_target"


# ── config_json snapshot ─────────────────────────────────────────────────


def test_config_snapshot_never_carries_api_key() -> None:
    """The snapshot is the source of truth (§5.2): it holds the full rubric/judge/attribution, and never api_key."""
    resolved = _resolver().resolve(
        _profile(
            rubric_ref="business.runs.ieval.amap_chat.rubric:AmapChatRubricV2",
            judge=JudgeProfileConfig(target=f"{_BASE}/api/pipelines/777/trigger"),
        ),
        _META,
    )
    snapshot = resolved.config_snapshot()
    assert snapshot["rubric_ref"] == (
        "business.runs.ieval.amap_chat.rubric:AmapChatRubricV2"
    )
    assert snapshot["judge"]["target"] == f"{_BASE}/api/pipelines/777/trigger"
    assert snapshot["judge"]["execution"] == "async"
    assert snapshot["judge"]["dims"] == "D0,D1,D2,D3,D4,D5,D6"
    assert [c["id"] for c in snapshot["rubric"]] == ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
    # Key names align with load_rubric_from_config / load_aggregation_from_config (F7)
    assert snapshot["aggregation"]["method"] == "weighted_avg"
    # Scoring-rule link: the Rubric class rule_link_url enters config via the snapshot (the source that fills report slots)
    assert snapshot["rule_link_url"] == (
        "https://alidocs.dingtalk.com/i/nodes/14dA3GK8gwAG5B1mHnbDXZPG89ekBD76"
    )
    # api_key leak check (frozen item §2)
    import json

    assert "ak-test" not in json.dumps(snapshot)
    assert "api_key" not in json.dumps(snapshot)


def test_config_snapshot_rule_link_url_none_for_meta_rubric() -> None:
    """No rubric_ref mapping (fall back to meta.rubric) → rule_link_url is None (the business did not configure it)."""
    resolved = _resolver().resolve(_profile(), _META)
    assert resolved.config_snapshot()["rule_link_url"] is None


# ── judge_execution_policy guard (20260923 search_eval incident regression)──────


_SEARCH_EVAL_RUBRIC_REF = "business.runs.search_eval.rubric:SearchEvalRubric"


def _search_eval_profile(**overrides: object) -> BenchmarkRuntimeProfile:
    base: dict[str, object] = {
        "benchmark_name": "search_eval",
        "rubric_ref": _SEARCH_EVAL_RUBRIC_REF,
    }
    base.update(overrides)
    return BenchmarkRuntimeProfile(**base)  # type: ignore[arg-type]


def test_policy_rejects_explicit_async_for_search_eval() -> None:
    """SearchEvalRubric declares a local policy: an explicit async judge → rejected with 400."""
    with pytest.raises(ProfileResolutionError) as exc:
        _resolver().resolve(
            _search_eval_profile(
                judge=JudgeProfileConfig(
                    execution="async",
                    target=f"{_BASE}/api/pipelines/777/trigger",
                ),
            ),
            _META,
        )
    assert exc.value.reason == "judge_execution_policy_violation"


def test_policy_rejects_default_async_inference_for_search_eval() -> None:
    """Incident regression: an omitted judge → derived as async from the Aone default pipeline → rejected.

    20260923: when the search_eval judge config was missing it was derived as async + the e2e pipeline,
    and every search_ai_card dimension failed because the ALL_DIMENSIONS=['D0'..'D6'] allowlist rejected them.
    """
    with pytest.raises(ProfileResolutionError) as exc:
        _resolver().resolve(_search_eval_profile(), _META)
    assert exc.value.reason == "judge_execution_policy_violation"


def test_policy_allows_sync_for_search_eval() -> None:
    """Explicit sync local evaluation → allowed (the correct setup for search_eval)."""
    resolved = _resolver().resolve(
        _search_eval_profile(
            judge=JudgeProfileConfig(execution="sync", provider="local"),
        ),
        _META,
    )
    assert resolved.judge.execution == "sync"
    assert resolved.judge.provider == "local"


def test_policy_absent_rubric_async_unaffected() -> None:
    """No policy declared (ieval_external / AmapChatRubricV2) → async production semantics stay unchanged."""
    resolved = _resolver().resolve(
        _profile(
            benchmark_name="ieval_external",
            rubric_ref="business.runs.ieval.amap_chat.rubric:AmapChatRubricV2",
        ),
        _META,
    )
    assert resolved.judge.execution == "async"
    assert resolved.judge.provider == "external"
