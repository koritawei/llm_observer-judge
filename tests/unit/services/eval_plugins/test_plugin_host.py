"""Unit tests: generic plugin host resolution + registry wiring for every category."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain import eval_plugin_config as cfg_mod
from benchmark.domain.eval_plugin_config import (
    PLUGIN_CATEGORIES,
    EvalPluginsConfig,
    PluginFilter,
)
from benchmark.domain.plugin_binding import PluginBinding
from benchmark.evaluators.base import EvaluatorResult
from benchmark.attribution.badcase.attribution_trigger_policy import (
    RubricV2TriggerPolicy,
    clear_trigger_policies,
    get_default_trigger_policy,
    register_trigger_policy,
    resolve_trigger_policy,
    should_trigger_attribution,
)
from benchmark.attribution.badcase.dispatch_policy import (
    get_default_dispatch_policy,
    resolve_dispatch_policy,
    should_dispatch_notification,
)
from benchmark.runs.common.attribution_stats import (
    clear_attribution_stats_providers,
    resolve_attribution_stats_provider,
)
from benchmark.runs.common.trace_url_builder import (
    default_item_trace_identity,
    default_trace_identity,
    resolve_item_trace_identity_extractor,
    resolve_trace_identity_extractor,
)
from benchmark.reporting.framework.fusion import ScoreFuser
from benchmark.reporting.framework.schema import AggregationConfig, Criterion
from benchmark.plugins.class_loader import (
    PluginClassLoadError,
    instantiate_plugin,
    load_plugin_class,
)
from benchmark.plugins.host import PluginResolution, is_plugin_disabled, resolve_plugin
from benchmark.plugins.remote_adapters import (
    RemoteAttributionStatsProvider,
    RemoteDispatchPolicy,
    RemoteFusionFunction,
    RemoteItemTraceIdentityExtractor,
    RemoteTraceIdentityExtractor,
    RemoteTriggerPolicy,
)
from benchmark.plugins.remote_client import RemotePluginClient, RemotePluginError

_REMOTE_URL = "https://biz.example/plugins"


def _use_config(monkeypatch: pytest.MonkeyPatch, **categories: PluginFilter) -> None:
    monkeypatch.setattr(
        cfg_mod,
        "fetch_eval_plugins",
        lambda **_k: EvalPluginsConfig(**categories),
    )


def _remote_binding(name: str, **extra: Any) -> PluginBinding:
    return PluginBinding(name=name, mode="remote", endpoint_url=_REMOTE_URL, **extra)


def _client(invoke_result: Any = None, *, error: Exception | None = None) -> MagicMock:
    client = MagicMock(spec=RemotePluginClient)
    if error is not None:
        client.invoke = AsyncMock(side_effect=error)
        client.invoke_sync = MagicMock(side_effect=error)
    else:
        client.invoke = AsyncMock(return_value=invoke_result)
        client.invoke_sync = MagicMock(return_value=invoke_result)
    return client


@pytest.fixture(autouse=True)
def _isolate_registries() -> Iterator[None]:
    yield
    clear_trigger_policies()
    clear_attribution_stats_providers()


# ── config schema ────────────────────────────────────────────────────────────


def test_plugin_categories_match_config_fields() -> None:
    assert set(PLUGIN_CATEGORIES) == set(EvalPluginsConfig.model_fields)


def test_new_categories_parse_from_remote_config_payload() -> None:
    parsed = EvalPluginsConfig.model_validate(
        {
            "attribution_trigger_policies": {
                "bindings": [
                    {"name": "ieval_external", "mode": "remote", "endpoint_url": _REMOTE_URL}
                ],
            },
            "fusion_functions": {"disabled": ["mod:fn"]},
        },
    )
    assert parsed.attribution_trigger_policies.binding_for("ieval_external") is not None
    assert not parsed.fusion_functions.allows("mod:fn")


# ── generic host ─────────────────────────────────────────────────────────────


class TestResolvePlugin:
    def test_no_binding_prefers_registry_then_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_config(monkeypatch)
        registered = object()
        default = object()
        assert (
            resolve_plugin(
                "dispatch_policies",
                "svc",
                resolution=PluginResolution(
                    registry={"svc": registered},
                    default=default,
                    remote_factory=lambda _b: object(),
                ),
            )
            is registered
        )
        assert (
            resolve_plugin(
                "dispatch_policies",
                "other",
                resolution=PluginResolution(
                    registry={"svc": registered},
                    default=default,
                    remote_factory=lambda _b: object(),
                ),
            )
            is default
        )

    def test_empty_name_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch)
        default = object()
        assert (
            resolve_plugin(
                "dispatch_policies",
                "",
                resolution=PluginResolution(
                    registry={},
                    default=default,
                    remote_factory=lambda _b: object(),
                ),
            )
            is default
        )

    def test_remote_binding_wins_over_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(
            monkeypatch,
            dispatch_policies=PluginFilter(bindings=[_remote_binding("svc")]),
        )
        remote = object()
        assert (
            resolve_plugin(
                "dispatch_policies",
                "svc",
                resolution=PluginResolution(
                    registry={"svc": object()},
                    default=object(),
                    remote_factory=lambda _b: remote,
                ),
            )
            is remote
        )

    def test_disabled_binding_forces_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(
            monkeypatch,
            dispatch_policies=PluginFilter(disabled=["svc"]),
        )
        default = object()
        assert (
            resolve_plugin(
                "dispatch_policies",
                "svc",
                resolution=PluginResolution(
                    registry={"svc": object()},
                    default=default,
                    remote_factory=lambda _b: object(),
                ),
            )
            is default
        )
        assert is_plugin_disabled("dispatch_policies", "svc")
        assert not is_plugin_disabled("dispatch_policies", "other")

    def test_class_ref_loads_local_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(
            monkeypatch,
            attribution_trigger_policies=PluginFilter(
                bindings=[
                    PluginBinding(
                        name="biz",
                        mode="local",
                        class_ref=(
                            "benchmark.attribution.badcase.attribution_trigger_policy:"
                            "RubricV2TriggerPolicy"
                        ),
                    ),
                ],
            ),
        )
        policy = resolve_trigger_policy("biz")
        assert isinstance(policy, RubricV2TriggerPolicy)
        assert policy is not get_default_trigger_policy()

    def test_class_ref_failure_raises_without_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_config(
            monkeypatch,
            attribution_trigger_policies=PluginFilter(
                bindings=[PluginBinding(name="biz", mode="local", class_ref="os.path:join")],
            ),
        )
        with pytest.raises(PluginClassLoadError, match="whitelist"):
            resolve_trigger_policy("biz")

    def test_class_ref_failure_falls_back_to_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_config(
            monkeypatch,
            attribution_trigger_policies=PluginFilter(
                bindings=[
                    PluginBinding(
                        name="biz",
                        mode="local",
                        class_ref="os.path:join",
                        fallback_local=True,
                    ),
                ],
            ),
        )
        registered = RubricV2TriggerPolicy(max_exclusive=1.0)
        register_trigger_policy("biz", registered)
        assert resolve_trigger_policy("biz") is registered


# ── class loader ─────────────────────────────────────────────────────────────


def test_instantiate_plugin_returns_function_for_function_ref() -> None:
    fn = instantiate_plugin("business.runs.ieval.amap_chat.services:rubric_v2_fuse")
    assert callable(fn) and not isinstance(fn, type)


def test_load_plugin_class_rejects_function_ref() -> None:
    with pytest.raises(PluginClassLoadError, match="not a class"):
        load_plugin_class("business.runs.ieval.amap_chat.services:rubric_v2_fuse")


# ── trigger / dispatch policies ─────────────────────────────────────────────


class TestTriggerPolicyPlugin:
    def test_registry_identity_preserved_without_remote_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_config(monkeypatch)
        policy = RubricV2TriggerPolicy(max_exclusive=0.5)
        register_trigger_policy("biz", policy)
        assert resolve_trigger_policy("biz") is policy
        assert resolve_trigger_policy("unknown") is get_default_trigger_policy()

    def test_remote_binding_yields_remote_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(
            monkeypatch,
            attribution_trigger_policies=PluginFilter(bindings=[_remote_binding("ieval_external")]),
        )
        assert isinstance(resolve_trigger_policy("ieval_external"), RemoteTriggerPolicy)

    async def test_async_call_uses_remote_client(self) -> None:
        client = _client({"result": False})
        policy = RemoteTriggerPolicy(
            _remote_binding("biz"),
            fallback=get_default_trigger_policy(),
            client=client,
        )
        assert await should_trigger_attribution(policy, fused_score=0.1) is False
        client.invoke.assert_awaited_once()
        client.invoke_sync.assert_not_called()
        kw = client.invoke.await_args.kwargs
        assert (kw["kind"], kw["action"]) == ("attribution_trigger_policy", "should_trigger")
        assert kw["payload"] == {"fused_score": 0.1, "criterion_scores": {}}

    async def test_remote_failure_uses_fallback_when_enabled(self) -> None:
        policy = RemoteTriggerPolicy(
            _remote_binding("biz", fallback_local=True),
            fallback=get_default_trigger_policy(),
            client=_client(error=RemotePluginError("down")),
        )
        assert await should_trigger_attribution(policy, fused_score=0.5) is True
        assert await should_trigger_attribution(policy, fused_score=1.9) is False

    async def test_remote_failure_raises_without_fallback(self) -> None:
        policy = RemoteTriggerPolicy(
            _remote_binding("biz"),
            fallback=get_default_trigger_policy(),
            client=_client(error=RemotePluginError("down")),
        )
        with pytest.raises(RemotePluginError):
            await should_trigger_attribution(policy, fused_score=0.5)

    async def test_local_policy_called_synchronously(self) -> None:
        assert await should_trigger_attribution(RubricV2TriggerPolicy(), fused_score=1.0) is True


class TestDispatchPolicyPlugin:
    def test_remote_binding_yields_remote_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(
            monkeypatch,
            dispatch_policies=PluginFilter(bindings=[_remote_binding("amap_chat")]),
        )
        assert isinstance(resolve_dispatch_policy("amap_chat"), RemoteDispatchPolicy)
        assert resolve_dispatch_policy("other") is get_default_dispatch_policy()

    async def test_async_call_and_fallback(self) -> None:
        ok = RemoteDispatchPolicy(
            _remote_binding("svc"),
            fallback=get_default_dispatch_policy(),
            client=_client({"allow": False}),
        )
        assert await should_dispatch_notification(ok, grade="POOR") is False
        broken = RemoteDispatchPolicy(
            _remote_binding("svc", fallback_local=True),
            fallback=get_default_dispatch_policy(),
            client=_client(error=RemotePluginError("down")),
        )
        assert await should_dispatch_notification(broken, grade="POOR") is True
        assert await should_dispatch_notification(broken, grade="GOOD") is False


# ── attribution stats ────────────────────────────────────────────────────────


class TestAttributionStatsPlugin:
    def test_remote_binding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(
            monkeypatch,
            attribution_stats_providers=PluginFilter(bindings=[_remote_binding("search_eval")]),
        )
        assert isinstance(
            resolve_attribution_stats_provider("search_eval"),
            RemoteAttributionStatsProvider,
        )

    def test_disabled_skips_lazy_ensure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(
            monkeypatch,
            attribution_stats_providers=PluginFilter(disabled=["search_eval"]),
        )
        clear_attribution_stats_providers()
        assert resolve_attribution_stats_provider("search_eval") is None

    async def test_remote_collect_fallback_empty(self) -> None:
        provider = RemoteAttributionStatsProvider(
            _remote_binding("search_eval", fallback_local=True),
            client=_client(error=RemotePluginError("down")),
        )
        assert await provider.collect("run-1") == {}


# ── trace identity extractors ────────────────────────────────────────────────


class TestTraceIdentityPlugins:
    def test_defaults_without_remote_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch)
        assert resolve_trace_identity_extractor("x") is default_trace_identity
        assert resolve_item_trace_identity_extractor("x") is default_item_trace_identity

    def test_remote_bindings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(
            monkeypatch,
            trace_identity_extractors=PluginFilter(bindings=[_remote_binding("ieval_external")]),
            item_trace_identity_extractors=PluginFilter(bindings=[_remote_binding("search_eval")]),
        )
        callback_extractor = resolve_trace_identity_extractor("ieval_external")
        item_extractor = resolve_item_trace_identity_extractor("search_eval")
        assert isinstance(callback_extractor, RemoteTraceIdentityExtractor)
        assert isinstance(item_extractor, RemoteItemTraceIdentityExtractor)
        assert item_extractor.kind == "item_trace_identity_extractor"

    async def test_remote_extract_payload_and_parse(self) -> None:
        client = _client({"project_id": "p1", "trace_id": "t1"})
        extractor = RemoteItemTraceIdentityExtractor(_remote_binding("search_eval"), client=client)
        assert await extractor.acall({"item_id": "i1"}) == ("p1", "t1")
        kw = client.invoke.await_args.kwargs
        assert kw["payload"] == {"item": {"item_id": "i1"}}

    async def test_remote_extract_failure_returns_none_pair(self) -> None:
        extractor = RemoteTraceIdentityExtractor(
            _remote_binding("ieval_external"),
            client=_client(error=RemotePluginError("down")),
        )
        payload = SimpleNamespace(project_id="p", trace_id="t")
        assert extractor(payload) == (None, None)

    async def test_remote_extract_failure_uses_fallback(self) -> None:
        extractor = RemoteTraceIdentityExtractor(
            _remote_binding("ieval_external", fallback_local=True),
            fallback=lambda p: (p.project_id, p.trace_id),
            client=_client(error=RemotePluginError("down")),
        )
        assert await extractor.acall(SimpleNamespace(project_id="p", trace_id="t")) == ("p", "t")


# ── fusion function ──────────────────────────────────────────────────────────

_FUSION_REF = "business.runs.ieval.amap_chat.services:rubric_v2_fuse"


def _rubric() -> list[Criterion]:
    ref = "business.evaluators.example.agent:ExampleAgent"
    return [
        Criterion(id="a", name="a", weight=1.0, evaluator_ref=ref),
        Criterion(id="b", name="b", weight=1.0, evaluator_ref=ref),
    ]


def _scores() -> dict[str, EvaluatorResult]:
    return {"a": EvaluatorResult(score=1.0), "b": EvaluatorResult(score=0.0)}


class TestFusionFunctionPlugin:
    async def test_remote_fusion_used_in_afuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(
            monkeypatch,
            fusion_functions=PluginFilter(bindings=[_remote_binding(_FUSION_REF)]),
        )
        client = _client({"score": 0.25})
        monkeypatch.setattr(RemotePluginClient, "invoke", client.invoke)
        agg = AggregationConfig(method="weighted_avg", fusion_fn_ref=_FUSION_REF)
        assert await ScoreFuser.afuse(_scores(), _rubric(), agg) == 0.25
        kw = client.invoke.await_args.kwargs
        assert (kw["kind"], kw["action"]) == ("fusion_function", "fuse")
        assert kw["payload"]["scores"]["a"] == {"score": 1.0, "reason": None, "failed": False}

    async def test_remote_fusion_failure_falls_back_to_method(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_config(
            monkeypatch,
            fusion_functions=PluginFilter(
                bindings=[_remote_binding(_FUSION_REF, fallback_local=True)]
            ),
        )
        monkeypatch.setattr(
            RemotePluginClient, "invoke", AsyncMock(side_effect=RemotePluginError("x"))
        )
        agg = AggregationConfig(method="weighted_avg", fusion_fn_ref=_FUSION_REF)
        assert await ScoreFuser.afuse(_scores(), _rubric(), agg) == pytest.approx(0.5)

    async def test_remote_fusion_failure_raises_without_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use_config(
            monkeypatch,
            fusion_functions=PluginFilter(bindings=[_remote_binding(_FUSION_REF)]),
        )
        monkeypatch.setattr(
            RemotePluginClient, "invoke", AsyncMock(side_effect=RemotePluginError("x"))
        )
        agg = AggregationConfig(method="weighted_avg", fusion_fn_ref=_FUSION_REF)
        with pytest.raises(RemotePluginError):
            await ScoreFuser.afuse(_scores(), _rubric(), agg)

    def test_remote_fusion_rejects_non_numeric_score(self) -> None:
        fn = RemoteFusionFunction(_remote_binding(_FUSION_REF), client=_client({"score": "n/a"}))
        with pytest.raises(RemotePluginError, match="numeric"):
            fn(_scores(), _rubric(), AggregationConfig(method="weighted_avg"), lambda _c: 0.0)

    async def test_disabled_ref_falls_back_to_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch, fusion_functions=PluginFilter(disabled=[_FUSION_REF]))
        agg = AggregationConfig(method="weighted_avg", fusion_fn_ref=_FUSION_REF)
        assert await ScoreFuser.afuse(_scores(), _rubric(), agg) == pytest.approx(0.5)

    async def test_no_binding_imports_ref_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_config(monkeypatch)
        agg = AggregationConfig(method="weighted_avg", fusion_fn_ref=_FUSION_REF)
        sync_result = ScoreFuser.fuse(_scores(), _rubric(), agg)
        assert await ScoreFuser.afuse(_scores(), _rubric(), agg) == sync_result
