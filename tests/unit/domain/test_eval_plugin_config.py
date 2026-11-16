"""Unit tests for RemoteConfig eval plugin host (filters + bindings)."""

import pytest
from benchmark.domain.eval_plugin_config import (
    EvalPluginsConfig,
    PluginFilter,
    filter_enabled_names,
    is_benchmark_component_enabled,
    is_output_provider_enabled,
    resolve_plugin_binding,
)
from benchmark.domain.plugin_binding import PluginBinding


def test_plugin_filter_wildcard_allow_all() -> None:
    filt = PluginFilter()
    assert filt.allows("ieval_external")
    assert filt.allows("search_eval")


def test_plugin_filter_disabled_wins() -> None:
    filt = PluginFilter(enabled=["*"], disabled=["search_eval"])
    assert filt.allows("ieval_external")
    assert not filt.allows("search_eval")


def test_plugin_filter_explicit_enabled() -> None:
    filt = PluginFilter(enabled=["expected_output"], disabled=[])
    assert filt.allows("expected_output")
    assert not filt.allows("agent_chat")


def test_binding_enabled_false_blocks() -> None:
    filt = PluginFilter(
        bindings=[PluginBinding(name="agent_chat", mode="local", enabled=False)],
    )
    assert not filt.allows("agent_chat")


def test_fetch_defaults_allow_all(monkeypatch: pytest.MonkeyPatch) -> None:
    # fetch_eval_plugins imports lazily (breaking the benchmark.settings ↔ benchmark.services cycle),
    # so patch the real source module rather than a module attribute of eval_plugin_config.
    import benchmark.settings.remote_config_store as store

    monkeypatch.setattr(store, "fetch_cached_settings", lambda *_a, **_k: None)
    assert is_benchmark_component_enabled("anything")
    assert is_output_provider_enabled("expected_output")
    assert resolve_plugin_binding("output_providers", "expected_output") is None


def test_filter_enabled_names(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmark.domain.eval_plugin_config as mod

    monkeypatch.setattr(
        mod,
        "fetch_eval_plugins",
        lambda **_k: EvalPluginsConfig(
            benchmark_components=PluginFilter(enabled=["a", "c"], disabled=["c"]),
        ),
    )
    assert filter_enabled_names(["a", "b", "c"], kind="benchmark_components") == ["a"]
