"""Eval plugin runtime: local / remote resolution and enablement.

RemoteConfig JSON（dataId ``agent-eval.eval.plugins``）::

    {
      "benchmark_components": { "enabled": ["*"], "disabled": [] },
      "output_providers": {
        "enabled": ["*"],
        "disabled": [],
        "bindings": [
          { "name": "expected_output", "mode": "local" },
          {
            "name": "agent_chat",
            "mode": "remote",
            "endpoint_url": "https://biz.example/plugins/output",
            "timeout_ms": 60000,
            "fallback_local": true
          }
        ]
      },
      "parse_interceptors": {
        "enabled": ["*"],
        "bindings": [
          {
            "name": "business.runs.search_eval.parse:SearchEvalParseInterceptor",
            "mode": "remote",
            "endpoint_url": "https://biz.example/plugins/parse"
          }
        ]
      },
      "report_providers": {
        "bindings": [
          {
            "name": "search_eval",
            "mode": "remote",
            "endpoint_url": "https://biz.example/plugins/report"
          }
        ]
      },
      "attribution_trigger_policies": {
        "bindings": [
          { "name": "ieval_external", "mode": "remote",
            "endpoint_url": "https://biz.example/plugins/trigger", "fallback_local": true }
        ]
      },
      "dispatch_policies": {
        "bindings": [
          { "name": "amap_chat", "mode": "local",
            "class_ref": "benchmark.attribution.badcase.dispatch_policy:MyDispatchPolicy" }
        ]
      },
      "attribution_stats_providers": { "bindings": [] },
      "trace_identity_extractors": { "bindings": [] },
      "item_trace_identity_extractors": { "bindings": [] },
      "fusion_functions": {
        "bindings": [
          { "name": "business.runs.ieval.amap_chat.services:rubric_v2_fuse",
            "mode": "remote", "endpoint_url": "https://biz.example/plugins/fuse" }
        ]
      }
    }

Policy / extractor / stats categories use the **benchmark component name** as ``name``
(dispatch uses service_name); fusion uses ``fusion_fn_ref`` as ``name``. Resolution order
for each category: RemoteConfig binding (remote / class_ref) → in-process ``register_*``
registry → framework default. ``binding.enabled=false`` skips the business registration
and forces the framework default.

Compatibility: when an old payload has only ``enabled`` / ``disabled`` filters, every
plugin stays ``local``.
"""

from typing import Any, Literal, get_args

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field, field_validator

from benchmark.domain.plugin_binding import PluginBinding

logger = get_logger(__name__)

EVAL_PLUGINS_DATA_ID = "agent-eval.eval.plugins"
EVAL_PLUGINS_GROUP = "agent-eval.service.group"
_PLUGIN_TTL_SEC = 10
_WILDCARD = "*"

PluginCategory = Literal[
    "benchmark_components",
    "output_providers",
    "parse_interceptors",
    "report_providers",
    "report_payload_plugins",
    "case_display_plugins",
    "attribution_trigger_policies",
    "dispatch_policies",
    "attribution_stats_providers",
    "trace_identity_extractors",
    "item_trace_identity_extractors",
    "fusion_functions",
    "notification_providers",
    "trace_backends",
]

#: Every plugin category (one-to-one with :class:`EvalPluginsConfig` fields).
PLUGIN_CATEGORIES: tuple[PluginCategory, ...] = get_args(PluginCategory)


class PluginFilter(BaseModel):
    """Enable/disable filter plus optional per-name bindings."""

    model_config = {"extra": "ignore"}

    enabled: list[str] = Field(default_factory=lambda: [_WILDCARD])
    disabled: list[str] = Field(default_factory=list)
    bindings: list[PluginBinding] = Field(default_factory=list)

    @field_validator("enabled", "disabled", mode="before")
    @classmethod
    def _normalize_names(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            name = str(item or "").strip()
            if name:
                out.append(name)
        return out

    def allows(self, name: str) -> bool:
        key = (name or "").strip()
        if not key:
            return False
        disabled = {n.strip() for n in self.disabled}
        if key in disabled:
            return False
        # An explicit binding.enabled=false is also rejected.
        for binding in self.bindings:
            if binding.name == key and not binding.enabled:
                return False
        enabled = [n.strip() for n in self.enabled]
        if not enabled or _WILDCARD in enabled:
            return True
        return key in set(enabled)

    def binding_for(self, name: str) -> PluginBinding | None:
        key = (name or "").strip()
        for binding in self.bindings:
            if binding.name == key:
                return binding
        return None


class EvalPluginsConfig(BaseModel):
    """Top-level RemoteConfig plugin config."""

    model_config = {"extra": "ignore"}

    benchmark_components: PluginFilter = Field(default_factory=PluginFilter)
    output_providers: PluginFilter = Field(default_factory=PluginFilter)
    parse_interceptors: PluginFilter = Field(default_factory=PluginFilter)
    report_providers: PluginFilter = Field(default_factory=PluginFilter)
    report_payload_plugins: PluginFilter = Field(default_factory=PluginFilter)
    case_display_plugins: PluginFilter = Field(default_factory=PluginFilter)
    attribution_trigger_policies: PluginFilter = Field(default_factory=PluginFilter)
    dispatch_policies: PluginFilter = Field(default_factory=PluginFilter)
    attribution_stats_providers: PluginFilter = Field(default_factory=PluginFilter)
    trace_identity_extractors: PluginFilter = Field(default_factory=PluginFilter)
    item_trace_identity_extractors: PluginFilter = Field(default_factory=PluginFilter)
    fusion_functions: PluginFilter = Field(default_factory=PluginFilter)
    notification_providers: PluginFilter = Field(default_factory=PluginFilter)
    trace_backends: PluginFilter = Field(default_factory=PluginFilter)

    def category(self, kind: PluginCategory) -> PluginFilter:
        return getattr(self, kind)


def _parse_config(raw: dict[str, Any] | None) -> EvalPluginsConfig:
    if not raw:
        return EvalPluginsConfig()
    try:
        return EvalPluginsConfig.model_validate(raw)
    except Exception:
        logger.warning("eval_plugins.invalid_remote_config_payload, fallback to allow-all")
        return EvalPluginsConfig()


def fetch_eval_plugins(*, use_cache: bool = True) -> EvalPluginsConfig:
    """Read RemoteConfig plugin config (short TTL). On failure, fall back to all-local enabled."""
# Lazy import breaks the cycle: the ``benchmark.settings`` package also loads the whole
# ``benchmark.services`` tree, and several of those modules import ``eval_plugins.host``,
# which points back here. This module stays a domain leaf (top level depends only on
# plugin_binding) so an entry that imports this module first (ops scripts / config
# self-check) does not hit a partially-initialized package.
    from benchmark.settings.remote_config_store import fetch_cached_settings  # noqa: PLC0415
    from benchmark.settings.setting import settings  # noqa: PLC0415

    raw = fetch_cached_settings(
        settings.eval_plugins_remote_config_data_id,
        settings.eval_plugins_remote_config_group,
        use_cache=use_cache,
        ttl=_PLUGIN_TTL_SEC,
    )
    return _parse_config(raw if isinstance(raw, dict) else None)


def is_benchmark_component_enabled(name: str, *, use_cache: bool = True) -> bool:
    return fetch_eval_plugins(use_cache=use_cache).benchmark_components.allows(name)


def is_output_provider_enabled(name: str, *, use_cache: bool = True) -> bool:
    return fetch_eval_plugins(use_cache=use_cache).output_providers.allows(name)


def resolve_plugin_binding(
    kind: PluginCategory,
    name: str,
    *,
    use_cache: bool = True,
) -> PluginBinding | None:
    """Resolve the binding for ``name`` in a category. ``None`` when there is no explicit binding (default local)."""
    filt = fetch_eval_plugins(use_cache=use_cache).category(kind)
    if not filt.allows(name):
        return PluginBinding(name=name, mode="local", enabled=False)
    return filt.binding_for(name)


def filter_enabled_names(
    names: list[str],
    *,
    kind: PluginCategory,
    use_cache: bool = True,
) -> list[str]:
    filt = fetch_eval_plugins(use_cache=use_cache).category(kind)
    return [n for n in names if filt.allows(n)]


__all__ = [
    "EVAL_PLUGINS_DATA_ID",
    "EVAL_PLUGINS_GROUP",
    "PLUGIN_CATEGORIES",
    "EvalPluginsConfig",
    "PluginCategory",
    "PluginFilter",
    "fetch_eval_plugins",
    "filter_enabled_names",
    "is_benchmark_component_enabled",
    "is_output_provider_enabled",
    "resolve_plugin_binding",
]
