"""Attribution-stats plugin registry — register a report attribution-stats provider by benchmark component name (§8).

Background: the report's "attribution analysis" region historically had only the ieval
angle (the ``primary_issue_category`` distribution of ``attribution_detail_json``).
Businesses differ in data source and angle (ieval = distribution of Aone attribution
callback fields; search_eval = distribution of unsatisfied dimensions in the six-axis
detail embedded in the judge). This registry turns that logic into plugins:

- a business registers its own :class:`AttributionStatsProvider` from the component
  module (import side effect);
- RemoteConfig ``eval.plugins.attribution_stats_providers.bindings`` can switch a
  component to ``remote`` / ``class_ref``, which outranks the registry; ``enabled=false``
  forces the default path;
- the aggregation layer (``_aggregate_external_metrics``) resolves by the run's benchmark
  name: registered → call the provider; unregistered → fall back to the live default
  (``BenchmarkTraceRecordWriter.aggregate_attribution``, the original ieval path);
- an unregistered business behaves exactly as it does in production today.
"""

import importlib
from typing import Any, Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.plugins.host import PluginResolution, is_plugin_disabled, resolve_plugin
from benchmark.plugins.remote_adapters import RemoteAttributionStatsProvider

logger = get_logger("agent_eval.benchmark.common.attribution_stats")

_CATEGORY = "attribution_stats_providers"


@runtime_checkable
class AttributionStatsProvider(Protocol):
    """Attribution-stats provider protocol (registered by benchmark component name).

    The return shape matches what the report consumes:
    ``{key: {field_distributions: {field: {value: count}},
    value_labels: {field: {value: label}}, total_attributed: int}}``.
    """

    async def collect(self, run_id: str) -> dict[str, dict[str, Any]]:
        """Collect attribution stats for this run (an empty dict means the report does not render the attribution
        region).
        """
        ...


#: benchmark component name → attribution-stats provider (unregistered = the live default path).
_ATTRIBUTION_STATS_REGISTRY: dict[str, AttributionStatsProvider] = {}


def register_attribution_stats_provider(
    benchmark_name: str,
    provider: AttributionStatsProvider,
) -> None:
    """A business registers its own attribution-stats provider (same name overwrites; idempotent)."""
    _ATTRIBUTION_STATS_REGISTRY[benchmark_name] = provider
    logger.info(
        "attribution_stats.registered benchmark=%s provider=%s",
        benchmark_name,
        type(provider).__name__,
    )


#: When unregistered, try to restore the component's side-effect registration (aligned with a regenerate cold start).
_LAZY_ENSURE_BY_BENCHMARK: dict[str, str] = {
    "search_eval": "business.runs.search_eval.attribution_stats",
}


def resolve_attribution_stats_provider(
    benchmark_name: str | None,
) -> AttributionStatsProvider | None:
    """Resolve a provider by benchmark component name: RemoteConfig binding → registry → lazy load → None.

    ``None`` means fall back to the default path. If a known business (for example search_eval)
    has an empty registry entry, lazy-load and rerun the component's
    ``register_attribution_stats_provider`` (``importlib.reload``). That avoids a long-lived
    process whose registry was cleared or never mounted showing a blank attribution region
    while regenerate (``create_app``) still works. Do not lazy-load when RemoteConfig
    explicitly disables the provider.
    """
    if not benchmark_name:
        return None
    provider = resolve_plugin(
        _CATEGORY,
        benchmark_name,
        resolution=PluginResolution(
            registry=_ATTRIBUTION_STATS_REGISTRY,
            default=None,
            remote_factory=RemoteAttributionStatsProvider,
        ),
    )
    if provider is not None or is_plugin_disabled(_CATEGORY, benchmark_name):
        return provider
    return _lazy_ensure_registered(benchmark_name)


def _lazy_ensure_registered(benchmark_name: str) -> AttributionStatsProvider | None:
    module_path = _LAZY_ENSURE_BY_BENCHMARK.get(benchmark_name)
    if not module_path:
        return None
    try:
        module = importlib.import_module(module_path)
        # The module may already be in the process; reload so the register side effect runs again
        importlib.reload(module)
    except Exception:
        logger.warning(
            "attribution_stats.lazy_ensure_failed benchmark=%s module=%s",
            benchmark_name,
            module_path,
            exc_info=True,
        )
        return None
    return _ATTRIBUTION_STATS_REGISTRY.get(benchmark_name)


def clear_attribution_stats_providers() -> None:
    """Clear the registry (for test isolation)."""
    _ATTRIBUTION_STATS_REGISTRY.clear()


__all__ = [
    "AttributionStatsProvider",
    "clear_attribution_stats_providers",
    "register_attribution_stats_provider",
    "resolve_attribution_stats_provider",
]
