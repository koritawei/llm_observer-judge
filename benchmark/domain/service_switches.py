"""Service switches — loaded from the RemoteConfig center and checked live on each action.

Service-level switches, including the benchmark / dataset processing switches
(**two independent switches**), are managed in RemoteConfig (not frozen into Settings
at startup). Each processing action reads the latest state.

RemoteConfig JSON (dataId ``agent-eval.service.switches``)::

    {
        "label_studio_enabled": false,
        "benchmark_disabled": false,
        "dataset_disabled": false
    }

Fallback chain (**RemoteConfig > env var > default True**)::

    1. RemoteConfig has ``benchmark_disabled`` / ``dataset_disabled`` → use it;
    2. Field missing / RemoteConfig unavailable → env vars
       ``AGENT_EVAL_BENCHMARK_DISABLED`` / ``AGENT_EVAL_DATASET_DISABLED``;
    3. Neither is set → ``True`` (do not process new benchmark or dataset data).

``is_benchmark_disabled`` / ``is_dataset_disabled`` are called on every processing
action (create-style HTTP guards, daily schedule triggers, and so on). They go through
the short TTL cache of ``fetch_cached_settings`` (``_SWITCH_TTL_SEC``) to balance
freshness against RemoteConfig network cost, and they do not use the 300s parsed cache
in ``benchmark.domain._cache`` (switches must take effect quickly).

Effect: when a switch is on (=true, the default), "tasks that have not started are all
rejected" (create endpoints 503 / schedule skipped). A run already created, and the jobs
already in its queue, are "tasks already running" and finish unaffected. The benchmark
and dataset switches are independent.

Old config is not compatible: the historical combined field ``benchmark_dataset_disabled``
and the env var ``AGENT_EVAL_BENCHMARK_DATASET_DISABLED`` are never read. A RemoteConfig
payload that has only the old field is treated as unset and takes the default ``True``
(do not process). Publish the new fields ``benchmark_disabled`` / ``dataset_disabled``.

Usage::

    from benchmark.domain.service_switches import is_benchmark_disabled

    if is_benchmark_disabled():
        ...  # reject processing
"""

from collections.abc import Callable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.settings.setting import (
    is_benchmark_disabled_runtime,
    is_dataset_disabled_runtime,
    settings,
)

logger = get_logger(__name__)

#: RemoteConfig coordinates for service switches (historical defaults; overridable via settings env).
SERVICE_SWITCHES_DATA_ID = "agent-eval.service.switches"
SERVICE_SWITCHES_GROUP = "agent-eval.service.group"

#: RemoteConfig read TTL (seconds) for switch checks. Under per-action live detection,
#: a short TTL balances how fast a switch takes effect against RemoteConfig network cost.
_SWITCH_TTL_SEC = 10

#: Field names of the benchmark / dataset processing switches in RemoteConfig (two switches).
_BENCHMARK_DISABLED_KEY = "benchmark_disabled"
_DATASET_DISABLED_KEY = "dataset_disabled"


def _to_bool(value: object) -> bool:
    """Lenient bool parse: a bool is returned as-is; strings match true/1/yes (else False)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _is_disabled(
    key: str,
    env_fallback: Callable[[], bool],
    *,
    use_cache: bool,
) -> bool:
    """Shared switch path: RemoteConfig field → env-var fallback (default True)."""
    raw = fetch_cached_settings(
        settings.service_switches_remote_config_data_id,
        settings.service_switches_remote_config_group,
        use_cache=use_cache,
        ttl=_SWITCH_TTL_SEC,
    )
    if raw and key in raw:
        return _to_bool(raw[key])
    # RemoteConfig unavailable / field missing → env-var fallback (unset defaults to True).
    return env_fallback()


def is_benchmark_disabled(*, use_cache: bool = True) -> bool:
    """Current benchmark processing switch (checked live on each action).

    Fallback: RemoteConfig ``benchmark_disabled`` → env var
    ``AGENT_EVAL_BENCHMARK_DISABLED`` → ``True`` (do not process by default).

    Args:
        use_cache: Whether to use the ``_SWITCH_TTL_SEC`` TTL cache, default ``True``.
            ``False`` forces a direct RemoteConfig refresh (tests / forced refresh).
    """
    return _is_disabled(
        _BENCHMARK_DISABLED_KEY,
        is_benchmark_disabled_runtime,
        use_cache=use_cache,
    )


def is_dataset_disabled(*, use_cache: bool = True) -> bool:
    """Current dataset processing switch (checked live on each action).

    Fallback: RemoteConfig ``dataset_disabled`` → env var
    ``AGENT_EVAL_DATASET_DISABLED`` → ``True`` (do not process by default).

    Args:
        use_cache: Whether to use the ``_SWITCH_TTL_SEC`` TTL cache, default ``True``.
            ``False`` forces a direct RemoteConfig refresh (tests / forced refresh).
    """
    return _is_disabled(
        _DATASET_DISABLED_KEY,
        is_dataset_disabled_runtime,
        use_cache=use_cache,
    )


__all__ = [
    "SERVICE_SWITCHES_DATA_ID",
    "SERVICE_SWITCHES_GROUP",
    "is_benchmark_disabled",
    "is_dataset_disabled",
]
