"""Unified plugin host: resolution (RemoteConfig binding → registry → default) and async calls.

Business extension-point resolution order (high → low):

1. RemoteConfig ``eval.plugins.<category>.bindings`` (remote / class_ref)
2. In-process ``register_*`` registry
3. Framework default implementation

Each registry module (trigger policy / dispatch policy / attribution stats /
trace identity / fusion) delegates ``resolve_*`` to :func:`resolve_plugin`
and only supplies its own registry, default, and remote-adapter factory.

Policy / extractor protocols are themselves synchronous methods (local implementations are pure computation). A remote adapter also
provides ``a<method>`` / ``acall`` async variants; async call sites pick them automatically via :func:`call_plugin` /
:func:`call_plugin_callable`, so remote HTTP does not block the event loop.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.eval_plugin_config import (
    PLUGIN_CATEGORIES,
    PluginCategory,
    resolve_plugin_binding,
)
from benchmark.domain.plugin_binding import PluginBinding
from benchmark.plugins.class_loader import (
    PluginClassLoadError,
    instantiate_plugin,
)

logger = get_logger(__name__)

T = TypeVar("T")


# ── resolve ──────────────────────────────────────────────────────────────────


def _instantiate_class_ref(binding: PluginBinding, *, category: str) -> object | None:
    """Load ``class_ref``; on failure ``fallback_local`` decides whether to swallow the error and fall back or to raise."""
    try:
        return instantiate_plugin(binding.class_ref or "", what=category)
    except PluginClassLoadError as exc:
        logger.warning(
            "plugin.class_ref_failed category=%s name=%s err=%s fallback=%s",
            category,
            binding.name,
            exc,
            binding.fallback_local,
        )
        if binding.fallback_local:
            return None
        raise


@dataclass(frozen=True)
class PluginResolution(Generic[T]):
    """Fallback sources for plugin resolution (the argument object of :func:`resolve_plugin`).

    Attributes:
        registry: in-process registry (priority 2).
        default: default implementation (lowest priority; also returned when RemoteConfig explicitly disables the plugin).
        remote_factory: remote-adapter factory for a ``mode=remote`` binding.
    """

    registry: Mapping[str, T]
    default: T
    remote_factory: Callable[[PluginBinding], T]


def resolve_plugin(
    category: PluginCategory,
    name: str | None,
    *,
    resolution: PluginResolution[T],
) -> T:
    """RemoteConfig binding → in-process registry → default implementation.

    - ``binding.enabled=false``: RemoteConfig explicitly disables the business implementation → return ``default`` directly;
    - ``mode=remote`` with an ``endpoint_url`` → ``remote_factory(binding)``;
    - ``mode=local`` with a ``class_ref`` → load the business class / function without a release;
    - otherwise (no binding / class_ref fallback) → registry hit → default.
    """
    key = (name or "").strip()
    if not key:
        return resolution.default
    binding = resolve_plugin_binding(category, key)
    if binding is not None:
        if not binding.enabled:
            logger.info("plugin.disabled_by_remote_config category=%s name=%s", category, key)
            return resolution.default
        if binding.is_remote_active():
            return resolution.remote_factory(binding)
        if binding.class_ref:
            loaded = _instantiate_class_ref(binding, category=category)
            if loaded is not None:
                return loaded  # type: ignore[return-value]
    registered = resolution.registry.get(key)
    return registered if registered is not None else resolution.default


def is_plugin_disabled(category: PluginCategory, name: str | None) -> bool:
    """Whether RemoteConfig explicitly disabled this business plugin."""
    key = (name or "").strip()
    if not key:
        return False
    binding = resolve_plugin_binding(category, key)
    return binding is not None and not binding.enabled


def list_plugin_categories() -> tuple[PluginCategory, ...]:
    """For docs / discovery: every plugin category."""
    return PLUGIN_CATEGORIES


# ── async calls ──────────────────────────────────────────────────────────────


async def call_plugin(plugin: object, method: str, /, *args: object, **kwargs: object) -> object:
    """Call ``plugin.<method>``; if a ``plugin.a<method>`` coroutine exists, await that instead."""
    async_variant = getattr(plugin, f"a{method}", None)
    if async_variant is not None:
        return await async_variant(*args, **kwargs)
    return getattr(plugin, method)(*args, **kwargs)


async def call_plugin_callable(plugin: Callable[..., T], /, *args: object, **kwargs: object) -> T:
    """Call a callable plugin (function / ``__call__``); when the remote adapter provides ``acall``, await that."""
    async_variant = getattr(plugin, "acall", None)
    if async_variant is not None:
        return await async_variant(*args, **kwargs)
    return plugin(*args, **kwargs)


__all__ = [
    "PluginResolution",
    "call_plugin",
    "call_plugin_callable",
    "is_plugin_disabled",
    "list_plugin_categories",
    "resolve_plugin",
]
