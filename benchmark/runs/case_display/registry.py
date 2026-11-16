"""Case-display plugin registry and resolution.

Ways to extend a business (pick one):

1. **Exact key**: ``register_case_display_plugin(plugin, keys=("search_eval_v2",))``
   — used as soon as ``benchmark_name`` hits.
2. **matches probe**: the plugin implements ``matches(request)`` and is chosen by ``priority`` descending.
3. **RemoteConfig** ``case_display_plugins`` binding (``class_ref`` / remote) —
   the same resolution chain as the other eval plugins.

Resolution order: streaming is fixed to trace → RemoteConfig / exact key → matches(priority) → default.

Builtin plugins are loaded by the orchestration layer (the ``case_display`` package entry)
via ``ensure_builtin_plugins``. The default implementation lives in the sibling ``default_plugin``
module (not the ``plugins`` package), which avoids a ``registry`` ↔ ``plugins`` cycle and a
lazy import inside a function.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.plugin_binding import PluginBinding
from benchmark.runs.case_display.default_plugin import DefaultCaseDisplayPlugin
from benchmark.runs.case_display.views import CaseDisplayRequest, CaseDisplayResult
from benchmark.plugins.host import PluginResolution, resolve_plugin

logger = get_logger(__name__)

_CATEGORY = "case_display_plugins"
_NOTIFICATION_SOURCE_TRACE = "trace"


class RemoteCaseDisplayPlugin:
    """RemoteConfig ``mode=remote`` adapter: v1 falls back to the local default.

    Lives in the registry so it does not take another business-file slot; a real remote
    plugin should use ``mode=local`` + ``class_ref``.
    """

    name = "remote_case_display"
    priority = 0

    def __init__(self, binding: PluginBinding) -> None:
        self._binding = binding
        self.name = binding.name or self.name

    def matches(self, request: CaseDisplayRequest) -> bool:
        del request
        return True

    def normalize(self, request: CaseDisplayRequest) -> CaseDisplayResult:
        logger.warning(
            "case_display.remote_fallback name=%s endpoint=%s",
            self._binding.name,
            self._binding.endpoint_url,
        )
        return DefaultCaseDisplayPlugin().normalize(request)


@runtime_checkable
class CaseDisplayPlugin(Protocol):
    """Case-display plugin for one business.

    To add a business: implement this protocol → ``register_case_display_plugin`` (or a RemoteConfig class_ref).
    The orchestration layer / WebUI does not need to change.
    """

    name: str
    #: Priority during probe matching; higher wins. An exact key hit ignores priority.
    priority: int

    def matches(self, request: CaseDisplayRequest) -> bool:
        """Whether this plugin claims the case (probe runs when the exact key missed)."""
        ...

    def normalize(self, request: CaseDisplayRequest) -> CaseDisplayResult:
        """Produce the three-column view for the WebUI."""
        ...


@dataclass(slots=True)
class _RegistryState:
    exact: dict[str, CaseDisplayPlugin] = field(default_factory=dict)
    matchers: list[CaseDisplayPlugin] = field(default_factory=list)
    default: CaseDisplayPlugin | None = None
    builtins_loaded: bool = False


_STATE = _RegistryState()


def register_case_display_plugin(
    plugin: CaseDisplayPlugin,
    *,
    keys: Sequence[str] = (),
    as_default: bool = False,
) -> CaseDisplayPlugin:
    """Register a plugin. ``keys`` bind an exact ``benchmark_name`` hit; ``as_default`` makes it the fallback."""
    for raw in keys:
        key = (raw or "").strip()
        if not key:
            continue
        _STATE.exact[key] = plugin
    if plugin not in _STATE.matchers:
        _STATE.matchers.append(plugin)
        _STATE.matchers.sort(key=lambda item: (-int(item.priority), item.name))
    if as_default:
        _STATE.default = plugin
    logger.info(
        "case_display.registered name=%s keys=%s priority=%s default=%s",
        plugin.name,
        list(keys),
        plugin.priority,
        as_default,
    )
    return plugin


def clear_case_display_plugins() -> None:
    """For tests: clear the registry (call ``ensure_builtin_plugins`` again afterwards)."""
    _STATE.exact.clear()
    _STATE.matchers.clear()
    _STATE.default = None
    _STATE.builtins_loaded = False


def mark_builtins_loaded() -> None:
    """Mark builtin plugin loading as finished."""
    _STATE.builtins_loaded = True


def builtins_loaded() -> bool:
    return _STATE.builtins_loaded


def _remote_factory(binding: object) -> CaseDisplayPlugin:
    return RemoteCaseDisplayPlugin(binding)  # type: ignore[arg-type]


def _default_plugin() -> CaseDisplayPlugin:
    if _STATE.default is not None:
        return _STATE.default
    return DefaultCaseDisplayPlugin()


def _match_plugin(
    request: CaseDisplayRequest,
    default: CaseDisplayPlugin,
) -> CaseDisplayPlugin:
    for plugin in _STATE.matchers:
        # The default plugin has priority=0 and matches always; skip it while probing and keep it as the final fallback.
        if plugin is _STATE.default or plugin.priority <= 0:
            continue
        if plugin.matches(request):
            return plugin
    return default


def resolve_case_display_plugin(request: CaseDisplayRequest) -> CaseDisplayPlugin:
    """RemoteConfig → exact key → matches(priority) → default.

    The caller must load builtin plugins first (the package entry ``normalize_case_display`` /
    ``resolve_case_display_plugin`` wrappers call ``ensure_builtin_plugins``).
    """
    default = _default_plugin()
    ctx = request.context
    if ctx.notification_source == _NOTIFICATION_SOURCE_TRACE:
        for plugin in _STATE.matchers:
            if plugin.matches(request) and plugin is not _STATE.default:
                # The trace plugin has the highest priority and matches only aone_source=trace.
                return plugin
        return default

    matched = _match_plugin(request, default)
    key = (ctx.benchmark_name or "").strip()
    if not key:
        return matched
    return resolve_plugin(
        _CATEGORY,
        key,
        resolution=PluginResolution(
            registry=_STATE.exact,
            default=matched,
            remote_factory=_remote_factory,
        ),
    )


__all__ = [
    "CaseDisplayPlugin",
    "builtins_loaded",
    "clear_case_display_plugins",
    "mark_builtins_loaded",
    "register_case_display_plugin",
    "resolve_case_display_plugin",
]
