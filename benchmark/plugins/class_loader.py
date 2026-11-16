"""Load a local plugin reference (``ref = module:Name``).

An allowlist prefix restricts business code to agreed packages inside agent-eval, avoiding arbitrary imports.
``Name`` may be a class (used after instantiation) or a function (for example ``fusion_fn_ref``, used directly).
"""

import importlib

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger(__name__)

#: Allowed ref module prefixes
ALLOWED_PLUGIN_MODULE_PREFIXES: tuple[str, ...] = (
    "benchmark.",
    "business.",
)


class PluginClassLoadError(ValueError):
    """The ref is illegal, not on the allowlist, failed to import, or has the wrong shape."""


def _split_ref(ref: str, *, what: str) -> tuple[str, str]:
    text = (ref or "").strip()
    module_path, sep, attr_name = text.partition(":")
    module_path, attr_name = module_path.strip(), attr_name.strip()
    if not sep or not module_path or not attr_name:
        raise PluginClassLoadError(f"{what} must be 'module:Name', got {ref!r}")
    if not module_path.startswith(ALLOWED_PLUGIN_MODULE_PREFIXES):
        raise PluginClassLoadError(f"{what} module not in whitelist: {module_path!r}")
    return module_path, attr_name


def load_plugin_object(ref: str, *, what: str = "class_ref") -> object:
    """Load the object (class or function) named by ``module:Name``; raise :class:`PluginClassLoadError` on failure."""
    module_path, attr_name = _split_ref(ref, what=what)
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise PluginClassLoadError(f"{what} import failed: {module_path!r}: {exc}") from exc
    obj = getattr(module, attr_name, None)
    if obj is None:
        raise PluginClassLoadError(f"{what} attribute not found: {module_path}:{attr_name}")
    return obj


def load_plugin_class(ref: str, *, what: str = "class_ref") -> type:
    """Load ``module:Class``; raise :class:`PluginClassLoadError` when the target is not a class."""
    obj = load_plugin_object(ref, what=what)
    if not isinstance(obj, type):
        raise PluginClassLoadError(f"{what} is not a class: {ref!r}")
    return obj


def instantiate_plugin(ref: str, *, what: str = "class_ref") -> object:
    """Class → instantiate with no args; function → return as-is; any other shape raises."""
    obj = load_plugin_object(ref, what=what)
    if isinstance(obj, type):
        try:
            return obj()
        except Exception as exc:
            raise PluginClassLoadError(f"{what} instantiation failed: {ref!r}: {exc}") from exc
    if callable(obj):
        return obj
    raise PluginClassLoadError(f"{what} is neither a class nor callable: {ref!r}")


__all__ = [
    "ALLOWED_PLUGIN_MODULE_PREFIXES",
    "PluginClassLoadError",
    "instantiate_plugin",
    "load_plugin_class",
    "load_plugin_object",
]
