"""Loading of builtin case-display plugins.

Registered business: trace (``aone_source=trace``). Other unregistered names use default.
trace does not bind an items exact key, so a batch ``output.result`` shape is not treated as displayText.
"""

from benchmark.runs.case_display.default_plugin import DefaultCaseDisplayPlugin
from benchmark.runs.case_display.plugins.trace import TraceCaseDisplayPlugin
from benchmark.runs.case_display.registry import (
    builtins_loaded,
    clear_case_display_plugins,
    mark_builtins_loaded,
    register_case_display_plugin,
)


def ensure_builtin_plugins() -> None:
    """Register builtin plugins idempotently."""
    if builtins_loaded():
        return
    register_case_display_plugin(TraceCaseDisplayPlugin())
    register_case_display_plugin(DefaultCaseDisplayPlugin(), as_default=True)
    mark_builtins_loaded()


def reset_builtin_plugins() -> None:
    """For tests: clear plugins and allow them to be loaded again."""
    clear_case_display_plugins()


__all__ = [
    "DefaultCaseDisplayPlugin",
    "TraceCaseDisplayPlugin",
    "ensure_builtin_plugins",
    "reset_builtin_plugins",
]
