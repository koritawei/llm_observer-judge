"""Case-list display plugins: fold DB / report payloads into views the WebUI can render.

Quick business integration (no ``run_case_page`` edit required)
==============================================================

1. Add a plugin class implementing :class:`CaseDisplayPlugin` (``matches`` + ``normalize``)
2. At process startup::

       register_case_display_plugin(MyPlugin(), keys=("my_benchmark",))

   or attach ``class_ref`` on RemoteConfig ``eval.plugins.case_display_plugins.bindings``
3. Optionally raise ``priority`` so ``matches`` wins the probe when there is no exact key

The orchestration entry is still :func:`normalize_case_display` (called by ``run_case_page``).
This package loads builtin plugins before resolution, breaking the ``registry`` ↔ ``plugins`` cycle.
"""

from collections.abc import Mapping

from benchmark.runs.case_display.plugins import ensure_builtin_plugins
from benchmark.runs.case_display.registry import (
    CaseDisplayPlugin,
    clear_case_display_plugins,
    register_case_display_plugin,
)
from benchmark.runs.case_display.registry import (
    resolve_case_display_plugin as _resolve_registered_plugin,
)
from benchmark.runs.case_display.views import (
    CaseAttributionView,
    CaseDisplayContext,
    CaseDisplayRequest,
    CaseDisplayResult,
    CaseJudgeDimension,
    CaseJudgeView,
)


def resolve_case_display_plugin(request: CaseDisplayRequest) -> CaseDisplayPlugin:
    """Public resolution entry: load builtins first, then consult the registry."""
    ensure_builtin_plugins()
    return _resolve_registered_plugin(request)


def normalize_case_display(
    case: Mapping[str, object],
    ctx: CaseDisplayContext,
) -> tuple[str, CaseJudgeView | None, CaseAttributionView | None]:
    """Return ``(output_markdown, judge_view, attribution_view)``."""
    request = CaseDisplayRequest(case=case, context=ctx)
    plugin = resolve_case_display_plugin(request)
    result = plugin.normalize(request)
    return result.output, result.judge, result.attribution


__all__ = [
    "CaseAttributionView",
    "CaseDisplayContext",
    "CaseDisplayPlugin",
    "CaseDisplayRequest",
    "CaseDisplayResult",
    "CaseJudgeDimension",
    "CaseJudgeView",
    "clear_case_display_plugins",
    "normalize_case_display",
    "register_case_display_plugin",
    "resolve_case_display_plugin",
]
