"""Code attribution. TaskPack in, ResultPack out. Display replacement is done by AnalyzeAndApply."""

from benchmark.attribution.code.models import ModuleConfig, ResultPack, RunTimings, TaskPack
from benchmark.attribution.code.pipeline import (
    AnalyzeAndApply,
    CodeAttributionCase,
    CodeAttributionOutcome,
)
from benchmark.attribution.code.runner import (
    CodeAttributionModule,
    module_config_from_settings,
    run,
)

__all__ = [
    "AnalyzeAndApply",
    "CodeAttributionCase",
    "CodeAttributionModule",
    "CodeAttributionOutcome",
    "ModuleConfig",
    "ResultPack",
    "RunTimings",
    "TaskPack",
    "module_config_from_settings",
    "run",
]
