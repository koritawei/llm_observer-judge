"""Assemblable entry for code attribution. Analyze and replace display content. Later stages attach after ResultPack."""

from dataclasses import asdict, dataclass
from typing import Any

from benchmark.attribution.code.document import TaskSource, apply_code_attribution, build_task
from benchmark.attribution.code.runner import (
    CodeAttributionModule,
    module_config_from_settings,
)


@dataclass(frozen=True)
class CodeAttributionCase:
    """Input for one code-attribution run. document is the attribution document to replace."""

    case_id: str
    query: str
    tool_calls_json: str
    document: dict[str, Any]
    llm_calls_json: str = ""
    last_generation_input: str = ""
    response_text: str = ""


@dataclass(frozen=True)
class CodeAttributionOutcome:
    """When applied is false, document matches the input."""

    applied: bool
    document: dict[str, Any]


class AnalyzeAndApply:
    """Zero-arg assemblable class: build a TaskPack, run analysis, and replace display fields on success."""

    def __init__(self, module: CodeAttributionModule | None = None) -> None:
        self._module = (
            module if module is not None else CodeAttributionModule(module_config_from_settings())
        )

    def run(self, case: CodeAttributionCase) -> CodeAttributionOutcome:
        document = dict(case.document)
        raw = document.get("raw_json")
        if isinstance(raw, dict):
            document["raw_json"] = dict(raw)
        task = build_task(
            document,
            TaskSource(
                case_id=case.case_id,
                query=case.query,
                tool_calls_json=case.tool_calls_json,
                llm_calls_json=case.llm_calls_json,
                last_generation_input=case.last_generation_input,
                response_text=case.response_text,
            ),
        )
        result = self._module.run(task)
        applied = apply_code_attribution(document, asdict(result))
        return CodeAttributionOutcome(applied=applied, document=document)
