"""Display plugin for streaming (trace_pipeline_records)."""

from collections.abc import Mapping

from benchmark.runs.case_display.helpers import (
    as_mapping,
    as_text,
    attribution_table,
    component_display_text,
    empty_attribution,
    empty_judge,
    judge_from_extra_top,
    raw_attribution_blob,
    raw_judge_blob,
)
from benchmark.runs.case_display.views import (
    CaseAttributionView,
    CaseDisplayRequest,
    CaseDisplayResult,
    CaseJudgeView,
)

_NOTIFICATION_SOURCE_TRACE = "trace"


class TraceCaseDisplayPlugin:
    """amap_agent_benchmark streaming: displayText + extra-dimension judge + table attribution."""

    name = "trace"
    priority = 100

    def matches(self, request: CaseDisplayRequest) -> bool:
        # Claim only the streaming source; never claim the items side (those use default / other business plugins).
        return request.context.notification_source == _NOTIFICATION_SOURCE_TRACE

    def normalize(self, request: CaseDisplayRequest) -> CaseDisplayResult:
        case = request.case
        return CaseDisplayResult(
            output=_trace_output(case),
            judge=_trace_judge(case),
            attribution=_trace_attribution(case),
        )


def _trace_output(case: Mapping[str, object]) -> str:
    for root_key in ("trace_meta_json", "trace_meta", "output"):
        root = case.get(root_key)
        if isinstance(root, str):
            text = root.strip()
            if text:
                return text
            continue
        blob = as_mapping(root)
        if blob is None:
            continue
        output_map = blob if root_key == "output" else as_mapping(blob.get("output"))
        if output_map is None:
            continue
        text = component_display_text(output_map)
        if text:
            return text
    return as_text(case.get("display_output"))


def _trace_judge(case: Mapping[str, object]) -> CaseJudgeView | None:
    blob = raw_judge_blob(case)
    if blob is None:
        return None
    view = judge_from_extra_top(blob)
    return None if empty_judge(view) else view


def _trace_attribution(case: Mapping[str, object]) -> CaseAttributionView | None:
    blob = raw_attribution_blob(case)
    if blob is None:
        return None
    view = attribution_table(blob)
    return None if empty_attribution(view) else view


__all__ = ["TraceCaseDisplayPlugin"]
