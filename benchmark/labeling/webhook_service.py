"""LS webhook 回调处理 — 从合并 annotation 同时写两个 bucket。"""

from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.labeling.project_resolver import ProjectResolver
from benchmark.services.trace_record_service import TraceRecordService

logger = get_logger("agent_eval.label_studio.webhook_service")


class LabelStudioWebhookService:
    def __init__(
        self,
        *,
        project_resolver: ProjectResolver,
        trace_record_service: TraceRecordService,
    ) -> None:
        self._project_resolver = project_resolver
        self._trace_record_service = trace_record_service

    async def handle_annotation(
        self,
        annotation: dict,
        project: dict | None,
        task: dict | None = None,
    ) -> dict:
        pid = (project or {}).get("id")
        if pid is None:
            return {
                "status": "invalid",
                "reason": "missing project id",
                "_http_status": 422,
            }
        if not self._project_resolver.is_known_project(pid):
            return {
                "status": "ignored",
                "reason": "unknown project",
                "_http_status": 200,
            }

        task_obj = task or {}
        task_data = task_obj.get("data") or {}

        trace_record_id = task_data.get("trace_record_id")
        trace_id = task_data.get("trace_id") or ""

        if not trace_record_id:
            return {
                "status": "invalid",
                "reason": "missing trace_record_id in task_data",
                "_http_status": 422,
            }

        judge_data, attribution_data = self._parse_combined_annotation(annotation)

        record = await self._trace_record_service.update_ls_annotation(
            record_id=trace_record_id,
            judge_annotation_data=judge_data,
            attribution_annotation_data=attribution_data,
        )
        if record is None:
            return {"status": "not_found", "_http_status": 404}
        return {"status": "ok", "caseId": trace_id}

    @staticmethod
    def _parse_combined_annotation(annotation: dict) -> tuple[dict, dict]:
        """从合并 annotation 中分别解析 judge 修正和归因修正。

        LS annotation.result 是 list 类型，但 TraceRecord Validator
        要求 triage_result / attribution_result 为 dict。包装为
        ``{"raw": result}`` 以满足 ``_check_dict`` 约束。
        """
        result = annotation.get("result", [])
        completed_by = annotation.get("completed_by", "")
        annotator = str(
            completed_by.get("email") or completed_by.get("id")
            if isinstance(completed_by, dict)
            else completed_by
        )
        annotated_at = annotation.get("updated_at") or annotation.get("created_at")
        parsed = _extract_annotation_fields(result)

        judge_data: dict[str, Any] = {
            "triage_result": {"raw": result},
            "human_judge_score_correct": parsed.get("judge_score_correct"),
            "human_corrected_score": parsed.get("corrected_score"),
            "human_corrected_reason": parsed.get("corrected_reason"),
            "human_triage_level_correct": parsed.get("triage_level_correct"),
            "human_corrected_triage_level": parsed.get("corrected_triage_level"),
            "triage_annotator": annotator,
            "triage_annotated_at": annotated_at,
        }
        attribution_data: dict[str, Any] = {
            "attribution_result": {"raw": result},
            "human_primary_cause": parsed.get("primary_cause"),
            "human_owner_group": parsed.get("owner_group"),
            "human_attribution_correct": parsed.get("machine_attribution_correct"),
            "annotator": annotator,
            "annotated_at": annotated_at,
        }
        return (judge_data, attribution_data)


def _extract_annotation_fields(result: list[dict]) -> dict[str, str | None]:
    """从 LS annotation result list 中按 from_name 提取标注员填写的值。

    LS annotation result 结构:
    ``[{from_name, to_name, type, value: {choices|text|...}}, ...]``。
    按 ``from_name`` 查找并提取 ``value`` 中的 choices 或 text。
    """
    lookup: dict[str, dict] = {}
    for item in result:
        name = item.get("from_name", "")
        if name:
            lookup[name] = item

    def _choice(name: str) -> str | None:
        item = lookup.get(name)
        if not item:
            return None
        choices = item.get("value", {}).get("choices", [])
        return choices[0] if choices else None

    def _text(name: str) -> str | None:
        item = lookup.get(name)
        if not item:
            return None
        texts = item.get("value", {}).get("text", [])
        return texts[0] if texts else None

    fields: dict[str, str | None] = {}
    fields["judge_score_correct"] = _choice("judge_score_correct")
    fields["corrected_score"] = _text("corrected_score")
    fields["corrected_reason"] = _text("corrected_reason")
    fields["triage_level_correct"] = _choice("triage_level_correct")
    fields["corrected_triage_level"] = _choice("corrected_triage_level")
    fields["machine_attribution_correct"] = _choice("attribution_correct")
    fields["primary_cause"] = _text("corrected_root_cause")
    fields["owner_group"] = _text("corrected_owner_group")
    return fields
