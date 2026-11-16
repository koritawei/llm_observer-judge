"""Case review 直推 Dispatcher — 归因完成后一次性下发到 Label Studio。"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.attribution.badcase.schemas import AttributionResult
from benchmark.labeling.client import LabelStudioClient
from benchmark.labeling.constants import JSON_PREVIEW_MAX_CHARS, STAGE_BADCASE_REVIEW
from benchmark.labeling.context import get_triage_judge_context
from benchmark.labeling.project_resolver import ProjectIdentity, ProjectResolver
from benchmark.services.trace_record_service import TraceRecordService

logger = get_logger("agent_eval.label_studio.dispatcher")


@dataclass(frozen=True)
class CaseReviewPayload:
    """Case review 推送到 LS 的合并 payload 数据。"""

    trace_id: str
    project_id: str
    agent_name: str
    evaluator_name: str
    trace_record_id: str
    evaluator_request: Any
    attribution: AttributionResult
    judge_score: str
    judge_reason: str
    triage_level: str
    extra: str


@dataclass(frozen=True)
class _PushContext:
    """_push 的执行上下文（payload 为已构建的 LS task payload）。"""

    trace_id: str
    project_id: str
    agent_name: str
    record_id: str
    payload: dict[str, Any]
    dataset_name: str | None = None
    run_name: str | None = None


class LabelStudioDispatcher:
    """单次直推模式：归因完成后构造合并 payload → import task。

    合并 payload 包含:
      - judge 评分: score, reason（从 ContextVar 读取，零 DB 查询）
      - triage 分级: triage_level（从 ContextVar 读取）
      - 归因结果: root_cause, category, confidence, reasoning（从 attribution 参数）
      - agent 上下文: input/output 摘要（从 evaluator_request 提取）
    """

    def __init__(
        self,
        *,
        client: LabelStudioClient,
        project_resolver: ProjectResolver,
        trace_record_service: TraceRecordService,
    ) -> None:
        self._client = client
        self._project_resolver = project_resolver
        self._trace_record_service = trace_record_service

    async def dispatch_case_review(
        self,
        *,
        evaluator_name: str,
        evaluator_request: Any,  # noqa: ANN401 — evaluator 请求为动态载荷（不同 evaluator 类型不同）
        attribution: AttributionResult,
    ) -> bool:
        """归因完成后一次性下发到 LS。失败吞异常。"""
        trace_id = getattr(evaluator_request, "trace_id", None) or "unknown"
        # after_eval 已保证 project_id 非 None（None 时提前返回不触发 triage），
        # "default" 仅为防御性兜底。
        project_id = getattr(evaluator_request, "project_id", None) or "default"
        agent_name = getattr(evaluator_request, "agent_name", None) or "unknown"
        metadata = getattr(evaluator_request, "metadata", None) or {}
        trace_record_id = metadata.get("trace_record_id", "")
        dataset_name = metadata.get("dataset_name")
        run_name = metadata.get("dataset_run_name")

        ctx = get_triage_judge_context()
        judge_score = ctx.score if ctx else ""
        judge_reason = ctx.reason if ctx else ""
        triage_level = ctx.triage_level if ctx else ""
        extra = ctx.extra if ctx else ""

        review_payload = CaseReviewPayload(
            trace_id=trace_id,
            project_id=project_id,
            agent_name=agent_name,
            evaluator_name=evaluator_name,
            trace_record_id=trace_record_id,
            evaluator_request=evaluator_request,
            attribution=attribution,
            judge_score=judge_score,
            judge_reason=judge_reason,
            triage_level=triage_level,
            extra=extra,
        )
        task_payload = self._build_case_review_payload(review_payload)
        return await self._push(
            _PushContext(
                trace_id=trace_id,
                project_id=project_id,
                agent_name=agent_name,
                record_id=trace_record_id,
                payload=task_payload,
                dataset_name=dataset_name,
                run_name=run_name,
            )
        )

    async def _push(self, ctx: _PushContext) -> bool:
        try:
            ls_project_id = await self._project_resolver.resolve(
                ProjectIdentity(
                    agent_name=ctx.agent_name,
                    project_id=ctx.project_id,
                    dataset_name=ctx.dataset_name,
                    run_name=ctx.run_name,
                ),
            )
            await asyncio.to_thread(
                self._client.import_tasks,
                ls_project_id,
                [ctx.payload],
            )
            if ctx.record_id:
                await self._trace_record_service.update_ls_push_time(
                    record_id=ctx.record_id,
                )
            logger.info("ls_dispatch.success trace_id=%s", ctx.trace_id)
            return True
        except Exception:
            logger.warning(
                "ls_dispatch.failed trace_id=%s",
                ctx.trace_id,
                exc_info=True,
            )
            return False

    def _build_case_review_payload(self, payload: CaseReviewPayload) -> dict:
        raw_input = getattr(payload.evaluator_request, "input", None)
        raw_output = getattr(payload.evaluator_request, "output", None)
        input_preview = self._json_preview(raw_input)
        output_preview = self._json_preview(raw_output)

        metadata = getattr(payload.evaluator_request, "metadata", None) or {}
        data: dict[str, Any] = {
            "trace_record_id": payload.trace_record_id,
            "trace_id": payload.trace_id,
            "project_id": payload.project_id,
            "agent_name": payload.agent_name,
            "evaluator_name": payload.evaluator_name,
            "stage": STAGE_BADCASE_REVIEW,
            "agent_input": input_preview,
            "agent_output": output_preview,
            "judge_score": payload.judge_score,
            "judge_reason": payload.judge_reason,
            "triage_level": payload.triage_level,
            "extra": payload.extra,
            "root_cause": payload.attribution.root_cause,
            "category": payload.attribution.category,
            "confidence": str(payload.attribution.confidence),
            "reasoning": payload.attribution.reasoning or "",
        }
        # Additive dataset fields — only when present (dataset-eval path).
        if metadata.get("dataset_name") is not None:
            data["dataset_name"] = metadata["dataset_name"]
        if metadata.get("dataset_run_name") is not None:
            data["dataset_run_name"] = metadata["dataset_run_name"]
        if metadata.get("dataset_item_id") is not None:
            data["dataset_item_id"] = metadata["dataset_item_id"]
        return {"data": data}

    @staticmethod
    def _json_preview(value: Any, *, max_chars: int = JSON_PREVIEW_MAX_CHARS) -> str:  # noqa: ANN401 — 通用 JSON 预览工具，输入为任意值
        if value is None:
            return ""
        if isinstance(value, str):
            return value[:max_chars]
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
            return text[:max_chars]
        except (TypeError, ValueError):
            return str(value)[:max_chars]
