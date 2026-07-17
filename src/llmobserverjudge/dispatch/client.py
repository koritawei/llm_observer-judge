"""Dispatch Client — Annotation / attribution publish path (Option X)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from llmobserverjudge.contracts.idempotency import annotation_key, attribution_key
from llmobserverjudge.contracts.models import AnnotationTask
from llmobserverjudge.contracts.topics import Component, Topic
from llmobserverjudge.dispatch.config_loader import DispatchConfigLoader
from llmobserverjudge.dispatch.config_models import DispatchSection
from llmobserverjudge.dispatch.routing import (
    find_eval_routing_rule,
    should_trigger_annotation,
    should_trigger_attribution,
)
from llmobserverjudge.shared.mq import MessageQueue


class DispatchClient:
    """Used by Eval Graph nodes — never bypass with raw MQ from eval_worker."""

    COMPONENT = Component.DISPATCH

    def __init__(
        self,
        mq: MessageQueue,
        config_loader: DispatchConfigLoader | None = None,
    ) -> None:
        self.mq = mq
        self.config_loader = config_loader or DispatchConfigLoader()

    async def publish_annotation(self, task: AnnotationTask) -> None:
        payload = task.model_dump(mode="json")
        if not payload.get("idempotency_key"):
            payload["idempotency_key"] = annotation_key(task.trace_id, task.eval_agent_id)
        await self.mq.publish(self.COMPONENT, Topic.ANNOTATION_JOBS, payload)

    async def publish_attribution(
        self,
        *,
        trace_id: str,
        tenant_id: str,
        eval_agent_id: str,
        attribution_agent: str,
        priority: str,
        score_snapshot: dict,
        trace_summary: dict | None = None,
    ) -> None:
        await self.mq.publish(
            self.COMPONENT,
            Topic.ATTRIBUTION_JOBS,
            {
                "trace_id": trace_id,
                "tenant_id": tenant_id,
                "eval_agent_id": eval_agent_id,
                "attribution_agent": attribution_agent,
                "priority": priority,
                "score_snapshot": score_snapshot,
                "trace_summary": trace_summary or {},
                "idempotency_key": attribution_key(trace_id, eval_agent_id),
            },
        )

    async def route_post_eval(
        self,
        *,
        trace_id: str,
        tenant_id: str,
        eval_agent_id: str,
        score: float | None,
        confidence: float | None = None,
        label: str | None = None,
        conflict_detected: bool = False,
        is_bad_case: bool | None = None,
        trace_summary: dict | None = None,
        config: DispatchSection | None = None,
    ) -> dict[str, Any]:
        """Apply eval_routing_rules → attribution / annotation (Option X).

        Good cases (``is_bad_case=False``) never publish attribution.
        """
        cfg = config or self.config_loader.config
        rule = find_eval_routing_rule(eval_agent_id, cfg)
        result: dict[str, Any] = {
            "attribution_published": False,
            "annotation_published": False,
            "rule_found": rule is not None,
        }
        if rule is None:
            return result

        if should_trigger_attribution(rule, score=score, is_bad_case=is_bad_case):
            await self.publish_attribution(
                trace_id=trace_id,
                tenant_id=tenant_id,
                eval_agent_id=eval_agent_id,
                attribution_agent=rule.attribution_agent,
                priority=rule.attribution_priority.value,
                score_snapshot={
                    "score": score,
                    "label": label,
                    "is_bad_case": is_bad_case,
                },
                trace_summary=trace_summary,
            )
            result["attribution_published"] = True

        if should_trigger_annotation(
            rule.annotation_trigger,
            score=score,
            confidence=confidence,
            threshold=rule.bad_case_threshold,
        ):
            task = AnnotationTask(
                annotation_id=str(uuid4()),
                trace_id=trace_id,
                tenant_id=tenant_id,
                eval_agent_id=eval_agent_id,
                trigger_reason=rule.annotation_trigger.value,
                auto_score=score,
                auto_label=label,
                conflict_detected=conflict_detected,
            )
            await self.publish_annotation(task)
            result["annotation_published"] = True

        return result
