"""Eval Worker — consumes eval_jobs only.

Hard constraint (SDD-EV-27): MUST NOT publish to eval_jobs.
Annotation / attribution go through DispatchClient only.
"""

from __future__ import annotations

import logging

from llmobserverjudge.contracts.models import EvalJob, GraphState
from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.dashboard.events import PipelineEvent, PipelineEventBus
from llmobserverjudge.dashboard.store import EvalResultRecord, EvalResultStore
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.eval_worker.graph import run_eval_graph
from llmobserverjudge.eval_worker.judges import JudgeModelRouter, build_judge_router
from llmobserverjudge.shared.mq import MQ_MSG_ID, MessageQueue

logger = logging.getLogger(__name__)


class EvalWorker:
    """Consumer of eval_jobs. No Topic.EVAL_JOBS producer API on this class."""

    FORBIDDEN_PRODUCE_TOPICS = frozenset({Topic.EVAL_JOBS})

    def __init__(
        self,
        mq: MessageQueue,
        dispatch_client: DispatchClient | None = None,
        judge_router: JudgeModelRouter | None = None,
        *,
        eval_results: EvalResultStore | None = None,
        event_bus: PipelineEventBus | None = None,
        prompt_service: object | None = None,
        idempotency: object | None = None,
    ) -> None:
        self.mq = mq
        self.dispatch_client = dispatch_client or DispatchClient(mq)
        self.judge_router = judge_router or build_judge_router(include_env_http=True)
        self.eval_results = eval_results
        self.event_bus = event_bus
        self.prompt_service = prompt_service
        self.idempotency = idempotency
        self._seen: set[str] = set()

    async def _pull(self, max_batch: int) -> list[dict]:
        messages = await self.mq.drain(Topic.EVAL_JOBS, max_batch)
        reclaim = getattr(self.mq, "reclaim_stale", None)
        if reclaim is not None and len(messages) < max_batch:
            stale = await reclaim(Topic.EVAL_JOBS, max_count=max_batch - len(messages))
            messages.extend(stale)
        return messages

    async def _already_done(self, key: str) -> bool:
        if key in self._seen:
            return True
        if self.idempotency is not None:
            try_mark = getattr(self.idempotency, "try_mark", None)
            if try_mark is not None:
                acquired = await try_mark(f"eval:{key}")
                if not acquired:
                    self._seen.add(key)
                    return True
                # acquired lock for this job — proceed
                return False
            if await self.idempotency.exists(f"eval:{key}"):
                self._seen.add(key)
                return True
        get_by = getattr(self.eval_results, "get_by_idempotency", None)
        if get_by is not None:
            existing = await get_by(key)
            if existing is not None:
                self._seen.add(key)
                return True
        return False

    async def process_once(self, max_batch: int = 10) -> list[GraphState]:
        messages = await self._pull(max_batch)
        states: list[GraphState] = []
        ack_ids: list[str] = []
        for raw in messages:
            msg_id = raw.pop(MQ_MSG_ID, None)
            try:
                job = EvalJob.model_validate(raw)
                if await self._already_done(job.idempotency_key):
                    if msg_id:
                        ack_ids.append(str(msg_id))
                    continue
                self._seen.add(job.idempotency_key)
                if self.idempotency is not None and not hasattr(
                    self.idempotency, "try_mark"
                ):
                    await self.idempotency.mark(f"eval:{job.idempotency_key}")
                if self.event_bus is not None:
                    await self.event_bus.publish(
                        PipelineEvent(
                            trace_id=job.trace_id,
                            tenant_id=job.tenant_id,
                            node="EvalRouteNode",
                            status="running",
                        )
                    )
                state = await run_eval_graph(
                    job,
                    judge_router=self.judge_router,
                    dispatch_client=self.dispatch_client,
                    prompt_service=self.prompt_service,  # type: ignore[arg-type]
                )
                if self.event_bus is not None:
                    await self.event_bus.publish(
                        PipelineEvent(
                            trace_id=state.trace_id,
                            tenant_id=state.tenant_id,
                            node="EvalRouteNode",
                            status="completed",
                        )
                    )
                if self.eval_results is not None:
                    save_kw = {}
                    save = self.eval_results.save
                    try:
                        await save(  # type: ignore[call-arg]
                            EvalResultRecord(
                                trace_id=state.trace_id,
                                tenant_id=state.tenant_id,
                                project_id=getattr(state, "project_id", None)
                                or getattr(job, "project_id", None)
                                or "default",
                                final_score=state.final_score,
                                bad_case_detected=state.bad_case_detected,
                                eval_scores=list(state.eval_scores),
                                eval_execution_layers=list(state.eval_execution_layers),
                                eval_results=list(state.eval_results),
                                node_errors=dict(state.node_errors),
                                optional_skipped=list(state.optional_skipped),
                                prompt_version_id=state.prompt_version_id,
                                execution_graph=list(
                                    getattr(state, "execution_graph", None) or []
                                ),
                            ),
                            idempotency_key=job.idempotency_key,
                        )
                    except TypeError:
                        await save(
                            EvalResultRecord(
                                trace_id=state.trace_id,
                                tenant_id=state.tenant_id,
                                project_id=getattr(state, "project_id", None)
                                or getattr(job, "project_id", None)
                                or "default",
                                final_score=state.final_score,
                                bad_case_detected=state.bad_case_detected,
                                eval_scores=list(state.eval_scores),
                                eval_execution_layers=list(state.eval_execution_layers),
                                eval_results=list(state.eval_results),
                                node_errors=dict(state.node_errors),
                                optional_skipped=list(state.optional_skipped),
                                prompt_version_id=state.prompt_version_id,
                                execution_graph=list(
                                    getattr(state, "execution_graph", None) or []
                                ),
                            )
                        )
                        _ = save_kw
                if self.event_bus is not None:
                    from llmobserverjudge.eval_worker.execution_graph import (
                        map_score_status_to_ui,
                    )

                    by_result = {
                        str(r.get("agent_id")): r
                        for r in (state.eval_results or [])
                        if r.get("agent_id")
                    }
                    layers = state.eval_execution_layers or [
                        [c.get("agent_id") or "judge" for c in (state.eval_configs or [])]
                    ]
                    for layer in layers:
                        for name in layer:
                            r = by_result.get(str(name)) or {}
                            if name in state.optional_skipped:
                                st = "skipped"
                            elif name in state.node_errors:
                                st = "failed"
                            else:
                                st = map_score_status_to_ui(
                                    str(r.get("score_status") or "completed"),
                                    is_bad_case=bool(r.get("is_bad_case")),
                                )
                            await self.event_bus.publish(
                                PipelineEvent(
                                    trace_id=state.trace_id,
                                    tenant_id=state.tenant_id,
                                    node=str(name),
                                    status=st,
                                )
                            )
                    await self.event_bus.publish(
                        PipelineEvent(
                            trace_id=state.trace_id,
                            tenant_id=state.tenant_id,
                            node="ScoreNode",
                            status="completed",
                            detail={
                                "final_score": state.final_score,
                                "bad_case_detected": state.bad_case_detected,
                            },
                        )
                    )
                    if state.bad_case_detected:
                        await self.event_bus.publish(
                            PipelineEvent(
                                trace_id=state.trace_id,
                                tenant_id=state.tenant_id,
                                node="Attribution",
                                status="pending",
                            )
                        )
                    else:
                        await self.event_bus.publish(
                            PipelineEvent(
                                trace_id=state.trace_id,
                                tenant_id=state.tenant_id,
                                node="Attribution",
                                status="skipped",
                                detail={"skip_reason": "good_case"},
                            )
                        )
                    ann_status = "pending" if state.annotation_task_count else "skipped"
                    await self.event_bus.publish(
                        PipelineEvent(
                            trace_id=state.trace_id,
                            tenant_id=state.tenant_id,
                            node="Annotation",
                            status=ann_status,
                        )
                    )
                states.append(state)
                if msg_id:
                    ack_ids.append(str(msg_id))
            except Exception:
                logger.exception("eval_message_failed")
        if ack_ids:
            await self.mq.ack(Topic.EVAL_JOBS, ack_ids)
        return states


def main() -> None:
    from llmobserverjudge.runtime.workers import main as worker_main

    worker_main(["eval"])
