"""Attribution Worker — consumes attribution_jobs only (SDD-EV-24).

Must NOT publish eval_jobs. Checkpoint only for ReAct (ADR-0004).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from llmobserverjudge.attribution_worker.checkpoint import InMemoryCheckpointStore
from llmobserverjudge.attribution_worker.context import (
    AttributionContext,
    InMemoryAttributionContextStore,
)
from llmobserverjudge.attribution_worker.llm_policy import react_policy_from_env
from llmobserverjudge.attribution_worker.react import ReActAttributor
from llmobserverjudge.attribution_worker.store import (
    AttributionStore,
    InMemoryAttributionStore,
)
from llmobserverjudge.contracts.models import AttributionJob, AttributionResult
from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.shared.mq import MQ_MSG_ID, MessageQueue

logger = logging.getLogger(__name__)


class AttributionWorker:
    FORBIDDEN_PRODUCE_TOPICS = frozenset({Topic.EVAL_JOBS})

    def __init__(
        self,
        mq: MessageQueue,
        *,
        store: AttributionStore | None = None,
        context_store: InMemoryAttributionContextStore | None = None,
        attributor: ReActAttributor | None = None,
        checkpoint: Any | None = None,
        idempotency: object | None = None,
        max_concurrent: int = 4,
        use_env_react_policy: bool = False,
    ) -> None:
        self.mq = mq
        self.store = store or InMemoryAttributionStore()
        self.context_store = context_store or InMemoryAttributionContextStore()
        self.idempotency = idempotency
        ckpt = checkpoint or InMemoryCheckpointStore()
        if attributor is not None:
            self.attributor = attributor
        elif use_env_react_policy:
            self.attributor = ReActAttributor(
                policy=react_policy_from_env(),
                checkpoint=ckpt,
            )
        else:
            self.attributor = ReActAttributor(checkpoint=ckpt)
        self._sem = asyncio.Semaphore(max_concurrent)
        self._seen: set[str] = set()

    async def _pull(self, max_batch: int) -> list[dict]:
        messages = await self.mq.drain(Topic.ATTRIBUTION_JOBS, max_batch)
        reclaim = getattr(self.mq, "reclaim_stale", None)
        if reclaim is not None and len(messages) < max_batch:
            stale = await reclaim(
                Topic.ATTRIBUTION_JOBS, max_count=max_batch - len(messages)
            )
            messages.extend(stale)
        return messages

    async def process_once(self, max_batch: int = 10) -> list[AttributionResult]:
        messages = await self._pull(max_batch)
        results: list[AttributionResult] = []
        ack_ids: list[str] = []

        async def one(raw: dict) -> tuple[str | None, AttributionResult | None]:
            msg_id = raw.pop(MQ_MSG_ID, None)
            async with self._sem:
                try:
                    result = await self._process_job(raw)
                    return (str(msg_id) if msg_id else None, result)
                except Exception:
                    logger.exception("attribution_batch_item_failed")
                    return (None, None)

        done = await asyncio.gather(*[one(m) for m in messages])
        for msg_id, item in done:
            if msg_id:
                ack_ids.append(msg_id)
            if item is not None:
                results.append(item)
        if ack_ids:
            await self.mq.ack(Topic.ATTRIBUTION_JOBS, ack_ids)
        return results

    async def _process_job(self, raw: dict) -> AttributionResult | None:
        job = AttributionJob.model_validate(raw)
        idem_key = f"attr:{job.idempotency_key}"
        existing = await self.store.get_by_idempotency(job.idempotency_key)
        if existing is not None:
            self._seen.add(job.idempotency_key)
            return existing
        if self.idempotency is not None and await self.idempotency.exists(idem_key):
            got = await self.store.get_by_idempotency(job.idempotency_key)
            if got is not None:
                self._seen.add(job.idempotency_key)
                return got

        ctx = await self._build_context(job)
        result = await self.attributor.analyze(job, ctx)
        await self.store.save(result, idempotency_key=job.idempotency_key)
        if self.idempotency is not None:
            await self.idempotency.mark(idem_key)
        self._seen.add(job.idempotency_key)
        return result

    async def _build_context(self, job: AttributionJob) -> AttributionContext:
        enriched = await self.context_store.get(job.tenant_id, job.trace_id)
        summary = dict(job.trace_summary or {})
        spans: list[dict] = []
        if enriched:
            summary = {**enriched.trace_summary, **summary}
            spans = list(enriched.spans)
        return AttributionContext(
            tenant_id=job.tenant_id,
            trace_id=job.trace_id,
            trace_summary=summary,
            spans=spans,
            score_snapshot=dict(job.score_snapshot or {}),
            eval_agent_id=job.eval_agent_id,
        )


def main() -> None:
    from llmobserverjudge.runtime.workers import main as worker_main

    worker_main(["attribution"])
