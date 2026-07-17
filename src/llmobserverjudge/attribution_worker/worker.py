"""Attribution Worker — consumes attribution_jobs only (SDD-EV-24).

Must NOT publish eval_jobs. Checkpoint only for ReAct (ADR-0004).
"""

from __future__ import annotations

import asyncio
import logging

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
from llmobserverjudge.shared.mq import MessageQueue

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
        max_concurrent: int = 4,
        use_env_react_policy: bool = False,
    ) -> None:
        self.mq = mq
        self.store = store or InMemoryAttributionStore()
        self.context_store = context_store or InMemoryAttributionContextStore()
        if attributor is not None:
            self.attributor = attributor
        elif use_env_react_policy:
            self.attributor = ReActAttributor(
                policy=react_policy_from_env(),
                checkpoint=InMemoryCheckpointStore(),
            )
        else:
            self.attributor = ReActAttributor(checkpoint=InMemoryCheckpointStore())
        self._sem = asyncio.Semaphore(max_concurrent)
        self._seen: set[str] = set()

    async def process_once(self, max_batch: int = 10) -> list[AttributionResult]:
        messages = await self.mq.drain(Topic.ATTRIBUTION_JOBS, max_batch)
        results: list[AttributionResult] = []

        async def one(raw: dict) -> AttributionResult | None:
            async with self._sem:
                return await self._process_job(raw)

        done = await asyncio.gather(*[one(m) for m in messages], return_exceptions=True)
        for item in done:
            if isinstance(item, Exception):
                logger.exception("attribution_batch_item_failed: %s", item)
                continue
            if item is not None:
                results.append(item)
        return results

    async def _process_job(self, raw: dict) -> AttributionResult | None:
        job = AttributionJob.model_validate(raw)
        if job.idempotency_key in self._seen:
            return None
        existing = await self.store.get_by_idempotency(job.idempotency_key)
        if existing is not None:
            self._seen.add(job.idempotency_key)
            return existing

        ctx = await self._build_context(job)
        result = await self.attributor.analyze(job, ctx)
        await self.store.save(result, idempotency_key=job.idempotency_key)
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
