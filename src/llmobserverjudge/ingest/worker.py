"""Ingest Worker — consume spans, durable write, outbox intent (no eval_jobs)."""

from __future__ import annotations

import asyncio
import logging

from llmobserverjudge.contracts.idempotency import span_key
from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.dashboard.events import PipelineEvent, PipelineEventBus
from llmobserverjudge.dashboard.store import SamplingStatsStore
from llmobserverjudge.ingest.idempotency import (
    IdempotencyStore,
    InMemoryIdempotencyStore,
)
from llmobserverjudge.ingest.nodes import IngestState, SamplingConfig
from llmobserverjudge.ingest.outbox import InMemoryOutbox, OutboxStore
from llmobserverjudge.ingest.pipeline import IngestPipeline
from llmobserverjudge.shared.mq import MQ_MSG_ID, MessageQueue
from llmobserverjudge.shared.storage import InMemorySpanStore, SpanStore

logger = logging.getLogger(__name__)


class IngestWorker:
    def __init__(
        self,
        mq: MessageQueue,
        outbox: OutboxStore | None = None,
        pipeline: IngestPipeline | None = None,
        idempotency: IdempotencyStore | None = None,
        store: SpanStore | None = None,
        *,
        sampling_config: SamplingConfig | None = None,
        force_sample: bool | None = None,
        sampling_stats: SamplingStatsStore | None = None,
        event_bus: PipelineEventBus | None = None,
    ) -> None:
        self.mq = mq
        self.outbox = outbox or InMemoryOutbox()
        self.store = store or InMemorySpanStore()
        self.idempotency = idempotency or InMemoryIdempotencyStore()
        self.pipeline = pipeline or IngestPipeline(
            mq=mq,
            store=self.store,
            sampling_config=sampling_config,
            force_sample=force_sample,
        )
        self.sampling_stats = sampling_stats
        self.event_bus = event_bus
        self._stop = asyncio.Event()

    async def _pull(self, max_batch: int) -> list[dict]:
        messages = await self.mq.drain(Topic.SPANS, max_batch)
        reclaim = getattr(self.mq, "reclaim_stale", None)
        if reclaim is not None and len(messages) < max_batch:
            stale = await reclaim(Topic.SPANS, max_count=max_batch - len(messages))
            messages.extend(stale)
        return messages

    async def process_once(self, max_batch: int = 50) -> int:
        messages = await self._pull(max_batch)
        handled = 0
        ack_ids: list[str] = []
        for msg in messages:
            msg_id = msg.pop(MQ_MSG_ID, None)
            key = span_key(
                str(msg["tenant_id"]),
                str(msg["span_id"]),
                project_id=str(msg.get("project_id") or "default"),
            )
            try_mark = getattr(self.idempotency, "try_mark", None)
            if try_mark is not None:
                acquired = await try_mark(key)
                if not acquired:
                    if msg_id:
                        ack_ids.append(str(msg_id))
                    continue
            elif await self.idempotency.exists(key):
                if msg_id:
                    ack_ids.append(str(msg_id))
                continue
            try:
                if self.event_bus is not None:
                    await self.event_bus.publish(
                        PipelineEvent(
                            trace_id=str(msg.get("trace_id") or ""),
                            tenant_id=str(msg.get("tenant_id") or ""),
                            node="SamplingNode",
                            status="running",
                        )
                    )
                state = await self.pipeline.run(IngestState(span=msg))
                if try_mark is None:
                    await self.idempotency.mark(key)
                handled += 1
                if self.sampling_stats is not None:
                    await self.sampling_stats.record(
                        sampled=bool(state.sampled), reason=state.sample_reason
                    )
                if self.event_bus is not None:
                    await self.event_bus.publish(
                        PipelineEvent(
                            trace_id=state.trace_id,
                            tenant_id=state.tenant_id,
                            node="SamplingNode",
                            status="completed" if state.sampled else "skipped",
                            detail={"reason": state.sample_reason},
                        )
                    )
                    if state.written:
                        await self.event_bus.publish(
                            PipelineEvent(
                                trace_id=state.trace_id,
                                tenant_id=state.tenant_id,
                                node="WriteNode",
                                status="completed",
                                detail={"span_id": state.span_id},
                            )
                        )
                if state.sampled and state.written:
                    await self.outbox.insert_trace_closed(
                        tenant_id=state.tenant_id,
                        trace_id=state.trace_id,
                        trace_summary=state.trace_summary(),
                        prompt_version_id=msg.get("prompt_version_id"),
                        project_id=str(msg.get("project_id") or "default"),
                    )
                elif state.written:
                    logger.debug(
                        "span durable but not sampled reason=%s trace=%s",
                        state.sample_reason,
                        state.trace_id,
                    )
                if msg_id:
                    ack_ids.append(str(msg_id))
            except Exception:
                logger.exception("ingest_message_failed span_id=%s", msg.get("span_id"))
                # leave unacked for reclaim
        if ack_ids:
            await self.mq.ack(Topic.SPANS, ack_ids)
        return handled

    async def run_forever(self, *, poll_interval: float = 0.05, max_batch: int = 50) -> None:
        while not self._stop.is_set():
            n = await self.process_once(max_batch)
            if n == 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=poll_interval)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop.set()


def main() -> None:
    from llmobserverjudge.runtime.workers import main as worker_main

    worker_main(["ingest"])
