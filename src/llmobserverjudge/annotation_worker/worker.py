"""Annotation platform consumer — fire-and-forget ingest to PENDING (SDD-EV-25).

Must NOT publish eval_jobs. Does not wait for human completion on MQ path.
"""

from __future__ import annotations

import logging

from llmobserverjudge.annotation_worker.store import (
    AnnotationStore,
    InMemoryAnnotationStore,
)
from llmobserverjudge.contracts.idempotency import annotation_key
from llmobserverjudge.contracts.models import AnnotationTask, HumanAnnotation
from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.shared.mq import MessageQueue

logger = logging.getLogger(__name__)


class AnnotationWorker:
    """Consumes annotation_jobs → upsert HumanAnnotation(PENDING)."""

    FORBIDDEN_PRODUCE_TOPICS = frozenset({Topic.EVAL_JOBS})

    def __init__(
        self,
        mq: MessageQueue,
        store: AnnotationStore | None = None,
    ) -> None:
        self.mq = mq
        self.store = store or InMemoryAnnotationStore()
        self._seen: set[str] = set()

    async def process_once(self, max_batch: int = 100) -> list[HumanAnnotation]:
        messages = await self.mq.drain(Topic.ANNOTATION_JOBS, max_batch)
        out: list[HumanAnnotation] = []
        for raw in messages:
            try:
                row = await self._ingest(raw)
                if row is not None:
                    out.append(row)
            except Exception:
                logger.exception("annotation_ingest_failed id=%s", raw.get("annotation_id"))
        return out

    async def _ingest(self, raw: dict) -> HumanAnnotation | None:
        task = AnnotationTask.model_validate(raw)
        key = task.idempotency_key or annotation_key(task.trace_id, task.eval_agent_id)
        if key in self._seen:
            existing = await self.store.get_by_idempotency(key)
            return existing
        record = HumanAnnotation(
            annotation_id=task.annotation_id,
            trace_id=task.trace_id,
            tenant_id=task.tenant_id,
            eval_agent_id=task.eval_agent_id,
            trigger_reason=task.trigger_reason,
            auto_score=task.auto_score,
            auto_label=task.auto_label,
            conflict_detected=task.conflict_detected,
            idempotency_key=key,
        )
        saved = await self.store.upsert_pending(record)
        self._seen.add(key)
        return saved


def main() -> None:
    from llmobserverjudge.runtime.workers import main as worker_main

    worker_main(["annotation"])
