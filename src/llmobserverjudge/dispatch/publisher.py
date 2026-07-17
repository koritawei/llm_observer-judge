"""Dispatch — sole publisher of eval_jobs (R-02 Option B)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from llmobserverjudge.contracts.idempotency import eval_config_hash, eval_job_key
from llmobserverjudge.contracts.models import EvalJob
from llmobserverjudge.contracts.topics import Component, Topic, eval_jobs_priority_topic
from llmobserverjudge.dispatch.config_loader import DispatchConfigLoader
from llmobserverjudge.dispatch.routing import match_eval_configs
from llmobserverjudge.ingest.outbox import OutboxStore
from llmobserverjudge.shared.mq import MessageQueue

logger = logging.getLogger(__name__)


class DispatchPublisher:
    """Reads unpublished outbox rows → enrich → publish eval_jobs → mark published."""

    COMPONENT = Component.DISPATCH

    def __init__(
        self,
        mq: MessageQueue,
        outbox: OutboxStore,
        config_loader: DispatchConfigLoader | None = None,
    ) -> None:
        self.mq = mq
        self.outbox = outbox
        self.config_loader = config_loader or DispatchConfigLoader()

    async def publish_pending(self, limit: int = 100) -> int:
        # Hold snapshot for this batch (CoW: in-flight keeps old rules)
        cfg = self.config_loader.config
        rows = await self.outbox.claim_unpublished(limit)
        count = 0
        for row in rows:
            try:
                configs, priority = match_eval_configs(row["trace_summary"], cfg)
                config_dicts = [c.model_dump() for c in configs]
                job = EvalJob(
                    trace_id=row["trace_id"],
                    tenant_id=row["tenant_id"],
                    project_id=str(
                        row.get("project_id")
                        or (row.get("trace_summary") or {}).get("project_id")
                        or "default"
                    ),
                    trace_summary=row["trace_summary"],
                    prompt_version_id=row.get("prompt_version_id"),
                    eval_configs=configs,
                    priority=priority,
                    idempotency_key=eval_job_key(
                        row["trace_id"],
                        config_dicts,
                        tenant_id=row["tenant_id"],
                        project_id=str(
                            row.get("project_id")
                            or (row.get("trace_summary") or {}).get("project_id")
                            or "default"
                        ),
                    ),
                    eval_config_hash=eval_config_hash(config_dicts),
                    created_at=datetime.now(UTC),
                    outbox_id=row["id"],
                )
                payload = job.model_dump(mode="json")
                await self.mq.publish(self.COMPONENT, Topic.EVAL_JOBS, payload)
                if cfg.publish_priority_shards:
                    shard = eval_jobs_priority_topic(priority.value)
                    await self.mq.publish(self.COMPONENT, shard, payload)
                marked = await self.outbox.mark_published(row["id"])
                if marked:
                    count += 1
            except Exception:
                release = getattr(self.outbox, "release_claim", None)
                if release is not None:
                    await release(row["id"])
                else:
                    unclaim = getattr(self.outbox, "unclaim", None)
                    if unclaim is not None:
                        await unclaim(row["id"])
                await self.outbox.bump_attempt(row["id"])
                logger.exception("dispatch_publish_failed outbox_id=%s", row.get("id"))
        return count


def main() -> None:
    from llmobserverjudge.runtime.workers import main as worker_main

    worker_main(["dispatch"])
