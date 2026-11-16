"""Declared-state mirror persistence for benchmark_kafka_tasks (§5.3 revision: RemoteConfig is the source of truth + a DB mirror).

Role: **mirror / audit snapshot**, not the source of truth — RemoteConfig ``agent-eval.benchmark.streaming``
is still the only source of truth; after each successful reconciliation round the reconciler upserts declared tasks into
``benchmark_kafka_tasks`` for task-list queries / audit / future P6 agent-park
page reuse. Soft-delete audit is carried by RemoteConfig publish history (§5.3); this table keeps only the current
declared state (removing a config entry = deleting the row).

Fail-soft: a missing table / DB failure only logs WARNING and returns ``False``, and does not block reconciliation or
consumption (persistence is a mirror, not the source of truth).

Multi-node: concurrent upserts of the same ``task_id`` by each node's reconciler are idempotent
(``update_or_create``); brief cross-node inconsistency inside a config-change window converges on later rounds
(after TTL expiry every node persists the same RemoteConfig declared state).
"""

import hashlib
import json
from datetime import UTC, datetime

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.models import BenchmarkKafkaTask
from benchmark.streaming.tasks import StreamingTaskSpec

logger = get_logger("agent_eval.benchmark.streaming.persistence")

#: Spec fingerprint length (SHA256 hex prefix; used for change detection).
_FINGERPRINT_LEN = 16


def spec_fingerprint(spec: StreamingTaskSpec) -> str:
    """Fingerprint of the declared content (first 16 hex of SHA256; used for cross-node reconciliation / change detection)."""
    payload = json.dumps(
        {
            "topics": sorted(spec.topics),
            "group_id": spec.group_id,
            "state": spec.state,
            "judge_class": spec.judge_class,
            "judge_execution": spec.judge_execution,
            "triage_class": spec.triage_class,
            "attribution_class": spec.attribution_class,
            "attribution_execution": spec.attribution_execution,
            "rubric_class": spec.rubric_class,
            "sampling": spec.sampling,
            "daily_report_enabled": spec.daily_report_enabled,
            "shadow": spec.shadow,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:_FINGERPRINT_LEN]


async def sync_streaming_tasks(
    specs: dict[str, StreamingTaskSpec],
    *,
    source: str = "remote_config",
    now: datetime | None = None,
) -> bool:
    """Sync the declared-task mirror into ``benchmark_kafka_tasks`` (idempotent, fail-soft).

    Semantics: upsert every declared item (including state / spec fields / fingerprint / reconciliation time),
    and delete rows that are not in the declaration (removing a config entry = deleting the task, §6.2).

    Args:
        specs: declared task set (``name → StreamingTaskSpec``).
        source: config-source marker (``remote_config`` declarative / ``env`` cold-start fallback).
        now: reconciliation time (defaults to current UTC; tests may inject it).

    Returns:
        ``True`` = sync succeeded; ``False`` = failure was degraded (WARNING log, not blocking).
    """
    try:
        reconciled_at = now or datetime.now(UTC)
        keep = set(specs)
        async for row in BenchmarkKafkaTask.all():
            if row.task_id not in keep:
                await row.delete()
        for name, spec in specs.items():
            await BenchmarkKafkaTask.update_or_create(
                task_id=name,
                defaults={
                    "source": source,
                    "state": spec.state,
                    "topics": list(spec.topics),
                    "group_id": spec.group_id,
                    "judge_class": spec.judge_class,
                    "attribution_class": spec.attribution_class,
                    "rubric_class": spec.rubric_class,
                    "sampling": spec.sampling,
                    "daily_report_enabled": spec.daily_report_enabled,
                    "shadow": spec.shadow,
                    "spec_fingerprint": spec_fingerprint(spec),
                    "last_reconciled_at": reconciled_at,
                },
            )
    except Exception as exc:
        logger.warning(
            "streaming task mirror sync failed (non-blocking): %s",
            exc,
        )
        return False
    logger.info(
        "streaming task mirror synced: source=%s tasks=%d names=%s",
        source,
        len(specs),
        sorted(specs),
    )
    return True


__all__ = [
    "spec_fingerprint",
    "sync_streaming_tasks",
]
