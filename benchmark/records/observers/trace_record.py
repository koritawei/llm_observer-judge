"""TraceRecord lifecycle observer.

Runs domain-level logic (observability logs) when a TraceRecord is saved or deleted.
Registered automatically at application startup via ``DatabaseConfig.observers``.

Usage:
    The observer is discovered and registered by ``register_database()``.
    There is no need to call ``TraceRecord.set_observer()`` by hand.
"""

from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.models.observer import ModelObserver

from benchmark.records.models.trace_record import TraceRecord, clip_char_value

logger = get_logger("agent_eval.observers.trace_record")


class TraceRecordObserver(ModelObserver[TraceRecord]):
    """Lifecycle observer for TraceRecord.

    Triggered directly by the post-save call inside the ``TraceRecord.save()`` override.

    Responsibilities:
      - **on_save**: only sync the ``notification_job_status`` scalar column (extracted from
        the ``notification_job_json`` bucket). The ``triage_level`` / ``root_cause`` scalar
        columns are now written directly by the service layer (see
        :meth:`TraceRecordService.update_triage_level` /
        :meth:`TraceRecordService.update_attribution`) and are no longer synced from the
        bucket by the observer — that removes the mismatch of "triage/attribution stuffing
        values back into the judge/attribution bucket".
      - **on_delete**: record the trace delete event, for audit.

    Note:
      ``agent_name`` is a string field. Metadata is loaded from RemoteConfig by
      ``benchmark.domain.agent_config``; it does not need to be registered on save.
    """

    async def on_save(self, instance: TraceRecord) -> None:
        """Post-save: sync the ``notification_job_status`` scalar column and emit a pipeline log.

        Only ``notification_job_status`` is handled: extract it from the ``notification_job_json``
        bucket and write the scalar column with ``filter().update()``, bypassing the ``save()``
        lifecycle so it does not recurse. Truncate to the model ``_meta.max_length`` (same source
        as ``TraceRecord.save``). ``triage_level`` / ``root_cause`` are not synced here — the
        service layer writes them explicitly.
        """
        updates: dict[str, Any] = {}
        if instance.notification_job_json:
            val = instance.notification_job_json.get("notification_job_status")
            if isinstance(val, str):
                val = clip_char_value(
                    TraceRecord, "notification_job_status", val, record_id=instance.id,
                )
            if instance.notification_job_status != val:
                instance.notification_job_status = val
                updates["notification_job_status"] = val
        if updates:
            await TraceRecord.filter(id=instance.id).update(**updates)
        logger.debug(
            "trace_record saved: trace_id=%s status=%s project_id=%s evaluator=%s",
            instance.trace_id,
            instance.status,
            instance.project_id,
            instance.evaluator_name,
        )

    async def on_delete(self, instance: TraceRecord) -> None:
        """Fires after a TraceRecord is deleted and records the delete event."""
        logger.info(
            "trace_record deleted: trace_id=%s",
            instance.trace_id,
        )


__all__ = ["TraceRecordObserver"]
