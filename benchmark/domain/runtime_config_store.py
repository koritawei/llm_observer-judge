"""DB snapshots for eval config / ODPS jobs / scheduled tasks.

Kafka sampling, export scheduling, and batch scheduling read the in-memory snapshot
through synchronous APIs. The write API and the background refresh load the database
into the snapshot. Callers no longer read RemoteConfig (evaluators / scheduled_task).
ODPS export is the exception: at runtime RemoteConfig ``agent-eval.odps.export`` is still
the base, and a DB job with the same name overrides it (see
:func:`benchmark.domain.odps_export_config.fetch_odps_export_config`).
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from tortoise.exceptions import IntegrityError
from tortoise.transactions import in_transaction

from benchmark.records.models.runtime_config_record import (
    KIND_CUSTOMIZATION_SEQUENCE,
    KIND_EVALUATOR,
    KIND_INSTANCE_SEQUENCE,
    KIND_ODPS_JOB,
    KIND_OSS_UPLOAD_SEQUENCE,
    KIND_SCHEDULED_TASK,
    SCHEMA_VERSION,
    SCOPE_GLOBAL,
    CustomizationKey,
    RuntimeConfigRecord,
    instance_sequence_record_key,
    oss_upload_sequence_record_key,
    parse_evaluator_key_tail,
    split_evaluator_record_key,
)
from benchmark.domain.external_daily_config import ExternalDailyConfig
from benchmark.domain.odps_export_config import OdpsExportConfig, OdpsExportJob
from benchmark.domain.runtime_snapshot import (
    current_eval_config_file as read_eval_config_file,
)
from benchmark.domain.runtime_snapshot import (
    current_external_daily_config as read_external_daily_config,
)
from benchmark.domain.runtime_snapshot import (
    current_odps_export_config as read_odps_export_config,
)
from benchmark.domain.runtime_snapshot import (
    replace_eval_config_file as write_eval_config_file,
)
from benchmark.domain.runtime_snapshot import (
    replace_external_daily_config as write_external_daily_config,
)
from benchmark.domain.runtime_snapshot import (
    replace_odps_export_config as write_odps_export_config,
)
from benchmark.domain.scheduled_task import (
    ScheduledTaskDocument,
    external_daily_config_from_documents,
)
from benchmark.domain.schemas import EvalConfigEntry, EvalConfigFile

logger = get_logger(__name__)

_REFRESH_INTERVAL_S = 30.0

#: Listeners after an eval-snapshot refresh (sidecar MetaQ hot reconcile, and so on). Failures are isolated.
EvalSnapshotListener = Callable[[], Awaitable[None]]
_eval_snapshot_listeners: list[EvalSnapshotListener] = []


class _RefreshState:
    """Background refresh-task state. Config snapshots are held only by runtime_snapshot."""

    def __init__(self) -> None:
        self.refresh_task: asyncio.Task[None] | None = None


_state = _RefreshState()


def register_eval_snapshot_listener(listener: EvalSnapshotListener) -> None:
    """Register an eval-snapshot refresh callback. The same listener is recorded only once."""
    if listener not in _eval_snapshot_listeners:
        _eval_snapshot_listeners.append(listener)


def unregister_eval_snapshot_listener(listener: EvalSnapshotListener) -> None:
    """Remove an eval-snapshot refresh callback. A no-op when it was not registered."""
    try:
        _eval_snapshot_listeners.remove(listener)
    except ValueError:
        return


async def _notify_eval_snapshot_listeners() -> None:
    for listener in tuple(_eval_snapshot_listeners):
        try:
            await listener()
        except Exception:
            logger.exception("eval_snapshot_listener_failed")


def current_eval_config_file() -> EvalConfigFile:
    """Return the in-process eval-config snapshot."""
    snapshot = read_eval_config_file()
    return snapshot if isinstance(snapshot, EvalConfigFile) else EvalConfigFile()


def current_odps_export_config() -> OdpsExportConfig:
    """Return the in-process ODPS export-config snapshot."""
    snapshot = read_odps_export_config()
    return snapshot if isinstance(snapshot, OdpsExportConfig) else OdpsExportConfig()


def current_external_daily_config() -> ExternalDailyConfig:
    """Return the in-process batch scheduled-task snapshot (kind=scheduled_task)."""
    snapshot = read_external_daily_config()
    return snapshot if isinstance(snapshot, ExternalDailyConfig) else ExternalDailyConfig()


def replace_eval_config_file(config_file: EvalConfigFile) -> None:
    """Replace the eval-config snapshot. Shared by tests and refresh."""
    write_eval_config_file(config_file)


def replace_odps_export_config(config: OdpsExportConfig) -> None:
    """Replace the ODPS config snapshot. Shared by tests and refresh."""
    write_odps_export_config(config)


def replace_external_daily_config(config: ExternalDailyConfig) -> None:
    """Replace the scheduled-task snapshot. Shared by tests and refresh."""
    write_external_daily_config(config)


def _payload(row: RuntimeConfigRecord) -> dict[str, object]:
    if isinstance(row.payload, dict):
        return row.payload
    return {}


async def upsert_runtime_config(
    *,
    kind: str,
    record_key: str,
    payload: dict[str, Any],
    scope_key: str,
) -> RuntimeConfigRecord:
    """Write by record_key. A soft-deleted row is revived by the same key and revision increments.

    Existence is expressed only by the soft-delete columns (``is_deleted`` / ``deleted_at``),
    the same as benchmark_runs. Reviving a row clears both columns, the ``restore`` semantics.
    """
    row = await RuntimeConfigRecord.filter(record_key=record_key).first()
    if row is None:
        return await RuntimeConfigRecord.create(
            kind=kind,
            record_key=record_key,
            payload=payload,
            schema_version=SCHEMA_VERSION,
            revision=1,
            scope_key=scope_key,
        )
    row.kind = kind
    row.payload = payload
    row.schema_version = SCHEMA_VERSION
    row.revision = int(row.revision) + 1
    row.scope_key = scope_key
    row.is_deleted = False
    row.deleted_at = None
    await row.save()
    return row


_SEQUENCE_MAX_ATTEMPTS = 5


async def _allocate_sequence_id(record_key: str, kind: str) -> int:
    """Allocate a counter id: a transaction plus a row lock increments atomically from 1.

    The first insert retries on a unique-key collision. The counter row stores ``last_id``
    in its payload. A business failure after allocation skips numbers — monotonic, not
    contiguous (same semantics as the OSS upload_id). Other keys already on the payload
    are kept.
    """
    for _attempt in range(_SEQUENCE_MAX_ATTEMPTS):
        try:
            async with in_transaction() as conn:
                row = (
                    await RuntimeConfigRecord.filter(record_key=record_key)
                    .select_for_update()
                    .using_db(conn)
                    .first()
                )
                if row is None:
                    await RuntimeConfigRecord.create(
                        kind=kind,
                        record_key=record_key,
                        payload={"last_id": 1},
                        schema_version=SCHEMA_VERSION,
                        revision=1,
                        scope_key=SCOPE_GLOBAL,
                        using_db=conn,
                    )
                    return 1
                payload = dict(_payload(row))
                next_id = int(payload.get("last_id", 0)) + 1
                payload["last_id"] = next_id
                row.payload = payload
                await row.save(using_db=conn, update_fields=["payload", "updated_at"])
                return next_id
        except IntegrityError:
            continue
    msg = f"sequence allocation failed for {record_key}"
    raise RuntimeError(msg)


async def next_instance_sequence() -> int:
    """Allocate the global sequence for an internal multi-instance agent name (never reused).

    Same pattern as the OSS upload_id: a transaction plus a row lock increments
    atomically. A save failure after allocation skips numbers — uniqueness first, not
    contiguity (design §12.2).
    """
    return await _allocate_sequence_id(
        instance_sequence_record_key(),
        KIND_INSTANCE_SEQUENCE,
    )


async def next_oss_upload_id(partition_date: str) -> int:
    """Allocate today's OSS upload sequence (increments from 1, so upload order is visible).

    The counter is an oss_upload_sequence row in ``runtime_config_records`` (one row per
    day, payload.last_id). A transaction plus a row lock increments it atomically, so
    concurrent nodes do not collide. The first concurrent insert retries on a unique-key
    collision. An upload failure after allocation skips numbers — order first, not contiguity.
    """
    return await _allocate_sequence_id(
        oss_upload_sequence_record_key(partition_date),
        KIND_OSS_UPLOAD_SEQUENCE,
    )


async def peek_customization_sequence(key: CustomizationKey) -> dict[str, Any]:
    """Read the business-customization counter payload (empty dict when the row is missing).

    The payload holds ``last_id`` (highest contentVersion allocated) and ``last_ref``
    (the stage ref of the last successful commit, so an identical upload can be reused).
    """
    row = await RuntimeConfigRecord.filter(record_key=key.record_key()).first()
    if row is None:
        return {}
    return dict(_payload(row))


async def next_customization_version(key: CustomizationKey) -> int:
    """Allocate the next contentVersion for a stage (monotonic per stage).

    Same pattern as the OSS upload_id: a transaction plus a row lock, and failures skip
    numbers. If the Git commit fails the number is abandoned and the next upload takes a
    new one (directory versions may skip).
    """
    return await _allocate_sequence_id(
        key.record_key(),
        KIND_CUSTOMIZATION_SEQUENCE,
    )


async def record_customization_ref(key: CustomizationKey, ref: dict[str, Any]) -> None:
    """Write back the latest ref after a successful Git commit (the read source for idempotent reuse and audit)."""
    row = await RuntimeConfigRecord.filter(record_key=key.record_key()).first()
    if row is None:
        return
    payload = dict(_payload(row))
    payload["last_ref"] = ref
    row.payload = payload
    await row.save(update_fields=["payload", "updated_at"])


async def retire_runtime_config(kind: str, record_key: str) -> bool:
    """Soft-delete a config that is still active. The row stays, and saving the same key revives it."""
    row = await RuntimeConfigRecord.filter(
        kind=kind,
        record_key=record_key,
        deleted_at__isnull=True,
    ).first()
    if row is None:
        return False
    await row.delete()
    return True


async def restore_runtime_config(kind: str, record_key: str) -> bool:
    """Restore a soft-deleted config (the inverse of retire, via the base ``restore``).

    Returns False, and does not raise, when the row does not exist (never registered) or
    is already active. Disable/restore linkage tolerates intermediate states and is idempotent.
    """
    row = await RuntimeConfigRecord.filter(kind=kind, record_key=record_key).first()
    if row is None:
        return False
    if row.deleted_at is None and not row.is_deleted:
        return False
    await row.restore()
    return True


async def refresh_eval_config_snapshot() -> EvalConfigFile:
    """Rebuild the eval-config snapshot from active evaluator rows in ``runtime_config_records``.

    After the refresh, notify listeners (for example an immediate sidecar MetaQ reconcile)
    so an evaluator coming online or offline does not wait for the 30s cycle.
    """
    grouped: dict[str, list[EvalConfigEntry]] = {}
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVALUATOR,
        deleted_at__isnull=True,
    )
    for row in rows:
        project_id, _, _, _ = split_evaluator_record_key(row.record_key)
        grouped.setdefault(project_id, []).append(EvalConfigEntry.model_validate(_payload(row)))
    config_file = EvalConfigFile(configs=grouped)
    replace_eval_config_file(config_file)
    await _notify_eval_snapshot_listeners()
    return config_file


async def retire_sibling_evaluator_entries(
    project_id: str,
    evaluator_name: str,
    keep_agent_name: str,
    *,
    keep_registration_id: str = "",
) -> int:
    """Soft-delete variant entries under the same evaluator that must not stay active together.

    - **With a ``registration_id`` / stream composite tail (``reg__si__stream``)**: retire
      only variants that share that key_tail and differ in ``agent_name``. Other
      registrations (v56 vs v57) and other streams are kept, so one stem can run several
      prompts × several data sources in parallel.
    - **No registration / stream (the default slot)**: retire unsuffixed entries under the
      same stem that use another ``agent_name``. Suffixed multi-instance slots are left alone.
    """
    keep = keep_agent_name.strip()
    keep_tail = keep_registration_id.strip()
    keep_reg, keep_stream = parse_evaluator_key_tail(keep_tail)
    retired = 0
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVALUATOR,
        deleted_at__isnull=True,
    )
    for row in rows:
        row_project, row_evaluator, row_agent, row_tail = split_evaluator_record_key(row.record_key)
        if row_project != project_id or row_evaluator != evaluator_name:
            continue
        row_reg, row_stream = parse_evaluator_key_tail(row_tail)
        if keep_tail:
            # Exact slot: only the same tail (including stream) is a sibling agentName variant.
            if row_tail != keep_tail:
                continue
            if row_agent.strip() == keep:
                continue
        else:
            if row_reg or row_stream:
                continue
            if row_agent.strip() == keep:
                continue
        await retire_runtime_config(KIND_EVALUATOR, row.record_key)
        retired += 1
    if retired:
        logger.info(
            "retired sibling evaluator entries: project=%s evaluator=%s "
            "keep_tail=%s keep_reg=%s keep_stream=%s keep_agent=%s retired=%d",
            project_id,
            evaluator_name,
            keep_tail or "<none>",
            keep_reg or "<none>",
            keep_stream or "<none>",
            keep or "<empty>",
            retired,
        )
    return retired


async def refresh_odps_export_snapshot() -> OdpsExportConfig:
    """Rebuild the ODPS config snapshot from odps_job rows in ``runtime_config_records``."""
    jobs: list[OdpsExportJob] = []
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_ODPS_JOB,
        deleted_at__isnull=True,
    )
    for row in rows:
        jobs.append(OdpsExportJob.model_validate(_payload(row)))
    config = OdpsExportConfig(jobs=jobs)
    replace_odps_export_config(config)
    return config


async def refresh_external_daily_snapshot() -> ExternalDailyConfig:
    """Rebuild the batch scheduled-task snapshot from scheduled_task rows.

    RemoteConfig ``agent-eval.benchmark.external-daily`` is fully shielded: this snapshot
    is the only config source for ``DailyExternalBenchmarkScheduler``. A row that fails
    to parse is skipped and the other tasks are kept.
    """
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_SCHEDULED_TASK,
        deleted_at__isnull=True,
    )
    documents: list[ScheduledTaskDocument] = []
    for row in rows:
        try:
            documents.append(ScheduledTaskDocument.model_validate(_payload(row)))
        except Exception:
            logger.warning(
                "scheduled task document invalid, skip: key=%s",
                row.record_key,
                exc_info=True,
            )
    config = external_daily_config_from_documents(documents)
    replace_external_daily_config(config)
    return config


async def refresh_runtime_config_snapshots() -> None:
    """Refresh the three snapshots (evaluator / odps_job / scheduled_task). On failure, keep the previous ones."""
    try:
        await refresh_eval_config_snapshot()
        await refresh_odps_export_snapshot()
        await refresh_external_daily_snapshot()
    except Exception:
        logger.exception("runtime config snapshot refresh failed")


async def start_runtime_config_refresh() -> None:
    """Load once at startup, then refresh from the DB every 30 seconds."""
    await refresh_runtime_config_snapshots()

    async def _loop() -> None:
        while True:
            await asyncio.sleep(_REFRESH_INTERVAL_S)
            await refresh_runtime_config_snapshots()

    _state.refresh_task = asyncio.create_task(_loop(), name="runtime-config-refresh")


async def stop_runtime_config_refresh() -> None:
    """Cancel the refresh task on shutdown."""
    task = _state.refresh_task
    _state.refresh_task = None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return
