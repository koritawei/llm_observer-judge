"""DB materialization of the unified agent view (derived read model on ``runtime_config_records``).

The source of truth for view assembly stays in code and config (static component
declarations / page documents / the evaluator registry). This module only writes the
assembled result into the existing config table as ``kind=agent_view`` so catalog APIs
can read the DB directly. No new table: ``payload`` stores the full ``BenchmarkDefinition``
JSON plus ``source`` / ``generated_at`` markers; ``record_key`` is the canonical agent
name. The execution path must not read this table (execution truth is the
``ProfileResolver`` run snapshot); a stale row only affects display.

Sync semantics: an unchanged payload is not written (so multi-node startup does not
bump revision); agent names that disappear from the assembly are soft-deleted; a
single-row failure is logged and does not block the rest of the sync.
"""

from datetime import UTC, datetime
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.models.runtime_config_record import (
    KIND_AGENT_VIEW,
    RuntimeConfigRecord,
    agent_view_record_key,
)
from benchmark.domain.runtime_config_store import retire_runtime_config, upsert_runtime_config
from benchmark.runs.schema import BenchmarkDefinition

logger = get_logger(__name__)

#: Derived marker keys in the payload (stripped when restoring a BenchmarkDefinition).
_MARKER_KEYS = ("source", "generated_at")


def _comparable(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "generated_at"}


async def sync_agent_views(
    entries: list[tuple[BenchmarkDefinition, str]],
) -> dict[str, int]:
    """Sync assembled agent views into the DB (write on change, soft-delete disappearances).

    ``entries`` is ``(view row, source)``; source is ``static`` / ``dynamic`` /
    ``evaluator``. Returns written / unchanged / retired counts for the startup reconcile log.
    """
    counts = {"written": 0, "unchanged": 0, "retired": 0}
    rows = await RuntimeConfigRecord.filter(kind=KIND_AGENT_VIEW)
    existing = {row.record_key: row for row in rows}
    seen: set[str] = set()
    for definition, source in entries:
        key = agent_view_record_key(definition.agent_name)
        seen.add(key)
        payload: dict[str, Any] = {
            **definition.model_dump(mode="json"),
            "source": source,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        row = existing.get(key)
        if (
            row is not None
            and row.deleted_at is None
            and isinstance(row.payload, dict)
            and _comparable(row.payload) == _comparable(payload)
        ):
            counts["unchanged"] += 1
            continue
        try:
            await upsert_runtime_config(
                kind=KIND_AGENT_VIEW,
                record_key=key,
                payload=payload,
                scope_key=f"agent:{definition.agent_name}",
            )
            counts["written"] += 1
        except Exception:
            logger.exception("agent view sync failed: agent=%s", definition.agent_name)
    for key, row in existing.items():
        if key in seen or row.deleted_at is not None:
            continue
        if await retire_runtime_config(KIND_AGENT_VIEW, key):
            counts["retired"] += 1
    logger.info(
        "agent views synced: written=%d unchanged=%d retired=%d total=%d",
        counts["written"],
        counts["unchanged"],
        counts["retired"],
        len(seen),
    )
    return counts


async def list_agent_view_definitions() -> list[BenchmarkDefinition]:
    """Read agent-view rows from the DB (empty when there are no rows or the table is down).

    A single payload that fails to parse is skipped and does not take down the whole catalog.
    """
    try:
        rows = await RuntimeConfigRecord.filter(
            kind=KIND_AGENT_VIEW,
            deleted_at__isnull=True,
        )
    except Exception:
        logger.warning("agent view table unavailable, falling back to assembly")
        return []
    definitions: list[BenchmarkDefinition] = []
    for row in rows:
        if not isinstance(row.payload, dict):
            continue
        payload = {k: v for k, v in row.payload.items() if k not in _MARKER_KEYS}
        try:
            definitions.append(BenchmarkDefinition.model_validate(payload))
        except Exception:
            logger.warning("agent view row parse failed: record_key=%s", row.record_key)
    return sorted(definitions, key=lambda item: item.name)


__all__ = ["list_agent_view_definitions", "sync_agent_views"]
