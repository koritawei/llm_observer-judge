"""Read-only lookups for registration docs / T+1 report items (no console / RemoteConfig reverse deps).

For lightweight assemblers such as ``daily_report_catalog`` to import at top level without
pulling in the heavy graph and import cycles of ``benchmark_console`` / ``trace_report_config``.
"""

from typing import Any

from benchmark.records.models.runtime_config_record import (
    KIND_BENCHMARK_AGENT,
    KIND_TRACE_REPORT,
    RuntimeConfigRecord,
    benchmark_agent_record_key,
)


def _dict_payload(payload: object) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


async def find_agent_scope_by_registration_id(agent_id: str) -> str | None:
    """Read ``agentScope`` by registration id; ``None`` when missing or blank."""
    key = agent_id.strip()
    if not key:
        return None
    row = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        record_key=benchmark_agent_record_key(key),
        deleted_at__isnull=True,
    ).first()
    if row is None:
        return None
    payload = _dict_payload(row.payload)
    scope = payload.get("agent_scope") or payload.get("agentScope")
    if isinstance(scope, str) and scope.strip():
        return scope.strip()
    return None


async def list_active_trace_report_registration_ids() -> list[str | None]:
    """``agent_registration_id`` on active ``kind=trace_report`` rows (nullable).

    ``None`` / an empty string means the item is unbound; callers use the legacy default scope.
    """
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_TRACE_REPORT,
        deleted_at__isnull=True,
    )
    result: list[str | None] = []
    for row in rows:
        payload = _dict_payload(row.payload)
        raw = payload.get("agent_registration_id")
        if raw is None:
            result.append(None)
            continue
        text = str(raw).strip()
        result.append(text or None)
    return result


__all__ = [
    "find_agent_scope_by_registration_id",
    "list_active_trace_report_registration_ids",
]
