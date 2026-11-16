"""Materializing the unified agent view: once synced into runtime_config_records it can be read straight from the DB.

Covers: the first write, no rewrite when nothing changed (revision does not increase), soft-delete when the assembly disappears,
and reading back a BenchmarkDefinition (derived markers stripped).
"""

from benchmark.records.models.runtime_config_record import (
    KIND_AGENT_VIEW,
    RuntimeConfigRecord,
    agent_view_record_key,
)
from benchmark.domain.agent_view_store import list_agent_view_definitions, sync_agent_views
from benchmark.runs.schema import BenchmarkDefinition


def _definition(agent_name: str, description: str = "v1") -> BenchmarkDefinition:
    return BenchmarkDefinition(
        name=agent_name,
        agent_name=agent_name,
        service_name=agent_name,
        dataset_name="",
        project_id="",
        criteria=[],
        aggregation_method="weighted_avg",
        description=description,
        evaluators=[agent_name],
    )


async def _row(agent_name: str) -> RuntimeConfigRecord | None:
    return await RuntimeConfigRecord.filter(
        kind=KIND_AGENT_VIEW,
        record_key=agent_view_record_key(agent_name),
    ).first()


async def test_sync_writes_and_reads_back() -> None:
    counts = await sync_agent_views([(_definition("search_ai_card"), "static")])
    assert counts == {"written": 1, "unchanged": 0, "retired": 0}

    row = await _row("search_ai_card")
    assert row is not None and row.deleted_at is None
    assert row.scope_key == "agent:search_ai_card"
    assert isinstance(row.payload, dict)
    assert row.payload["source"] == "static"
    assert row.payload["name"] == "search_ai_card"

    stored = await list_agent_view_definitions()
    assert [item.agent_name for item in stored] == ["search_ai_card"]
    assert stored[0].description == "v1"


async def test_list_benchmarks_prefers_db_rows() -> None:
    """The catalog API prefers materialized rows: when the DB has rows, it no longer assembles them in memory."""
    from benchmark.routes.benchmark import list_benchmarks

    await sync_agent_views([(_definition("vehicle_agent"), "evaluator")])
    definitions = await list_benchmarks()
    assert [item.agent_name for item in definitions] == ["vehicle_agent"]


async def test_sync_skips_unchanged_and_retires_missing() -> None:
    await sync_agent_views([(_definition("amap_agent"), "static")])
    first = await _row("amap_agent")
    assert first is not None

    counts = await sync_agent_views([(_definition("amap_agent"), "static")])
    assert counts["unchanged"] == 1 and counts["written"] == 0
    again = await _row("amap_agent")
    assert again is not None and again.revision == first.revision

    counts = await sync_agent_views([])
    assert counts["retired"] == 1
    retired = await _row("amap_agent")
    assert retired is not None and retired.deleted_at is not None
    assert await list_agent_view_definitions() == []
