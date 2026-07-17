"""Factory for memory vs SQL persistence drivers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from llmobserverjudge.annotation_worker.store import InMemoryAnnotationStore
from llmobserverjudge.attribution_worker.store import InMemoryAttributionStore
from llmobserverjudge.benchmark.store import InMemoryBenchmarkStore
from llmobserverjudge.eval_worker.prompt import InMemoryPromptService
from llmobserverjudge.ingest.outbox import InMemoryOutbox
from llmobserverjudge.shared.db import (
    SchemaMode,
    ensure_schema,
    make_engine,
    make_session_factory,
    resolve_database_url,
)
from llmobserverjudge.shared.sql_annotation import SqlAnnotationStore
from llmobserverjudge.shared.sql_attribution import SqlAttributionStore
from llmobserverjudge.shared.sql_benchmark import SqlBenchmarkReportStore
from llmobserverjudge.shared.sql_outbox import SqlOutboxStore
from llmobserverjudge.shared.sql_prompt import SqlPromptService

Driver = Literal["memory", "sql"]


@dataclass
class PersistenceBundle:
    driver: Driver
    engine: AsyncEngine | None
    session_factory: async_sessionmaker[AsyncSession] | None
    outbox: Any
    annotations: Any
    attributions: Any
    prompts: Any
    benchmark_reports: Any
    schema_mode: SchemaMode | None = None


async def create_persistence(
    driver: Driver | str | None = None,
    *,
    database_url: str | None = None,
    schema_mode: SchemaMode | str | None = None,
) -> PersistenceBundle:
    """Create store bundle.

    - memory: existing in-memory drivers (default)
    - sql: SQLite/Postgres via LLMJUDGE_DATABASE_URL
      (default sqlite+aiosqlite:///:memory: with StaticPool)
    - schema_mode: create_all | alembic | none
      (default: alembic for Postgres, create_all for SQLite)
    """
    chosen = (driver or os.environ.get("LLMJUDGE_PERSISTENCE") or "memory").lower()
    if chosen in ("postgres", "postgresql", "sqlite"):
        chosen = "sql"
    if chosen == "memory":
        return PersistenceBundle(
            driver="memory",
            engine=None,
            session_factory=None,
            outbox=InMemoryOutbox(),
            annotations=InMemoryAnnotationStore(),
            attributions=InMemoryAttributionStore(),
            prompts=InMemoryPromptService(),
            benchmark_reports=InMemoryBenchmarkStore(),
            schema_mode=None,
        )

    url = resolve_database_url(database_url)
    engine = make_engine(url)
    applied = await ensure_schema(engine, mode=schema_mode, database_url=url)
    sf = make_session_factory(engine)
    bench = InMemoryBenchmarkStore()
    # reports go to SQL; jobs/tasks remain memory for STEP-11 (batch tables later)
    sql_reports = SqlBenchmarkReportStore(sf)

    class _HybridBenchmarkStore(InMemoryBenchmarkStore):
        async def save_report(self, report):  # type: ignore[no-untyped-def]
            await sql_reports.save_report(report)
            await super().save_report(report)

        async def get_report(self, report_id: str):  # type: ignore[no-untyped-def]
            return await sql_reports.get_report(report_id) or await super().get_report(
                report_id
            )

        async def get_report_by_batch(self, batch_job_id: str):  # type: ignore[no-untyped-def]
            return await sql_reports.get_report_by_batch(
                batch_job_id
            ) or await super().get_report_by_batch(batch_job_id)

    return PersistenceBundle(
        driver="sql",
        engine=engine,
        session_factory=sf,
        outbox=SqlOutboxStore(sf),
        annotations=SqlAnnotationStore(sf),
        attributions=SqlAttributionStore(sf),
        prompts=SqlPromptService(sf),
        benchmark_reports=_HybridBenchmarkStore(),
        schema_mode=applied,
    )
