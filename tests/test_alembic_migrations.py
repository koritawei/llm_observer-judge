"""STEP-13: Alembic upgrade/downgrade + schema_mode selection."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from llmobserverjudge.shared.db import (
    ensure_schema,
    is_postgres_url,
    resolve_schema_mode,
)
from llmobserverjudge.shared.migrate import upgrade_head, downgrade_base
from llmobserverjudge.shared.persistence import create_persistence


def test_resolve_schema_mode_defaults() -> None:
    assert resolve_schema_mode(database_url="sqlite+aiosqlite:///./x.db") == "create_all"
    assert (
        resolve_schema_mode(database_url="postgresql+asyncpg://u:p@h/db") == "alembic"
    )
    assert resolve_schema_mode("none") == "none"
    assert is_postgres_url("postgresql+asyncpg://localhost/llmjudge")


@pytest.mark.asyncio
async def test_alembic_upgrade_downgrade_sqlite(tmp_path: Path) -> None:
    db = tmp_path / "migrate.db"
    url = f"sqlite+aiosqlite:///{db}"
    upgrade_head(url)

    engine = create_async_engine(url)
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
    assert "pending_eval_jobs" in tables
    assert "human_annotations" in tables
    assert "alembic_version" in tables
    async with engine.connect() as conn:
        ver = (await conn.execute(text("select version_num from alembic_version"))).scalar_one()
    assert ver == "c3d9e2f5a7b1"
    assert "spans" in tables
    assert "eval_results" in tables
    assert "prompt_experiments" in tables
    assert "attribution_checkpoints" in tables
    assert "idempotency_keys" in tables
    await engine.dispose()

    downgrade_base(url)
    engine2 = create_async_engine(url)
    async with engine2.connect() as conn:
        tables2 = await conn.run_sync(lambda c: inspect(c).get_table_names())
    assert "pending_eval_jobs" not in tables2
    await engine2.dispose()


@pytest.mark.asyncio
async def test_create_persistence_alembic_mode(tmp_path: Path) -> None:
    db = tmp_path / "persist.db"
    url = f"sqlite+aiosqlite:///{db}"
    bundle = await create_persistence("sql", database_url=url, schema_mode="alembic")
    assert bundle.schema_mode == "alembic"
    row_id = await bundle.outbox.insert_trace_closed(
        tenant_id="t1",
        trace_id="0123456789abcdef0123456789abcdef",
        trace_summary={"status": "OK"},
    )
    assert row_id
    rows = await bundle.outbox.fetch_unpublished()
    assert len(rows) == 1
    if bundle.engine:
        await bundle.engine.dispose()


@pytest.mark.asyncio
async def test_ensure_schema_none_skips(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    url = f"sqlite+aiosqlite:///{db}"
    engine = create_async_engine(url)
    mode = await ensure_schema(engine, mode="none", database_url=url)
    assert mode == "none"
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
    assert "pending_eval_jobs" not in tables
    await engine.dispose()
