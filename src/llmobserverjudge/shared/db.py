"""Async SQLAlchemy engine helpers (SQLite tests / Postgres prod)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

SchemaMode = Literal["create_all", "alembic", "none"]


class Base(DeclarativeBase):
    pass


def resolve_database_url(url: str | None = None) -> str:
    return url or os.environ.get("LLMJUDGE_DATABASE_URL") or "sqlite+aiosqlite:///:memory:"


def is_postgres_url(url: str) -> bool:
    scheme = urlparse(url).scheme or ""
    return scheme.startswith("postgresql")


def resolve_schema_mode(
    mode: SchemaMode | str | None = None, *, database_url: str | None = None
) -> SchemaMode:
    """Default: alembic for Postgres, create_all for SQLite/dev."""
    if mode:
        m = str(mode).lower()
        if m in ("create_all", "alembic", "none"):
            return m  # type: ignore[return-value]
        raise ValueError(f"unknown schema mode: {mode}")
    env = (os.environ.get("LLMJUDGE_SCHEMA_MODE") or "").lower()
    if env in ("create_all", "alembic", "none"):
        return env  # type: ignore[return-value]
    url = resolve_database_url(database_url)
    return "alembic" if is_postgres_url(url) else "create_all"


def make_engine(url: str | None = None, *, echo: bool = False) -> AsyncEngine:
    """Create engine. Default: in-memory SQLite with StaticPool (shared)."""
    db_url = resolve_database_url(url)
    kwargs: dict = {"echo": echo}
    if db_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in db_url:
            kwargs["poolclass"] = StaticPool
    elif is_postgres_url(db_url):
        # Production-friendly pool defaults (override via URL query / engine opts later)
        kwargs.setdefault("pool_pre_ping", True)
    return create_async_engine(db_url, **kwargs)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Dev shortcut: metadata.create_all (not for production Postgres)."""
    from llmobserverjudge.shared import tables  # noqa: F401 — register mappers

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def ensure_schema(
    engine: AsyncEngine,
    *,
    mode: SchemaMode | str | None = None,
    database_url: str | None = None,
) -> SchemaMode:
    """Apply schema via create_all or Alembic upgrade head."""
    url = resolve_database_url(database_url or str(engine.url))
    chosen = resolve_schema_mode(mode, database_url=url)
    if chosen == "none":
        return chosen
    if chosen == "alembic":
        if ":memory:" in url:
            logger.warning(
                "alembic cannot target sqlite :memory: reliably; falling back to create_all"
            )
            await init_db(engine)
            return "create_all"
        from llmobserverjudge.shared.migrate import upgrade_head

        # Alembic manages its own engine; dispose caller's pool not required
        upgrade_head(url)
        logger.info("schema_applied mode=alembic url_scheme=%s", urlparse(url).scheme)
        return chosen
    await init_db(engine)
    logger.info("schema_applied mode=create_all")
    return chosen


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
