"""Shared pytest fixtures for agent-eval unit tests.

Provides isolated Tortoise ORM contexts per test session on per-test
SQLite files, and PostgreSQL-tagged fixtures that skip when PG env
vars are absent.

Uses libs/data_models (DatabaseConfig + register_database) for unified
ORM initialization with automatic observer registration.
"""

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from benchmark.records.database import create_database_config
from benchmark.records.models.benchmark_kafka_task import BenchmarkKafkaTask
from benchmark.records.models.benchmark_report_provider import BenchmarkReportProvider
from benchmark.records.models.benchmark_run import BenchmarkRun
from benchmark.records.models.trace_record import TraceRecord
from benchmark.services import TraceRecordService
from benchmark.libs.models.database import register_database
from tortoise import Tortoise
from tortoise.backends.base.executor import EXECUTOR_CACHE

# ── helpers ─────────────────────────────────────────────────────────────


def _per_test_db_path() -> str:
    """Per-test sqlite file path (one file per test)."""

    base = Path(os.environ.get("AGENT_EVAL_TEST_DB_DIR", "/tmp/agent_eval_test"))  # noqa: TID251
    base.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{base / f'unit-{uuid.uuid4().hex[:8]}.sqlite3'}"


def _clear_observers() -> None:
    """Clear model observers so register_database() can re-register them.

    ``ObserverMixin.set_observer()`` raises ``RuntimeError`` if an observer
    is already set.  Tests may re-initialise Tortoise multiple times (sqlite
    → PG → sqlite), so we clear observers before every ``register_database``
    call.
    """

    for model in (
        TraceRecord,
        BenchmarkRun,
        BenchmarkKafkaTask,
        BenchmarkReportProvider,
    ):
        model.clear_observer()


async def _tortoise_init_for(db_url: str) -> None:
    """Initialise Tortoise + register observers via libs/data_models."""

    _clear_observers()
    config = create_database_config(db_url=db_url, environment="test")
    await register_database(None, config)
    await Tortoise.generate_schemas(safe=True)


# ── SQLite autouse fixture ──────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _tortoise_init(request, db_url: str) -> AsyncIterator[None]:
    """Initialise Tortoise per test on the running event loop.

    Skipped when:
      * the test class is PG-tagged (``TestPg*``); the ``pg_*_repo``
        fixtures own the Tortoise lifecycle for those.
      * the test does not need ORM (non-repository tests).
    """

    cls_name = getattr(request.cls, "__name__", "") or ""
    if cls_name.startswith("TestPg"):
        try:
            yield
        finally:
            pass
        return

    requested = set(getattr(request, "fixturenames", ()) or ())
    module_path = str(request.fspath) if request.fspath else ""
    sqlite_markers = (
        "test_case.py",
        "test_trace_record.py",
        "test_evaluator_service.py",
        "services/triage_attribution/test_service.py",
        "data_model/test_observers.py",
        "test_runtime_config_record.py",
        "test_benchmark_console.py",
        "test_agent_view_store.py",
        "test_oss_upload_sequence.py",
        "test_customization",
        "test_agent_evaluation_counts.py",
        "test_agent_scope",
    )
    needs_sqlite = (
        "repo" in requested
        or "case_repo_with_jobs" in requested
        or any(marker in module_path for marker in sqlite_markers)
    )
    if not needs_sqlite:
        try:
            yield
        finally:
            pass
        return

    # Close any leftover connection from a prior test and clear the
    # executor SQL cache so sqlite/PG dialects don't cross-pollinate.
    try:
        await Tortoise.close_connections()
    except Exception:
        pass

    EXECUTOR_CACHE.clear()

    await _tortoise_init_for(db_url)
    try:
        yield
    finally:
        await Tortoise.close_connections()
        # SQLite WAL mode leaves sidecar files; clean them up.
        db_path = Path(db_url.replace("sqlite:///", ""))
        for suffix in ("", "-wal", "-shm"):
            target = db_path if suffix == "" else db_path.with_name(db_path.name + suffix)
            try:
                target.unlink()
            except FileNotFoundError:
                pass


@pytest.fixture()
def db_url() -> str:
    """Per-test database URL (a fresh sqlite file under /tmp)."""

    base = Path(os.environ.get("AGENT_EVAL_TEST_DB_DIR", "/tmp/agent_eval_test"))  # noqa: TID251
    base.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{base / f'case-{uuid.uuid4().hex[:8]}.sqlite3'}"


# ── PostgreSQL fixtures ─────────────────────────────────────────────────
# These tests are skipped when the env vars are absent or the broker is
# unreachable; in CI they run against the test Postgres cluster.


def _pg_env() -> dict[str, str] | None:
    """Resolve PostgreSQL connection params, or None if any are missing."""

    required = ("PG_HOST", "PG_USER", "PG_PASSWORD", "PG_DATABASE")
    if not all(os.environ.get(k) for k in required):  # noqa: TID251
        return None
    return {
        "host": os.environ["PG_HOST"],  # noqa: TID251
        "user": os.environ["PG_USER"],  # noqa: TID251
        "password": os.environ["PG_PASSWORD"],  # noqa: TID251
        "db": os.environ["PG_DATABASE"],  # noqa: TID251
    }


def _skip_if_no_pg() -> None:
    if _pg_env() is None:
        pytest.skip("PostgreSQL env vars not set (PG_HOST/PG_USER/PG_PASSWORD/PG_DATABASE)")


async def _reset_tortoise_for_pg() -> None:
    """Hard-reset Tortoise state before re-initialising against Postgres.

    Clears observers, closes connections, resets the app registry, and
    flushes the executor SQL cache so PG-flavoured queries are rebuilt.
    """

    _clear_observers()
    await Tortoise.close_connections()
    await Tortoise._reset_apps()
    EXECUTOR_CACHE.clear()


async def _pg_init(db_url: str) -> None:
    """Initialise Tortoise against Postgres with all models + observers."""

    config = create_database_config(db_url=db_url, environment="test")
    await register_database(None, config)
    await Tortoise.generate_schemas(safe=True)


@pytest_asyncio.fixture()
async def pg_trace_repo() -> AsyncIterator[TraceRecordService]:
    """Per-test PostgreSQL TraceRecordService (skipped when no PG env)."""

    _skip_if_no_pg()
    pg = _pg_env()
    db_url = f"postgres://{pg['user']}:{pg['password']}@{pg['host']}/{pg['db']}"
    await _reset_tortoise_for_pg()
    await _pg_init(db_url)
    await _truncate_pg_tables(["trace_pipeline_records"])
    yield TraceRecordService()
    await Tortoise.close_connections()


async def _truncate_pg_tables(tables: list[str]) -> None:
    """TRUNCATE the listed tables (CASCADE) so PG tests start clean."""

    try:
        conn = Tortoise.get_connection("default")
    except Exception:
        return
    for table in tables:
        async with conn.acquire_connection() as raw_conn:
            await raw_conn.execute(f'TRUNCATE TABLE "{table}" CASCADE')
