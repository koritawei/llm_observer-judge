"""Database configuration for agent-eval, built on libs/data_models.

Usage:
    from benchmark.records.database import create_database_config
    from benchmark.libs.models.database import register_database, close_database

    config = create_database_config(db_url="sqlite:///./data.db")
    await register_database(app, config)
"""

from benchmark.libs.models.database import DatabaseConfig


def _normalize_db_scheme(db_url: str) -> str:
    """Normalize a standard ``postgresql[+driver]://`` scheme to the ``postgres://`` tortoise knows.

    tortoise-orm registers only 3 PostgreSQL schemes:

    - ``postgres://`` — default asyncpg
    - ``asyncpg://`` — explicit asyncpg
    - ``psycopg://`` — explicit psycopg3

    It does not accept ``postgresql://`` or the SQLAlchemy-style ``postgresql+asyncpg://`` /
    ``postgresql+psycopg://`` (that raises ``ConfigurationError: Unknown DB scheme: postgresql``),
    and the driver cannot be selected with a URL suffix — tortoise decides internally based on
    whether importing ``psycopg`` succeeds at runtime.

    ``AGENT_EVAL_DATABASE_URL`` injected by ops often uses the standard spelling. This normalizes
    the prefix once so all three standard spellings start on the default asyncpg path.
    """
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://", "postgresql://"):
        if db_url.startswith(prefix):
            return "postgres://" + db_url[len(prefix) :]
    return db_url


def create_database_config(
    *,
    db_url: str,
    environment: str = "production",
) -> DatabaseConfig:
    """Build the agent-eval DatabaseConfig.

    Register every TraceRecord / Benchmark ORM model module on ``DatabaseConfig``.
    ``data_models.database.register_database()`` initializes them at application startup.

    Args:
        db_url: database URL (for example ``sqlite:///./data.db`` or a PostgreSQL URL)
        environment: runtime environment id (``production`` / ``test``, and so on);
                     it affects schema generation and initialization
    """

    return DatabaseConfig(
        environment=environment,
        models=[
            "benchmark.records.models.trace_record",
            "benchmark.records.models.benchmark_run",
            "benchmark.records.models.benchmark_run_item",
            "benchmark.records.models.benchmark_report_provider",
            "benchmark.records.models.benchmark_kafka_task",
            "benchmark.records.models.runtime_config_record",
            "benchmark.records.models.cluster_report_job",
        ],
        observers=lambda: ["benchmark.records.observers"],
        db_url=_normalize_db_scheme(db_url),
    )


__all__ = ["create_database_config"]
