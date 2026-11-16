#!/usr/bin/env python3
"""Ops: reset the PG database schema — drop every table and recreate them.

Discover table names from the data_model model definitions, run DROP TABLE IF EXISTS ... CASCADE,
then rebuild every table / index / constraint with Tortoise.generate_schemas().

Dry-run by default; it really runs only with ``--execute``.

Usage (prefer the wrapper script; production prepends agentgraph to PYTHONPATH)::

  bash scripts/reset_schema.sh
  bash scripts/reset_schema.sh --execute
"""

import argparse
import asyncio
from typing import Any

import _bootstrap  # sys.path/.env/disable-MQ side effects must run before app.*
from benchmark.settings.setting import Settings
from benchmark.records.database import create_database_config
from tortoise import Tortoise

del _bootstrap  # imported only for side effects: already executed, release the module name


async def _discover_table_names() -> list[str]:
    """Discover every table name from the Tortoise model registry."""
    tables: list[str] = []
    for _app_name, models in Tortoise.apps.items():
        for _model_name, model_cls in models.items():
            meta = getattr(model_cls, "_meta", None)
            if meta is None:
                continue
            # skip abstract base classes
            if getattr(meta, "abstract", False):
                continue
            table = getattr(meta, "db_table", None)
            if table:
                tables.append(table)
    return tables


async def _drop_tables(conn: Any, tables: list[str], *, execute: bool) -> int:
    """DROP TABLE IF EXISTS ... CASCADE, one table at a time."""
    dropped = 0
    for table in tables:
        print(f"  DROP TABLE IF EXISTS {table} CASCADE;")
        if execute:
            await conn.execute_script(f'DROP TABLE IF EXISTS "{table}" CASCADE;')
            dropped += 1
    return dropped


async def _amain(args: argparse.Namespace) -> int:
    settings = Settings()
    db_url = settings.database_url
    if "sqlite" in db_url:
        print(f"DB: {db_url}")
    else:
        # hide the password
        safe = db_url.split("@")[-1] if "@" in db_url else db_url
        print(f"DB: ...@{safe}")

    config = create_database_config(
        db_url=db_url,
        environment=settings.environment,
    )

    # Initialize Tortoise ORM (do not register observers; schema operations only)
    await Tortoise.init(config=config.build_config())

    try:
        # 1. Discover table names
        tables = await _discover_table_names()
        if not tables:
            print("No tables found in model registry.")
            return 1

        print(f"\nDiscovered {len(tables)} table(s):")
        for t in sorted(tables):
            print(f"  - {t}")

        # 2. Drop
        print(f"\n--- Drop tables ({'EXECUTE' if args.execute else 'DRY-RUN'}) ---")
        conn = Tortoise.get_connection("default")
        dropped = await _drop_tables(conn, tables, execute=args.execute)
        print(f"Dropped: {dropped}")

        # 3. Create
        print(f"\n--- Generate schemas ({'EXECUTE' if args.execute else 'DRY-RUN'}) ---")
        if args.execute:
            await Tortoise.generate_schemas(safe=True)
            print("Schemas generated successfully.")
        else:
            print("[dry-run] skip generate_schemas")

        if not args.execute:
            print("\n[dry-run] no changes made. Add --execute to apply.")
        else:
            print("\n✅ Done.")
        return 0
    finally:
        await Tortoise.close_connections()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset PG schema: drop all tables and recreate from data_model",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually execute (default: dry-run)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
